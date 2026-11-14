/**
 * Byte-bounded transcript pagination and opaque resume cursors.
 *
 * @module @deepseek-ai/dsh-tool-history-recall/pagination
 */

import { Buffer } from 'node:buffer'
import type { SessionSeq } from '@deepseek-ai/dsh-session'
import { SessionSeq as seqOf } from '@deepseek-ai/dsh-session'
import { cursorInvalid, pageBudgetTooSmall } from './errors.ts'
import type {
  CompactionCheckpoint,
  RenderedRecord,
  RecallCursor,
} from './types.ts'

/** Version tag embedded in every opaque cursor. */
const CURSOR_VERSION = 1

/** Bytes reserved for the continuation footer and cursor framing. */
const FOOTER_RESERVE_BYTES = 384

/** Smallest text slice a page will emit before continuing inside one event. */
const MIN_CONTENT_SLICE_BYTES = 16

interface PageInput {
  readonly checkpoint: CompactionCheckpoint
  readonly seqs: readonly SessionSeq[]
  readonly cursor: RecallCursor | undefined
  readonly maxPageBytes: number
  readonly load: (fromIndex: number) => { records: readonly RenderedRecord[]; nextIndex: number }
}

/**
 * Render one bounded page of a checkpoint's shadowed transcript.
 * @param input - checkpoint, optional cursor, loader, and deployment bounds.
 * @returns page text and the cursor that resumes at the next unread character.
 */
export function renderPage(input: PageInput): { text: string; next: RecallCursor | undefined } {
  const { checkpoint, seqs, cursor, maxPageBytes } = input
  const startIndex = cursor === undefined ? 0 : seqs.indexOf(cursor.s)
  if (startIndex === -1) {
    throw cursorInvalid(`event #${String(cursor?.s)} is not inside checkpoint c${String(checkpoint.summarySeq)}`)
  }
  let index = startIndex
  let offset = cursor?.o ?? 0
  if (offset < 0 || !Number.isSafeInteger(offset)) throw cursorInvalid('character offset must be a non-negative integer')

  const header = `Compacted checkpoint c${String(checkpoint.summarySeq)}: shadows #${String(checkpoint.shadowedRange.start)}–#${String(checkpoint.shadowedRange.end)}, `
    + `${String(seqs.length)} events, ~${String(checkpoint.shadowedTokenCount)} tokens`
    + `${checkpoint.superseded ? ', superseded by a later checkpoint' : ''}.\n\n`
  const headerBytes = Buffer.byteLength(header, 'utf8')
  if (headerBytes + FOOTER_RESERVE_BYTES + MIN_CONTENT_SLICE_BYTES > maxPageBytes) {
    throw pageBudgetTooSmall(maxPageBytes)
  }
  let remaining = maxPageBytes - headerBytes - FOOTER_RESERVE_BYTES
  let body = ''
  let next: RecallCursor | undefined
  let loadedIndex = index
  let chunk: readonly RenderedRecord[] = []

  while (index < seqs.length) {
    if (index >= loadedIndex + chunk.length) {
      const loaded = input.load(index)
      loadedIndex = index
      chunk = loaded.records
      if (loaded.nextIndex <= index) break
    }
    const record = chunk[index - loadedIndex]
    if (record === undefined) break
    if (record.seq !== seqs[index]) {
      throw cursorInvalid(`event #${String(record.seq)} does not match the checkpoint's span at #${String(seqs[index])}`)
    }
    const recordLength = codePointLength(record.text)
    if (offset > recordLength) {
      throw cursorInvalid(`character offset ${String(offset)} exceeds event #${String(record.seq)} length ${String(recordLength)}`)
    }
    if (offset === recordLength) {
      index += 1
      offset = 0
      continue
    }
    const prefix = offset === 0
      ? `[#${String(record.seq)} ${record.label}]\n`
      : `[#${String(record.seq)} ${record.label} · continues]\n`
    const text = sliceCodePoints(record.text, offset)
    const separator = body === '' ? '' : '\n\n'
    const candidateBytes = Buffer.byteLength(separator + prefix, 'utf8') + Buffer.byteLength(text, 'utf8')
    if (candidateBytes <= remaining) {
      body += separator + prefix + text
      remaining -= candidateBytes
      index += 1
      offset = 0
      continue
    }
    const available = remaining - Buffer.byteLength(separator + prefix, 'utf8')
    if (available < MIN_CONTENT_SLICE_BYTES) {
      if (body === '') throw pageBudgetTooSmall(maxPageBytes)
      next = { v: 1, c: checkpoint.summarySeq, s: record.seq, o: offset }
      break
    }
    const slice = truncateToBytes(text, available)
    if (slice.text === '') {
      if (body === '') throw pageBudgetTooSmall(maxPageBytes)
      next = { v: 1, c: checkpoint.summarySeq, s: record.seq, o: offset }
      break
    }
    body += separator + prefix + slice.text
    next = {
      v: 1,
      c: checkpoint.summarySeq,
      s: record.seq,
      o: offset + slice.codePoints,
    }
    break
  }

  if (next === undefined && index < seqs.length) {
    const seq = seqs[index]
    if (seq !== undefined) next = { v: 1, c: checkpoint.summarySeq, s: seq, o: 0 }
  }
  const footer = next === undefined
    ? `\n\nEnd of checkpoint c${String(checkpoint.summarySeq)}.`
    : `\n\nContinues at event #${String(next.s)}, character ${String(next.o)}. Pass checkpoint ${String(next.c)} with cursor ${encodeCursor(next)} to read on.`
  const text = header + body + footer
  if (Buffer.byteLength(text, 'utf8') > maxPageBytes) {
    throw pageBudgetTooSmall(maxPageBytes)
  }
  return { text, next }
}

/**
 * Encode one resume position as an opaque base64url cursor.
 * @param cursor - validated checkpoint, event, and character offset.
 * @returns URL-safe cursor text.
 */
export function encodeCursor(cursor: RecallCursor): string {
  return Buffer.from(JSON.stringify(cursor), 'utf8').toString('base64url')
}

/**
 * Decode and validate one model-supplied resume cursor.
 * @param value - opaque cursor text previously returned by {@link encodeCursor}.
 * @returns the resume position.
 * @throws when the cursor is not this format or names invalid coordinates.
 */
export function decodeCursor(value: string): RecallCursor {
  if (value.length > 4_096) throw cursorInvalid('cursor text is longer than 4096 characters')
  let parsed: unknown
  try {
    parsed = JSON.parse(Buffer.from(value, 'base64url').toString('utf8'))
  } catch {
    throw cursorInvalid('not a base64url-encoded JSON cursor')
  }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    throw cursorInvalid('cursor payload must be an object')
  }
  const record = parsed as Record<string, unknown>
  if (record['v'] !== CURSOR_VERSION) throw cursorInvalid(`unsupported cursor version ${String(record['v'])}`)
  const checkpoint = record['c']
  const seq = record['s']
  const offset = record['o']
  if (typeof checkpoint !== 'number' || !Number.isSafeInteger(checkpoint) || checkpoint < 0) {
    throw cursorInvalid('checkpoint seq must be a non-negative safe integer')
  }
  if (typeof seq !== 'number' || !Number.isSafeInteger(seq) || seq < 0) {
    throw cursorInvalid('event seq must be a non-negative safe integer')
  }
  if (typeof offset !== 'number' || !Number.isSafeInteger(offset) || offset < 0) {
    throw cursorInvalid('character offset must be a non-negative safe integer')
  }
  return {
    v: CURSOR_VERSION,
    c: seqOf(checkpoint),
    s: seqOf(seq),
    o: offset,
  }
}

/** Number of Unicode code points in one string. */
export function codePointLength(value: string): number {
  let length = 0
  for (const _character of value) length += 1
  return length
}

/** Slice by Unicode code point, never splitting a surrogate pair. */
function sliceCodePoints(value: string, start: number): string {
  if (start === 0) return value
  let index = 0
  let output = ''
  for (const character of value) {
    if (index >= start) output += character
    index += 1
  }
  return output
}

/** Truncate to at most the given UTF-8 byte count without splitting one code point. */
function truncateToBytes(value: string, maxBytes: number): { text: string; codePoints: number } {
  let bytes = 0
  let codePoints = 0
  let output = ''
  for (const character of value) {
    const characterBytes = Buffer.byteLength(character, 'utf8')
    if (bytes + characterBytes > maxBytes) break
    bytes += characterBytes
    codePoints += 1
    output += character
  }
  return { text: output, codePoints }
}
