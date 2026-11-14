/**
 * Session-scoped access to saved full-output evidence named by a tool result.
 *
 * A shell result may contain `[output truncated; full output: <path>]`. The
 * log only holds the retained view, but the spill file holds the original. This
 * module lets history_read consume that file only when the exact path is named
 * by a tool result in the caller's own session and the file resolves under the
 * subprocess spill temp root.
 *
 * @module @deepseek-ai/dsh-tool-history-recall/evidence
 */

import { readFile, realpath, stat } from 'node:fs/promises'
import { basename, dirname, isAbsolute, resolve, sep } from 'node:path'
import { tmpdir } from 'node:os'
import type { SessionId } from '@deepseek-ai/dsh-session'
import { HarnessError } from '@deepseek-ai/dsh-llm'
import type {
  RecallQuery,
  ResolvedHistoryRecallConfig,
} from './types.ts'

const EVIDENCE_PATTERN = /\[output truncated; full output: ([^\]\r\n]+)\]/iu

/**
 * Collect exact spill paths named by tool results in one session log.
 * @param events - raw session events.
 * @returns paths in the order first seen.
 */
export function evidencePaths(events: readonly { type: string; data?: unknown }[]): Set<string> {
  const paths = new Set<string>()
  for (const event of events) {
    if (event.type !== 'tool/result') continue
    const message = (event.data as { message?: { content?: readonly { type?: string; text?: string }[] } } | undefined)?.message
    if (message?.content === undefined) continue
    for (const block of message.content) {
      if (block.type !== 'text' || typeof block.text !== 'string') continue
      const match = EVIDENCE_PATTERN.exec(block.text)
      if (match?.[1] !== undefined) paths.add(match[1].trim())
    }
  }
  return paths
}

/**
 * Read one bounded page of a saved tool-output evidence file.
 * @param query - caller session reader.
 * @param sessionId - caller session id.
 * @param requestedPath - exact spill path named by a tool result.
 * @param offset - character offset into the saved text.
 * @param config - page and evidence bounds.
 * @param signal - optional cancellation.
 * @returns bounded evidence page.
 */
export async function readEvidence(
  query: RecallQuery,
  sessionId: SessionId,
  requestedPath: string,
  offset: number,
  config: ResolvedHistoryRecallConfig,
  signal?: AbortSignal,
): Promise<string> {
  signal?.throwIfAborted()
  if (!Number.isSafeInteger(offset) || offset < 0) {
    throw new HarnessError('history_read evidenceOffset must be a non-negative integer', 'HISTORY_EVIDENCE_INVALID')
  }
  const text = await loadEvidenceText(query, sessionId, requestedPath, config, signal)
  const characters = Array.from(text)
  if (offset > characters.length) {
    throw new HarnessError(
      `history_read evidenceOffset ${String(offset)} exceeds the saved text length ${String(characters.length)}`,
      'HISTORY_EVIDENCE_INVALID',
    )
  }
  const page = sliceByBytes(characters.slice(offset), config.maxPageBytes)
  const next = offset + page.codePoints
  const more = next < characters.length
  const header = `Saved tool output evidence (${String(characters.length)} characters) at offset ${String(offset)}:\n\n`
  const footer = more
    ? `\n\nContinues at offset ${String(next)}; call history_read with evidence and evidenceOffset=${String(next)}.`
    : '\n\nEnd of saved evidence.'
  return header + page.text + footer
}

/**
 * Search one saved tool-output evidence file for a literal phrase.
 * @param query - caller session reader.
 * @param sessionId - caller session id.
 * @param requestedPath - exact spill path named by a tool result.
 * @param text - literal phrase.
 * @param config - search and evidence bounds.
 * @param signal - optional cancellation.
 * @returns matching snippets or a complete-miss notice.
 */
export async function searchEvidence(
  query: RecallQuery,
  sessionId: SessionId,
  requestedPath: string,
  text: string,
  config: ResolvedHistoryRecallConfig,
  signal?: AbortSignal,
): Promise<string> {
  const trimmed = text.trim()
  if (trimmed.length === 0 || trimmed.length > 2_048) {
    throw new HarnessError('history_search evidence query must be 1..2048 characters', 'HISTORY_SEARCH_INVALID_QUERY')
  }
  const body = await loadEvidenceText(query, sessionId, requestedPath, config, signal)
  const needle = trimmed.toLowerCase()
  const lower = body.toLowerCase()
  const hits: string[] = []
  let cursor = 0
  while (hits.length < config.maxSearchHits) {
    const index = lower.indexOf(needle, cursor)
    if (index === -1) break
    const start = Math.max(0, index - 120)
    const end = Math.min(body.length, index + needle.length + 120)
    hits.push(`- offset ${String(index)}: …${body.slice(start, end).replace(/\s+/gu, ' ').trim()}…`)
    cursor = index + needle.length
  }
  const header = `Saved tool output evidence search for "${trimmed}" (${String(Array.from(body).length)} characters): ${String(hits.length)} match(es).`
  return hits.length === 0
    ? `${header}\n\nNo literal match in the saved evidence.`
    : `${header}\n\n${hits.join('\n')}`
}

/**
 * Scan one saved evidence text for a literal phrase.
 * @param path - original spill path.
 * @param body - saved text.
 * @param text - literal phrase.
 * @param maxHits - maximum snippets.
 * @returns formatted hit lines.
 */
export function scanEvidenceText(
  path: string,
  body: string,
  text: string,
  maxHits: number,
): string[] {
  const needle = text.toLowerCase()
  const lower = body.toLowerCase()
  const hits: string[] = []
  let cursor = 0
  while (hits.length < maxHits) {
    const index = lower.indexOf(needle, cursor)
    if (index === -1) break
    const start = Math.max(0, index - 120)
    const end = Math.min(body.length, index + needle.length + 120)
    hits.push(`- evidence ${path} offset ${String(index)}: …${body.slice(start, end).replace(/\s+/gu, ' ').trim()}…`)
    cursor = index + needle.length
  }
  return hits
}

/**
 * Read a saved evidence file without another session scan.
 * Callers must first establish that the path is named by their session.
 * @param path - spill path named by a tool result.
 * @param config - evidence bounds.
 * @returns saved text.
 */
export async function readEvidenceText(path: string, config: ResolvedHistoryRecallConfig): Promise<string> {
  const file = await validateSpillPath(path)
  const info = await stat(file)
  if (!info.isFile()) throw missing(path)
  if (info.size > config.maxEvidenceBytes) {
    throw new HarnessError(
      `history_read evidence file is ${String(info.size)} bytes, above maxEvidenceBytes ${String(config.maxEvidenceBytes)}; narrow the source or raise the bound`,
      'HISTORY_EVIDENCE_TOO_LARGE',
    )
  }
  return readFile(file, 'utf8')
}

async function loadEvidenceText(
  query: RecallQuery,
  sessionId: SessionId,
  requestedPath: string,
  config: ResolvedHistoryRecallConfig,
  signal?: AbortSignal,
): Promise<string> {
  signal?.throwIfAborted()
  const log = await query.readRawLog(sessionId, signal)
  const allowed = evidencePaths(log.events)
  if (!allowed.has(requestedPath)) {
    throw new HarnessError(
      'history_read evidence path is not named by a tool result in this session',
      'HISTORY_EVIDENCE_INVALID',
    )
  }
  const file = await validateSpillPath(requestedPath)
  signal?.throwIfAborted()
  const info = await stat(file)
  if (!info.isFile()) throw missing(requestedPath)
  if (info.size > config.maxEvidenceBytes) {
    throw new HarnessError(
      `history_read evidence file is ${String(info.size)} bytes, above maxEvidenceBytes ${String(config.maxEvidenceBytes)}; narrow the source or raise the bound`,
      'HISTORY_EVIDENCE_TOO_LARGE',
    )
  }
  const text = await readFile(file, 'utf8')
  signal?.throwIfAborted()
  return text
}

/** Resolve and constrain one shell spill path to the private subprocess temp root. */
async function validateSpillPath(path: string): Promise<string> {
  if (!isAbsolute(path)) throw invalid(path)
  let real: string
  let root: string
  try {
    real = await realpath(resolve(path))
    root = await realpath(tmpdir())
  } catch {
    throw missing(path)
  }
  const parent = dirname(real)
  if (!real.startsWith(`${root}${sep}`) || !basename(parent).startsWith('dsh-subprocess-')) {
    throw new HarnessError(
      'history_read evidence path is not a session spill artifact under the subprocess temp root',
      'HISTORY_EVIDENCE_INVALID',
    )
  }
  return real
}

function sliceByBytes(characters: readonly string[], maxBytes: number): { text: string; codePoints: number } {
  let bytes = 0
  let count = 0
  let text = ''
  for (const character of characters) {
    const size = Buffer.byteLength(character, 'utf8')
    if (bytes + size > maxBytes) break
    bytes += size
    count += 1
    text += character
  }
  return { text, codePoints: count }
}

function missing(path: string): HarnessError {
  return new HarnessError(
    `history_read evidence file is missing or unreadable: ${path}`,
    'HISTORY_EVIDENCE_MISSING',
  )
}

function invalid(path: string): HarnessError {
  return new HarnessError(
    `history_read evidence path is invalid: ${path}`,
    'HISTORY_EVIDENCE_INVALID',
  )
}
