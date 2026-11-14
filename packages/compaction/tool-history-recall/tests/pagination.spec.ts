import { Buffer } from 'node:buffer'
import { describe, expect, it } from 'vitest'
import { SessionSeq } from '@deepseek-ai/dsh-session'
import { codePointLength, decodeCursor, encodeCursor, renderPage } from '../src/pagination.ts'
import type { CompactionCheckpoint, RecallCursor, RenderedRecord } from '../src/types.ts'

/**
 * Unit coverage for the exported pagination primitives. Checkpoint transcripts
 * are reached end to end through the tool suites; this suite pins the guards
 * that only the exported functions themselves can reach: deployment bounds that
 * no tool config can produce, cursor coordinates that decoded cursors can carry
 * but the encoder never emits, and the loader's contiguous-batch contract.
 */

const CHECKPOINT: CompactionCheckpoint = {
  summarySeq: SessionSeq(4),
  shadowedRange: { start: SessionSeq(1), end: SessionSeq(2) },
  shadowedSeqs: [SessionSeq(1), SessionSeq(2)],
  shadowedTokenCount: 10,
  replacementSeq: SessionSeq(5),
  state: 'completed',
  superseded: false,
}

function record(seq: number, text: string, label = 'user'): RenderedRecord {
  return { seq: SessionSeq(seq), originalSeq: SessionSeq(seq), label, text }
}

/** The production loader shape: contiguous batches that always advance. */
function contiguousLoader(records: readonly RenderedRecord[]) {
  return (fromIndex: number) => {
    const batch = records.slice(fromIndex, fromIndex + 20)
    return { records: batch, nextIndex: fromIndex + batch.length }
  }
}

interface RenderOverrides {
  seqs?: readonly SessionSeq[]
  cursor?: RecallCursor
  maxPageBytes?: number
  records?: readonly RenderedRecord[]
  load?: (fromIndex: number) => { records: readonly RenderedRecord[]; nextIndex: number }
}

function render(overrides: RenderOverrides = {}) {
  const records = overrides.records ?? []
  return renderPage({
    checkpoint: CHECKPOINT,
    seqs: overrides.seqs ?? CHECKPOINT.shadowedSeqs,
    cursor: overrides.cursor,
    maxPageBytes: overrides.maxPageBytes ?? 4_096,
    load: overrides.load ?? contiguousLoader(records),
  })
}

function opaque(payload: unknown): string {
  return Buffer.from(JSON.stringify(payload), 'utf8').toString('base64url')
}

describe('byte-bounded transcript paging', () => {
  it('rejects a cursor naming an event outside the checkpoint span', () => {
    expect(() => render({ cursor: { v: 1, c: SessionSeq(4), s: SessionSeq(99), o: 0 } }))
      .toThrow(/is not inside checkpoint c4/u)
  })

  it('rejects a cursor with a negative or fractional character offset', () => {
    for (const offset of [-1, 1.5]) {
      expect(() => render({ cursor: { v: 1, c: SessionSeq(4), s: SessionSeq(1), o: offset } }))
        .toThrow(/character offset must be a non-negative integer/u)
    }
  })

  it('rejects a page budget too small to hold framing plus one character', () => {
    expect(() => render({ maxPageBytes: 256 }))
      .toThrow(/too small for transcript framing/u)
  })

  it('rejects a page budget that cannot hold one record prefix plus a content slice', () => {
    const records = [record(1, 'z'.repeat(400), 'x'.repeat(200))]
    expect(() => render({ records, load: contiguousLoader(records), maxPageBytes: 512 }))
      .toThrow(/too small for transcript framing/u)
  })

  it('continues at the next event when the remaining budget cannot hold a slice', () => {
    const first = record(1, 'tiny')
    const second = record(2, 'z'.repeat(400), 'x'.repeat(100))
    const { text, next } = render({
      records: [first, second],
      load: contiguousLoader([first, second]),
      maxPageBytes: 520,
    })
    expect(text).toContain('[#1 user]\ntiny')
    expect(text).not.toContain('[#2')
    expect(next).toEqual({ v: 1, c: SessionSeq(4), s: SessionSeq(2), o: 0 })
  })

  it('rejects a page loader that makes no progress', () => {
    expect(() => render({ load: fromIndex => ({ records: [], nextIndex: fromIndex }) }))
      .toThrow(/did not advance/u)
  })

  it('rejects a page loader batch that omits the record at its own index', () => {
    const records = [record(1, 'short record')]
    expect(() => render({
      records,
      load: fromIndex => ({ records: fromIndex === 0 ? records : [], nextIndex: fromIndex + 20 }),
    })).toThrow(/omitted the record/u)
  })

  it('rejects a loader record that does not match the checkpoint span', () => {
    const records = [record(7, 'wrong event')]
    expect(() => render({ records, load: contiguousLoader(records) }))
      .toThrow(/does not match the checkpoint's span/u)
  })

  it('never splits a surrogate pair at the byte bound', () => {
    const records = [record(1, '😀'.repeat(300))]
    const { text, next } = render({ seqs: [SessionSeq(1)], records, load: contiguousLoader(records), maxPageBytes: 1_024 })
    expect(text).not.toContain('\uFFFD')
    expect(next?.s).toBe(SessionSeq(1))
    expect(next?.o ?? 0).toBeGreaterThan(0)
    expect(text).toContain('😀')
  })
})

describe('opaque resume cursors', () => {
  it('round-trips a resume position through its encoding', () => {
    const cursor: RecallCursor = { v: 1, c: SessionSeq(4), s: SessionSeq(1), o: 7 }
    expect(decodeCursor(encodeCursor(cursor))).toEqual(cursor)
  })

  it('rejects a payload that is not a base64url JSON object', () => {
    expect(() => decodeCursor('-')).toThrow(/not a base64url-encoded JSON cursor/u)
    expect(() => decodeCursor(opaque(null))).toThrow(/must be an object/u)
    expect(() => decodeCursor(opaque([1, 2]))).toThrow(/must be an object/u)
    expect(() => decodeCursor('a'.repeat(4_097))).toThrow(/longer than 4096 characters/u)
  })

  it('rejects a payload naming an unsupported version', () => {
    expect(() => decodeCursor(opaque({ v: 2, c: 1, s: 1, o: 0 }))).toThrow(/unsupported cursor version/u)
  })

  it('rejects a payload whose coordinates are not non-negative safe integers', () => {
    expect(() => decodeCursor(opaque({ v: 1, c: '1', s: 1, o: 0 }))).toThrow(/checkpoint seq must be/u)
    expect(() => decodeCursor(opaque({ v: 1, c: 1, s: 1.5, o: 0 }))).toThrow(/event seq must be/u)
    expect(() => decodeCursor(opaque({ v: 1, c: 1, s: 1, o: -1 }))).toThrow(/character offset must be/u)
  })

  it('counts code points rather than UTF-16 units', () => {
    expect(codePointLength('a😀b')).toBe(3)
    expect(codePointLength('')).toBe(0)
  })
})
