/**
 * Turn a checkpoint's shadowed seqs into rendered transcript records through
 * bounded session-query reads.
 *
 * @module @deepseek-ai/dsh-tool-history-recall/records
 */

import type { SessionEvent, SessionId } from '@deepseek-ai/dsh-session'
import { SessionQueryError } from '@deepseek-ai/dsh-session-query'
import { readEvents } from './checkpoints.ts'
import { evidenceMissing } from './errors.ts'
import { prunedSourceSeq, renderRecord } from './render.ts'
import type { CompactionCheckpoint, RecallQuery, RenderedRecord } from './types.ts'

/** Records loaded per bounded range request. */
export const RECORD_BATCH = 20

/**
 * Load and render the next records of one checkpoint.
 * @param query - bounded session-query reader.
 * @param sessionId - session that owns the checkpoint.
 * @param checkpoint - checkpoint whose shadowed seqs are rendered.
 * @param fromIndex - first shadowed-seq position to load.
 * @param signal - optional cancellation.
 * @returns rendered records in shadowed-seq order.
 * @throws when a requested original is absent or cannot carry a transcript.
 */
export async function loadRecords(
  query: RecallQuery,
  sessionId: SessionId,
  checkpoint: CompactionCheckpoint,
  fromIndex: number,
  signal?: AbortSignal,
): Promise<readonly RenderedRecord[]> {
  const seqs = checkpoint.shadowedSeqs.slice(fromIndex, fromIndex + RECORD_BATCH)
  if (seqs.length === 0) return []
  let events: SessionEvent[]
  try {
    events = await readEvents(query, sessionId, seqs, signal)
  } catch (error) {
    if (error instanceof SessionQueryError && error.code === 'SESSION_QUERY_EVENT_NOT_FOUND') {
      throw evidenceMissing(seqs)
    }
    throw error
  }
  const bySeq = new Map<number, SessionEvent>(events.map(event => [event.seq, event]))
  const extra = events
    .map(event => prunedSourceSeq(event))
    .filter((seq): seq is NonNullable<typeof seq> => seq !== undefined && !bySeq.has(seq))
  if (extra.length > 0) {
    try {
      for (const event of await readEvents(query, sessionId, [...new Set(extra)].sort((left, right) => left - right), signal)) {
        bySeq.set(event.seq, event)
      }
    } catch (error) {
      if (error instanceof SessionQueryError && error.code === 'SESSION_QUERY_EVENT_NOT_FOUND') {
        throw evidenceMissing(extra)
      }
      throw error
    }
  }
  return seqs.map((seq) => {
    const event = bySeq.get(seq)
    if (event === undefined) throw evidenceMissing([seq])
    const source = prunedSourceSeq(event)
    const original = source === undefined ? undefined : bySeq.get(source)
    if (original === undefined) return renderRecord(event)
    const rendered = renderRecord(original)
    return {
      seq,
      label: `${rendered.label} · pruned replacement #${String(seq)}; showing cited original #${String(original.seq)}`,
      text: rendered.text,
    }
  })
}
