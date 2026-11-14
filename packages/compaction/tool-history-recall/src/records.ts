/**
 * Turn a checkpoint's shadowed seqs into rendered transcript records from one
 * already-loaded event array.
 *
 * @module @deepseek-ai/dsh-tool-history-recall/records
 */

import type { SessionEvent } from '@deepseek-ai/dsh-session'
import { evidenceInvalid, evidenceMissing } from './errors.ts'
import { prunedSourceSeq, renderRecord } from './render.ts'
import type { CompactionCheckpoint, RenderedRecord } from './types.ts'

/** Records rendered per page chunk. */
const RECORD_BATCH = 20

/** Maximum prune-citation hops followed from one shadowed replacement. */
const MAX_PRUNE_HOPS = 8

/** Read-only seq-to-event index shared by every record/search page in one call. */
export type SessionEventIndex = ReadonlyMap<number, SessionEvent>

/**
 * Build the one seq index a recall/search call reuses for all its batches.
 * @param events - raw events in log order.
 * @returns seq-to-event index.
 */
export function indexSessionEvents(events: readonly SessionEvent[]): SessionEventIndex {
  const bySeq = new Map<number, SessionEvent>()
  for (const event of events) bySeq.set(event.seq, event)
  return bySeq
}

/**
 * Render the next records of one checkpoint.
 * @param bySeq - shared seq index for the complete log.
 * @param checkpoint - checkpoint whose shadowed seqs are rendered.
 * @param fromIndex - first shadowed-seq position to render.
 * @returns rendered records in shadowed-seq order, plus the next index.
 * @throws when a requested original is absent or cannot carry a transcript.
 */
export function loadRecordsFromEvents(
  bySeq: SessionEventIndex,
  checkpoint: CompactionCheckpoint,
  fromIndex: number,
): { records: readonly RenderedRecord[]; nextIndex: number } {
  const seqs = checkpoint.shadowedSeqs.slice(fromIndex, fromIndex + RECORD_BATCH)
  if (seqs.length === 0) return { records: [], nextIndex: fromIndex }
  const records = seqs.map((seq) => {
    const event = bySeq.get(seq)
    if (event === undefined) throw evidenceMissing([seq])
    const original = resolveOriginal(bySeq, event)
    if (original.seq === event.seq) return renderRecord(event)
    const rendered = renderRecord(original)
    return {
      seq,
      originalSeq: original.seq,
      label: `${rendered.label} · pruned replacement #${String(seq)}; showing cited original #${String(original.seq)}`,
      text: rendered.text,
    }
  })
  return { records, nextIndex: fromIndex + records.length }
}

function resolveOriginal(bySeq: SessionEventIndex, event: SessionEvent): SessionEvent {
  let current = event
  const seen = new Set<number>([event.seq])
  for (let hop = 0; hop < MAX_PRUNE_HOPS; hop += 1) {
    const source = prunedSourceSeq(current)
    if (source === undefined) return current
    if (seen.has(source)) throw evidenceInvalid(event.seq, 'prune citation chain is cyclic')
    seen.add(source)
    const next = bySeq.get(source)
    if (next === undefined) throw evidenceMissing([source])
    current = next
  }
  throw evidenceInvalid(event.seq, `prune citation chain exceeds ${String(MAX_PRUNE_HOPS)} hops`)
}
