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

/**
 * Render the next records of one checkpoint.
 * @param events - complete logged events in seq order.
 * @param checkpoint - checkpoint whose shadowed seqs are rendered.
 * @param fromIndex - first shadowed-seq position to render.
 * @returns rendered records in shadowed-seq order, plus the next index.
 * @throws when a requested original is absent or cannot carry a transcript.
 */
export function loadRecordsFromEvents(
  events: readonly SessionEvent[],
  checkpoint: CompactionCheckpoint,
  fromIndex: number,
): { records: readonly RenderedRecord[]; nextIndex: number } {
  const seqs = checkpoint.shadowedSeqs.slice(fromIndex, fromIndex + RECORD_BATCH)
  if (seqs.length === 0) return { records: [], nextIndex: fromIndex }
  const bySeq = new Map<number, SessionEvent>(events.map(event => [event.seq, event]))
  const records = seqs.map((seq) => {
    const event = bySeq.get(seq)
    if (event === undefined) throw evidenceMissing([seq])
    const original = resolveOriginal(bySeq, event)
    if (original.seq === event.seq) return renderRecord(event)
    const rendered = renderRecord(original)
    return {
      seq,
      label: `${rendered.label} 路 pruned replacement #${String(seq)}; showing cited original #${String(original.seq)}`,
      text: rendered.text,
    }
  })
  return { records, nextIndex: fromIndex + records.length }
}

function resolveOriginal(bySeq: ReadonlyMap<number, SessionEvent>, event: SessionEvent): SessionEvent {
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
