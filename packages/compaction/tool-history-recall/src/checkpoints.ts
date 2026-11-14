/**
 * Durable checkpoint discovery over one bounded session-query observation.
 *
 * @module @deepseek-ai/dsh-tool-history-recall/checkpoints
 */

import type { SessionEvent, SessionId } from '@deepseek-ai/dsh-session'
import type {} from '@deepseek-ai/dsh-compaction'
import type {
  CompactionCheckpoint,
  RecallQuery,
} from './types.ts'
import { checkpointNotFound } from './errors.ts'

/**
 * List every durable compaction checkpoint in log order. A checkpoint stays
 * listed after a later compaction shadows its replacement node, because its
 * originals remain addressable in the log.
 * @param query - bounded session-query reader.
 * @param sessionId - caller session id.
 * @param signal - optional cancellation.
 * @returns checkpoints in ascending summary seq order.
 */
export async function listCheckpoints(
  query: RecallQuery,
  sessionId: SessionId,
  signal?: AbortSignal,
): Promise<CompactionCheckpoint[]> {
  const log = await query.readSession(sessionId, signal)
  return checkpointsFromEvents(log.events)
}

/**
 * Derive checkpoints from one already-loaded event array.
 * @param events - complete logged events in seq order.
 * @returns checkpoints in ascending summary seq order.
 */
export function checkpointsFromEvents(events: readonly SessionEvent[]): CompactionCheckpoint[] {
  const bySeq = new Map<number, SessionEvent>(events.map(event => [event.seq, event]))
  const summaries = events.filter((event): event is SessionEvent<'compaction/summary'> => event.type === 'compaction/summary')
  const shadowedByAny = new Set<number>()
  for (const summary of summaries) {
    for (const seq of summary.data.shadowedSeqs) shadowedByAny.add(seq)
  }
  return summaries.map((summary) => {
    const next = bySeq.get(summary.seq + 1)
    const replacementSeq = next !== undefined
      && next.type === 'user/message'
      && next.surfaceOp !== 'append'
      && next.sourceEventSeqs?.includes(summary.seq) === true
      ? next.seq
      : undefined
    return {
      summarySeq: summary.seq,
      shadowedRange: summary.data.shadowedRange,
      shadowedSeqs: summary.data.shadowedSeqs,
      shadowedTokenCount: summary.data.shadowedTokenCount,
      replacementSeq,
      superseded: replacementSeq !== undefined && shadowedByAny.has(replacementSeq),
    }
  })
}

/**
 * Resolve one checkpoint by summary seq from an already-loaded event array.
 * @param events - complete logged events in seq order.
 * @param summarySeq - checkpoint summary seq.
 * @returns the resolved checkpoint.
 * @throws when the event is absent or is not a compaction summary.
 */
export function findCheckpoint(
  events: readonly SessionEvent[],
  summarySeq: number,
): CompactionCheckpoint {
  if (!Number.isSafeInteger(summarySeq) || summarySeq < 0) throw checkpointNotFound(summarySeq)
  const checkpoint = checkpointsFromEvents(events).find(candidate => candidate.summarySeq === summarySeq)
  if (checkpoint === undefined) throw checkpointNotFound(summarySeq)
  return checkpoint
}
