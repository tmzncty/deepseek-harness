/**
 * Durable checkpoint discovery over bounded session-query reads.
 *
 * @module @deepseek-ai/dsh-tool-history-recall/checkpoints
 */

import type { SessionEvent, SessionId, SessionSeq } from '@deepseek-ai/dsh-session'
import { SessionSeq as seqOf } from '@deepseek-ai/dsh-session'
import { SessionQueryError } from '@deepseek-ai/dsh-session-query'
import type {} from '@deepseek-ai/dsh-compaction'
import type {
  CompactionCheckpoint,
  RecallQuery,
} from './types.ts'
import { checkpointNotFound, evidenceMissing } from './errors.ts'

/** Maximum events requested from one range read; the default service window. */
const READ_BATCH = 10

/**
 * Read explicit seqs through the bounded service API, falling back to
 * single-event reads when a deployment configured a smaller window.
 * @param query - bounded session-query reader.
 * @param sessionId - session that owns the events.
 * @param seqs - strictly ascending unique seqs.
 * @param signal - optional cancellation.
 * @returns events aligned with `seqs`.
 */
export async function readEvents(
  query: RecallQuery,
  sessionId: SessionId,
  seqs: readonly SessionSeq[],
  signal?: AbortSignal,
): Promise<SessionEvent[]> {
  const events: SessionEvent[] = []
  for (let start = 0; start < seqs.length; start += READ_BATCH) {
    const chunk = seqs.slice(start, start + READ_BATCH)
    try {
      const observation = await query.readEventRange({ sessionId, seqs: chunk }, signal)
      events.push(...observation.events)
    } catch (error) {
      if (!(error instanceof SessionQueryError) || error.code !== 'SESSION_QUERY_INVALID_WINDOW') throw error
      for (const seq of chunk) {
        signal?.throwIfAborted()
        events.push((await query.readEvent({ sessionId, seq }, signal)).target)
      }
    }
  }
  return events
}

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
  const records = await query.listEvents(sessionId)
  signal?.throwIfAborted()
  const summarySeqs = records
    .filter(record => record.type === 'compaction/summary')
    .map(record => record.seq)
  if (summarySeqs.length === 0) return []
  const recordSeqs = new Set(records.map(record => record.seq))
  const replacementSeqs = summarySeqs
    .map(seq => seqOf(seq + 1))
    .filter(seq => recordSeqs.has(seq))
  const readSeqs = [...new Set([...summarySeqs, ...replacementSeqs])]
    .sort((left, right) => left - right)
  const events = await readEvents(query, sessionId, readSeqs, signal)
    .catch((error: unknown) => {
      if (error instanceof SessionQueryError && error.code === 'SESSION_QUERY_EVENT_NOT_FOUND') {
        throw evidenceMissing(summarySeqs)
      }
      throw error
    })
  const bySeq = new Map<number, SessionEvent>(events.map(event => [event.seq, event]))
  const summaries = summarySeqs
    .map(seq => bySeq.get(seq))
    .filter((event): event is SessionEvent<'compaction/summary'> => event?.type === 'compaction/summary')
  const shadowedByAny = new Set<number>()
  for (const summary of summaries) {
    for (const seq of summary.data.shadowedSeqs) shadowedByAny.add(seq)
  }
  const missing = summarySeqs.filter(seq => !summaries.some(summary => summary.seq === seq))
  if (missing.length > 0) throw evidenceMissing(missing)
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
 * Read one checkpoint summary event by seq without scanning the log.
 * @param query - bounded session-query reader.
 * @param sessionId - caller session id.
 * @param summarySeq - checkpoint summary seq.
 * @param signal - optional cancellation.
 * @returns the checkpoint payload.
 * @throws when the event is absent or is not a compaction summary.
 */
export async function readCheckpoint(
  query: RecallQuery,
  sessionId: SessionId,
  summarySeq: number,
  signal?: AbortSignal,
): Promise<CompactionCheckpoint> {
  if (!Number.isSafeInteger(summarySeq) || summarySeq < 0) throw checkpointNotFound(summarySeq)
  const event = await query.readEvent({ sessionId, seq: seqOf(summarySeq) }, signal)
    .then(window => window.target)
    .catch((error: unknown) => {
      if (error instanceof SessionQueryError && error.code === 'SESSION_QUERY_EVENT_NOT_FOUND') {
        throw checkpointNotFound(summarySeq)
      }
      throw error
    })
  if (event.type !== 'compaction/summary') throw checkpointNotFound(summarySeq)
  return {
    summarySeq: event.seq,
    shadowedRange: event.data.shadowedRange,
    shadowedSeqs: event.data.shadowedSeqs,
    shadowedTokenCount: event.data.shadowedTokenCount,
    replacementSeq: undefined,
    superseded: false,
  }
}

/**
 * Resolve one checkpoint by summary seq, defaulting to the latest checkpoint.
 * @param checkpoints - checkpoints in ascending summary seq order.
 * @param requestedSeq - model-supplied summary seq, or undefined for latest.
 * @returns the resolved checkpoint.
 * @throws when no checkpoint exists or the requested seq is not one.
 */
export function resolveCheckpoint(
  checkpoints: readonly CompactionCheckpoint[],
  requestedSeq: number | undefined,
): CompactionCheckpoint {
  if (requestedSeq !== undefined && (!Number.isSafeInteger(requestedSeq) || requestedSeq < 0)) {
    throw checkpointNotFound(requestedSeq)
  }
  if (requestedSeq === undefined) {
    const latest = checkpoints.at(-1)
    if (latest === undefined) throw checkpointNotFound(0)
    return latest
  }
  const checkpoint = checkpoints.find(candidate => candidate.summarySeq === requestedSeq)
  if (checkpoint === undefined) throw checkpointNotFound(requestedSeq)
  return checkpoint
}

/**
 * Render the compact directory every first page carries so the model can
 * discover sibling and superseded checkpoints without another tool call.
 * @param checkpoints - all checkpoints in log order.
 * @param limit - maximum entries to name.
 * @returns one line, or an empty string when only one checkpoint exists.
 */
export function checkpointDirectory(
  checkpoints: readonly CompactionCheckpoint[],
  limit: number,
): string {
  if (checkpoints.length <= 1 || limit <= 0) return ''
  const latest = checkpoints.at(-1)?.summarySeq
  const shown = checkpoints.slice(-limit)
  const entries = shown.map((checkpoint) => {
    const state = checkpoint.summarySeq === latest
      ? 'latest'
      : checkpoint.superseded ? 'superseded' : 'ready'
    return `c${String(checkpoint.summarySeq)} #${String(checkpoint.shadowedRange.start)}–#${String(checkpoint.shadowedRange.end)} (${state})`
  })
  const omitted = checkpoints.length - shown.length
  return `Checkpoints: ${entries.join(', ')}${omitted > 0 ? `, ${String(omitted)} older omitted` : ''}`
}
