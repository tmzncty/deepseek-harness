/**
 * Model-facing operations over one asynchronous session-query observation.
 *
 * @module @deepseek-ai/dsh-tool-history-recall/recall
 */

import type { SessionId } from '@deepseek-ai/dsh-session'
import { checkpointsFromEvents, findCheckpoint } from './checkpoints.ts'
import { cursorInvalid } from './errors.ts'
import { evidencePaths } from './evidence.ts'
import { decodeCursor, renderPage } from './pagination.ts'
import { indexSessionEvents, loadRecordsFromEvents } from './records.ts'
import type {
  CompactionCheckpoint,
  RecallQuery,
  ResolvedHistoryRecallConfig,
} from './types.ts'

/**
 * List compacted spans in the session.
 * @param query - bounded session-query reader.
 * @param sessionId - caller session id.
 * @param limit - maximum spans to name, newest last.
 * @param signal - optional cancellation.
 * @returns model-facing directory text, or a no-compaction notice.
 */
export async function readCheckpointDirectory(
  query: RecallQuery,
  sessionId: SessionId,
  limit: number,
  signal?: AbortSignal,
): Promise<string> {
  const log = await query.readRawLog(sessionId, signal)
  return formatDirectory(checkpointsFromEvents(log.events), limit, [...evidencePaths(log.events)])
}

function formatDirectory(
  checkpoints: readonly CompactionCheckpoint[],
  limit: number,
  evidence: readonly string[],
): string {
  const lines: string[] = []
  if (checkpoints.length === 0) {
    lines.push('This session has no compacted checkpoints yet.')
  } else {
    const shown = limit <= 0 ? [] : checkpoints.slice(-limit)
    const omitted = checkpoints.length - shown.length
    lines.push(`Compacted checkpoints (${String(checkpoints.length)}):`)
    for (const checkpoint of shown) {
      lines.push(
        `- c${String(checkpoint.summarySeq)}: shadows #${String(checkpoint.shadowedRange.start)}–#${String(checkpoint.shadowedRange.end)}, `
          + `${String(checkpoint.shadowedSeqs.length)} events, ~${String(checkpoint.shadowedTokenCount)} tokens (${checkpoint.state})`,
      )
    }
    if (omitted > 0) lines.push(`- (${String(omitted)} older checkpoint(s) omitted)`)
    lines.push('', 'Read one span with history_read({checkpoint: <c-number>}); without a checkpoint this tool lists them.')
  }
  if (evidence.length > 0) {
    lines.push('', `Saved full-output evidence (${String(evidence.length)}):`)
    for (const path of evidence.slice(-limit <= 0 ? 0 : limit)) {
      lines.push(`- ${path} (read with history_read evidence; search with history_search evidence)`)
    }
  }
  return lines.join('\n')
}

/**
 * Read one bounded page of a checkpoint's original transcript, or list the
 * session's checkpoints when no checkpoint is requested.
 * @param query - bounded session-query reader.
 * @param sessionId - caller session id.
 * @param checkpointSeq - requested checkpoint summary seq, or undefined to list.
 * @param cursorText - opaque continuation cursor from a prior page.
 * @param config - resolved page bounds.
 * @param signal - optional cancellation.
 * @returns model-facing transcript page or checkpoint directory.
 */
export async function readHistory(
  query: RecallQuery,
  sessionId: SessionId,
  checkpointSeq: number | undefined,
  cursorText: string | undefined,
  config: ResolvedHistoryRecallConfig,
  signal?: AbortSignal,
): Promise<string> {
  const log = await query.readRawLog(sessionId, signal)
  const bySeq = indexSessionEvents(log.events)
  if (checkpointSeq === undefined) {
    return formatDirectory(
      checkpointsFromEvents(log.events, bySeq),
      config.maxListedCheckpoints,
      [...evidencePaths(log.events)],
    )
  }
  const checkpoint = findCheckpoint(log.events, checkpointSeq, bySeq)
  const cursor = cursorText === undefined ? undefined : decodeCursor(cursorText)
  if (cursor !== undefined && cursor.c !== checkpoint.summarySeq) {
    throw cursorInvalid(
      `the cursor belongs to checkpoint c${String(cursor.c)}, not c${String(checkpoint.summarySeq)}`,
    )
  }
  return renderPage({
    checkpoint,
    seqs: checkpoint.shadowedSeqs,
    cursor,
    maxPageBytes: config.maxPageBytes,
    load: fromIndex => loadRecordsFromEvents(bySeq, checkpoint, fromIndex),
  }).text
}
