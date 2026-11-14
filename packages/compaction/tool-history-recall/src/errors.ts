/**
 * Typed failures surfaced to the model when a recall request cannot be honored.
 *
 * @module @deepseek-ai/dsh-tool-history-recall/errors
 */

import { HarnessError } from '@deepseek-ai/dsh-llm'

/** A checkpoint seq was not found in the caller's durable log. */
export function checkpointNotFound(seq: number): HarnessError {
  return new HarnessError(
    `no compaction checkpoint #${String(seq)} exists in this session; call history_read without a checkpoint to list them`,
    'HISTORY_CHECKPOINT_NOT_FOUND',
  )
}

/** The checkpoint's shadowed span cites events the loaded log no longer holds. */
export function evidenceMissing(seqs: readonly number[]): HarnessError {
  return new HarnessError(
    `compacted history cites ${String(seqs.length)} event(s) absent from the loaded session log: ${seqs.join(', ')}; the durable log is incomplete and this span cannot be read exactly`,
    'HISTORY_EVIDENCE_MISSING',
  )
}

/** A shadowed seq resolved to an event that cannot carry a transcript message. */
export function evidenceInvalid(seq: number, type: string): HarnessError {
  return new HarnessError(
    `compacted history cites event #${String(seq)} (${type}), which is not a message-bearing surface event; the checkpoint record is not readable as a transcript`,
    'HISTORY_EVIDENCE_INVALID',
  )
}

/** A resume cursor is malformed, stale, or belongs to another checkpoint. */
export function cursorInvalid(detail: string): HarnessError {
  return new HarnessError(
    `history_read cursor is invalid: ${detail}; call history_read without a cursor to restart at the checkpoint's first page`,
    'HISTORY_CURSOR_INVALID',
  )
}

/** The caller tool execution is not bound to an agent session. */
export function noCallerSession(): HarnessError {
  return new HarnessError(
    'history_read requires an agent-bound session; this execution has no caller session',
    'HISTORY_RECALL_NO_SESSION',
  )
}

/** The configured page budget cannot hold framing plus one character of content. */
export function pageBudgetTooSmall(maxPageBytes: number): HarnessError {
  return new HarnessError(
    `history recall page budget (${String(maxPageBytes)} bytes) is too small for transcript framing; raise maxPageBytes`,
    'HISTORY_PAGE_BUDGET_TOO_SMALL',
  )
}

/** The internal page loader violated its contiguous-record contract. */
export function pageLoaderInvalid(detail: string): HarnessError {
  return new HarnessError(
    `history recall page loader returned an invalid batch: ${detail}`,
    'HISTORY_PAGE_LOAD_INVALID',
  )
}
