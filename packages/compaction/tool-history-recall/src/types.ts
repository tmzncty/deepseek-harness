/**
 * Configuration, checkpoint, and pagination types for model-facing compacted
 * history recall.
 *
 * @module @deepseek-ai/dsh-tool-history-recall/types
 */

import type { SessionId, SessionSeq } from '@deepseek-ai/dsh-session'
import type { SessionLogSnapshot } from '@deepseek-ai/dsh-session-query'

/** Deployment-owned bounds for recall reads and searches. */
export interface HistoryRecallConfig {
  /** Maximum UTF-8 bytes in one transcript page. Defaults to 16384. */
  maxPageBytes?: number
  /** Maximum literal-search hits returned by one call. Defaults to 20. */
  maxSearchHits?: number
  /** Maximum rendered bytes one search scans. Defaults to 2000000. */
  maxSearchBytes?: number
  /** Maximum checkpoints listed in one page header. Defaults to 8. */
  maxListedCheckpoints?: number
}

/** Fully resolved, validated recall bounds. */
export interface ResolvedHistoryRecallConfig {
  readonly maxPageBytes: number
  readonly maxSearchHits: number
  readonly maxSearchBytes: number
  readonly maxListedCheckpoints: number
}

/** One durable compaction checkpoint and the surface span it shadows. */
export interface CompactionCheckpoint {
  /** Seq of the log-only `compaction/summary` event that owns this checkpoint. */
  readonly summarySeq: SessionSeq
  /** Surface-position span recorded by the summary event. */
  readonly shadowedRange: { readonly start: SessionSeq; readonly end: SessionSeq }
  /** Authoritative shadowed surface nodes, in surface order. */
  readonly shadowedSeqs: readonly SessionSeq[]
  /** Estimated tokens the summary priced as shadowed. */
  readonly shadowedTokenCount: number
  /** Seq of the replacement `user/message`, when the checkpoint landed completely. */
  readonly replacementSeq: SessionSeq | undefined
  /** Whether a later compaction shadowed this checkpoint's replacement node. */
  readonly superseded: boolean
}

/** Opaque resume position inside one checkpoint transcript. */
export interface RecallCursor {
  readonly v: 1
  readonly c: SessionSeq
  readonly s: SessionSeq
  readonly o: number
}

/** One rendered surface event in a recall transcript. */
export interface RenderedRecord {
  readonly seq: SessionSeq
  readonly label: string
  readonly text: string
}

/** One literal-search hit inside a compacted span. */
export interface HistorySearchHit {
  readonly checkpointSeq: SessionSeq
  readonly seq: SessionSeq
  readonly label: string
  readonly snippet: string
}

/**
 * The one asynchronous session read this consumer needs. `SessionQueryEngine`
 * satisfies it; tests provide an in-memory implementation.
 */
export interface RecallQuery {
  readSession(sessionId: SessionId, signal?: AbortSignal): Promise<SessionLogSnapshot>
}
