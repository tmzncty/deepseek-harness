/**
 * Load-time validation for deployment-owned recall bounds.
 *
 * @module @deepseek-ai/dsh-tool-history-recall/config
 */

import type { HistoryRecallConfig, ResolvedHistoryRecallConfig } from './types.ts'

/** Default maximum UTF-8 bytes in one transcript page. */
export const DEFAULT_MAX_PAGE_BYTES = 16_384
/** Default maximum literal-search hits. */
export const DEFAULT_MAX_SEARCH_HITS = 20
/** Default maximum rendered bytes scanned by one search. */
export const DEFAULT_MAX_SEARCH_BYTES = 2_000_000
/** Default maximum checkpoints named in one page header. */
export const DEFAULT_MAX_LISTED_CHECKPOINTS = 8

/**
 * Resolve and validate recall configuration.
 * @param config - untrusted plugin configuration after Loader normalization.
 * @returns detached immutable bounds.
 */
export function resolveConfig(config: HistoryRecallConfig = {}): ResolvedHistoryRecallConfig {
  const maxPageBytes = config.maxPageBytes ?? DEFAULT_MAX_PAGE_BYTES
  const maxSearchHits = config.maxSearchHits ?? DEFAULT_MAX_SEARCH_HITS
  const maxSearchBytes = config.maxSearchBytes ?? DEFAULT_MAX_SEARCH_BYTES
  const maxListedCheckpoints = config.maxListedCheckpoints ?? DEFAULT_MAX_LISTED_CHECKPOINTS
  assertInteger('maxPageBytes', maxPageBytes, 1024, 1_048_576)
  assertInteger('maxSearchHits', maxSearchHits, 1, 1_000)
  assertInteger('maxSearchBytes', maxSearchBytes, 1_024, 100_000_000)
  assertInteger('maxListedCheckpoints', maxListedCheckpoints, 0, 64)
  return { maxPageBytes, maxSearchHits, maxSearchBytes, maxListedCheckpoints }
}

function assertInteger(name: string, value: number, min: number, max: number): void {
  if (!Number.isSafeInteger(value) || value < min || value > max) {
    throw new TypeError(`tool-history-recall: ${name} must be a safe integer in [${String(min)}, ${String(max)}]`)
  }
}
