/**
 * Literal search over the compacted spans of one session log.
 *
 * @module @deepseek-ai/dsh-tool-history-recall/search
 */

import { Buffer } from 'node:buffer'
import type { SessionId } from '@deepseek-ai/dsh-session'
import { HarnessError } from '@deepseek-ai/dsh-llm'
import { checkpointsFromEvents, findCheckpoint } from './checkpoints.ts'
import { evidencePaths, readEvidenceText, scanEvidenceText } from './evidence.ts'
import { indexSessionEvents, loadRecordsFromEvents } from './records.ts'
import type {
  HistorySearchHit,
  RecallQuery,
  ResolvedHistoryRecallConfig,
} from './types.ts'

const MAX_QUERY_CHARS = 2_048
const SNIPPET_CONTEXT_CHARS = 120
const SNIPPET_MAX_CHARS = 240

/**
 * Search every selected compacted span for one literal, case-insensitive
 * phrase. The scan is bounded by bytes and hit count; coverage is reported so
 * the model can tell a complete miss from a capped scan.
 * @param query - bounded session-query reader.
 * @param sessionId - caller session id.
 * @param text - literal text to find.
 * @param checkpointSeq - restrict the scan to one checkpoint, or undefined for all.
 * @param config - resolved scan bounds.
 * @param signal - optional cancellation.
 * @returns model-facing hit list with coverage metadata.
 */
export async function searchHistory(
  query: RecallQuery,
  sessionId: SessionId,
  text: string,
  checkpointSeq: number | undefined,
  config: ResolvedHistoryRecallConfig,
  signal?: AbortSignal,
): Promise<string> {
  const trimmed = text.trim()
  if (trimmed.length === 0) {
    throw new HarnessError('history_search query must contain non-whitespace text', 'HISTORY_SEARCH_INVALID_QUERY')
  }
  if (trimmed.length > MAX_QUERY_CHARS) {
    throw new HarnessError(
      `history_search query is longer than ${String(MAX_QUERY_CHARS)} characters; use a shorter literal phrase`,
      'HISTORY_SEARCH_INVALID_QUERY',
    )
  }
  const log = await query.readRawLog(sessionId, signal)
  const bySeq = indexSessionEvents(log.events)
  const all = checkpointsFromEvents(log.events, bySeq)
  const evidence = checkpointSeq === undefined ? [...evidencePaths(log.events)] : []
  if (all.length === 0 && evidence.length === 0) {
    return 'This session has no compacted checkpoints or saved evidence, so there is no compacted history to search.'
  }
  const selected = checkpointSeq === undefined ? all : [findCheckpoint(log.events, checkpointSeq, bySeq)]
  const needle = trimmed.toLowerCase()
  const hits: HistorySearchHit[] = []
  const evidenceHits: string[] = []
  const seenOriginals = new Set<number>()
  let duplicatesSkipped = 0
  let checkpointsScanned = 0
  let evidenceScanned = 0
  let eventsScanned = 0
  let bytesScanned = 0
  let capped = false

  for (const checkpoint of selected) {
    checkpointsScanned += 1
    let fromIndex = 0
    while (fromIndex < checkpoint.shadowedSeqs.length) {
      if (signal?.aborted === true) signal.throwIfAborted()
      const loaded = loadRecordsFromEvents(bySeq, checkpoint, fromIndex)
      fromIndex = loaded.nextIndex
      for (const record of loaded.records) {
        const bytes = Buffer.byteLength(record.text, 'utf8')
        if (bytesScanned + bytes > config.maxSearchBytes) {
          capped = true
          break
        }
        bytesScanned += bytes
        eventsScanned += 1
        const index = record.text.toLowerCase().indexOf(needle)
        if (index === -1) continue
        if (seenOriginals.has(record.originalSeq)) {
          duplicatesSkipped += 1
          continue
        }
        seenOriginals.add(record.originalSeq)
        hits.push({
          checkpointSeq: checkpoint.summarySeq,
          seq: record.seq,
          label: record.label,
          snippet: snippet(record.text, index, needle.length),
        })
        if (hits.length >= config.maxSearchHits) {
          capped = true
          break
        }
      }
      if (capped) break
    }
    if (capped) break
  }

  for (const path of evidence) {
    if (hits.length + evidenceHits.length >= config.maxSearchHits) {
      capped = true
      break
    }
    const body = await readEvidenceText(path, config)
    const bytes = Buffer.byteLength(body, 'utf8')
    if (bytesScanned + bytes > config.maxSearchBytes) {
      capped = true
      break
    }
    bytesScanned += bytes
    evidenceScanned += 1
    evidenceHits.push(...scanEvidenceText(path, body, trimmed, config.maxSearchHits - hits.length - evidenceHits.length))
    if (hits.length + evidenceHits.length >= config.maxSearchHits) capped = true
  }

  const coverage = `Scanned ${String(checkpointsScanned)} of ${String(selected.length)} checkpoint(s), `
    + `${String(evidenceScanned)} evidence file(s), ${String(eventsScanned)} events, ${formatBytes(bytesScanned)} for "${trimmed}": `
    + `${String(hits.length + evidenceHits.length)} match(es)`
    + `${duplicatesSkipped === 0 ? '' : `, ${String(duplicatesSkipped)} duplicate citation(s) skipped`}.`
  const lines = [coverage]
  for (const hit of hits) {
    lines.push('', `- c${String(hit.checkpointSeq)} #${String(hit.seq)} ${hit.label}: ${hit.snippet}`)
  }
  for (const hit of evidenceHits) lines.push('', hit)
  if (hits.length + evidenceHits.length === 0) {
    lines.push('', 'No literal match. The scan is case-insensitive but literal; try a shorter exact phrase, or read the checkpoint transcript directly.')
  }
  if (capped) {
    lines.push('', 'Scan stopped at a deployment bound; narrow the query or pass a checkpoint to search one span.')
  }
  return lines.join('\n')
}

function snippet(text: string, index: number, needleLength: number): string {
  const start = Math.max(0, index - SNIPPET_CONTEXT_CHARS)
  const end = Math.min(text.length, index + needleLength + SNIPPET_CONTEXT_CHARS)
  const prefix = start === 0 ? '' : '…'
  const suffix = end === text.length ? '' : '…'
  const body = text.slice(start, end).replace(/\s+/gu, ' ').trim()
  const clipped = body.length <= SNIPPET_MAX_CHARS ? body : `${body.slice(0, SNIPPET_MAX_CHARS - 1)}…`
  return `${prefix}${clipped}${suffix}`
}

function formatBytes(bytes: number): string {
  if (bytes < 1_024) return `${String(bytes)} B`
  if (bytes < 1_048_576) return `${(bytes / 1_024).toFixed(1)} KB`
  return `${(bytes / 1_048_576).toFixed(1)} MB`
}
