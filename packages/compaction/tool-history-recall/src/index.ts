/**
 * Model-facing, bounded recall of the original events behind a compaction
 * checkpoint. The tools read the caller's append-only Session log through the
 * paginated session-query service; they never rewrite the summary, the surface,
 * or the request prefix.
 *
 * @module @deepseek-ai/dsh-tool-history-recall
 */

import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import { HarnessError } from '@deepseek-ai/dsh-llm'
import { defineTool, type ToolRunContext } from '@deepseek-ai/dsh-tools'
import {
  DEFAULT_MAX_LISTED_CHECKPOINTS,
  DEFAULT_MAX_PAGE_BYTES,
  DEFAULT_MAX_SEARCH_BYTES,
  DEFAULT_MAX_SEARCH_HITS,
  resolveConfig,
} from './config.ts'
import { noCallerSession } from './errors.ts'
import { readHistory } from './recall.ts'
import { searchHistory } from './search.ts'
import type { HistoryRecallConfig, ResolvedHistoryRecallConfig } from './types.ts'

/** Cordis plugin name used by Loader diagnostics. */
export const name = 'tool-history-recall'

/** Capability services required by the model-facing consumer. */
export const inject = ['tools', 'systemPrompt', 'sessionQuery']

/** Schemastery config for Loader defaults and generated configuration docs. */
export const Config: z<HistoryRecallConfig> = z.object({
  maxPageBytes: z.number().step(1).min(1_024).max(1_048_576).default(DEFAULT_MAX_PAGE_BYTES),
  maxSearchHits: z.number().step(1).min(1).max(1_000).default(DEFAULT_MAX_SEARCH_HITS),
  maxSearchBytes: z.number().step(1).min(1_024).max(100_000_000).default(DEFAULT_MAX_SEARCH_BYTES),
  maxListedCheckpoints: z.number().step(1).min(0).max(64).default(DEFAULT_MAX_LISTED_CHECKPOINTS),
})

const TEXT_OUTPUT = {
  schema: { type: 'string' as const },
  render: (_args: unknown, value: string) => [{ type: 'text' as const, text: value }],
}

const PROMPT_TEXT =
  'Long sessions may replace older conversation with a compacted checkpoint. Use history_read to list this '
  + 'session\'s checkpoints and page through the exact original events one shadows, and history_search to find '
  + 'a literal phrase in those originals. Recalled text arrives as ordinary tool output; the checkpoint summary '
  + 'remains the current context.'

interface ReadArgs {
  checkpoint?: number
  cursor?: string
}

interface SearchArgs {
  query: string
  checkpoint?: number
}

/** Register the two recall tools and their shared model guidance. */
export function apply(ctx: Context, config: HistoryRecallConfig): void {
  const resolved = resolveConfig(config)
  ctx.systemPrompt.section({
    name: 'tool:history-recall',
    order: ctx.systemPrompt.getSectionOrder('TOOL_HISTORY_RECALL'),
    text: PROMPT_TEXT,
  })

  ctx.tools.register(defineTool({
    name: 'history_read',
    description:
      'List this session\'s compacted checkpoints, or page through the exact original events a checkpoint shadows. '
      + 'Omit checkpoint to list them; pass checkpoint (and the returned cursor) to read a page.',
    parameters: {
      checkpoint: { type: 'integer', description: 'Compaction checkpoint seq to read. Omit to list checkpoints.' },
      cursor: { type: 'string', description: 'Continuation cursor returned by the previous page of this checkpoint.' },
    },
    output: TEXT_OUTPUT,
    isConcurrencySafe: () => true,
    execute: (args, exec) => executeRead(ctx, args, exec, resolved),
    presentCall: args => ({
      card: 'generic',
      kind: 'read',
      title: args.checkpoint === undefined ? 'List compacted history' : `Read compacted history c${String(args.checkpoint)}`,
      rawInput: args,
    }),
  }))

  ctx.tools.register(defineTool({
    name: 'history_search',
    description:
      'Search the original events of this session\'s compacted spans for one literal, case-insensitive phrase. '
      + 'Returns snippets with checkpoint and event seq; pass checkpoint to scan one span.',
    parameters: {
      query: { type: 'string', required: true, description: 'Literal phrase to find in compacted originals.' },
      checkpoint: { type: 'integer', description: 'Restrict the scan to one checkpoint seq. Omit to scan all.' },
    },
    output: TEXT_OUTPUT,
    isConcurrencySafe: () => true,
    execute: (args, exec) => executeSearch(ctx, args, exec, resolved),
    presentCall: args => ({
      card: 'generic',
      kind: 'search',
      title: 'Search compacted history',
      rawInput: args.query,
    }),
  }))
}

function callerSessionId(exec: ToolRunContext) {
  const session = exec.agent?.session
  if (session === undefined) throw noCallerSession()
  return session.id
}

async function executeRead(
  ctx: Context,
  args: ReadArgs,
  exec: ToolRunContext,
  config: ResolvedHistoryRecallConfig,
): Promise<string> {
  if (args.cursor !== undefined && args.checkpoint === undefined) {
    throw new HarnessError(
      'history_read cursor requires the checkpoint it was returned for',
      'HISTORY_CURSOR_INVALID',
    )
  }
  return readHistory(
    ctx.sessionQuery,
    callerSessionId(exec),
    args.checkpoint,
    args.cursor,
    config,
    exec.signal,
  )
}

async function executeSearch(
  ctx: Context,
  args: SearchArgs,
  exec: ToolRunContext,
  config: ResolvedHistoryRecallConfig,
): Promise<string> {
  return searchHistory(
    ctx.sessionQuery,
    callerSessionId(exec),
    args.query,
    args.checkpoint,
    config,
    exec.signal,
  )
}
