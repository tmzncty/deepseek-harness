/**
 * Exact transcript rendering for one shadowed surface event. The renderer reads
 * durable event payloads rather than the current surface projection, so it
 * works after compaction, after restart, and for events a later compaction
 * shadowed again.
 *
 * @module @deepseek-ai/dsh-tool-history-recall/render
 */

import type { ContentBlock, ToolResultMessage } from '@deepseek-ai/dsh-llm'
import type { SessionEvent, SessionSeq } from '@deepseek-ai/dsh-session'
import { isCompactCheckpointSource } from '@deepseek-ai/dsh-compaction/checkpoint'
import type {} from '@deepseek-ai/dsh-compaction'
import type { RenderedRecord } from './types.ts'
import { evidenceInvalid } from './errors.ts'

/**
 * Render one shadowed surface event as a labeled transcript record.
 * @param event - durable event at the shadowed seq.
 * @returns the exact record text and its role label.
 * @throws when the event cannot carry a transcript message.
 */
export function renderRecord(event: SessionEvent): RenderedRecord {
  const rendered = renderEvent(event)
  return {
    seq: event.seq,
    label: rendered.label,
    text: rendered.text,
  }
}

/**
 * The single original seq a tool-result prune replacement cites, when the
 * replacement's visible content is not the exact original. Compaction summary
 * replacements cite many sources and are not returned.
 * @param event - event to inspect.
 * @returns the cited original seq, or undefined when this is not a prune replacement.
 */
export function prunedSourceSeq(event: SessionEvent): SessionSeq | undefined {
  if (event.type !== 'tool/result' || event.surfaceOp === 'append') return undefined
  const sources = event.sourceEventSeqs
  if (sources?.length !== 1) return undefined
  const source = sources[0]
  return source === undefined || source === event.seq ? undefined : source
}

function renderEvent(event: SessionEvent): { label: string; text: string } {
  switch (event.type) {
    case 'user/message':
      return {
        label: isCompactCheckpointSource(event.data.source)
          ? 'user · prior compaction checkpoint'
          : 'user',
        text: renderContent(event.data.content),
      }
    case 'assistant/message':
      return {
        label: event.data.interrupted === true ? 'assistant · interrupted' : 'assistant',
        text: renderContent(event.data.message.content),
      }
    case 'tool/result':
      return {
        label: toolLabel(event.data.message),
        text: renderToolResult(event.data.message, event.data.error?.reason),
      }
    case 'developer/message':
      return { label: 'developer', text: renderContent(event.data.message.content) }
    case 'system/message':
      return { label: 'system', text: renderContent(event.data.message.content) }
    default:
      throw evidenceInvalid(event.seq, event.type)
  }
}

function toolLabel(message: ToolResultMessage): string {
  const call = String(message.toolCallId)
  return message.isError === true ? `tool result (call ${call}) · error` : `tool result (call ${call})`
}

function renderToolResult(message: ToolResultMessage, reason: string | undefined): string {
  const text = renderContent(message.content)
  if (reason === undefined || reason === '') return text
  return text === '(empty)' ? `Error: ${reason}` : `${text}\n\nError: ${reason}`
}

/**
 * Render content blocks as plain transcript text. Reasoning is omitted because
 * it never entered a later model request; every block that carried visible
 * model input is preserved verbatim.
 * @param blocks - durable message content.
 * @returns text with one line per non-text block marker, or `(empty)`.
 */
export function renderContent(blocks: readonly ContentBlock[]): string {
  const parts: string[] = []
  for (const block of blocks) {
    switch (block.type) {
      case 'text':
        if (block.text !== '') parts.push(block.text)
        break
      case 'tool-call':
        parts.push(
          `Tool call: ${block.name} (call ${String(block.id)})\n${block.arguments}`,
        )
        break
      case 'image':
        parts.push(`[image: ${attachmentName(block.attachment.name)} (${String(block.attachment.width)}×${String(block.attachment.height)}, ${String(block.attachment.bytes)} bytes)]`)
        break
      case 'file':
        parts.push(`[file: ${attachmentName(block.attachment.name)} (${String(block.attachment.bytes)} bytes)]`)
        break
      case 'reasoning':
        break
      default:
        break
    }
  }
  return parts.length === 0 ? '(empty)' : parts.join('\n\n')
}

function attachmentName(name: string | undefined): string {
  const normalized = name?.replace(/\s+/gu, ' ').trim() ?? ''
  return normalized === '' ? 'attachment' : normalized
}
