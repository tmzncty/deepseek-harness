import { describe, expect, it } from 'vitest'
import { brandString } from '@deepseek-ai/dsh-brand'
import {
  createDeveloperMessage,
  createMessage,
  createSystemMessage,
  createToolResultMessage,
  type ContentBlock,
  type ToolCallId,
} from '@deepseek-ai/dsh-llm'
import { Session, SessionSeq, type SessionId } from '@deepseek-ai/dsh-session'
import { indexSessionEvents, loadRecordsFromEvents } from '../src/records.ts'
import { prunedSourceSeq, renderContent, renderRecord } from '../src/render.ts'
import type { CompactionCheckpoint } from '../src/types.ts'

/**
 * Unit coverage for the transcript renderer and the prune-citation predicate.
 * The tool suites prove a checkpoint page renders end to end; this suite pins
 * the event kinds and block kinds a single page is unlikely to contain at once,
 * and the citation shapes that make a tool result a readable prune replacement.
 */

const SESSION_ID = brandString<SessionId>('render-unit')

describe('exact transcript rendering', () => {
  it('labels developer, system, and interrupted assistant messages distinctly', () => {
    const session = Session.create(SESSION_ID)
    session.append('turn/start', { turn: 1 })
    const developer = session.append('developer/message', {
      turn: 1,
      step: 1,
      message: createDeveloperMessage({
        content: [{ type: 'text', text: 'DEVELOPER RULES' }],
        source: { kind: 'user' },
      }),
    }, { surfaceOp: 'append' })
    const system = session.append('system/message', {
      turn: 1,
      step: 1,
      message: createSystemMessage('SYSTEM PROMPT'),
    }, { surfaceOp: 'append' })
    const interrupted = session.append('assistant/message', {
      turn: 1,
      step: 1,
      interrupted: true,
      stream: [],
      message: createMessage({
        role: 'assistant',
        content: [{ type: 'text', text: 'CUT OFF MID-STREAM' }],
        source: { kind: 'model', provider: 'test', model: 'test' },
      }),
    }, { surfaceOp: 'append' })

    expect(renderRecord(developer)).toMatchObject({
      seq: developer.seq,
      originalSeq: developer.seq,
      label: 'developer',
      text: 'DEVELOPER RULES',
    })
    expect(renderRecord(system)).toMatchObject({ label: 'system', text: 'SYSTEM PROMPT' })
    expect(renderRecord(interrupted)).toMatchObject({ label: 'assistant · interrupted', text: 'CUT OFF MID-STREAM' })
  })

  it('labels a failed tool result and appends its durable error reason', () => {
    const session = Session.create(SESSION_ID)
    session.append('turn/start', { turn: 1 })
    session.append('step/start', { turn: 1, step: 1 })
    const callId = brandString<ToolCallId>('render-call')
    session.append('tool/call', { turn: 1, step: 1, callId, name: 'pwsh', arguments: '{}' })
    const failure = { name: 'HarnessError', code: 'E_PWSH' }
    const partial = session.append('tool/result', {
      turn: 1,
      step: 1,
      error: { ...failure, reason: 'disk exploded' },
      message: createToolResultMessage({
        callId,
        content: [{ type: 'text', text: 'PARTIAL OUTPUT' }],
        isError: true,
      }),
    }, { surfaceOp: 'append' })
    const emptyOutput = session.append('tool/result', {
      turn: 1,
      step: 1,
      error: { ...failure, reason: 'nothing to report' },
      message: createToolResultMessage({ callId, content: [], isError: true }),
    }, { surfaceOp: 'append' })
    const reasonless = session.append('tool/result', {
      turn: 1,
      step: 1,
      error: { ...failure, reason: '' },
      message: createToolResultMessage({
        callId,
        content: [{ type: 'text', text: 'KEPT TEXT' }],
        isError: true,
      }),
    }, { surfaceOp: 'append' })

    expect(renderRecord(partial)).toMatchObject({
      label: 'tool result (call render-call) · error',
      text: 'PARTIAL OUTPUT\n\nError: disk exploded',
    })
    expect(renderRecord(emptyOutput)).toMatchObject({ text: 'Error: nothing to report' })
    expect(renderRecord(reasonless)).toMatchObject({ text: 'KEPT TEXT' })
  })

  it('renders visible blocks, names attachments, and drops invisible blocks', () => {
    const content = [
      { type: 'text', text: 'VISIBLE TEXT' },
      { type: 'text', text: '' },
      { type: 'tool-call', id: brandString<ToolCallId>('call-7'), name: 'pwsh', arguments: '{"command":"ls"}' },
      { type: 'reasoning', text: 'PRIVATE THOUGHTS' },
      {
        type: 'image',
        attachment: {
          attachmentId: 'image-1',
          mediaType: 'image/png',
          bytes: 2_048,
          width: 640,
          height: 480,
          name: '  build   shot \n',
        },
      },
      { type: 'file', attachment: { attachmentId: 'file-1', name: '', bytes: 12 } },
      { type: 'image', attachment: { attachmentId: 'image-2', mediaType: 'image/png', bytes: 16, width: 1, height: 1 } },
      { type: 'tool-removal', toolName: 'pwsh' },
    ] as unknown as ContentBlock[]

    const text = renderContent(content)
    expect(text).toContain('VISIBLE TEXT')
    expect(text).toContain('Tool call: pwsh (call call-7)\n{"command":"ls"}')
    expect(text).not.toContain('PRIVATE THOUGHTS')
    expect(text).toContain('[image: build shot (640×480, 2048 bytes)]')
    expect(text).toContain('[image: attachment (1×1, 16 bytes)]')
    expect(text).toContain('[file: attachment (12 bytes)]')
    expect(text).not.toContain('tool-removal')
  })

  it('renders an empty transcript record as an explicit emptiness marker', () => {
    expect(renderContent([])).toBe('(empty)')
    expect(renderContent([{ type: 'text', text: '' }])).toBe('(empty)')
  })
})

describe('record loading and prune citation resolution', () => {
  it('treats only a single, self-excluding, earlier citation as a prune source', () => {
    const session = Session.create(SESSION_ID)
    session.append('turn/start', { turn: 1 })
    session.append('step/start', { turn: 1, step: 1 })
    const callId = brandString<ToolCallId>('prune-unit')
    session.append('tool/call', { turn: 1, step: 1, callId, name: 'pwsh', arguments: '{}' })
    const result = (text: string) => createToolResultMessage({ callId, content: [{ type: 'text', text }], isError: false })
    const original = session.append('tool/result', { turn: 1, step: 1, message: result('ORIGINAL') }, { surfaceOp: 'append' })
    const appended = session.append('tool/result', { turn: 1, step: 1, message: result('APPENDED') }, { surfaceOp: 'append' })
    const replacement = session.append('tool/result', {
      turn: 1,
      step: 1,
      message: { ...original.data.message, content: [{ type: 'text', text: 'PRUNED' }] },
    }, {
      surfaceOp: { op: 'replace', startSeq: original.seq, endSeq: original.seq },
      sourceEventSeqs: [original.seq],
    })

    expect(prunedSourceSeq(original)).toBeUndefined()
    expect(prunedSourceSeq(appended)).toBeUndefined()
    expect(prunedSourceSeq(replacement)).toBe(original.seq)
    expect(prunedSourceSeq({ ...replacement, sourceEventSeqs: [original.seq, appended.seq] })).toBeUndefined()
    expect(prunedSourceSeq({ ...replacement, sourceEventSeqs: [replacement.seq] })).toBeUndefined()
  })

  it('returns no records when a batch starts past the end of the shadowed span', () => {
    const checkpoint: CompactionCheckpoint = {
      summarySeq: SessionSeq(9),
      shadowedRange: { start: SessionSeq(1), end: SessionSeq(1) },
      shadowedSeqs: [SessionSeq(1)],
      shadowedTokenCount: 1,
      replacementSeq: undefined,
      state: 'unfinished',
      superseded: false,
    }
    expect(loadRecordsFromEvents(indexSessionEvents([]), checkpoint, checkpoint.shadowedSeqs.length))
      .toEqual({ records: [], nextIndex: 1 })
  })
})
