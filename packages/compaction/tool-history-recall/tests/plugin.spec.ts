import { describe, expect, it } from 'vitest'
import type { Context } from '@deepseek-ai/cordis'
import { brandString } from '@deepseek-ai/dsh-brand'
import type { CompactionId } from '@deepseek-ai/dsh-compaction'
import { compactCheckpointSource } from '@deepseek-ai/dsh-compaction/checkpoint'
import { createMessage, createUserMessage } from '@deepseek-ai/dsh-llm'
import { Session, SessionLogOffset, type SessionId } from '@deepseek-ai/dsh-session'
import type { ToolDefinition, ToolRunContext } from '@deepseek-ai/dsh-tools'
import * as HistoryRecall from '../src/index.ts'

const SESSION_ID = brandString<SessionId>('plugin-session')

function fixture() {
  const session = Session.create(SESSION_ID)
  session.append('turn/start', { turn: 1 })
  const user = session.append('user/message', createUserMessage({
    content: [{ type: 'text', text: 'CONSTRAINT: keep C: free.' }],
    source: { kind: 'user' },
  }), { surfaceOp: 'append' })
  session.append('step/start', { turn: 1, step: 1 })
  session.append('assistant/message', {
    stream: [],
    turn: 1,
    step: 1,
    message: createMessage({
      role: 'assistant',
      content: [{ type: 'text', text: 'Noted.' }],
      source: { kind: 'model', provider: 'test', model: 'test' },
    }),
  }, { surfaceOp: 'append' })
  session.append('step/end', { turn: 1, step: 1 })
  session.append('turn/end', { turn: 1, reason: { kind: 'completed' } })
  const compactionId = brandString<CompactionId>('c1')
  const start = session.append('compaction/start', { compactionId, turn: null })
  const summary = session.append('compaction/summary', {
    compactionId,
    summary: [{ type: 'text', text: 'checkpoint' }],
    rawOutput: [{ type: 'text', text: 'checkpoint' }],
    llmStreamCall: true,
    shadowedRange: { start: user.seq, end: user.seq },
    shadowedSeqs: [user.seq],
    shadowedTokenCount: 10,
    provider: 'test',
    model: 'test',
  })
  session.append('user/message', createUserMessage({
    content: [{ type: 'text', text: '<compacted-summary>checkpoint</compacted-summary>' }],
    source: compactCheckpointSource(compactionId),
  }), {
    surfaceOp: { op: 'replace', startSeq: user.seq, endSeq: user.seq },
    sourceEventSeqs: [start.seq, summary.seq, user.seq],
  })
  session.append('compaction/end', { compactionId, turn: null })
  return { session, summarySeq: summary.seq }
}

function fakeQuery(session: Session) {
  const events = session.snapshotEvents()
  return {
    async readRawLog() {
      return {
        session: session.header,
        inheritedEventCount: SessionLogOffset(0),
        events: [...events],
        capturedThroughSeq: events.at(-1)?.seq ?? null,
      }
    },
  }
}

function harness() {
  const { session, summarySeq } = fixture()
  const tools: ToolDefinition[] = []
  const sections: Array<{ name: string; text: string }> = []
  const ctx = {
    systemPrompt: {
      getSectionOrder: () => 2310,
      section: (section: { name: string; text: string }) => {
        sections.push(section)
        return () => {}
      },
    },
    tools: {
      register: (tool: ToolDefinition) => {
        tools.push(tool)
        return () => {}
      },
    },
    sessionQuery: fakeQuery(session),
  } as unknown as Context
  HistoryRecall.apply(ctx, {})
  const exec = {
    agent: { session: { id: SESSION_ID } },
    signal: new AbortController().signal,
  } as unknown as ToolRunContext
  return { tools, sections, exec, summarySeq }
}

describe('history recall plugin', () => {
  it('registers two read-only tools and one prompt section', () => {
    const { tools, sections } = harness()
    expect(tools.map(tool => tool.name)).toEqual(['history_read', 'history_search'])
    expect(tools.every(tool => tool.isConcurrencySafe?.(
      tool.name === 'history_search' ? { query: 'x' } : {},
    ) === true)).toBe(true)
    expect(sections).toHaveLength(1)
    expect(sections[0]?.text).toContain('history_read')
    expect(sections[0]?.text).toContain('history_search')
  })

  it('lists checkpoints and pages originals through the registered tools', async () => {
    const { tools, exec, summarySeq } = harness()
    const read = tools.find(tool => tool.name === 'history_read')
    if (read === undefined) throw new Error('missing history_read')
    const listed = await read.execute({}, exec)
    expect(String(listed)).toContain(`c${String(summarySeq)}`)
    const page = await read.execute({ checkpoint: summarySeq }, exec)
    expect(String(page)).toContain('CONSTRAINT: keep C: free.')
    await expect(read.execute({ cursor: 'eyJ2IjoxfQ' }, exec)).rejects.toMatchObject({ code: 'HISTORY_CURSOR_INVALID' })
  })

  it('keeps the fixed prompt and schema cost small', () => {
    const { tools, sections } = harness()
    const promptChars = sections[0]?.text.length ?? 0
    const schemaChars = tools.reduce((total, tool) => total + JSON.stringify(tool.parameters).length, 0)
    expect(promptChars + schemaChars).toBeLessThan(4_000)
  })
})
