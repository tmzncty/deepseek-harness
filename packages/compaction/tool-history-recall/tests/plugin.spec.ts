import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import type { Context } from '@deepseek-ai/cordis'
import { brandString } from '@deepseek-ai/dsh-brand'
import type { CompactionId } from '@deepseek-ai/dsh-compaction'
import { compactCheckpointSource } from '@deepseek-ai/dsh-compaction/checkpoint'
import { createMessage, createToolResultMessage, createUserMessage, type ToolCallId } from '@deepseek-ai/dsh-llm'
import { Session, SessionLogOffset, type SessionId, type SessionSeq } from '@deepseek-ai/dsh-session'
import type { ToolDefinition, ToolRunContext } from '@deepseek-ai/dsh-tools'
import * as HistoryRecall from '../src/index.ts'

const SESSION_ID = brandString<SessionId>('plugin-session')

/** Temp paths created by one case; removed afterwards. */
const temporary: string[] = []
afterEach(async () => {
  for (const path of temporary.splice(0)) await rm(path, { recursive: true, force: true })
})

/** A spill artifact under a subprocess-shaped temp directory. */
async function spillFile(body: string): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), 'dsh-subprocess-'))
  temporary.push(root)
  const path = join(root, 'stdout.log')
  await writeFile(path, body, 'utf8')
  return path
}

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

/** One session whose single tool result names a saved full-output spill path. */
function spillSession(spillPath: string): Session {
  const session = Session.create(SESSION_ID)
  session.append('turn/start', { turn: 1 })
  session.append('step/start', { turn: 1, step: 1 })
  const callId = brandString<ToolCallId>('spill-call')
  session.append('tool/call', { turn: 1, step: 1, callId, name: 'pwsh', arguments: '{}' })
  session.append('tool/result', {
    turn: 1,
    step: 1,
    message: createToolResultMessage({
      callId,
      content: [{ type: 'text', text: `[output truncated; full output: ${spillPath}]\n\nretained tail` }],
      isError: false,
    }),
  }, { surfaceOp: 'append' })
  return session
}

function harnessFor(session: Session, summarySeq?: SessionSeq) {
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
    agent: { session: { id: session.id } },
    signal: new AbortController().signal,
  } as unknown as ToolRunContext
  return { tools, sections, exec, summarySeq }
}

function harness() {
  const { session, summarySeq } = fixture()
  return harnessFor(session, summarySeq)
}

function toolNamed(tools: readonly ToolDefinition[], name: string): ToolDefinition {
  const tool = tools.find(candidate => candidate.name === name)
  if (tool === undefined) throw new Error(`harness did not register ${name}`)
  return tool
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

  it('renders checkpoint text and presents both tools at call time', () => {
    const { tools } = harness()
    const read = toolNamed(tools, 'history_read')
    const search = toolNamed(tools, 'history_search')

    expect(read.output.render({}, 'PAGE TEXT')).toEqual([{ type: 'text', text: 'PAGE TEXT' }])
    expect(read.presentCall?.({})).toMatchObject({ card: 'generic', kind: 'read', title: 'List compacted history' })
    expect(read.presentCall?.({ checkpoint: 7, cursor: 'abc' })).toMatchObject({
      card: 'generic',
      kind: 'read',
      title: 'Read compacted history c7',
    })
    expect(search.presentCall?.({ query: 'needle' })).toMatchObject({
      card: 'generic',
      kind: 'search',
      title: 'Search compacted history',
      rawInput: 'needle',
    })
  })

  it('searches compacted originals through the registered tool', async () => {
    const { tools, exec } = harness()
    const found = await toolNamed(tools, 'history_search').execute({ query: 'C: free' }, exec)
    expect(String(found)).toContain('match(es)')
    expect(String(found)).toContain('CONSTRAINT')
  })

  it('rejects a recall request that has no caller session', async () => {
    const { tools } = harness()
    const unbound = { signal: new AbortController().signal } as unknown as ToolRunContext
    await expect(toolNamed(tools, 'history_read').execute({}, unbound))
      .rejects.toMatchObject({ code: 'HISTORY_RECALL_NO_SESSION' })
    await expect(toolNamed(tools, 'history_search').execute({ query: 'C:' }, unbound))
      .rejects.toMatchObject({ code: 'HISTORY_RECALL_NO_SESSION' })
  })

  it('rejects a cursor that arrives without the checkpoint it belongs to', async () => {
    const { tools, exec } = harness()
    await expect(toolNamed(tools, 'history_read').execute({ cursor: 'abc' }, exec))
      .rejects.toMatchObject({ code: 'HISTORY_CURSOR_INVALID' })
  })

  it('rejects an out-of-range deployment bound while registering', () => {
    const unbound = {} as unknown as Context
    expect(() => HistoryRecall.apply(unbound, { maxPageBytes: 512 })).toThrow(TypeError)
    expect(() => HistoryRecall.apply(unbound, { maxListedCheckpoints: 65 })).toThrow(/maxListedCheckpoints/u)
  })
})

describe('saved evidence tool wiring', () => {
  it('pages a session-named spill artifact through history_read', async () => {
    const file = await spillFile(`EVIDENCE-BODY${'x'.repeat(20_000)}TAIL-MARKER`)
    const { tools, exec } = harnessFor(spillSession(file))
    const read = toolNamed(tools, 'history_read')

    const first = String(await read.execute({ evidence: file }, exec))
    expect(first).toContain('Saved tool output evidence')
    expect(first).toContain('Continues at offset')
    const next = Number(/Continues at offset (\d+)/u.exec(first)?.[1])
    expect(Number.isSafeInteger(next)).toBe(true)

    const later = String(await read.execute({ evidence: file, evidenceOffset: next }, exec))
    expect(later).toContain(`at offset ${String(next)}:`)
    expect(later).toContain('TAIL-MARKER')
  })

  it('rejects saved evidence combined with a checkpoint or a cursor', async () => {
    const file = await spillFile('evidence-body')
    const { tools, exec } = harnessFor(spillSession(file))
    const read = toolNamed(tools, 'history_read')

    await expect(read.execute({ evidence: file, checkpoint: 1 }, exec))
      .rejects.toMatchObject({ code: 'HISTORY_EVIDENCE_INVALID' })
    await expect(read.execute({ evidence: file, cursor: 'eyJ2IjoxfQ' }, exec))
      .rejects.toMatchObject({ code: 'HISTORY_EVIDENCE_INVALID' })
  })

  it('searches a session-named spill artifact through history_search', async () => {
    const file = await spillFile('retained head\nSAVED-NEEDLE in the saved original')
    const { tools, exec } = harnessFor(spillSession(file))

    const found = String(await toolNamed(tools, 'history_search').execute({ query: 'saved-needle', evidence: file }, exec))
    expect(found).toContain('Saved tool output evidence search for "saved-needle"')
    expect(found).toContain('1 match(es)')
    expect(found).toMatch(/- offset 14: …retained head SAVED-NEEDLE/u)
  })
})
