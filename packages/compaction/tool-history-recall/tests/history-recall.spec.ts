import { Buffer } from 'node:buffer'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import { brandString } from '@deepseek-ai/dsh-brand'
import type { CompactionId } from '@deepseek-ai/dsh-compaction'
import { compactCheckpointSource } from '@deepseek-ai/dsh-compaction/checkpoint'
import {
  createMessage,
  createToolResultMessage,
  createUserMessage,
  type ToolCallId,
} from '@deepseek-ai/dsh-llm'
import {
  SESSION_FORMAT_VERSION,
  Session,
  SessionLogOffset,
  SessionSeq,
  type SessionEvent,
  type SessionHeader,
  type SessionId,
} from '@deepseek-ai/dsh-session'
import * as JsonlSessionPersistence from '@deepseek-ai/dsh-session-persistence-jsonl'
import type { SessionLogSnapshot } from '@deepseek-ai/dsh-session-query'
import { resolveConfig } from '../src/config.ts'
import { listCheckpoints } from '../src/checkpoints.ts'
import { readCheckpointDirectory, readHistory } from '../src/recall.ts'
import { searchHistory } from '../src/search.ts'
import type { RecallQuery } from '../src/types.ts'

const QUERY_CONFIG = resolveConfig({ maxPageBytes: 1_024, maxSearchHits: 5, maxSearchBytes: 100_000 })
const SESSION_ID = brandString<SessionId>('recall-test')

interface Conversation {
  readonly session: Session
  readonly constraintSeq: SessionSeq
  readonly assistantSeq: SessionSeq
  readonly resultSeq: SessionSeq
  readonly revisionSeq: SessionSeq
}

function conversation(): Conversation {
  const session = Session.create(SESSION_ID)
  session.append('turn/start', { turn: 1 })
  const constraint = session.append('user/message', createUserMessage({
    content: [{ type: 'text', text: 'CONSTRAINT: never delete C: files. Use X: for build output.' }],
    source: { kind: 'user' },
  }), { surfaceOp: 'append' })
  session.append('step/start', { turn: 1, step: 1 })
  const callId = brandString<ToolCallId>('call-1')
  const assistant = session.append('assistant/message', {
    stream: [],
    turn: 1,
    step: 1,
    message: createMessage({
      role: 'assistant',
      content: [
        { type: 'text', text: 'I will inspect the build directory.' },
        { type: 'tool-call', id: callId, name: 'pwsh', arguments: '{"command":"Get-ChildItem"}' },
      ],
      source: { kind: 'model', provider: 'test', model: 'test' },
    }),
  }, { surfaceOp: 'append' })
  session.append('tool/call', { turn: 1, step: 1, callId, name: 'pwsh', arguments: '{"command":"Get-ChildItem"}' })
  const result = session.append('tool/result', {
    turn: 1,
    step: 1,
    message: createToolResultMessage({
      callId,
      content: [{ type: 'text', text: 'RESULT: build output lives on X:.' }],
      isError: false,
    }),
  }, { surfaceOp: 'append' })
  session.append('step/end', { turn: 1, step: 1 })
  session.append('turn/end', { turn: 1, reason: { kind: 'completed' } })
  session.append('turn/start', { turn: 2 })
  const revision = session.append('user/message', createUserMessage({
    content: [{ type: 'text', text: 'REVISION: X: is now the only build disk; keep C: free.' }],
    source: { kind: 'user' },
  }), { surfaceOp: 'append' })
  session.append('step/start', { turn: 2, step: 1 })
  session.append('assistant/message', {
    stream: [],
    turn: 2,
    step: 1,
    message: createMessage({
      role: 'assistant',
      content: [{ type: 'text', text: 'Understood; I will keep C: free.' }],
      source: { kind: 'model', provider: 'test', model: 'test' },
    }),
  }, { surfaceOp: 'append' })
  session.append('step/end', { turn: 2, step: 1 })
  session.append('turn/end', { turn: 2, reason: { kind: 'completed' } })
  return {
    session,
    constraintSeq: constraint.seq,
    assistantSeq: assistant.seq,
    resultSeq: result.seq,
    revisionSeq: revision.seq,
  }
}

function appendBulk(session: Session): void {
  session.append('turn/start', { turn: 3 })
  session.append('user/message', createUserMessage({
    content: [{ type: 'text', text: 'BULK '.repeat(600) }],
    source: { kind: 'user' },
  }), { surfaceOp: 'append' })
  session.append('step/start', { turn: 3, step: 1 })
  session.append('assistant/message', {
    stream: [],
    turn: 3,
    step: 1,
    message: createMessage({
      role: 'assistant',
      content: [{ type: 'text', text: 'Bulk acknowledged.' }],
      source: { kind: 'model', provider: 'test', model: 'test' },
    }),
  }, { surfaceOp: 'append' })
  session.append('step/end', { turn: 3, step: 1 })
  session.append('turn/end', { turn: 3, reason: { kind: 'completed' } })
}

function compact(
  session: Session,
  shadowedSeqs: readonly SessionSeq[],
  summaryText: string,
  label: string,
): { summarySeq: SessionSeq; replacementSeq: SessionSeq } {
  const compactionId = brandString<CompactionId>(label)
  const start = session.append('compaction/start', { compactionId, turn: null })
  const summary = session.append('compaction/summary', {
    compactionId,
    summary: [{ type: 'text', text: summaryText }],
    rawOutput: [{ type: 'text', text: summaryText }],
    llmStreamCall: true,
    shadowedRange: { start: shadowedSeqs[0] ?? SessionSeq(0), end: shadowedSeqs.at(-1) ?? SessionSeq(0) },
    shadowedSeqs: [...shadowedSeqs],
    shadowedTokenCount: 50,
    provider: 'test',
    model: 'test',
  })
  const replacement = session.append('user/message', createUserMessage({
    content: [{ type: 'text', text: `<compacted-summary>${summaryText}</compacted-summary>` }],
    source: compactCheckpointSource(compactionId),
  }), {
    surfaceOp: { op: 'replace', startSeq: shadowedSeqs[0] ?? SessionSeq(0), endSeq: shadowedSeqs.at(-1) ?? SessionSeq(0) },
    sourceEventSeqs: [start.seq, summary.seq, ...shadowedSeqs],
  })
  session.append('compaction/end', { compactionId, turn: null })
  return { summarySeq: summary.seq, replacementSeq: replacement.seq }
}

class FakeQuery implements RecallQuery {
  readonly events: readonly SessionEvent[]
  readonly failSeq: SessionSeq | undefined

  constructor(events: readonly SessionEvent[], failSeq?: SessionSeq) {
    this.events = events
    this.failSeq = failSeq
  }

  async readSession(): Promise<SessionLogSnapshot> {
    const events = this.failSeq === undefined
      ? this.events
      : this.events.filter(event => event.seq !== this.failSeq)
    return {
      session: header(),
      inheritedEventCount: SessionLogOffset(0),
      events: [...events],
    }
  }
}

function header(): SessionHeader {
  return {
    version: SESSION_FORMAT_VERSION as SessionHeader['version'],
    id: SESSION_ID,
    createdAt: 1,
    isSeeded: false,
  }
}

const temporary: string[] = []
afterEach(async () => {
  for (const path of temporary.splice(0)) await rm(path, { recursive: true, force: true })
})

describe('history recall over compaction checkpoints', () => {
  it('lists checkpoints without any model-visible pointer in the summary', async () => {
    const { session } = conversation()
    const first = compact(session, [...session.surface.nodes], 'checkpoint one', 'c1')
    const query = new FakeQuery(session.snapshotEvents())
    const directory = await readCheckpointDirectory(query, SESSION_ID, 8)
    expect(directory).toContain(`c${String(first.summarySeq)}`)
    expect(directory).toContain('readable')
    expect(session.deriveMessages().some(message => message.content.some(block => block.type === 'text' && block.text.includes('CONSTRAINT')))).toBe(false)
  })

  it('pages exact originals with a working cursor and byte bound', async () => {
    const { session, constraintSeq, resultSeq } = conversation()
    appendBulk(session)
    const first = compact(session, [...session.surface.nodes], 'checkpoint one', 'c1')
    const query = new FakeQuery(session.snapshotEvents())
    let cursor: string | undefined
    let combined = ''
    for (let page = 0; page < 20; page += 1) {
      const text = await readHistory(query, SESSION_ID, first.summarySeq, cursor, QUERY_CONFIG)
      expect(Buffer.byteLength(text, 'utf8')).toBeLessThanOrEqual(QUERY_CONFIG.maxPageBytes)
      expect(text).not.toContain('\uFFFD')
      combined += text
      cursor = /cursor ([A-Za-z0-9_-]+)/u.exec(text)?.[1]
      if (cursor === undefined) break
    }
    expect(cursor).toBeUndefined()
    expect(combined).toContain('CONSTRAINT: never delete C: files')
    expect(combined).toContain('REVISION: X: is now the only build disk')
    expect(combined).toContain('RESULT: build output lives on X:')
    expect(combined).toContain('BULK')
    expect(combined).toContain(`#${String(constraintSeq)}`)
    expect(combined).toContain(`#${String(resultSeq)}`)
    expect(combined).toContain('Tool call: pwsh (call call-1)')
    expect(combined).toContain('tool result (call call-1)')
    expect(session.deriveMessages().some(message => message.content.some(block => block.type === 'text' && block.text.includes('CONSTRAINT')))).toBe(false)
  })

  it('keeps facts and user revisions across two compaction generations', async () => {
    const { session, constraintSeq, assistantSeq, resultSeq, revisionSeq } = conversation()
    const first = compact(session, [constraintSeq], 'checkpoint one', 'c1')
    // Surface order after the first replacement: [high-seq checkpoint,
    // assistant, tool result, revision, ...]. The second span is therefore
    // non-monotonic in seq order, which is the normal multi-generation shape.
    const second = compact(
      session,
      [first.replacementSeq, assistantSeq, resultSeq, revisionSeq],
      'checkpoint two merges checkpoint one',
      'c2',
    )
    const query = new FakeQuery(session.snapshotEvents())
    const checkpoints = await listCheckpoints(query, SESSION_ID)
    expect(checkpoints.map(checkpoint => checkpoint.summarySeq)).toEqual([first.summarySeq, second.summarySeq])
    expect(checkpoints[0]?.superseded).toBe(true)
    const older = await readHistory(query, SESSION_ID, first.summarySeq, undefined, QUERY_CONFIG)
    expect(older).toContain('CONSTRAINT: never delete C: files')
    expect(older).not.toContain('REVISION: X: is now the only build disk')
    const newer = await readHistory(query, SESSION_ID, second.summarySeq, undefined, QUERY_CONFIG)
    expect(newer).toContain('prior compaction checkpoint')
    expect(newer).toContain('RESULT: build output lives on X:')
    expect(newer).toContain('REVISION: X: is now the only build disk')
    expect(newer).not.toContain('CONSTRAINT: never delete C: files')
  })

  it('search finds literal content only present in a shadowed span', async () => {
    const { session } = conversation()
    const first = compact(session, [...session.surface.nodes], 'checkpoint one', 'c1')
    const query = new FakeQuery(session.snapshotEvents())
    const result = await searchHistory(query, SESSION_ID, 'never delete C: files', first.summarySeq, QUERY_CONFIG)
    expect(result).toContain('1 match')
    expect(result).toContain('CONSTRAINT')
  })

  it('keeps tool call and result pairing in the transcript', async () => {
    const { session } = conversation()
    const first = compact(session, [...session.surface.nodes], 'checkpoint one', 'c1')
    const query = new FakeQuery(session.snapshotEvents())
    const page = await readHistory(query, SESSION_ID, first.summarySeq, undefined, QUERY_CONFIG)
    expect(page).toContain('call-1')
    expect(page).toContain('Tool call: pwsh')
    expect(page).toContain('tool result (call call-1)')
  })

  it('names missing evidence instead of silently skipping it', async () => {
    const { session, resultSeq } = conversation()
    const first = compact(session, [...session.surface.nodes], 'checkpoint one', 'c1')
    const query = new FakeQuery(session.snapshotEvents(), resultSeq)
    await expect(readHistory(query, SESSION_ID, first.summarySeq, undefined, QUERY_CONFIG))
      .rejects.toMatchObject({ code: 'HISTORY_EVIDENCE_MISSING' })
  })

  it('rejects an unknown checkpoint and reports a session without compaction', async () => {
    const { session } = conversation()
    compact(session, [...session.surface.nodes], 'checkpoint one', 'c1')
    const query = new FakeQuery(session.snapshotEvents())
    await expect(readHistory(query, SESSION_ID, 999, undefined, QUERY_CONFIG))
      .rejects.toMatchObject({ code: 'HISTORY_CHECKPOINT_NOT_FOUND' })
    const empty = new FakeQuery([])
    await expect(readHistory(empty, SESSION_ID, undefined, undefined, QUERY_CONFIG))
      .resolves.toContain('no compacted checkpoints')
  })

  it('rejects a checkpoint whose cited event is not message-bearing', async () => {
    const session = Session.create(SESSION_ID)
    session.append('turn/start', { turn: 1 })
    const user = session.append('user/message', createUserMessage({
      content: [{ type: 'text', text: 'visible' }],
      source: { kind: 'user' },
    }), { surfaceOp: 'append' })
    const compactionId = brandString<CompactionId>('bad-evidence')
    const start = session.append('compaction/start', { compactionId, turn: null })
    const summary = session.append('compaction/summary', {
      compactionId,
      summary: [{ type: 'text', text: 'checkpoint' }],
      rawOutput: [{ type: 'text', text: 'checkpoint' }],
      llmStreamCall: true,
      shadowedRange: { start: user.seq, end: user.seq },
      shadowedSeqs: [SessionSeq(0)],
      shadowedTokenCount: 10,
      provider: 'test',
      model: 'test',
    })
    session.append('user/message', createUserMessage({
      content: [{ type: 'text', text: 'checkpoint' }],
      source: compactCheckpointSource(compactionId),
    }), {
      surfaceOp: { op: 'replace', startSeq: user.seq, endSeq: user.seq },
      sourceEventSeqs: [start.seq, summary.seq, user.seq],
    })
    session.append('compaction/end', { compactionId, turn: null })
    const query = new FakeQuery(session.snapshotEvents())
    await expect(readHistory(query, SESSION_ID, summary.seq, undefined, QUERY_CONFIG))
      .rejects.toMatchObject({ code: 'HISTORY_EVIDENCE_INVALID' })
  })

  it('follows a chained prune replacement to the deepest cited original', async () => {
    const session = Session.create(SESSION_ID)
    session.append('turn/start', { turn: 1 })
    session.append('user/message', createUserMessage({
      content: [{ type: 'text', text: 'read the tool output' }],
      source: { kind: 'user' },
    }), { surfaceOp: 'append' })
    session.append('step/start', { turn: 1, step: 1 })
    const callId = brandString<ToolCallId>('prune-call')
    session.append('tool/call', { turn: 1, step: 1, callId, name: 'pwsh', arguments: '{}' })
    const original = session.append('tool/result', {
      turn: 1,
      step: 1,
      message: createToolResultMessage({
        callId,
        content: [{ type: 'text', text: 'ORIGINAL-ONLY: the middle value is 417.' }],
        isError: false,
      }),
    }, { surfaceOp: 'append' })
    const first = session.append('tool/result', {
      turn: 1,
      step: 1,
      message: { ...original.data.message, content: [{ type: 'text', text: 'head [...] tail' }] },
    }, {
      surfaceOp: { op: 'replace', startSeq: original.seq, endSeq: original.seq },
      sourceEventSeqs: [original.seq],
    })
    session.append('compaction/prune', {
      shadowedRange: { start: original.seq, end: original.seq },
      shadowedSeqs: [original.seq],
      shadowedTokenCount: 1,
    })
    const second = session.append('tool/result', {
      turn: 1,
      step: 1,
      message: { ...original.data.message, content: [{ type: 'text', text: 'more pruned' }] },
    }, {
      surfaceOp: { op: 'replace', startSeq: first.seq, endSeq: first.seq },
      sourceEventSeqs: [first.seq],
    })
    session.append('compaction/prune', {
      shadowedRange: { start: first.seq, end: first.seq },
      shadowedSeqs: [first.seq],
      shadowedTokenCount: 1,
    })
    const compactionId = brandString<CompactionId>('prune-checkpoint')
    const start = session.append('compaction/start', { compactionId, turn: null })
    const summary = session.append('compaction/summary', {
      compactionId,
      summary: [{ type: 'text', text: 'checkpoint' }],
      rawOutput: [{ type: 'text', text: 'checkpoint' }],
      llmStreamCall: true,
      shadowedRange: { start: second.seq, end: second.seq },
      shadowedSeqs: [second.seq],
      shadowedTokenCount: 10,
      provider: 'test',
      model: 'test',
    })
    session.append('user/message', createUserMessage({
      content: [{ type: 'text', text: 'checkpoint' }],
      source: compactCheckpointSource(compactionId),
    }), {
      surfaceOp: { op: 'replace', startSeq: second.seq, endSeq: second.seq },
      sourceEventSeqs: [start.seq, summary.seq, second.seq],
    })
    session.append('compaction/end', { compactionId, turn: null })
    const query = new FakeQuery(session.snapshotEvents())
    const page = await readHistory(query, SESSION_ID, summary.seq, undefined, QUERY_CONFIG)
    expect(page).toContain('ORIGINAL-ONLY: the middle value is 417.')
    expect(page).toContain(`showing cited original #${String(original.seq)}`)
  })

  it('rejects a cursor from another checkpoint and a malformed cursor', async () => {
    const { session } = conversation()
    const first = compact(session, [...session.surface.nodes], 'checkpoint one', 'c1')
    const query = new FakeQuery(session.snapshotEvents())
    await expect(readHistory(query, SESSION_ID, first.summarySeq, 'not-a-cursor', QUERY_CONFIG))
      .rejects.toMatchObject({ code: 'HISTORY_CURSOR_INVALID' })
    await expect(readHistory(query, SESSION_ID, first.summarySeq, encodeCursorFor(999), QUERY_CONFIG))
      .rejects.toMatchObject({ code: 'HISTORY_CURSOR_INVALID' })
    await expect(readHistory(query, SESSION_ID, first.summarySeq, 'a'.repeat(5_000), QUERY_CONFIG))
      .rejects.toMatchObject({ code: 'HISTORY_CURSOR_INVALID' })
  })

  it('reconstructs the same transcript after a persistence restart', async () => {
    const { session } = conversation()
    const first = compact(session, [...session.surface.nodes], 'checkpoint one', 'c1')
    const dir = await mkdtemp(join(tmpdir(), 'history-recall-'))
    temporary.push(dir)
    const ctx = new Context()
    const fiber = await ctx.plugin(JsonlSessionPersistence, { root: dir, compression: 'none' })
    try {
      const handle = await ctx.sessionPersistence.create(header())
      await handle.append(session.snapshotEvents())
      await handle.close()
    } finally {
      await fiber.dispose()
    }
    const reopened = new Context()
    const reopenedFiber = await reopened.plugin(JsonlSessionPersistence, { root: dir, compression: 'none' })
    try {
      const handle = await reopened.sessionPersistence.open(SESSION_ID, 'read')
      const stored = (await handle.read()).events
      await handle.close()
      const query = new FakeQuery(stored)
      const page = await readHistory(query, SESSION_ID, first.summarySeq, undefined, QUERY_CONFIG)
      expect(page).toContain('CONSTRAINT: never delete C: files')
      expect(page).toContain('RESULT: build output lives on X:')
    } finally {
      await reopenedFiber.dispose()
    }
  })
})

function encodeCursorFor(seq: number): string {
  return Buffer.from(JSON.stringify({ v: 1, c: seq, s: 1, o: 0 }), 'utf8').toString('base64url')
}
