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
  type SessionId,
} from '@deepseek-ai/dsh-session'
import * as JsonlSessionPersistence from '@deepseek-ai/dsh-session-persistence-jsonl'
import { SessionQueryError } from '@deepseek-ai/dsh-session-query'
import type {
  SessionEventRangeObservation,
  SessionEventRecord,
  SessionEventReadRequest,
  SessionEventWindow,
} from '@deepseek-ai/dsh-session-query'
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
  readonly resultSeq: SessionSeq
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
  session.append('assistant/message', {
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
  session.append('user/message', createUserMessage({
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
  return { session, constraintSeq: constraint.seq, resultSeq: result.seq }
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

  async listEvents(sessionId: SessionId): Promise<SessionEventRecord[]> {
    return this.events.map(event => ({
      sessionId,
      seq: event.seq,
      type: event.type,
      time: event.time,
      surface: 'shadowed',
    }))
  }

  async readEvent(request: SessionEventReadRequest): Promise<SessionEventWindow> {
    const target = this.events[request.seq]
    if (target === undefined || this.failSeq === request.seq) {
      throw new SessionQueryError(`missing ${String(request.seq)}`, 'SESSION_QUERY_EVENT_NOT_FOUND')
    }
    const start = Math.max(0, request.seq - (request.before ?? 0))
    const end = Math.min(this.events.length - 1, request.seq + (request.after ?? 0))
    return {
      session: header(),
      inheritedEventCount: SessionLogOffset(0),
      target,
      events: this.events.slice(start, end + 1),
      startSeq: SessionSeq(start),
      endSeq: SessionSeq(end),
    }
  }

  async readEventRange(request: {
    sessionId: SessionId
    seqs: readonly SessionSeq[]
  }): Promise<SessionEventRangeObservation> {
    return {
      session: header(),
      inheritedEventCount: SessionLogOffset(0),
      events: request.seqs.map((seq) => {
        const event = this.events[seq]
        if (event === undefined || this.failSeq === seq) {
          throw new SessionQueryError(`missing ${String(seq)}`, 'SESSION_QUERY_EVENT_NOT_FOUND')
        }
        return event
      }),
    }
  }
}

function header() {
  return {
    version: SESSION_FORMAT_VERSION,
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
    const { session } = conversation()
    const first = compact(session, [...session.surface.nodes], 'checkpoint one', 'c1')
    const second = compact(
      session,
      [first.replacementSeq],
      'checkpoint two merges checkpoint one',
      'c2',
    )
    const query = new FakeQuery(session.snapshotEvents())
    const checkpoints = await listCheckpoints(query, SESSION_ID)
    expect(checkpoints.map(checkpoint => checkpoint.summarySeq)).toEqual([first.summarySeq, second.summarySeq])
    expect(checkpoints[0]?.superseded).toBe(true)
    const older = await readHistory(query, SESSION_ID, first.summarySeq, undefined, QUERY_CONFIG)
    expect(older).toContain('CONSTRAINT: never delete C: files')
    expect(older).toContain('REVISION: X: is now the only build disk')
    const newer = await readHistory(query, SESSION_ID, second.summarySeq, undefined, QUERY_CONFIG)
    expect(newer).toContain('prior compaction checkpoint')
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

  it('rejects a cursor from another checkpoint and a malformed cursor', async () => {
    const { session } = conversation()
    const first = compact(session, [...session.surface.nodes], 'checkpoint one', 'c1')
    const query = new FakeQuery(session.snapshotEvents())
    await expect(readHistory(query, SESSION_ID, first.summarySeq, 'not-a-cursor', QUERY_CONFIG))
      .rejects.toMatchObject({ code: 'HISTORY_CURSOR_INVALID' })
    await expect(readHistory(query, SESSION_ID, first.summarySeq, encodeCursorFor(999), QUERY_CONFIG))
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


