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
import type { SessionRawLogSnapshot } from '@deepseek-ai/dsh-session-query'
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

  async readRawLog(): Promise<SessionRawLogSnapshot> {
    const events = this.failSeq === undefined
      ? this.events
      : this.events.filter(event => event.seq !== this.failSeq)
    return {
      session: header(),
      inheritedEventCount: SessionLogOffset(0),
      events: [...events],
      capturedThroughSeq: events.at(-1)?.seq ?? null,
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

interface PruneChain {
  readonly session: Session
  readonly summarySeq: SessionSeq
  /** Tool results in the chain, oldest first: `chain[0]` is the original. */
  readonly chain: readonly SessionEvent[]
}

/**
 * A checkpoint whose shadowed node is the tail of a true prune-citation chain:
 * every replacement rewrites the previous node and cites exactly it, so the
 * original text is only reachable by following the citations.
 * @param hops - replacement links between the observed node and the original.
 */
function pruneChain(hops: number): PruneChain {
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
  const chain: SessionEvent[] = [original]
  let tail = original
  for (let hop = 0; hop < hops; hop += 1) {
    const replacement = session.append('tool/result', {
      turn: 1,
      step: 1,
      message: { ...original.data.message, content: [{ type: 'text', text: `pruned view ${String(hop)}` }] },
    }, {
      surfaceOp: { op: 'replace', startSeq: tail.seq, endSeq: tail.seq },
      sourceEventSeqs: [tail.seq],
    })
    session.append('compaction/prune', {
      shadowedRange: { start: tail.seq, end: tail.seq },
      shadowedSeqs: [tail.seq],
      shadowedTokenCount: 1,
    })
    chain.push(replacement)
    tail = replacement
  }
  const { summarySeq } = compact(session, [tail.seq], 'pruned checkpoint', `prune-${String(hops)}`)
  return { session, summarySeq, chain }
}

/**
 * Append a compaction summary over one span without any replacement node.
 * @param session - session receiving the checkpoint.
 * @param shadowedSeqs - span the summary claims to shadow.
 * @param label - compaction identity for the appended events.
 * @param close - whether to also append the matching `compaction/end`.
 * @returns the summary seq, which addresses the checkpoint.
 */
function appendSummary(
  session: Session,
  shadowedSeqs: readonly SessionSeq[],
  label: string,
  close: boolean,
): SessionSeq {
  const compactionId = brandString<CompactionId>(label)
  session.append('compaction/start', { compactionId, turn: null })
  const summary = session.append('compaction/summary', {
    compactionId,
    summary: [{ type: 'text', text: 'condensed' }],
    rawOutput: [{ type: 'text', text: 'condensed' }],
    llmStreamCall: true,
    shadowedRange: { start: shadowedSeqs[0] ?? SessionSeq(0), end: shadowedSeqs.at(-1) ?? SessionSeq(0) },
    shadowedSeqs: [...shadowedSeqs],
    shadowedTokenCount: 10,
    provider: 'test',
    model: 'test',
  })
  if (close) session.append('compaction/end', { compactionId, turn: null })
  return summary.seq
}

type UnfinishedShape = 'summary-last' | 'ended' | 'appended' | 'foreign-citation'

/**
 * A session whose single checkpoint never landed a replacement node, in the
 * four shapes a durable log can record: the summary is its last event, another
 * event follows it, the following user message was appended instead of
 * replacing the span, and the following user message replaced the span without
 * citing the summary.
 * @param shape - which of those logs to build.
 */
function unfinishedCheckpoint(shape: UnfinishedShape): Session {
  const { session, constraintSeq } = conversation()
  appendSummary(session, [constraintSeq], `unfinished-${shape}`, shape === 'ended')
  if (shape === 'summary-last' || shape === 'ended') return session
  const appended = shape === 'appended'
  const source = appended ? { kind: 'user' as const } : compactCheckpointSource(brandString<CompactionId>(`unfinished-${shape}`))
  const surface = appended
    ? { surfaceOp: 'append' as const }
    : {
        surfaceOp: { op: 'replace' as const, startSeq: constraintSeq, endSeq: constraintSeq },
        sourceEventSeqs: [constraintSeq],
      }
  session.append('user/message', createUserMessage({
    content: [{ type: 'text', text: appended ? 'appended after the summary' : 'replacement citing only the span' }],
    source,
  }), surface)
  return session
}

/** Opaque cursor text naming an arbitrary resume position. */
function cursorAt(checkpoint: number, seq: number, offset: number): string {
  return Buffer.from(JSON.stringify({ v: 1, c: checkpoint, s: seq, o: offset }), 'utf8').toString('base64url')
}

describe('history recall over compaction checkpoints', () => {
  it('lists checkpoints without any model-visible pointer in the summary', async () => {
    const { session } = conversation()
    const first = compact(session, [...session.surface.nodes], 'checkpoint one', 'c1')
    const query = new FakeQuery(session.snapshotEvents())
    const directory = await readCheckpointDirectory(query, SESSION_ID, 8)
    expect(directory).toContain(`c${String(first.summarySeq)}`)
    expect(directory).toContain('completed')
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
    const { session, summarySeq, chain } = pruneChain(2)
    const query = new FakeQuery(session.snapshotEvents())
    const page = await readHistory(query, SESSION_ID, summarySeq, undefined, QUERY_CONFIG)
    expect(page).toContain('ORIGINAL-ONLY: the middle value is 417.')
    expect(page).toContain(`showing cited original #${String(chain[0]?.seq)}`)
  })

  it('names a prune citation whose cited original is absent from the log', async () => {
    const { session, summarySeq, chain } = pruneChain(2)
    const firstReplacement = chain[1]
    if (firstReplacement === undefined) throw new Error('fixture lost its first replacement')
    const query = new FakeQuery(session.snapshotEvents(), firstReplacement.seq)
    await expect(readHistory(query, SESSION_ID, summarySeq, undefined, QUERY_CONFIG))
      .rejects.toMatchObject({ code: 'HISTORY_EVIDENCE_MISSING' })
  })

  it('rejects a cyclic prune citation chain', async () => {
    const { session, summarySeq, chain } = pruneChain(2)
    const firstReplacement = chain[1]
    const tail = chain[2]
    if (firstReplacement === undefined || tail === undefined) throw new Error('fixture is too short to cycle')
    // Only a corrupt or hand-edited durable log can cite a later event, so the
    // cycle is injected into the loaded events instead of appended.
    const cycled: SessionEvent[] = session.snapshotEvents().map(event => (
      event.seq === firstReplacement.seq
        ? { ...event, sourceEventSeqs: [tail.seq] } as unknown as SessionEvent
        : event
    ))
    await expect(readHistory(new FakeQuery(cycled), SESSION_ID, summarySeq, undefined, QUERY_CONFIG))
      .rejects.toMatchObject({ code: 'HISTORY_EVIDENCE_INVALID' })
  })

  it('rejects a prune citation chain longer than the hop bound', async () => {
    const { session, summarySeq } = pruneChain(9)
    await expect(readHistory(new FakeQuery(session.snapshotEvents()), SESSION_ID, summarySeq, undefined, QUERY_CONFIG))
      .rejects.toThrow(/exceeds 8 hops/u)
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

  it('marks a checkpoint unfinished when no replacement node landed', async () => {
    const shapes: readonly UnfinishedShape[] = ['summary-last', 'ended', 'appended', 'foreign-citation']
    for (const shape of shapes) {
      const query = new FakeQuery(unfinishedCheckpoint(shape).snapshotEvents())
      const checkpoints = await listCheckpoints(query, SESSION_ID)
      expect(checkpoints, shape).toHaveLength(1)
      expect(checkpoints[0], shape).toMatchObject({ state: 'unfinished', superseded: false })
      expect(checkpoints[0]?.replacementSeq, shape).toBeUndefined()
      await expect(readCheckpointDirectory(query, SESSION_ID, 8), shape).resolves.toContain('(unfinished)')
    }
  })

  it('bounds the checkpoint directory by its configured limit', async () => {
    const { session, constraintSeq, revisionSeq } = conversation()
    const first = compact(session, [constraintSeq], 'checkpoint one', 'c1')
    const second = compact(session, [revisionSeq], 'checkpoint two', 'c2')
    const query = new FakeQuery(session.snapshotEvents())

    const hidden = await readCheckpointDirectory(query, SESSION_ID, 0)
    expect(hidden).toContain('Compacted checkpoints (2):')
    expect(hidden).toContain('2 older checkpoint(s) omitted')
    expect(hidden).not.toContain(`c${String(first.summarySeq)}: shadows`)
    expect(hidden).not.toContain(`c${String(second.summarySeq)}: shadows`)

    const newest = await readCheckpointDirectory(query, SESSION_ID, 1)
    expect(newest).toContain(`c${String(second.summarySeq)}: shadows`)
    expect(newest).not.toContain(`c${String(first.summarySeq)}: shadows`)
    expect(newest).toContain('1 older checkpoint(s) omitted')
  })

  it('rejects a checkpoint seq that is not a non-negative safe integer', async () => {
    const { session } = conversation()
    compact(session, [...session.surface.nodes], 'checkpoint one', 'c1')
    const query = new FakeQuery(session.snapshotEvents())
    await expect(readHistory(query, SESSION_ID, -1, undefined, QUERY_CONFIG))
      .rejects.toMatchObject({ code: 'HISTORY_CHECKPOINT_NOT_FOUND' })
    await expect(readHistory(query, SESSION_ID, 1.5, undefined, QUERY_CONFIG))
      .rejects.toMatchObject({ code: 'HISTORY_CHECKPOINT_NOT_FOUND' })
  })

  it('validates a cursor against the span, the event, and the text it names', async () => {
    const { session, constraintSeq } = conversation()
    const first = compact(session, [constraintSeq], 'checkpoint one', 'c1')
    const query = new FakeQuery(session.snapshotEvents())
    const constraint = session.snapshotEvents().find(event => event.seq === constraintSeq)
    if (constraint?.type !== 'user/message') throw new Error('fixture is not a user message')
    const block = constraint.data.content[0]
    if (block?.type !== 'text') throw new Error('fixture has no text block')
    const length = [...block.text].length

    await expect(readHistory(query, SESSION_ID, first.summarySeq, cursorAt(first.summarySeq, constraintSeq + 500, 0), QUERY_CONFIG))
      .rejects.toMatchObject({ code: 'HISTORY_CURSOR_INVALID' })
    await expect(readHistory(query, SESSION_ID, first.summarySeq, cursorAt(first.summarySeq, constraintSeq, length + 1), QUERY_CONFIG))
      .rejects.toMatchObject({ code: 'HISTORY_CURSOR_INVALID' })

    const exhausted = await readHistory(query, SESSION_ID, first.summarySeq, cursorAt(first.summarySeq, constraintSeq, length), QUERY_CONFIG)
    expect(exhausted).toContain('End of checkpoint')
    expect(exhausted).not.toContain('Continues at event')
  })

  it('rejects a blank or over-long search query', async () => {
    const { session } = conversation()
    const first = compact(session, [...session.surface.nodes], 'checkpoint one', 'c1')
    const query = new FakeQuery(session.snapshotEvents())
    await expect(searchHistory(query, SESSION_ID, '   ', first.summarySeq, QUERY_CONFIG))
      .rejects.toMatchObject({ code: 'HISTORY_SEARCH_INVALID_QUERY' })
    await expect(searchHistory(query, SESSION_ID, 'x'.repeat(2_049), first.summarySeq, QUERY_CONFIG))
      .rejects.toMatchObject({ code: 'HISTORY_SEARCH_INVALID_QUERY' })
  })

  it('reports a session with no compacted span to search', async () => {
    const { session } = conversation()
    await expect(searchHistory(new FakeQuery(session.snapshotEvents()), SESSION_ID, 'C:', undefined, QUERY_CONFIG))
      .resolves.toContain('no compacted checkpoints')
  })

  it('searches every span and skips a duplicate citation of one original', async () => {
    const session = Session.create(SESSION_ID)
    session.append('turn/start', { turn: 1 })
    session.append('step/start', { turn: 1, step: 1 })
    const callId = brandString<ToolCallId>('duplicate-call')
    session.append('tool/call', { turn: 1, step: 1, callId, name: 'pwsh', arguments: '{}' })
    const original = session.append('tool/result', {
      turn: 1,
      step: 1,
      message: createToolResultMessage({
        callId,
        content: [{ type: 'text', text: 'DUPLICATE-NEEDLE lives only here.' }],
        isError: false,
      }),
    }, { surfaceOp: 'append' })
    // An interrupted compaction leaves the span addressable, the pruner then
    // rewrites the observed node, and a later compaction finalizes the pruned
    // view. Both spans still cite the same original, so the second is a
    // duplicate citation of it.
    const older = appendSummary(session, [original.seq], 'interrupted span', false)
    const pruned = session.append('tool/result', {
      turn: 1,
      step: 1,
      message: { ...original.data.message, content: [{ type: 'text', text: 'pruned view' }] },
    }, {
      surfaceOp: { op: 'replace', startSeq: original.seq, endSeq: original.seq },
      sourceEventSeqs: [original.seq],
    })
    const newer = compact(session, [pruned.seq], 'shadowing the pruned view', 'c2').summarySeq
    const result = await searchHistory(new FakeQuery(session.snapshotEvents()), SESSION_ID, 'duplicate-needle', undefined, QUERY_CONFIG)
    expect(result).toContain('Scanned 2 of 2 checkpoint(s)')
    expect(result).toContain('1 match(es)')
    expect(result).toContain('1 duplicate citation(s) skipped')
    expect(result).toContain(`- c${String(older)} #${String(original.seq)}`)
    expect(result).not.toContain(`- c${String(newer)}`)
  })

  it('caps a scan at the byte budget and reports the miss', async () => {
    const { session } = conversation()
    appendBulk(session)
    const first = compact(session, [...session.surface.nodes], 'checkpoint one', 'c1')
    const tight = resolveConfig({ maxPageBytes: 1_024, maxSearchHits: 5, maxSearchBytes: 1_024 })
    const result = await searchHistory(new FakeQuery(session.snapshotEvents()), SESSION_ID, 'BULK', first.summarySeq, tight)
    expect(result).toContain('Scan stopped at a deployment bound')
    expect(result).toContain('No literal match')
  })

  it('caps a scan at the hit bound', async () => {
    const session = Session.create(SESSION_ID)
    session.append('turn/start', { turn: 1 })
    const shadowed: SessionSeq[] = []
    for (let index = 0; index < 4; index += 1) {
      const message = session.append('user/message', createUserMessage({
        content: [{ type: 'text', text: `REPEATED-NEEDLE ${String(index)}` }],
        source: { kind: 'user' },
      }), { surfaceOp: 'append' })
      shadowed.push(message.seq)
    }
    const first = compact(session, shadowed, 'many matches', 'c1')
    const tight = resolveConfig({ maxPageBytes: 1_024, maxSearchHits: 2, maxSearchBytes: 100_000 })
    const result = await searchHistory(new FakeQuery(session.snapshotEvents()), SESSION_ID, 'repeated-needle', first.summarySeq, tight)
    expect(result).toContain('2 match(es)')
    expect(result).toContain('Scan stopped at a deployment bound')
  })

  it('bounds a distant snippet with ellipses and a length cap', async () => {
    const padding = 'x'.repeat(200)
    const session = Session.create(SESSION_ID)
    session.append('turn/start', { turn: 1 })
    const message = session.append('user/message', createUserMessage({
      content: [{ type: 'text', text: `${padding} FIND-ME ${padding}` }],
      source: { kind: 'user' },
    }), { surfaceOp: 'append' })
    const first = compact(session, [message.seq], 'long record', 'c1')
    const result = await searchHistory(new FakeQuery(session.snapshotEvents()), SESSION_ID, 'FIND-ME', first.summarySeq, QUERY_CONFIG)
    const hit = result.split('\n').find(line => line.startsWith('- c'))
    expect(hit).toMatch(/….*FIND-ME.*…/u)
    expect(hit?.length ?? 0).toBeLessThan(260)
    expect(result).not.toContain(padding)
  })

  it('reports the scanned size in KB and in MB', async () => {
    const bulk = conversation()
    appendBulk(bulk.session)
    const bulkCheckpoint = compact(bulk.session, [...bulk.session.surface.nodes], 'bulk checkpoint', 'c1')
    const kilobytes = await searchHistory(
      new FakeQuery(bulk.session.snapshotEvents()),
      SESSION_ID,
      'BULK',
      bulkCheckpoint.summarySeq,
      QUERY_CONFIG,
    )
    expect(kilobytes).toMatch(/\d+\.\d KB/u)

    const huge = Session.create(SESSION_ID)
    huge.append('turn/start', { turn: 1 })
    const message = huge.append('user/message', createUserMessage({
      content: [{ type: 'text', text: `${'y'.repeat(1_100_000)} BIG-NEEDLE` }],
      source: { kind: 'user' },
    }), { surfaceOp: 'append' })
    const hugeCheckpoint = compact(huge, [message.seq], 'huge checkpoint', 'c1')
    const roomy = resolveConfig({ maxPageBytes: 1_024, maxSearchHits: 5, maxSearchBytes: 5_000_000 })
    const megabytes = await searchHistory(
      new FakeQuery(huge.snapshotEvents()),
      SESSION_ID,
      'BIG-NEEDLE',
      hugeCheckpoint.summarySeq,
      roomy,
    )
    expect(megabytes).toMatch(/\d+\.\d MB/u)
  })

  it('stops a search when the caller aborts', async () => {
    const { session } = conversation()
    const first = compact(session, [...session.surface.nodes], 'checkpoint one', 'c1')
    const controller = new AbortController()
    controller.abort()
    await expect(searchHistory(
      new FakeQuery(session.snapshotEvents()),
      SESSION_ID,
      'CONSTRAINT',
      first.summarySeq,
      QUERY_CONFIG,
      controller.signal,
    )).rejects.toThrow()
  })
})

function encodeCursorFor(seq: number): string {
  return Buffer.from(JSON.stringify({ v: 1, c: seq, s: 1, o: 0 }), 'utf8').toString('base64url')
}
