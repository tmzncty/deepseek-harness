import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import { brandString } from '@deepseek-ai/dsh-brand'
import type { CompactionId } from '@deepseek-ai/dsh-compaction'
import { compactCheckpointSource } from '@deepseek-ai/dsh-compaction/checkpoint'
import { createMessage, createUserMessage } from '@deepseek-ai/dsh-llm'
import { SESSION_FORMAT_VERSION, Session, SessionSeq, type SessionId, type SessionHeader } from '@deepseek-ai/dsh-session'
import SessionStore from '@deepseek-ai/dsh-session'
import * as JsonlSessionPersistence from '@deepseek-ai/dsh-session-persistence-jsonl'
import SessionProjectionRegistry from '@deepseek-ai/dsh-session-projection'
import SqliteSessionQueryEngine from '@deepseek-ai/dsh-session-query-sqlite'
import { resolveConfig } from '../src/config.ts'
import { readHistory } from '../src/recall.ts'

const SESSION_ID = brandString<SessionId>('real-query-session')
const CONFIG = resolveConfig({ maxPageBytes: 1_024 })

async function liveContext(): Promise<Context> {
  const ctx = new Context()
  await ctx.plugin(SessionStore)
  await ctx.plugin(SessionProjectionRegistry)
  await ctx.plugin(SqliteSessionQueryEngine, { path: ':memory:' })
  return ctx
}

function persistedHeader(): SessionHeader {
  return {
    version: SESSION_FORMAT_VERSION,
    id: SESSION_ID,
    createdAt: 1,
    isSeeded: false,
  }
}

function buildConversation(session: Session): { summarySeq: SessionSeq; userSeq: SessionSeq } {
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
  return { summarySeq: summary.seq, userSeq: user.seq }
}

describe('history recall through the real session-query service', () => {
  it('lists and pages a live session checkpoint', async () => {
    const ctx = await liveContext()
    try {
      const session = ctx.sessions.create(SESSION_ID, { meta: { cwd: '/work' } })
      const { summarySeq, userSeq } = buildConversation(session)
      const directory = await readHistory(ctx.sessionQuery, SESSION_ID, undefined, undefined, CONFIG)
      expect(directory).toContain(`c${String(summarySeq)}`)
      const page = await readHistory(ctx.sessionQuery, SESSION_ID, summarySeq, undefined, CONFIG)
      expect(page).toContain('CONSTRAINT: keep C: free.')
      expect(page).toContain(`#${String(userSeq)}`)
    } finally {
      await ctx.fiber.dispose()
    }
  })

  it('pages a checkpoint from the persisted log in a fresh process context', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'history-recall-restart-'))
    try {
      const detached = Session.create(SESSION_ID)
      const { summarySeq } = buildConversation(detached)
      const writerContext = await liveContext()
      try {
        const persistence = await writerContext.plugin(JsonlSessionPersistence, { root: dir, compression: 'none' })
        const writer = await writerContext.sessionPersistence.create(persistedHeader())
        await writer.append(detached.snapshotEvents())
        await writer.close()
        await persistence.dispose()
      } finally {
        await writerContext.fiber.dispose()
      }

      const readerContext = await liveContext()
      try {
        await readerContext.plugin(JsonlSessionPersistence, { root: dir, compression: 'none' })
        const page = await readHistory(readerContext.sessionQuery, SESSION_ID, summarySeq, undefined, CONFIG)
        expect(page).toContain('CONSTRAINT: keep C: free.')
      } finally {
        await readerContext.fiber.dispose()
      }
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })
})
