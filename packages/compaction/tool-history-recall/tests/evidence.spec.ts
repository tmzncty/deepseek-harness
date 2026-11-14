import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { brandString } from '@deepseek-ai/dsh-brand'
import type { SessionId } from '@deepseek-ai/dsh-session'
import { resolveConfig } from '../src/config.ts'
import { evidencePaths, readEvidence, searchEvidence } from '../src/evidence.ts'
import { readCheckpointDirectory } from '../src/recall.ts'
import type { SessionRawLogSnapshot } from '@deepseek-ai/dsh-session-query'

const SESSION = brandString<SessionId>('evidence-session')
let root: string | undefined

afterEach(async () => {
  if (root !== undefined) await rm(root, { recursive: true, force: true })
  root = undefined
})

function snapshot(path: string): SessionRawLogSnapshot {
  return {
    session: { version: 4, id: SESSION, createdAt: 1, isSeeded: false } as never,
    inheritedEventCount: 0 as never,
    events: [{
      type: 'tool/result',
      seq: 1 as never,
      time: 1,
      data: { message: { content: [{ type: 'text', text: `tail\n[output truncated; full output: ${path}]` }] } },
    } as never],
    capturedThroughSeq: 1 as never,
  }
}

function query(path: string) {
  return { async readRawLog() { return snapshot(path) } }
}

describe('saved tool output evidence', () => {
  it('extracts and pages a session spill artifact', async () => {
    root = await mkdtemp(join(tmpdir(), 'dsh-subprocess-test-'))
    const file = join(root, 'stdout.log')
    await writeFile(file, `BEGIN-${'x'.repeat(2_000)}-END`, 'utf8')
    const paths = evidencePaths(snapshot(file).events as never)
    expect([...paths]).toEqual([file])
    const config = resolveConfig({ maxPageBytes: 1_024, maxEvidenceBytes: 8_192 })
    const first = await readEvidence(query(file) as never, SESSION, file, 0, config)
    expect(first).toContain('BEGIN-')
    expect(first).toContain('Continues at offset')
    const rest = await readEvidence(query(file) as never, SESSION, file, 1_000, config)
    expect(rest).toContain('at offset 1000')
    expect(rest).toContain('End of saved evidence')
  })

  it('rejects paths not named by the caller session', async () => {
    root = await mkdtemp(join(tmpdir(), 'dsh-subprocess-test-'))
    const file = join(root, 'stdout.log')
    const other = join(root, 'other.log')
    await writeFile(file, 'secret', 'utf8')
    await writeFile(other, 'other', 'utf8')
    const config = resolveConfig({ maxEvidenceBytes: 8_192 })
    await expect(readEvidence(query(other) as never, SESSION, file, 0, config))
      .rejects.toMatchObject({ code: 'HISTORY_EVIDENCE_INVALID' })
  })

  it('lists saved evidence even when no checkpoint exists', async () => {
    root = await mkdtemp(join(tmpdir(), 'dsh-subprocess-test-'))
    const file = join(root, 'stdout.log')
    await writeFile(file, 'evidence', 'utf8')
    const directory = await readCheckpointDirectory(query(file) as never, SESSION, 8)
    expect(directory).toContain('no compacted checkpoints')
    expect(directory).toContain(file)
    expect(directory).toContain('history_read evidence')
  })

  it('searches a saved evidence file and reports a complete miss', async () => {
    root = await mkdtemp(join(tmpdir(), 'dsh-subprocess-test-'))
    const file = join(root, 'stdout.log')
    await writeFile(file, `before NEEDLE-CODE-9K4X after\n${'x'.repeat(2_000)}`, 'utf8')
    const config = resolveConfig({ maxEvidenceBytes: 8_192, maxSearchHits: 2 })
    const found = await searchEvidence(query(file) as never, SESSION, file, 'needle-code', config)
    expect(found).toContain('1 match')
    expect(found).toContain('NEEDLE-CODE-9K4X')
    const miss = await searchEvidence(query(file) as never, SESSION, file, 'absent-token', config)
    expect(miss).toContain('No literal match')
    await expect(searchEvidence(query(file) as never, SESSION, file, ' ', config))
      .rejects.toMatchObject({ code: 'HISTORY_SEARCH_INVALID_QUERY' })
  })
})
