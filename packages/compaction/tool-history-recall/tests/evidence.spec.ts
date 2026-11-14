import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { brandString } from '@deepseek-ai/dsh-brand'
import { createToolResultMessage, type ToolCallId } from '@deepseek-ai/dsh-llm'
import {
  SESSION_FORMAT_VERSION,
  Session,
  SessionLogOffset,
  type SessionEvent,
  type SessionHeader,
  type SessionId,
} from '@deepseek-ai/dsh-session'
import { resolveConfig } from '../src/config.ts'
import {
  evidencePaths,
  readEvidence,
  readEvidenceText,
  scanEvidenceText,
  searchEvidence,
} from '../src/evidence.ts'
import { readCheckpointDirectory } from '../src/recall.ts'
import { searchHistory } from '../src/search.ts'
import type { RecallQuery } from '../src/types.ts'

const SESSION_ID = brandString<SessionId>('evidence-session')
const CONFIG = resolveConfig({ maxPageBytes: 1_024, maxEvidenceBytes: 8_192, maxSearchHits: 20 })

/** Temp paths created by one case; removed afterwards. */
const temporary: string[] = []
afterEach(async () => {
  for (const path of temporary.splice(0)) await rm(path, { recursive: true, force: true })
})

/** The notice `tool-pwsh` prefixes to a truncated stream. */
function notice(path: string): string {
  return `[output truncated; full output: ${path}]`
}

/** A spill directory named the way the subprocess temp root is. */
async function spillRoot(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), 'dsh-subprocess-'))
  temporary.push(root)
  return root
}

async function spillFile(name: string, body: string): Promise<string> {
  const path = join(await spillRoot(), name)
  await writeFile(path, body, 'utf8')
  return path
}

/** One session whose tool results carry the given retained-output texts. */
function sessionEvents(texts: readonly string[]): readonly SessionEvent[] {
  const session = Session.create(SESSION_ID)
  session.append('turn/start', { turn: 1 })
  session.append('step/start', { turn: 1, step: 1 })
  const callId = brandString<ToolCallId>('evidence-call')
  session.append('tool/call', { turn: 1, step: 1, callId, name: 'pwsh', arguments: '{}' })
  for (const text of texts) {
    session.append('tool/result', {
      turn: 1,
      step: 1,
      message: createToolResultMessage({ callId, content: [{ type: 'text', text }], isError: false }),
    }, { surfaceOp: 'append' })
  }
  return session.snapshotEvents()
}

function sessionQuery(texts: readonly string[]): RecallQuery {
  const events = sessionEvents(texts)
  const header: SessionHeader = {
    version: SESSION_FORMAT_VERSION,
    id: SESSION_ID,
    createdAt: 1,
    isSeeded: false,
  }
  return {
    async readRawLog() {
      return {
        session: header,
        inheritedEventCount: SessionLogOffset(0),
        events: [...events],
        capturedThroughSeq: events.at(-1)?.seq ?? null,
      }
    },
  }
}

describe('saved tool output evidence', () => {
  it('extracts spill paths only from the retained text a tool result names', () => {
    const events: { type: string; data?: unknown }[] = [
      { type: 'turn/start', data: { turn: 1 } },
      { type: 'tool/result' },
      { type: 'tool/result', data: {} },
      { type: 'tool/result', data: { message: {} } },
      {
        type: 'tool/result',
        data: {
          message: {
            content: [
              { type: 'tool-call' },
              { type: 'text', text: 42 },
              { type: 'text' },
              { type: 'text', text: 'no spill notice in this result' },
              { type: 'text', text: `retained tail\n${notice('  /tmp/spill/padded.log  ')}` },
            ],
          },
        },
      },
    ]

    expect([...evidencePaths(events)]).toEqual(['/tmp/spill/padded.log'])
  })

  it('pages a prefix-announced artifact across the byte bound without splitting characters', async () => {
    const body = `${'😀'.repeat(600)}TAIL-MARKER`
    const characters = Array.from(body).length
    const file = await spillFile('stdout.log', body)
    // `tool-pwsh` now puts the notice first so tail truncation cannot drop it.
    const query = sessionQuery([`${notice(file)}\n\nretained tail`])

    let offset = 0
    let combined = ''
    for (let page = 0; page < 12; page += 1) {
      const text = await readEvidence(query, SESSION_ID, file, offset, CONFIG)
      expect(text).toContain(`(${String(characters)} characters) at offset ${String(offset)}:`)
      combined += text
      const next = /Continues at offset (\d+)/u.exec(text)?.[1]
      if (next === undefined) {
        expect(text).toContain('End of saved evidence')
        break
      }
      offset = Number(next)
    }

    expect(combined).not.toContain('\uFFFD')
    expect(combined.match(/😀/gu)?.length).toBe(600)
    expect(combined).toContain('TAIL-MARKER')
  })

  it('rejects a negative, fractional, or past-the-end evidence offset', async () => {
    const file = await spillFile('bounds.log', 'short text')
    const query = sessionQuery([notice(file)])

    await expect(readEvidence(query, SESSION_ID, file, -1, CONFIG))
      .rejects.toMatchObject({ code: 'HISTORY_EVIDENCE_INVALID' })
    await expect(readEvidence(query, SESSION_ID, file, 1.5, CONFIG))
      .rejects.toMatchObject({ code: 'HISTORY_EVIDENCE_INVALID' })
    await expect(readEvidence(query, SESSION_ID, file, 11, CONFIG))
      .rejects.toThrow(/exceeds the saved text length 10/u)

    const exhausted = await readEvidence(query, SESSION_ID, file, 10, CONFIG)
    expect(exhausted).toContain('at offset 10:')
    expect(exhausted).toContain('End of saved evidence')
  })

  it('rejects a spill path the caller session never named', async () => {
    const file = await spillFile('named.log', 'secret evidence')
    const other = await spillFile('other.log', 'other output')

    await expect(readEvidence(sessionQuery([notice(other)]), SESSION_ID, file, 0, CONFIG))
      .rejects.toMatchObject({ code: 'HISTORY_EVIDENCE_INVALID' })
    await expect(searchEvidence(sessionQuery([notice(other)]), SESSION_ID, file, 'secret', CONFIG))
      .rejects.toMatchObject({ code: 'HISTORY_EVIDENCE_INVALID' })
  })

  it('rejects a relative, out-of-root, absent, or non-artifact evidence path', async () => {
    const relative = 'logs/relative.log'
    await expect(readEvidence(sessionQuery([notice(relative)]), SESSION_ID, relative, 0, CONFIG))
      .rejects.toMatchObject({ code: 'HISTORY_EVIDENCE_INVALID' })

    const outside = process.execPath
    await expect(readEvidence(sessionQuery([notice(outside)]), SESSION_ID, outside, 0, CONFIG))
      .rejects.toThrow(/not a session spill artifact/u)

    const absent = join(await spillRoot(), 'gone.log')
    await expect(readEvidence(sessionQuery([notice(absent)]), SESSION_ID, absent, 0, CONFIG))
      .rejects.toMatchObject({ code: 'HISTORY_EVIDENCE_MISSING' })

    const plainRoot = await mkdtemp(join(tmpdir(), 'plain-evidence-'))
    temporary.push(plainRoot)
    const plain = join(plainRoot, 'evidence.log')
    await writeFile(plain, 'saved outside a spill directory', 'utf8')
    await expect(readEvidence(sessionQuery([notice(plain)]), SESSION_ID, plain, 0, CONFIG))
      .rejects.toThrow(/not a session spill artifact/u)
  })

  it('rejects saved evidence above the byte bound or that is not a regular file', async () => {
    const large = await spillFile('large.log', 'x'.repeat(2_048))
    const tight = resolveConfig({ maxEvidenceBytes: 1_024 })
    await expect(readEvidence(sessionQuery([notice(large)]), SESSION_ID, large, 0, tight))
      .rejects.toMatchObject({ code: 'HISTORY_EVIDENCE_TOO_LARGE' })
    await expect(readEvidenceText(large, tight))
      .rejects.toMatchObject({ code: 'HISTORY_EVIDENCE_TOO_LARGE' })

    const directory = join(await spillRoot(), 'stdout.log')
    await mkdir(directory)
    await expect(readEvidence(sessionQuery([notice(directory)]), SESSION_ID, directory, 0, CONFIG))
      .rejects.toMatchObject({ code: 'HISTORY_EVIDENCE_MISSING' })
    await expect(readEvidenceText(directory, CONFIG))
      .rejects.toMatchObject({ code: 'HISTORY_EVIDENCE_MISSING' })
  })

  it('reads and scans saved text directly once ownership is established', async () => {
    const body = `head NEEDLE-B${'x'.repeat(300)}NEEDLE-B tail`
    const file = await spillFile('direct.log', body)
    await expect(readEvidenceText(file, CONFIG)).resolves.toBe(body)

    const hits = scanEvidenceText(file, body, 'needle-b', 2)
    expect(hits).toHaveLength(2)
    expect(hits[0]).toContain(`- evidence ${file} offset 5: …head NEEDLE-B`)
    expect(scanEvidenceText(file, body, 'absent-token', 2)).toEqual([])
    expect(scanEvidenceText(file, 'NEEDLE-B NEEDLE-B NEEDLE-B', 'needle-b', 2)).toHaveLength(2)
  })

  it('searches saved evidence for a literal phrase and reports a complete miss', async () => {
    const file = await spillFile('search.log', `before NEEDLE-CODE-9K4X after\n${'x'.repeat(2_000)}NEEDLE-CODE-9K4X again`)
    const query = sessionQuery([notice(file)])
    const tight = resolveConfig({ maxEvidenceBytes: 8_192, maxSearchHits: 2 })

    const found = await searchEvidence(query, SESSION_ID, file, 'needle-code', tight)
    expect(found).toContain('2 match(es)')
    expect(found).toContain('- offset 7: …before NEEDLE-CODE-9K4X after')

    const miss = await searchEvidence(query, SESSION_ID, file, 'absent-token', tight)
    expect(miss).toContain('No literal match in the saved evidence')

    await expect(searchEvidence(query, SESSION_ID, file, '   ', tight))
      .rejects.toMatchObject({ code: 'HISTORY_SEARCH_INVALID_QUERY' })
    await expect(searchEvidence(query, SESSION_ID, file, 'x'.repeat(2_049), tight))
      .rejects.toMatchObject({ code: 'HISTORY_SEARCH_INVALID_QUERY' })
  })

  it('lists saved evidence in the directory under any listing limit', async () => {
    const first = await spillFile('a.log', 'alpha')
    const second = await spillFile('b.log', 'beta')
    const query = sessionQuery([`${notice(first)}\n\nalpha tail`, `${notice(second)}\n\nbeta tail`])

    const listed = await readCheckpointDirectory(query, SESSION_ID, 8)
    expect(listed).toContain('no compacted checkpoints')
    expect(listed).toContain('Saved full-output evidence (2):')
    expect(listed).toContain(`- ${first} (read with history_read evidence; search with history_search evidence)`)

    const unlimited = await readCheckpointDirectory(query, SESSION_ID, 0)
    expect(unlimited).toContain('Saved full-output evidence (2):')
    expect(unlimited).toContain(`- ${first} (`)
    expect(unlimited).toContain(`- ${second} (`)

    // A negative listing limit names only the newest artifact.
    const negative = await readCheckpointDirectory(query, SESSION_ID, -1)
    expect(negative).toContain('Saved full-output evidence (2):')
    expect(negative).not.toContain(`- ${first} (`)
    expect(negative).toContain(`- ${second} (`)
  })

  it('scans saved evidence through the span search with hits, both caps, and a miss', async () => {
    const first = await spillFile('span-a.log', `before SPAN-NEEDLE after${'y'.repeat(200)}`)
    const second = await spillFile('span-b.log', 'SPAN-NEEDLE again')
    const both = sessionQuery([`${notice(first)}\n\ntail`, `${notice(second)}\n\ntail`])

    const scanned = await searchHistory(both, SESSION_ID, 'SPAN-NEEDLE', undefined, CONFIG)
    expect(scanned).toContain('2 evidence file(s)')
    expect(scanned).toContain(`- evidence ${first} offset 7:`)
    expect(scanned).toContain(`- evidence ${second} offset 0:`)
    expect(scanned).toContain('2 match(es)')

    // One hit already meets the cap, so the second file is never opened.
    const cappedHits = resolveConfig({ maxSearchHits: 1 })
    const capped = await searchHistory(both, SESSION_ID, 'SPAN-NEEDLE', undefined, cappedHits)
    expect(capped).toContain('1 match(es)')
    expect(capped).toContain('Scan stopped at a deployment bound')

    const large = await spillFile('big.log', 'z'.repeat(2_000))
    const cappedBytes = resolveConfig({ maxSearchBytes: 1_024 })
    const bytes = await searchHistory(
      sessionQuery([`${notice(large)}\n\ntail`]),
      SESSION_ID,
      'SPAN-NEEDLE',
      undefined,
      cappedBytes,
    )
    expect(bytes).toContain('0 evidence file(s)')
    expect(bytes).toContain('Scan stopped at a deployment bound')

    const miss = await searchHistory(both, SESSION_ID, 'absent-token', undefined, CONFIG)
    expect(miss).toContain('No literal match')
  })
})
