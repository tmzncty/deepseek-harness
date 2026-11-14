---
description: "Model-facing bounded recall of the original events a compaction checkpoint shadows, for agent developers and maintainers recovering constraints, decisions, errors, and tool results after long-session condensation."
kind: "package-reference"
---

# @deepseek-ai/dsh-tool-history-recall

English | [中文](README.zh.md)

## Summary

Use `dsh-tool-history-recall` to let the model list this session's compaction checkpoints, page through the exact original events one checkpoint shadows, and search those originals for a literal phrase. Each call takes one shallow `ctx.sessionQuery.readRawLog` observation and builds one seq index shared by every page batch, so compacted summaries stay the current context while recovered evidence arrives as ordinary, logged tool output. Model-visible reads are byte-bounded and cursor-resumable; the package never rewrites a summary, the surface, or the request prefix.

## Table of Contents

- [Use this package](#use-this-package)
- [Understand the implementation](#understand-the-implementation)
- [Further Exploration](#further-exploration)
- [Model Experience](#model-experience)
- [Known Limitations and Deferred Work](#known-limitations-and-deferred-work)

-----

<a id="use-this-package"></a>
## Use this package

Mount this package beside the compaction backend when the model should be able to recover what a checkpoint condensed. `dsh-base` mounts it in the shipped composition.

### Configuration

| Field | Default | Meaning |
|---|---|---|
| `maxPageBytes` | `16384` | Maximum UTF-8 bytes in one transcript page |
| `maxSearchHits` | `20` | Maximum literal-search hits returned by one call |
| `maxSearchBytes` | `2000000` | Maximum rendered bytes one search scans |
| `maxListedCheckpoints` | `8` | Maximum checkpoints named in a directory or page header |

### What the model can do

| Tool | What the model gets |
|---|---|
| `history_read` | Without `checkpoint`: this session's compacted checkpoints. With `checkpoint`: one bounded transcript page of the exact original events, plus a cursor when more remain. |
| `history_search` | Literal, case-insensitive snippets from every compacted span, or one checkpoint, with checkpoint and event seqs plus scan coverage. |

Checkpoints are addressed by the seq of their log-only `compaction/summary` event. The model obtains those seqs from the directory or a page header; no footer is added to the model-visible summary.

### Failures and recovery

Reads fail closed with `HarnessError` codes: `HISTORY_CHECKPOINT_NOT_FOUND` for an unknown checkpoint, `HISTORY_EVIDENCE_MISSING` when a cited event is absent from the loaded log, `HISTORY_EVIDENCE_INVALID` when a cited event cannot carry a transcript message, `HISTORY_CURSOR_INVALID` for a malformed or stale cursor, `HISTORY_PAGE_BUDGET_TOO_SMALL` for an unusable page budget, and `HISTORY_SEARCH_INVALID_QUERY` for an empty or oversized query. A missing event is never skipped silently.

-----

<a id="understand-the-implementation"></a>
## Understand the implementation

<details>
<summary>Implementation internals — click to expand</summary>

### Design philosophy

The package adds addressability and bounded paging, not another summary. It commits to four rules:

- **The log is the only source.** Transcripts come from durable events, not the current surface projection, so they survive later compactions and process restarts.
- **Paging is explicit.** `history_read` returns at most `maxPageBytes` UTF-8 bytes and a base64url cursor naming the checkpoint, event seq, and character offset. A cursor is validated against the checkpoint's current shadowed span before it is honored.
- **Prune replacements resolve to their original.** A tool-result prune replacement cites its original through `sourceEventSeqs`; the transcript shows that original and labels the cited replacement.
- **No new storage or index.** Checkpoints are discovered from `compaction/summary` events; spans come from `shadowedSeqs`.

### Source map

| File | Role |
|---|---|
| [`src/index.ts`](src/index.ts) | Plugin entry: config, prompt section, two tool registrations |
| [`src/checkpoints.ts`](src/checkpoints.ts) | Bounded checkpoint discovery and directory rendering |
| [`src/records.ts`](src/records.ts) | Shadowed-seq chunk loading and prune-original resolution |
| [`src/render.ts`](src/render.ts) | Durable event to labeled transcript record |
| [`src/pagination.ts`](src/pagination.ts) | UTF-8 byte budget, cursor encoding, and page framing |
| [`src/search.ts`](src/search.ts) | Bounded literal scan over compacted spans |

### Concurrency and cancellation

Both tools are read-only and concurrency-safe. Every `ctx.sessionQuery` call receives the tool execution's abort signal; an aborted scan stops before its next batch. The package registers no global state.

</details>

-----

<a id="further-exploration"></a>
## Further Exploration

- [Compaction](../compaction/README.md) — the checkpoint contract, `shadowedRange`, and `shadowedSeqs`.
- [Compaction basic](../compaction-basic/README.md) — the shipped backend that writes the checkpoints this package reads.
- [Session query](../../session-query/session-query/README.md) — the asynchronous session observation behind both tools.
- [Session persistence](../../session/session-persistence/README.md) — the durable log that survives restart.

-----

<a id="model-experience"></a>
## Model Experience

### System prompt guidance

#### What the model sees

One fixed section is present while the package is mounted.

##### Fixed guidance

```markdown
Long sessions may replace older conversation with a compacted checkpoint. Use history_read to list this session's checkpoints and page through the exact original events one shadows, and history_search to find a literal phrase in those originals. Recalled text arrives as ordinary tool output; the checkpoint summary remains the current context.
```

#### Token effect

One fixed short section on every request while mounted.

#### KV Cache effect

Prefix-stable while the section and tool visibility are unchanged. The package does not alter the compacted summary or any earlier request token.

### Tool schemas

#### What the model sees

Two flat schemas: `history_read(checkpoint?, cursor?)` and `history_search(query, checkpoint?)`. No model-controlled page size, offset, or scan bound is exposed. The generated [tool catalog](../../../docs/tool-catalog.md#deepseek-aidsh-tool-history-recall) owns the exact schema text.

#### Token effect

Two fixed read-only schemas on every request while visible.

#### KV Cache effect

Prefix-stable while tool visibility and definitions are unchanged.

### Tool results

#### What the model sees

`history_read` returns a checkpoint header, labeled transcript records, and either an end marker or a continuation cursor. `history_search` returns snippets with checkpoint and event seqs plus `scanned` coverage. Recovered text is ordinary tool output and is logged like any other result.

#### Token effect

Data-dependent and bounded by `maxPageBytes` per read and `maxSearchHits`/`maxSearchBytes` per search; recalled bytes remain in logged tool history until later compaction.

#### KV Cache effect

Append-only result text follows the reusable request prefix and does not invalidate earlier cache entries. The package never rewrites the summary checkpoint that precedes the tail.

### Dev Note

<details>
<summary>Working context for maintainers — click to expand</summary>

This Dev Note is non-authoritative; shipped behavior lives in the sections above and the code.

- **Saved evidence is an address, not a copy.** A shell result names a subprocess spill file; `history_read`/`history_search` accept that exact session-named path and read it only after checking the private `dsh-subprocess-*` temp root. The model-visible result stays bounded; the artifact may be removed by external temp cleanup, in which case the tool reports the evidence as missing rather than pretending it is complete.
- **Retention is bounded per file.** `maxEvidenceBytes` (default 16 MiB) rejects larger artifacts explicitly. There is no cross-session or arbitrary-path read.
- **The pwsh/cache boundary is separate.** `readRawLog` is still one observation; evidence reads are separate bounded file reads and do not change checkpoint semantics.

</details>

## Known Limitations and Deferred Work

- **Saved evidence can disappear.** A shell result names a subprocess spill file; the recall tools can read it only while that exact session-named artifact exists under the private temp root. External temp cleanup, a missing file, or a file above `maxEvidenceBytes` is reported explicitly; bytes that never reached the log or spill file remain unrecoverable.
- **One raw-log observation per call.** `ctx.sessionQuery.readRawLog` reads the complete raw log once without replay validation or per-event cloning; page bytes are still bounded by `maxPageBytes`, but peak memory grows with the raw log (an 85.8 MB / 114k-event fixture measured ~42 ms cold and ~8 MB extra RSS).
- **Checkpoint guidance is conditional.** Substantial summaries carry a self-directed recall hint and evidence-boundary reminder; short summaries keep the original framing, and no checkpoint seq pointer is embedded in the summary.
- **Literal search only.** Matching is case-insensitive and literal; there is no regex, semantic search, or structured query over JSON tool results.
- **One session, no cross-session recall.** The tools read only the caller's session. Cross-session history remains the experimental `dsh-tool-session-query` surface.
- **Not a compression backend.** The package does not choose ranges, summarize, or reduce tokens; it only exposes originals a backend already shadowed.
