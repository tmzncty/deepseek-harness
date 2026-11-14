---
description: "面向模型的、有界地回读压缩检查点所遮蔽的原始事件，供智能体开发者与维护者在长会话压缩后恢复约束、决定、错误和工具结果。"
kind: "package-reference"
---

# @deepseek-ai/dsh-tool-history-recall

[English](README.md) | 中文

## 摘要

`dsh-tool-history-recall` 让模型列出本会话的压缩检查点、分页读取某个检查点遮蔽的精确原始事件，并在这些原文中按字面量检索。两个只读工具每次调用通过 `ctx.sessionQuery` 异步观察一次仅追加的会话日志：压缩摘要仍是当前上下文，回读内容作为普通且已记录的工具体输出到达。模型可见读取按字节有界、可用游标续读；本包从不改写摘要、表面或请求前缀。

## 目录

- [使用本包](#use-this-package)
- [理解实现](#understand-the-implementation)
- [进一步阅读](#further-exploration)
- [模型体验](#model-experience)
- [已知限制与后续工作](#known-limitations-and-deferred-work)

-----

<a id="use-this-package"></a>
## 使用本包

当模型需要恢复检查点压缩掉的内容时，把本包与压缩后端一起挂载。`dsh-base` 已在交付组合中挂载它。

### 配置

| 字段 | 默认值 | 含义 |
|---|---|---|
| `maxPageBytes` | `16384` | 单个 transcript 页面的最大 UTF-8 字节数 |
| `maxSearchHits` | `20` | 单次字面量检索返回的最大命中数 |
| `maxSearchBytes` | `2000000` | 单次检索扫描的最大渲染字节数 |
| `maxListedCheckpoints` | `8` | 目录或页头中最多列出的检查点数 |

### 模型能做什么

| 工具 | 模型得到什么 |
|---|---|
| `history_read` | 不传 `checkpoint`：本会话的压缩检查点列表。传入 `checkpoint`：该检查点遮蔽的精确原始事件的一页，若还有剩余则附续读游标。 |
| `history_search` | 在所有压缩区段或单个检查点中按字面量、不区分大小写检索，返回带检查点与事件 seq 的片段及扫描覆盖信息。 |

检查点用其仅日志可见的 `compaction/summary` 事件 seq 寻址。模型从目录或页头获得这些 seq；模型可见的摘要文本不添加页脚。

### 失败与恢复

读取以 `HarnessError` 代码失败关闭：未知检查点为 `HISTORY_CHECKPOINT_NOT_FOUND`，被引用事件不在已加载日志中为 `HISTORY_EVIDENCE_MISSING`，事件不能承载 transcript 消息为 `HISTORY_EVIDENCE_INVALID`，游标损坏或过期失效为 `HISTORY_CURSOR_INVALID`，页面预算不可用为 `HISTORY_PAGE_BUDGET_TOO_SMALL`，查询为空或过长为 `HISTORY_SEARCH_INVALID_QUERY`。缺失事件绝不会被静默跳过。

-----

<a id="understand-the-implementation"></a>
## 理解实现

<details>
<summary>实现内部细节——点击展开</summary>

### 设计原则

本包增加的是可寻址性与有界分页，而不是另一份摘要。它遵守四条规则：

- **日志是唯一来源。** transcript 来自持久事件，而不是当前表面投影，因此能跨后续压缩和进程重启存活。
- **分页是显式的。** `history_read` 最多返回 `maxPageBytes` 个 UTF-8 字节，并给出 base64url 游标，记录检查点、事件 seq 和字符偏移；使用前会按检查点当前的遮蔽区段校验。
- **裁剪替换回溯到原文。** 工具结果裁剪替换通过 `sourceEventSeqs` 引用原文；transcript 展示原文并标注被引用的替换。
- **不新增存储或索引。** 检查点从 `compaction/summary` 事件发现，区段来自 `shadowedSeqs`。

### 源码地图

| 文件 | 职责 |
|---|---|
| [`src/index.ts`](src/index.ts) | 插件入口：配置、提示词章节、两个工具注册 |
| [`src/checkpoints.ts`](src/checkpoints.ts) | 有界检查点发现与目录渲染 |
| [`src/records.ts`](src/records.ts) | 遮蔽 seq 分块加载与裁剪原文解析 |
| [`src/render.ts`](src/render.ts) | 持久事件到带标签 transcript 记录 |
| [`src/pagination.ts`](src/pagination.ts) | UTF-8 字节预算、游标编码与页面框架 |
| [`src/search.ts`](src/search.ts) | 压缩区段上的有界字面量扫描 |

### 并发与取消

两个工具都是只读且并发安全的。每次 `ctx.sessionQuery` 调用都接收工具执行的中止信号；中止的扫描在下一批之前停止。本包不注册全局状态。

</details>

-----

<a id="further-exploration"></a>
## 进一步阅读

- [压缩](../compaction/README.zh.md) — 检查点约定、`shadowedRange` 与 `shadowedSeqs`。
- [基础压缩后端](../compaction-basic/README.zh.md) — 写出本包所读检查点的交付后端。
- [会话查询](../../session-query/session-query/README.zh.md) — 两个工具背后的异步会话观察服务。
- [会话持久化](../../session/session-persistence/README.zh.md) — 重启后仍存在的持久日志。

-----

<a id="model-experience"></a>
## 模型体验

### 系统提示词指引

#### 模型看到什么

挂载期间始终存在一段固定章节。

##### 固定指引

```markdown
Long sessions may replace older conversation with a compacted checkpoint. Use history_read to list this session's checkpoints and page through the exact original events one shadows, and history_search to find a literal phrase in those originals. Recalled text arrives as ordinary tool output; the checkpoint summary remains the current context.
```

#### Token 影响

挂载期间每次请求增加一段固定的短章节。

#### KV Cache 影响

只要章节与工具可见性不变，前缀保持稳定。本包不改写压缩摘要或任何更早的请求 token。

### 工具 schema

#### 模型看到什么

两个扁平 schema：`history_read(checkpoint?, cursor?)` 与 `history_search(query, checkpoint?)`。不向模型暴露页面大小、偏移或扫描上限。生成的[工具目录](../../../docs/tool-catalog.zh.md#deepseek-aidsh-tool-history-recall)拥有精确的 schema 文本。

#### Token 影响

工具可见期间每次请求增加两个固定只读 schema。

#### KV Cache 影响

只要工具可见性与定义不变，前缀保持稳定。

### 工具结果

#### 模型看到什么

`history_read` 返回检查点头部、带标签的 transcript 记录，以及结束标记或续读游标。`history_search` 返回带检查点与事件 seq 的片段以及 `scanned` 覆盖信息。回读文本是普通工具体输出，并与其他结果一样被记录。

#### Token 影响

依赖数据，单次读取受 `maxPageBytes` 限制，单次检索受 `maxSearchHits`/`maxSearchBytes` 限制；回读字节在后续压缩前保留在已记录的工具体历史中。

#### KV Cache 影响

只追加的结果文本位于可复用请求前缀之后，不会使更早的缓存条目失效。本包从不改写位于尾部之前的摘要检查点。

## 已知限制与后续工作

- **每次调用一份完整日志观察。** `ctx.sessionQuery.readSession` 目前会为一次工具调用物化完整逻辑日志；分页限制的是模型可见字节，不是该快照，因此超大日志的每次回读都会按其大小消耗内存。
- **被 Spill 截断的工具结果从未进入日志。** `dsh-spill-policy` 可能在追加 `tool/result` 事件之前把超大的工具输出替换为预览与定位符。本包只能渲染进入会话日志的字节；被移除的中段在保留期清理前只能从 spill 文件取得。
- **没有检查点页脚。** 检查点文本保持不变，因此模型从 `history_read` 的目录而不是摘要本身获知检查点 seq。增加确定性页脚属于单独的压缩格式决策。
- **仅字面量检索。** 匹配不区分大小写且按字面量执行；没有正则、语义检索或对 JSON 工具结果的结构化查询。
- **单会话，不跨会话回读。** 工具只读调用方会话。跨会话历史仍由实验性的 `dsh-tool-session-query` 提供。
- **不是压缩后端。** 本包不选择区段、不生成摘要、不减少 token；它只暴露后端已经遮蔽的原文。
