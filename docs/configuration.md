# 配置说明

配置 schema 当前版本为 `1`。所有层级默认拒绝未知字段，避免拼写错误被静默忽略。

## 优先级

生效顺序从低到高为：内置默认值、YAML 文件、CLI 覆盖、Python API 覆盖。映射递归合并，列表和标量整体替换。

```python
from corpuscore.config import load_config, parse_cli_overrides

config = load_config(
    "corpus.yaml",
    cli_overrides=parse_cli_overrides(["embedding.batch_size=32"]),
    api_overrides={"runtime": {"log_level": "DEBUG"}},
)
```

YAML 使用安全加载器，重复 key、未知字段、不支持的版本和不存在的 profile 引用都会抛出 `ConfigurationError`。

配置中的本地相对路径以 `corpus.yaml` 所在目录为基准，而不是运行命令时的当前目录。适用字段包括 runtime work/cache 目录、embedding 模型、Qdrant Local 数据目录和 reranker 模型目录。

模型内容固定后，可把一次完整校验得到的 SHA-256 写入 `embedding.model_checksum`。之后启动会直接使用这个不可变指纹，避免每个短生命周期 CLI 进程重新读取整个模型目录；该字段必须是 64 位十六进制字符串，模型文件变化后必须同步更新。未配置时仍会完整计算，保持安全默认值。

Query profile 是经过启动期校验的默认参数集合。Python `engine.retrieve(...)` 和 CLI `corpus-index query` 可以逐调用覆盖 filters、organizer、final_k、score threshold、rerank 候选数、MMR、每文档上限及邻居距离；不能覆盖 embedding、sparse 编码、parser/chunker 或向量库等索引结构参数。

完整字段示例见 [设计文档](design.md#112-完整配置示例)。

## 代码与实验性语义切块

代码文件默认路由到内置 `source_code` profile。Python 使用 AST 顶层符号边界；其他语言和语法不完整的 Python 使用指定 profile 回退：

```yaml
chunk_profiles:
  default:
    type: recursive
    chunk_size: 800
    chunk_overlap: 120
  source_code:
    type: syntax
    max_chunk_size: 1600
    fallback_profile: default
```

语义切块是显式启用的实验能力。它使用同一个本地 document embedding 模型比较相邻结构块，低于阈值或超过最大长度时断开；切块参数和 embedding 模型都会进入索引指纹：

```yaml
chunk_profiles:
  semantic_manual:
    type: semantic
    similarity_threshold: 0.45
    minimum_chunk_size: 200
    maximum_chunk_size: 1200
routing:
  - match:
      path_patterns: ["manuals/**"]
    use: semantic_manual
```

语义切块会额外执行一轮 document embedding，导入耗时和显存占用高于递归切块。阈值必须通过目标公司的固定评测集校准；模型缺失、向量维度异常或推理失败会中止该文档，不会静默改用另一种边界。

## 结构化格式默认切块

内置配置会按格式选择专用切块器：Markdown 按标题、CSV/XLSX/XLSM 按单行并重复表头、PDF/PPTX 按页、源代码按语法边界。显式 routing 或 sidecar profile 仍具有更高优先级。逐行表格切块更适合精确字段和数字查询；如果数据行很短且查询偏汇总，可提高 `max_rows_per_chunk` 来减少向量数量。

```yaml
chunk_profiles:
  table_rows:
    type: table_rows
    repeat_headers: true
    max_rows_per_chunk: 1
  page_aware:
    type: page_aware
    chunk_size: 1000
    chunk_overlap: 100
```

## 查询并发与模型微批

异步查询通过两个独立的有界队列控制本地 embedding 和 reranker 模型。短时间内到达的 embedding query 会合并为一次模型 `encode`；多个 rerank 请求会先展平 query-document 对，执行一次 `predict`，再按输入请求拆分结果：

```yaml
query_concurrency:
  queue_capacity: 256
  enqueue_timeout_seconds: 1.0
  execution_timeout_seconds: 60.0
  embedding_microbatch_size: 16
  embedding_wait_ms: 5
  embedding_workers: 1
  reranker_microbatch_size: 4
  reranker_wait_ms: 5
  reranker_workers: 1
```

队列满或等待结果超时会抛出错误码为 `concurrent_access` 的 `ConcurrentAccessError`，上层服务可据此返回忙碌状态或执行限次重试。GPU 部署建议从单 worker 开始，结合显存、吞吐和 P95/P99 延迟压测后调整；增加 worker 会提高并行推理数，也会增加显存和内存压力。微批等待窗口用于吞吐与首 token 前延迟之间的折中。

队列按 engine 实例和事件循环所有；企业异步服务应在应用启动时创建一个 engine，并在同一事件循环中复用，关闭时调用 `await engine.aclose()` 或使用 `async with`。多进程服务的每个 worker 各自创建 engine，共享同一个 Qdrant Server。

## 完全离线 Hybrid

启用内置稀疏编码后，每个 Qdrant point 同时保存 `dense` 和 `sparse` named vectors。编码器使用稳定哈希、次线性词频和 L2 归一化，支持中文单字/双字特征、英文词、数字、路径及 API/产品编号，不需要模型文件或联网下载。

```yaml
sparse_embedding:
  provider: hashed_lexical
  revision: "1"
  hash_space: 2147483647
  normalize: true
  include_cjk_bigrams: true

retrieval_profiles:
  fast:
    strategy: dense
    fetch_k: 8
    final_k: 5
  balanced:
    strategy: hybrid
    dense_fetch_k: 24
    sparse_fetch_k: 24
    fusion:
      type: rrf
      constant: 60
      dense_weight: 0.5
      sparse_weight: 0.5
    final_k: 6
```

`sparse_embedding` 未配置时保持 dense-only 索引。它属于索引指纹的一部分；启用、关闭或改变参数后必须重建索引，系统不会把不兼容的旧 collection 当作可用索引。

## 本地 Reranker

Reranker 仅在引用它的查询 profile 执行时延迟加载，不影响文档导入。模型只能从本地目录读取：

```yaml
rerankers:
  qwen_local:
    provider: qwen_cross_encoder
    model_path: ./models/Qwen3-Reranker-0.6B
    model_revision: pinned-local
    device: auto
    batch_size: 8
    max_length: 2048
    maximum_candidates: 100
    instruction: >-
      Given a user question, retrieve relevant passages that answer the question.
    score_mode: sigmoid
    failure_policy: return_unranked

retrieval_profiles:
  precise:
    strategy: dense
    fetch_k: 30
    reranker: qwen_local
    rerank_top_n: 20
    final_k: 6
```

`score_mode: sigmoid` 将 Qwen 输出的相关性 logit 转成 0–1 分数。`failure_policy: return_unranked` 会保留召回顺序并返回 warning；需要严格失败时改为 `fail`。`rerank_top_n` 必须不小于 `final_k`，且不能超过 `maximum_candidates`。

## 检索后处理

每个查询 profile 可独立配置阈值、MMR、文档多样性与邻居扩展：

```yaml
retrieval_profiles:
  diverse:
    strategy: dense
    fetch_k: 40
    score_threshold: 0.35
    mmr_lambda: 0.65
    mmr_fetch_k: 30
    maximum_chunks_per_document: 3
    neighbor_expansion: 1
    final_k: 8
```

处理顺序是 rerank → threshold → 每文档主命中限制 → MMR → `final_k` → neighbor expansion。邻居用于补齐上下文，保留 `origins: [neighbor]`，但没有伪造的检索分数和排名。`score_threshold` 和 `mmr_lambda` 与模型、语料强相关，部署方必须用自己的固定评测集校准。

## 结果组织器

`organizers` 支持以下类型：

- `flat`：保持独立切片。
- `context`：生成带稳定引用编号的 LLM 上下文；`merge_neighbors: true` 时先合并相邻切片。
- `grouped`：按文档分组，文档组按最佳命中顺序，组内按页码和 chunk index 排列。
- `merge_neighbors`：只合并 source、document 及双向相邻 ID 均一致的连续切片，citation 保留所有原 chunk ID。
- `parent`：将多个 child 命中折叠为完整 parent 正文，citation 保留触发它的 child ID。
- `diverse`：按文档轮询输出，避免同一文档的结果连续占满预算。
- `debug`：输出候选的各阶段分数、来源、是否入选及 `character_budget`、`token_budget`、`document_limit` 等决策原因。

```yaml
organizers:
  cited_context:
    type: context
    max_context_tokens: 6000
    merge_neighbors: true
    maximum_chunks_per_document: 3
    citation_style: numbered
  diagnostics:
    type: debug
    max_context_tokens: 20000
```

## 插件白名单

`enabled_plugins` 只接受 `<kind>:<entry-point-name>`，例如：

```yaml
enabled_plugins:
  - chunkers:company_manual
```

配置不能引用 Python 文件路径或任意 import string。

## Qdrant Server

公司问答服务、多进程 worker 或导入与查询分离部署应使用 Server 模式。服务仍可部署在完全断网的内网，本项目不会调用 Qdrant Cloud inference：

```yaml
vector_store:
  provider: qdrant
  mode: server
  path: null
  url: http://qdrant.internal:6333
  api_key_env: CORPUSCORE_QDRANT_API_KEY
  prefer_grpc: true
  timeout_seconds: 30
  pool_size: 20
  collection_alias: documents_active
  distance: cosine
  on_disk: true
```

API key 只从指定环境变量读取，不写入 YAML。变量未设置时 engine 初始化立即抛出 `ConfigurationError`。Server 和 Local 使用相同的索引指纹、named dense/sparse vectors、payload filters、staging collection 与 alias 回滚语义。
