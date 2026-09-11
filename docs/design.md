# Offline RAG Retriever 详细设计

状态：Draft  
版本：0.1  
日期：2026-09-11

## 1. 文档目的

本文定义 Offline RAG Retriever 的产品边界、核心抽象、数据协议、索引流程、查询流程、配置系统、扩展机制、并发模型、离线交付方式以及验收标准。

本文首先解决架构和接口稳定性问题。在数据协议与扩展边界确认之前，不开始大规模实现，避免把 LangChain、Qdrant 或特定解析库的接口泄露到整个代码库中。

## 2. 产品定位

Offline RAG Retriever 是一个可嵌入其他 Python 应用的检索引擎，而不是完整问答产品、SaaS 平台或带界面的知识库系统。

它应当完成以下工作：

1. 从目录、文件清单或 Python 对象读取文档。
2. 将不同格式的内容转换成统一文档结构。
3. 根据文档结构选择和执行切块策略。
4. 使用本地 embedding 模型生成向量。
5. 创建、维护、验证和迁移本地向量索引。
6. 执行同步或异步检索。
7. 可选地执行稀疏召回、结果融合和本地 rerank。
8. 对检索结果去重、扩展、合并并按上下文预算组织。
9. 返回内容、来源、页码、分数、索引版本和诊断信息。
10. 适配 LangChain `BaseRetriever`，供上层 RAG Chain 或 Agent 直接使用。

上层问答系统负责：

- 用户身份和会话管理。
- 多轮问题改写，除非显式注入 QueryTransformer。
- 调用生成模型。
- 最终答案模板、风格和安全策略。
- 将检索证据与生成答案呈现给最终用户。

## 3. 明确不做的内容

首个稳定版本不包含：

- Web UI 或桌面 UI。
- 多租户控制面或租户生命周期管理。
- 面向不同业务领域的自动路由。
- 内置聊天机器人。
- 内置通用 LLM 服务。
- 云端 embedding 或云端向量库作为默认依赖。
- 在同一个 collection 中混用不兼容的 embedding 模型。
- 静默联网下载模型、tokenizer 或运行时资源。
- 对所有文档格式做无条件、无损解析的承诺。

不支持的文件、损坏文件和无法可靠解析的内容必须显式报告，不能静默导入空内容。

## 4. 核心设计原则

### 4.1 核心协议归项目所有

内部核心数据类型由本项目定义。LangChain `Document`、Qdrant point、SentenceTransformer 输出只能出现在适配层，不能成为跨模块协议。

### 4.2 配置优先，代码可扩展

常见场景通过 YAML 配置完成；高级场景通过组合内置组件完成；极少数特殊场景通过受信任的 Python 插件完成。

### 4.3 结果可解释

每个结果必须能够追踪到原始文件、页码或位置、切片策略、索引版本及各检索阶段的分数。

### 4.4 导入可重复、可恢复

同一批文件重复导入不产生重复数据。导入中断后可恢复。更新失败时不删除仍可用的旧版本。

### 4.5 查询与组织解耦

召回候选、融合排序、rerank、相邻扩展和上下文组织是不同阶段。更换结果组织方式不能要求重新查询向量库。

### 4.6 离线行为可证明

离线不仅表示“正常情况下不联网”，还表示缺失本地模型时应立即失败，并可在断网测试中完整运行。

### 4.7 同一代码支持嵌入式和服务式向量库

开发、预览和串行 CLI 可以使用 Qdrant Local。接入并发问答服务时使用 Qdrant Server。业务接口保持一致。

## 5. 系统上下文

```text
                                 ┌─────────────────────┐
                                 │ Company QA / LLM App│
                                 └──────────┬──────────┘
                                            │ Python API / BaseRetriever
┌───────────────────────────────────────────▼────────────────────────────┐
│                         RetrievalEngine                               │
│                                                                       │
│ QueryProcessor → RetrievalStrategy → Fusion → Reranker → Organizer   │
│                                                                       │
│ IndexManager → Loader → Normalizer → Parser → Chunker → Embedding    │
└──────────────────────────────┬────────────────────────────────────────┘
                               │ VectorStorePort
                    ┌──────────▼──────────┐
                    │ Qdrant Local/Server │
                    └─────────────────────┘
```

CLI 与未来可选的 HTTP 服务均调用 `RetrievalEngine`，不能复制业务逻辑。

## 6. 交付物

### 6.1 Python 包

包名暂定为 `offline_rag`，提供：

- `RetrievalEngine`
- `IndexManager`
- 数据协议与异常类型
- 内置 loader、parser、chunker、retriever、reranker 和 organizer
- Qdrant、LangChain、SentenceTransformers 适配器
- 插件注册 API

### 6.2 索引 CLI

命令暂定为 `rag-index`，用于初始化、预览、导入、同步、验证、检查和备份。

### 6.3 LangChain 适配器

提供标准 `BaseRetriever` 行为，同时保留项目原生的丰富结果对象。

### 6.4 可选查询服务适配器

未来可提供 `rag-server`，但它是薄适配层，不属于核心检索逻辑。公司也可以直接在自己的 FastAPI、Django 或任务进程中引用 Python 包。

## 7. 推荐项目结构

```text
offline-rag/
├── docs/
│   ├── design.md
│   ├── configuration.md
│   ├── plugin-development.md
│   └── deployment.md
├── src/offline_rag/
│   ├── api/
│   │   ├── engine.py
│   │   └── options.py
│   ├── contracts/
│   │   ├── documents.py
│   │   ├── chunks.py
│   │   ├── retrieval.py
│   │   └── indexing.py
│   ├── ingestion/
│   │   ├── discovery.py
│   │   ├── loaders/
│   │   ├── normalizers/
│   │   ├── parsers/
│   │   ├── chunkers/
│   │   ├── processors/
│   │   ├── manifest.py
│   │   └── manager.py
│   ├── embeddings/
│   │   ├── base.py
│   │   ├── qwen.py
│   │   └── registry.py
│   ├── vectorstores/
│   │   ├── base.py
│   │   ├── qdrant.py
│   │   └── registry.py
│   ├── retrieval/
│   │   ├── query_processing.py
│   │   ├── strategies/
│   │   ├── fusion/
│   │   ├── rerankers/
│   │   ├── expanders/
│   │   ├── organizers/
│   │   └── engine.py
│   ├── integrations/
│   │   ├── langchain.py
│   │   └── service.py
│   ├── config/
│   ├── diagnostics/
│   ├── evaluation/
│   ├── cli/
│   └── exceptions.py
├── tests/
│   ├── unit/
│   ├── contract/
│   ├── integration/
│   ├── fixtures/
│   └── performance/
├── pyproject.toml
└── README.md
```

具体实现时可以压缩目录层级，但模块边界不能因此消失。

## 8. 公共 API

### 8.1 初始化

```python
from offline_rag import RetrievalEngine

engine = RetrievalEngine.from_config("rag.yaml")
```

初始化行为：

1. 解析并校验配置。
2. 注册内置与显式启用的第三方插件。
3. 加载索引元数据。
4. 检查索引与当前模型配置是否兼容。
5. 延迟或立即加载 embedding/reranker，取决于配置。
6. 禁止隐式网络下载。

### 8.2 导入接口

```python
report = engine.index.ingest_directory("./documents")
report = engine.index.ingest_files(paths)
report = engine.index.ingest_manifest("documents.yaml")
report = engine.index.ingest_documents(documents)
report = engine.index.sync_directory("./documents")
```

异步与流式版本：

```python
report = await engine.index.aingest_directory("./documents")

async for event in engine.index.astream_ingest("./documents"):
    handle(event)
```

### 8.3 查询接口

```python
result = engine.retrieve(
    query="如何安装系统？",
    profile="balanced",
    filters=None,
    organizer="context",
)
```

```python
result = await engine.aretrieve(query)
results = engine.batch_retrieve(queries)
results = await engine.abatch_retrieve(queries)
```

### 8.4 LangChain 接口

```python
retriever = engine.as_langchain_retriever(profile="balanced")
documents = retriever.invoke("如何安装系统？")
documents = await retriever.ainvoke("如何安装系统？")
```

LangChain 适配器只能返回标准 `Document`，详细分数和诊断信息通过 metadata 或项目原生接口获取。

## 9. 核心数据协议

### 9.1 SourceDescriptor

```python
@dataclass(frozen=True)
class SourceDescriptor:
    source_id: str
    uri: str
    relative_path: str | None
    media_type: str | None
    size_bytes: int | None
    modified_at: datetime | None
    content_hash: str
    metadata: Mapping[str, JSONValue]
```

`source_id` 必须稳定，不能依赖绝对安装目录。目录导入默认以规范化相对路径生成。

### 9.2 ParsedDocument

```python
@dataclass(frozen=True)
class ParsedDocument:
    document_id: str
    source: SourceDescriptor
    title: str | None
    blocks: Sequence[ContentBlock]
    language_hints: Sequence[str]
    metadata: Mapping[str, JSONValue]
    parser_name: str
    parser_version: str
```

`ContentBlock` 可以表示段落、标题、代码、表格、列表、页边界和图片说明，避免解析后立刻丢失结构。

### 9.3 Chunk

```python
@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    content: str
    embedding_text: str
    source_uri: str
    title: str | None
    heading_path: tuple[str, ...]
    page_start: int | None
    page_end: int | None
    char_start: int | None
    char_end: int | None
    chunk_index: int
    parent_id: str | None
    previous_id: str | None
    next_id: str | None
    token_count: int | None
    metadata: Mapping[str, JSONValue]
```

重要约束：

- `content` 是展示和交给生成模型的原文。
- `embedding_text` 可以加入标题路径、表头等检索上下文。
- 两者必须分开，不能为了提升召回而污染原文。
- `chunk_id` 必须可重复生成。
- 相邻关系必须局限在同一文档和同一版本。

### 9.4 RetrievalCandidate

```python
@dataclass
class RetrievalCandidate:
    chunk: Chunk
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    final_score: float | None = None
    rank: int | None = None
    origins: list[str] = field(default_factory=list)
```

### 9.5 RetrievalResult

```python
@dataclass(frozen=True)
class RetrievalResult:
    query: str
    processed_query: str
    expanded_queries: tuple[str, ...]
    hits: tuple[RetrievalCandidate, ...]
    context: str | None
    citations: tuple[Citation, ...]
    index_version: str
    embedding_fingerprint: str
    timings_ms: Mapping[str, float]
    warnings: tuple[str, ...]
```

生产默认结果可以关闭中间诊断字段，`debug=True` 时保留所有阶段分数。

## 10. 组件协议与注册表

### 10.1 SourceProvider

负责产生 `SourceDescriptor`，内置目录、文件列表和 manifest 实现。

```python
class SourceProvider(Protocol):
    def discover(self) -> Iterable[SourceDescriptor]: ...
```

### 10.2 DocumentLoader

负责读取字节、编码和基础文件容器，不承担复杂切块。

```python
class DocumentLoader(Protocol):
    def supports(self, source: SourceDescriptor) -> bool: ...
    def load(self, source: SourceDescriptor) -> LoadedContent: ...
```

### 10.3 DocumentParser

将 `LoadedContent` 转换成结构化 blocks。

```python
class DocumentParser(Protocol):
    def parse(self, loaded: LoadedContent) -> ParsedDocument: ...
```

### 10.4 Chunker

```python
class Chunker(Protocol):
    def split(self, document: ParsedDocument) -> Sequence[ChunkDraft]: ...
```

### 10.5 ChunkProcessor

用于标题上下文注入、短片段合并、去重和质量检查。

```python
class ChunkProcessor(Protocol):
    def process(self, chunks: Sequence[ChunkDraft]) -> Sequence[ChunkDraft]: ...
```

### 10.6 EmbeddingProvider

```python
class EmbeddingProvider(Protocol):
    @property
    def specification(self) -> EmbeddingSpecification: ...

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]: ...
    def embed_query(self, text: str) -> Vector: ...
    async def aembed_query(self, text: str) -> Vector: ...
```

查询和文档必须使用不同方法，以支持 Qwen3 的 query instruction。

### 10.7 VectorStorePort

```python
class VectorStorePort(Protocol):
    def ensure_index(self, spec: IndexSpecification) -> None: ...
    def upsert(self, records: Sequence[VectorRecord]) -> UpsertReport: ...
    def delete(self, chunk_ids: Sequence[str]) -> DeleteReport: ...
    def search(self, request: SearchRequest) -> Sequence[SearchHit]: ...
    async def asearch(self, request: SearchRequest) -> Sequence[SearchHit]: ...
```

### 10.8 RetrievalStrategy

```python
class RetrievalStrategy(Protocol):
    def retrieve(self, request: RetrievalRequest) -> Sequence[RetrievalCandidate]: ...
```

### 10.9 ResultOrganizer

```python
class ResultOrganizer(Protocol):
    def organize(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        budget: ContextBudget,
    ) -> OrganizedResult: ...
```

### 10.10 插件发现

第三方插件通过 Python package entry points 注册，例如：

```toml
[project.entry-points."offline_rag.chunkers"]
company_manual = "company_plugin:CompanyManualChunker"
```

配置文件只能引用已安装、已显式启用的插件名称，不能直接执行任意 `.py` 文件。

## 11. 配置系统

### 11.1 配置来源优先级

从高到低：

1. Python API 显式参数。
2. CLI 参数。
3. manifest 中的单文件配置。
4. 文件旁的 `filename.ext.rag.yaml`。
5. routing 规则。
6. profile 配置。
7. 全局默认值。

最终生效配置必须能够通过 `rag-index explain-config <file>` 输出。

### 11.2 完整配置示例

```yaml
version: 1

runtime:
  offline: true
  log_level: INFO
  work_dir: ./data
  cache_dir: ./data/cache

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

embedding:
  provider: qwen_sentence_transformers
  model_path: ./models/Qwen3-Embedding-0.6B
  model_revision: pinned-local
  device: auto
  dimension: 1024
  normalize: true
  batch_size: 16
  max_length: 1024
  query_instruction: >-
    Given a user question, retrieve relevant passages that answer the question.

sparse_embedding:
  provider: hashed_lexical
  revision: "1"
  hash_space: 2147483647
  normalize: true
  include_cjk_bigrams: true

vector_store:
  provider: qdrant
  mode: local
  path: ./data/qdrant
  url: null
  api_key_env: null
  prefer_grpc: false
  timeout_seconds: 30
  pool_size: null
  collection_alias: documents_active
  distance: cosine
  on_disk: false

ingestion:
  recursive: true
  follow_symlinks: false
  include_hidden: false
  ignore_patterns:
    - "**/~$*"
    - "**/*.tmp"
    - "**/.git/**"
  continue_on_error: true
  delete_missing_on_sync: true
  commit_batch_size: 128

chunk_profiles:
  default:
    type: recursive
    chunk_size: 800
    chunk_overlap: 120
    length_unit: token
    minimum_size: 80

  markdown_heading:
    type: heading_recursive
    max_chunk_size: 1000
    chunk_overlap: 100
    include_heading_in_embedding: true

  table_rows:
    type: table_rows
    repeat_headers: true
    max_rows_per_chunk: 10

  source_code:
    type: syntax
    fallback_profile: default

  parent_child:
    type: parent_child
    parent_size: 1400
    child_size: 350
    child_overlap: 60

routing:
  - match:
      extensions: [".md", ".markdown"]
    use: markdown_heading
  - match:
      extensions: [".csv", ".xlsx"]
    use: table_rows
  - match:
      extensions: [".py", ".js", ".ts", ".java", ".go"]
    use: source_code
  - match:
      filename_patterns: ["*faq*", "*FAQ*"]
    use: parent_child
  - match:
      extensions: [".pdf", ".docx", ".txt", ".html"]
    use: default

retrieval_profiles:
  fast:
    strategy: dense
    fetch_k: 8
    final_k: 5
    reranker: null
    organizer: flat

  balanced:
    strategy: hybrid
    dense_fetch_k: 24
    sparse_fetch_k: 24
    fusion:
      type: rrf
      constant: 60
    reranker: qwen_local
    rerank_top_n: 16
    final_k: 6
    organizer: context

  precise:
    strategy: hybrid
    dense_fetch_k: 50
    sparse_fetch_k: 50
    fusion:
      type: rrf
      constant: 60
    reranker: qwen_local
    rerank_top_n: 30
    score_threshold: 0.35
    mmr_lambda: 0.65
    mmr_fetch_k: 30
    maximum_chunks_per_document: 3
    neighbor_expansion: 1
    final_k: 8
    organizer: context

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

organizers:
  context:
    type: context
    max_context_tokens: 6000
    merge_neighbors: true
    maximum_chunks_per_document: 3
    citation_style: numbered
```

配置结构需通过版本化 schema 校验。未知字段默认报错，防止拼写错误被静默忽略。

## 12. 文档发现与导入方式

### 12.1 目录导入

```bash
rag-index build ./documents --recursive
```

扫描器必须处理：

- include/exclude glob。
- 文件扩展名大小写。
- 隐藏文件和临时文件。
- 软链接循环。
- 文件在扫描期间被删除或修改。
- 权限不足。
- 单文件大小限制和总大小预警。
- 相对路径规范化。

### 12.2 显式文件导入

```bash
rag-index build docs/*.pdf manuals/**/*.md
```

shell 未展开的 glob 由 CLI 自己处理，以确保 Windows 与 Linux 行为一致。

### 12.3 Manifest 导入

```yaml
documents:
  - path: contracts/main.pdf
    profile: parent_child
    metadata:
      category: contract
  - path: faq.md
    profile: markdown_heading
```

Manifest 可以给文件附加 metadata 或覆盖 profile，但不能覆盖索引级 embedding 规格。

### 12.4 Python 对象导入

允许导入本项目 `ParsedDocument` 和 LangChain `Document`。LangChain 文档若缺少稳定来源标识，调用者必须提供 namespace 或显式 ID 策略。

## 13. 解析与标准化

### 13.1 首批格式

第一阶段建议支持：

- `.txt`
- `.md` / `.markdown`
- `.pdf`，仅文本型 PDF
- `.docx`
- `.html` / `.htm`
- `.csv`
- `.json` / `.jsonl`

后续可选 extras：

- Excel
- PowerPoint
- OCR 和扫描 PDF
- 源代码语言解析器
- 邮件和归档文件

### 13.2 标准化规则

内置 normalizer 可独立启停：

- Unicode 规范化。
- 换行符统一。
- 非法控制字符清理。
- 重复空白压缩，但不破坏代码和表格。
- PDF 重复页眉页脚识别。
- 断行和连字符修复。
- 空 block 删除。
- 可疑乱码检测。

原始内容 hash 在标准化前计算，解析内容 hash 在标准化后计算，两者用途不同。

## 14. 切块体系

### 14.1 Recursive

通用 fallback。优先保持段落、句子和词语边界。中英文默认分隔符至少包括：

```text
\n\n, \n, 。, ！, ？, ；, ., !, ?, ;, ，, ,, 空格
```

长度单位推荐 token；字符长度可作为无 tokenizer 时的退化方案。

### 14.2 Heading Recursive

先按标题层级生成 section，再对过长 section 递归切分。标题路径写入 metadata，并可注入 `embedding_text`。

### 14.3 Page Aware

保留页边界。短页面可跨页合并，但 `page_start` 和 `page_end` 必须准确。表格跨页时优先保持表头与行语义。

### 14.4 Paragraph Packing

按顺序将自然段装入目标 token 预算。过短段落向相邻段落合并，过长段落使用 fallback splitter。

### 14.5 Sliding Window

提供确定性的覆盖率，适用于结构极差文本。该策略会增加切片数，只在显式配置时启用。

### 14.6 QA Pair

识别或接受结构化问答对，一个问答作为最小单元。问题可以加入 `embedding_text`，答案保留在 `content`。

### 14.7 Table Row

每个切片包含表名、表头和若干完整数据行。不得只保存脱离表头的数据值。

### 14.8 Syntax Aware

按类、函数、方法或逻辑代码块切分。解析失败时退回 Recursive，不能丢弃文件。

### 14.9 Parent Child

Child 用于向量召回，Parent 用于返回上下文。两者均有稳定 ID，Child 保存 `parent_id`。Parent 可存储在相同 collection 的 payload 或单独 docstore；第一版优先使用同一 Qdrant collection 中的类型字段。

### 14.10 Semantic

根据连续句子的 embedding 差异寻找边界。它计算成本高，受模型版本影响，且会使增量切片边界更易漂移，因此作为实验性插件实现，不作为默认策略。

### 14.11 Chunk 后处理

顺序默认如下：

1. 删除空切片。
2. 合并过短切片。
3. 强制切分超长切片。
4. 注入标题、表头等 embedding 上下文。
5. 规范化 metadata。
6. 计算 token 数。
7. 内容级去重。
8. 生成稳定 ID。
9. 建立 previous/next/parent 关系。
10. 执行质量规则并生成警告。

## 15. Embedding 设计

### 15.1 默认模型

默认基线：

- 模型：`Qwen/Qwen3-Embedding-0.6B`
- 运行方式：本地 SentenceTransformers
- 最大向量维度：1024
- 距离：cosine
- 输出向量归一化：开启
- 默认最大输入：1024 tokens，而非无条件使用模型最大上下文

模型目录通过本地路径指定。生产配置必须固定模型 revision 或文件校验和。

### 15.2 Query Instruction

Qwen3 查询端使用 instruction，文档端不使用。适配器必须保证：

```python
model.encode(queries, prompt_name="query", normalize_embeddings=True)
model.encode(documents, normalize_embeddings=True)
```

不能使用一个不区分 query/document 的包装器破坏该行为。

### 15.3 可替换模型

EmbeddingProvider 允许替换成其他本地模型，但以下任意属性变化都产生新索引：

- 模型文件。
- tokenizer。
- pooling 方法。
- query instruction。
- 向量维度。
- 是否归一化。
- 最大输入长度。
- 距离算法。

### 15.4 Index Fingerprint

索引指纹至少包含：

```text
model checksum
tokenizer checksum
dimension
normalization
distance
query instruction
parser versions
chunker configuration
chunk processor configuration
payload schema version
```

查询启动时必须校验指纹。配置不匹配时拒绝查询，并提示重建或切换兼容索引。

## 16. 向量数据库设计

### 16.1 默认后端

默认使用 Qdrant，并提供：

- `local`：预览、开发、CLI 串行使用。
- `server`：接入公司问答 AI、并发查询、导入与查询并行。

Qdrant Local 不能由多个进程同时打开同一存储目录。服务模式不能通过简单增加多个 Python worker 来共享 Local 目录。

### 16.2 Collection 策略

单部署默认只有一个 active collection，但允许为重建和回滚保留多个版本：

```text
documents_20260911_001
documents_20260912_001
alias: documents_active
```

不同 embedding 指纹不能写入同一 collection。

### 16.3 Payload Schema

每条检索向量至少包含：

```json
{
  "record_type": "child",
  "document_id": "...",
  "document_version": "...",
  "chunk_id": "...",
  "content": "...",
  "embedding_text": "...",
  "source_uri": "manual.pdf",
  "title": "安装要求",
  "heading_path": ["安装", "系统要求"],
  "page_start": 12,
  "page_end": 12,
  "char_start": 1402,
  "char_end": 1880,
  "chunk_index": 7,
  "parent_id": "...",
  "previous_id": "...",
  "next_id": "...",
  "content_hash": "...",
  "metadata": {}
}
```

### 16.4 Dense 与 Sparse

“单一向量数据库”和“只做 dense 检索”是两个不同概念。

本设计只使用一个 Qdrant，不要求 Elasticsearch。为覆盖编号、型号、日期、API 名称和专有词等精确匹配场景，允许在同一 point 中保存 dense 与 sparse named vectors。

支持三种检索模式：

- `dense`：最小依赖、语义召回。
- `sparse`：精确词项召回。
- `hybrid`：分别召回并融合，推荐用于通用企业资料。

若交付要求严格只允许 dense，可通过配置关闭 sparse，但验收报告必须明确其精确词检索限制。

## 17. 索引生命周期

### 17.1 状态机

每个文件导入状态：

```text
DISCOVERED → LOADED → PARSED → CHUNKED → EMBEDDED → INDEXED
      │          │         │          │           │
      └──────────┴─────────┴──────────┴───────────┴→ FAILED
```

状态日志至少记录 job ID、source、hash、阶段、耗时、切片数和错误摘要。

### 17.2 幂等 ID

建议：

```text
document_id = hash(canonical relative path or explicit source ID)
document_version = hash(raw content + parser configuration)
chunk_id = hash(document_id + document_version + chunk index + chunk content)
```

### 17.3 增量同步

同步行为：

- 新文件：解析、切块并写入。
- 未变化文件：跳过。
- 内容变化：构建新文档版本，成功后删除旧版本。
- 路径变化但内容相同：默认视为新文档；可选内容去重模式。
- 已删除文件：`sync` 模式删除其有效切片。
- 解析失败：保留旧索引并报告失败。

### 17.4 完整重建

完整重建使用 staging collection：

1. 创建新版本 collection。
2. 完整导入。
3. 校验记录数、索引规格和抽样查询。
4. 运行可选评测集。
5. 切换 active alias。
6. 保留上一版本供回滚。
7. 按保留策略清理更旧版本。

## 18. 查询流水线

默认阶段：

```text
Validate
  → Normalize Query
  → Optional Query Transform
  → Dense/Sparse Candidate Retrieval
  → Fusion
  → Optional Rerank
  → Score/Quality Filter
  → Deduplicate
  → Parent/Neighbor Expansion
  → Diversity Control
  → Result Organization
```

每个阶段记录耗时。`debug` 模式记录输入输出数量和分数变化。

## 19. 查询策略

### 19.1 Similarity

标准 dense cosine 检索，参数为 `fetch_k`、`final_k` 和可选 threshold。

### 19.2 Threshold

低于阈值的结果被删除。阈值不能跨模型照搬，应通过部署方自己的评测集校准。零结果是合法输出。

### 19.3 MMR

在相关性和多样性之间平衡，适合大量相似切片。MMR 的 lambda 与候选池大小属于 profile 配置。

### 19.4 Hybrid

Dense 和 sparse 各自召回候选，然后使用 RRF 或加权归一化融合。默认使用 RRF，避免直接比较量纲不同的原始分数。

### 19.5 Parent Child

向量命中 child，最终候选映射到 parent。多个 child 指向同一 parent 时合并证据并保留最佳分数与命中列表。

### 19.6 Neighbor Expansion

命中后按 `previous_id`、`next_id` 获取前后片段。扩展结果不是新命中，不应虚构独立检索分数。

### 19.7 Multi Query

允许注入外部 QueryTransformer，将一个问题转成多个等价表达。核心包不强制依赖 LLM；默认实现仅返回原问题。

### 19.8 Ensemble

允许组合多个策略，但第一版不允许跨不兼容 embedding collection 融合。候选使用来源标签记录由哪个策略召回。

## 20. Fusion 与 Rerank

### 20.1 Fusion

首批实现：

- Reciprocal Rank Fusion。
- 归一化加权融合。

RRF 默认公式：

```text
score(d) = sum(weight_i / (constant + rank_i(d)))
```

### 20.2 Reranker

默认可选模型为本地 `Qwen3-Reranker-0.6B`。接口必须允许关闭或替换。

Rerank 输入为 query 与候选 `content`/`embedding_text` 对。默认先召回较大候选集，再只对前 `rerank_top_n` 执行昂贵推理。

需要限制：

- 单请求最大候选数。
- 单候选最大 token 数。
- batch size。
- CPU/GPU 并发数。
- 超时后的退化策略。

若 reranker 失败，profile 可以配置为返回未重排结果或整体失败；默认返回未重排结果并附 warning。

## 21. 去重、扩展与多样性

### 21.1 去重

支持：

- 相同 chunk ID 去重。
- 相同 parent ID 合并。
- 内容 hash 去重。
- 高文本重叠近似去重，可选。

### 21.2 相邻合并

只有来源、文档版本和相邻关系均匹配时才合并。合并后页码范围和引用映射必须更新。

### 21.3 多样性约束

允许配置每个文档最多保留的片段数，避免一个长文档占据全部上下文。该约束在 rerank 后执行，不能影响早期召回率。

## 22. 结果组织

ResultOrganizer 只处理已经排好序的候选，不直接访问向量库。

### 22.1 FlatOrganizer

返回 Top-K 独立切片，适合搜索列表和调试。

### 22.2 GroupByDocumentOrganizer

按文档分组，组内按页码或 chunk index 排序，文档组按最佳命中分数排序。

### 22.3 MergeNeighborsOrganizer

合并连续片段，保留每个原始 chunk 的引用映射。

### 22.4 ParentOrganizer

返回 parent 内容，并附带触发该 parent 的 child 列表。

### 22.5 DiverseOrganizer

按每文档配额和 MMR 规则组织结果。

### 22.6 ContextOrganizer

生成可直接交给 LLM 的上下文：

```text
[证据 1]
来源：manual.pdf，第 12 页
章节：安装 > 系统要求
内容：……

[证据 2]
来源：faq.md
章节：常见问题
内容：……
```

需要支持：

- 最大 token 预算。
- 最大字符预算，作为无目标 tokenizer 时的退化方案。
- 每文档片段上限。
- 不在句子或表格行中间截断。
- 标题和来源开销纳入预算。
- 稳定的引用编号。
- 返回 citation 到 chunk/source/page 的结构化映射。

### 22.7 DebugOrganizer

返回所有阶段的候选、分数和删除原因，仅用于评测和诊断，不应作为默认生产响应。

## 23. Query Profile

调用者通过 profile 选择延迟与质量平衡，而不是理解所有底层参数。

建议内置：

- `fast`：dense、无 reranker、少量候选。
- `balanced`：hybrid、RRF、轻量 reranker、上下文组织。
- `precise`：大候选池、reranker、相邻扩展。
- `raw`：返回未经组织的向量库命中，便于调试。

Profile 是默认值集合，调用参数可以覆盖非结构性参数；不能在单次查询中覆盖 embedding 规格。

当前开放的逐调用覆盖项为 filters、organizer、final_k、score threshold、rerank_top_n、MMR 参数、每文档主命中上限和邻居距离。覆盖后重新执行边界及交叉约束校验，且不修改进程内冻结的 profile 对象。

## 24. 并发和运行模型

### 24.1 Python API

提供同步、异步和批量接口。异步接口不应简单用线程包装全部流程；向量库网络 I/O 使用异步 client，CPU/GPU 推理进入受控执行队列。

### 24.2 模型生命周期

- 模型按进程加载一次。
- 不能每次查询重新创建 SentenceTransformer。
- GPU 通常使用单模型 worker 和微批处理队列。
- 多个 Web worker 会复制模型显存，应由部署适配器限制。
- CPU 模式需要明确 PyTorch 线程数和请求并发上限，防止线程过量竞争。

### 24.3 Qdrant Local

Local 模式适合单进程开发和 CLI。一个进程内必须复用同一 client；不同进程不能同时打开同一目录。

### 24.4 Qdrant Server

接入公司问答 AI 时使用 Server 模式。查询服务、导入 CLI 和其他调用方通过 HTTP/gRPC 访问同一个 Qdrant Server。

### 24.5 导入与查询并行

- 增量 upsert 使用独立批次和限速。
- 重建使用 staging collection，不在 active collection 上进行大规模破坏性操作。
- 查询优先于导入任务。
- 写入完成确认后才能让新版本可见。
- collection 切换过程中 active alias 必须始终指向完整索引。

## 25. CLI 设计

### 25.1 初始化

```bash
rag-index init [target-directory]
```

生成示例配置和标准目录，不下载模型。

### 25.2 预览

```bash
rag-index preview ./documents
rag-index preview manual.pdf --profile parent_child --show-content
```

输出文件数、解析错误、切片数量、长度分布、过短/超长切片和抽样内容，不写向量库。

### 25.3 构建与同步

```bash
rag-index build ./documents
rag-index sync ./documents
rag-index rebuild ./documents
```

`build` 创建新索引；`sync` 增量同步 active 索引；`rebuild` 创建 staging 索引并切换。

### 25.4 诊断

```bash
rag-index validate
rag-index stats
rag-index inspect --source manual.pdf
rag-index explain-config manual.pdf
rag-index doctor
```

`doctor` 检查本地模型文件、依赖、设备、磁盘空间、索引指纹、数据库连接和离线设置。

### 25.5 查询调试

```bash
rag-index query "安装要求是什么" --profile balanced
rag-index query "ERR-1042" --organizer debug --json
```

该命令用于验证检索，不代替上层问答 AI。

### 25.6 备份和版本

```bash
rag-index versions
rag-index activate <index-version>
rag-index backup ./backups/index.snapshot
rag-index restore ./backups/index.snapshot
```

破坏性命令需要显式确认；自动化场景通过 `--yes` 明确选择。

## 26. 错误模型

所有公共异常继承 `OfflineRagError`，首批类型：

- `ConfigurationError`
- `OfflineResourceMissingError`
- `UnsupportedDocumentError`
- `DocumentLoadError`
- `DocumentParseError`
- `ChunkingError`
- `EmbeddingError`
- `IndexCompatibilityError`
- `VectorStoreError`
- `RetrievalError`
- `RerankerError`
- `ContextBudgetError`
- `ConcurrentAccessError`

批量导入使用结构化 `ItemFailure` 收集单文件错误，而不是只抛出第一个异常。不可恢复的索引规格错误应立即停止任务。

## 27. 离线交付

### 27.1 交付包

建议提供：

```text
offline-rag-bundle/
├── wheelhouse/
├── models/
│   ├── Qwen3-Embedding-0.6B/
│   └── Qwen3-Reranker-0.6B/
├── config/
├── install.sh
├── install.ps1
├── checksums.sha256
└── LICENSES/
```

Python 包按功能拆分 extras：

```text
offline-rag[qdrant,pdf,docx]
offline-rag[ocr]
offline-rag[code]
offline-rag[all]
```

### 27.2 离线强制规则

- 模型只接受本地路径。
- 设置 Hugging Face/Transformers 离线环境。
- 缺文件时立即报错，不尝试下载。
- `doctor` 校验模型权重、tokenizer、配置和许可证文件。
- CI 中增加禁网集成测试。

## 28. 安全与隐私

即使本项目不是多租户平台，也必须考虑：

- 默认不上传内容或遥测。
- 日志默认不记录完整文档和完整查询。
- 错误信息不泄露文档正文。
- 模型与依赖提供 checksum。
- 插件只能来自显式安装和启用的包。
- Qdrant Server 默认绑定 loopback 或私有地址。
- 生产模式支持 API key、TLS 和只读查询凭据。
- 向量库不直接暴露给最终用户。
- 上层 LLM 应把检索文档当作不受信任的数据，防范文档内 prompt injection。

如果同一家公司内部存在文档权限差异，虽然不需要多租户，仍需由调用方传入 ACL filter；第一版保留通用 filters 接口，但不实现身份系统。

## 29. 可观测性

### 29.1 日志

结构化日志字段：

```text
operation_id
job_id
query_id
index_version
source_id
stage
duration_ms
candidate_count
warning_code
error_code
```

### 29.2 指标

核心包提供回调或 metrics snapshot，不强制依赖 Prometheus：

- 导入文件数、失败数和跳过数。
- chunk 数和长度分布。
- embedding 吞吐和失败率。
- Qdrant 查询延迟。
- rerank 延迟。
- 总查询 P50/P95/P99。
- 候选数与最终结果数。
- 空结果率和低分结果率。
- 当前索引版本、文档数和 chunk 数。

### 29.3 Tracing Hook

提供无实现的事件回调接口，上层可以接入自己的 OpenTelemetry 或日志系统。完全离线部署不能默认连接外部 tracing 服务。

## 30. 评测体系

### 30.1 检索评测数据

```json
{
  "query": "出差住宿标准是多少？",
  "relevant_document_ids": ["..."],
  "relevant_chunk_ids": ["..."],
  "tags": ["zh", "policy"]
}
```

允许只有文档级标注，但 chunk 级标注优先。

### 30.2 指标

- Recall@K
- Precision@K
- MRR
- nDCG@K
- Hit Rate
- 空结果准确率
- P50/P95/P99 延迟
- 索引吞吐

### 30.3 必测查询类型

- 中文查询中文文档。
- 英文查询英文文档。
- 中文查询英文文档。
- 英文查询中文文档。
- 中英文混合问题。
- 精确编号、型号、日期、金额和 API 名称。
- 同义表达。
- 长问题与短问题。
- 模糊问题。
- 无答案问题。
- 多个近似重复文档。
- 答案跨相邻片段。

### 30.4 回归门槛

模型、切块、fusion、reranker 或数据库索引参数发生变化时必须运行相同评测集。不能只凭主观查看少数结果决定上线。

## 31. 测试策略

### 31.1 单元测试

- ID 生成稳定性。
- 配置合并和优先级。
- 文档标准化。
- 各切块策略边界。
- token 预算。
- fusion 算法。
- 去重与相邻合并。

### 31.2 协议测试

所有 EmbeddingProvider、VectorStorePort、Chunker、Reranker 和 Organizer 实现必须通过同一组 contract tests。

### 31.3 集成测试

- 从 fixture 文档到 Qdrant Local 的完整导入与查询。
- Qdrant Server 的同步/异步查询。
- 增量增加、修改和删除。
- 导入中断恢复。
- staging collection 切换和回滚。
- LangChain `invoke/ainvoke`。
- 断网运行。

### 31.4 并发测试

- 多个异步查询。
- 查询与增量导入并行。
- reranker 并发上限。
- 模型只加载一次。
- Qdrant Local 多进程冲突产生明确错误。

### 31.5 性能测试

测试数据规模至少覆盖 10K、100K 和 1M chunks。结果记录硬件、维度、索引配置、batch size、召回质量和延迟，不能只报告 QPS。

## 32. 版本与兼容性

分别维护：

- Python package version。
- config schema version。
- payload schema version。
- index format version。
- parser/chunker version。
- embedding fingerprint。

兼容性策略：

- patch：不改变协议和索引。
- minor：可增加可选字段和插件。
- major：允许公共 API 或配置不兼容变化。
- 索引不兼容时提供清晰迁移或重建提示，禁止隐式修改线上索引。

## 33. 实施阶段

### Phase 1：协议与最小垂直链路

- 核心 dataclass/Pydantic 协议。
- 配置 schema。
- TXT/Markdown loader。
- Recursive 与 Heading chunker。
- Qwen3 embedding 适配器。
- Qdrant Local 适配器。
- Dense 查询。
- Flat/Context organizer。
- `build`、`preview`、`query`、`doctor`。

验收：断网环境下完成目录导入并通过 Python API 与 CLI 查询。

### Phase 2：可靠导入和结构化文档

- PDF、DOCX、HTML、CSV。
- Manifest 和 routing。
- 增量同步、删除和失败恢复。
- Parent Child、Table Row、Page Aware。
- 索引指纹、staging collection 和回滚。
- 备份恢复。

### Phase 3：完整检索链路

- Sparse/Hybrid。
- RRF 和加权融合。
- Qwen3 reranker。
- MMR、neighbor expansion、去重和多样性。
- 多种 organizer。
- Query profile。
- LangChain BaseRetriever 适配。

### Phase 4：并发、评测和交付

- Qdrant Server 模式。
- async/batch API。
- 并发控制和微批处理。
- 评测工具。
- 性能基准。
- 离线 wheelhouse/model bundle。
- 安装、升级和兼容性文档。

### Phase 5：可选高级插件

- OCR。
- Excel、PPT、邮件和归档。
- 代码 AST 切块。
- Semantic chunker。
- HTTP 服务参考实现。
- 外部 QueryTransformer hook。

## 34. 首版验收标准

首个可用版本至少满足：

1. 在物理断网环境安装并运行。
2. 中英文 TXT/Markdown/PDF/DOCX 可导入。
3. 重复导入不产生重复 chunk。
4. 文件修改和删除能正确同步。
5. 所有命中能定位来源和页码或位置。
6. Qwen3 查询 instruction 与文档编码严格区分。
7. 配置与索引不兼容时拒绝查询。
8. 同步、异步、批量和 LangChain 查询接口行为一致。
9. 至少提供 dense、hybrid、parent-child 三种 profile。
10. 至少提供 flat、grouped、context 三种结果组织方式。
11. Qdrant Local 多进程访问给出明确指导；Qdrant Server 支持并发接入。
12. 导入中断后可重试，失败文件不破坏其他已导入内容。
13. 提供可执行的检索评测和回归报告。
14. 核心模块无 UI 和生成模型依赖。

## 35. 已确定决策

- 产品形态是可嵌入的检索库，CLI 是索引管理适配器。
- 每家公司本地部署独立实例，不建设多租户平台。
- 不按业务领域划分或路由，但按文档结构适配解析与切块。
- 使用 Python 和 LangChain，核心协议不直接绑定 LangChain 类型。
- 完全离线，中英文混合。
- 默认 Qwen3-Embedding-0.6B，1024 维、归一化 cosine。
- 默认向量库为 Qdrant，支持 Local 和 Server 两种模式。
- 不内置答案生成，向上层返回结构化证据和可直接使用的 context。
- 导入、召回、重排、扩展和组织均采用可配置插件架构。

## 36. 待确认决策

后续设计评审需要确认：

1. “纯向量数据库”是否允许在同一个 Qdrant 中启用 sparse vectors。本文建议允许，并且不引入第二个搜索数据库。
2. 首批强制支持的文档格式和 OCR 是否进入首版。
3. 最低支持硬件，包括 CPU、内存、是否要求无 GPU。
4. 单实例目标规模：文件数、总文本量、chunk 数和并发查询数。
5. 默认 profile 使用 dense 还是 hybrid。本文建议 `fast=dense`、`balanced=hybrid`。
6. Qdrant Server 是否随离线交付包提供 Docker Compose，还是只提供配置说明。
7. Windows 是否属于首版正式支持范围。
8. Python 最低版本及需要支持的 LangChain 版本区间。

这些问题不会改变核心协议方向，但会影响依赖、测试矩阵和首版工期。

## 37. 参考资料

- LangChain Qdrant integration: https://docs.langchain.com/oss/python/integrations/vectorstores/qdrant
- LangChain text splitters: https://docs.langchain.com/oss/python/integrations/splitters/index
- Qdrant production checklist: https://qdrant.tech/documentation/production-checklist/
- Qdrant security: https://qdrant.tech/documentation/security/
- Qwen3-Embedding-0.6B: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- BGE-M3: https://huggingface.co/BAAI/bge-m3
