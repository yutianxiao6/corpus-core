# Dense、Sparse、Hybrid 查询与结果组织

查询文本通过 Qwen3 query instruction 生成 dense 向量，文档编码不加 instruction；两条路径不会混用。配置 `sparse_embedding` 后还可使用纯 sparse 或 hybrid profile。Hybrid 分别召回两组候选，支持 RRF 或逐路 min-max 归一化后的加权融合，绝不直接相加两种量纲不同的原始分数。

```python
from offline_rag.config import load_config
from offline_rag.engine import OfflineRagEngine

config = load_config("rag.yaml")
with OfflineRagEngine.from_config(config) as engine:
    result = engine.query("离线安装需要什么环境？", profile="fast")

    # 只覆盖本次调用；不会修改 profile，也不能修改 embedding/index 规格
    precise = engine.retrieve(
        "ERR-1042 如何处理？",
        profile="balanced",
        filters={"metadata.department": "support"},
        organizer="debug",
        final_k=8,
        score_threshold=0.4,
        mmr_lambda=0.65,
        mmr_fetch_k=30,
        neighbor_expansion=1,
    )
```

完全离线的 hybrid 示例见[配置说明](configuration.md#完全离线-hybrid)。返回的每个候选保留 `dense_score`、`sparse_score`、`fusion_score` 和 `origins`，便于企业评测和检索问题排查。

配置 reranker 后，查询会按 `rerank_top_n` 扩大召回池，使用本地 Qwen3 CrossEncoder 重新打分，最后才裁剪到 `final_k`。`rerank_score` 保留在结果中；模型失败时按配置选择降级到原召回顺序并附 warning，或让查询失败。

查询 profile 还支持 `score_threshold`、`mmr_lambda`/`mmr_fetch_k`、`maximum_chunks_per_document` 和 `neighbor_expansion`。MMR 会对候选正文进行本地 dense 编码，换取更少重复的证据；邻居通过索引中的稳定相邻 ID 按批取回，不执行第二次相似度搜索。

组织阶段可返回独立切片、按文档分组、合并邻居、parent 正文、轮询多样性或 debug 明细。`RetrievalResult.groups` 保留结构化分组；`debug` 保留分数和预算决策；合并及 parent citation 始终指向原始证据 chunk，不指向临时合成 ID。

## LangChain 接入

`OfflineRagEngine` 可生成标准 `BaseRetriever`，直接用于 LangChain Runnable、Chain 或 Agent：

```python
from offline_rag import QueryOverrides

retriever = engine.as_langchain_retriever(
    profile="balanced",
    overrides=QueryOverrides(filters={"metadata.department": "support"}, final_k=6),
)
documents = retriever.invoke("退款条件是什么？")
documents = await retriever.ainvoke("退款条件是什么？")
```

每个 LangChain `Document` 的 `page_content` 是候选正文；metadata 包含 chunk/document/source/page、dense/sparse/fusion/rerank/final 分数、rank、origins、索引版本、embedding 指纹、原始 source metadata 和可用的 citation。metadata 会转换为普通 JSON 结构。适配器不拥有 engine 生命周期，应用关闭时仍由调用方关闭 engine。

原生接口提供行为一致的同步、异步与批量查询；批量结果顺序与输入查询严格一致：

```python
result = engine.retrieve("退款条件是什么？", profile="balanced")
result = await engine.aretrieve("退款条件是什么？", profile="balanced")

overrides = QueryOverrides(filters={"metadata.department": "support"}, final_k=5)
results = engine.batch_retrieve(queries, profile="balanced", overrides=overrides)
results = await engine.abatch_retrieve(queries, profile="balanced", overrides=overrides)
```

`aquery`/`aretrieve` 使用异步 embedding、reranker、Qdrant search/fetch，不会在线程中执行整条同步查询。异步 context manager 会关闭同步与异步数据库连接：

```python
async with OfflineRagEngine.from_config(config) as engine:
    result = await engine.aretrieve("退款条件是什么？")
```

`RetrievalResult` 同时返回排序后的 chunk、阶段耗时、索引版本和 embedding 指纹。使用 Context organizer 时还会返回可直接交给上层 LLM 的 context 与结构化 citations。

Qdrant Local 只适合单进程 CLI、预览和开发。同一个进程复用同一 `QdrantLocalVectorStore`；公司问答服务的多进程并发接入使用 `vector_store.mode: server`，每个应用进程连接同一个内网 Qdrant 服务与 active alias。

## CLI

```bash
rag-index init ./my-rag
rag-index preview ./documents --show-content
rag-index --config rag.yaml build ./documents --json
rag-index --config rag.yaml query "安装方法" --profile fast --json
rag-index --config rag.yaml query "ERR-1042" --profile balanced \
  --organizer debug --final-k 8 --score-threshold 0.4 \
  --filter metadata.department=support --json
rag-index --config rag.yaml doctor
```

`preview` 不写数据库。`build` 和 `query` 必须能找到配置指定的本地 embedding 模型；缺失时立即报错，不会联网下载。
