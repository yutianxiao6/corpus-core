# 配置说明

配置 schema 当前版本为 `1`。所有层级默认拒绝未知字段，避免拼写错误被静默忽略。

## 优先级

生效顺序从低到高为：内置默认值、YAML 文件、CLI 覆盖、Python API 覆盖。映射递归合并，列表和标量整体替换。

```python
from offline_rag.config import load_config, parse_cli_overrides

config = load_config(
    "rag.yaml",
    cli_overrides=parse_cli_overrides(["embedding.batch_size=32"]),
    api_overrides={"runtime": {"log_level": "DEBUG"}},
)
```

YAML 使用安全加载器，重复 key、未知字段、不支持的版本和不存在的 profile 引用都会抛出 `ConfigurationError`。

完整字段示例见 [设计文档](design.md#112-完整配置示例)。

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

## 插件白名单

`enabled_plugins` 只接受 `<kind>:<entry-point-name>`，例如：

```yaml
enabled_plugins:
  - chunkers:company_manual
```

配置不能引用 Python 文件路径或任意 import string。
