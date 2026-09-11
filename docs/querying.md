# Dense、Sparse、Hybrid 查询与结果组织

查询文本通过 Qwen3 query instruction 生成 dense 向量，文档编码不加 instruction；两条路径不会混用。配置 `sparse_embedding` 后还可使用纯 sparse 或 hybrid profile。Hybrid 分别召回两组候选，支持 RRF 或逐路 min-max 归一化后的加权融合，绝不直接相加两种量纲不同的原始分数。

```python
from offline_rag.config import load_config
from offline_rag.engine import OfflineRagEngine

config = load_config("rag.yaml")
with OfflineRagEngine.from_config(config) as engine:
    result = engine.query("离线安装需要什么环境？", profile="fast")
```

完全离线的 hybrid 示例见[配置说明](configuration.md#完全离线-hybrid)。返回的每个候选保留 `dense_score`、`sparse_score`、`fusion_score` 和 `origins`，便于企业评测和检索问题排查。

`RetrievalResult` 同时返回排序后的 chunk、阶段耗时、索引版本和 embedding 指纹。使用 Context organizer 时还会返回可直接交给上层 LLM 的 context 与结构化 citations。

Qdrant Local 只适合单进程 CLI、预览和开发。同一个进程复用同一 `QdrantLocalVectorStore`；公司问答服务的多进程并发接入应使用后续的 Qdrant Server 适配器。

## CLI

```bash
rag-index init ./my-rag
rag-index preview ./documents --show-content
rag-index --config rag.yaml build ./documents --json
rag-index --config rag.yaml query "安装方法" --profile fast --json
rag-index --config rag.yaml doctor
```

`preview` 不写数据库。`build` 和 `query` 必须能找到配置指定的本地 embedding 模型；缺失时立即报错，不会联网下载。
