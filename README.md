# Offline RAG Retriever

Offline RAG Retriever 是一个面向本地部署的通用 Python 检索组件。它负责文档发现、解析、切块、向量化、索引维护、检索、重排和上下文组织；上层问答系统只需要调用检索接口并接入自己的 LLM。

当前已经完成完整 P0/P1/P2 和前三项 P3：多格式解析及多策略切块、可恢复增量索引、Qdrant staging/alias 回滚、dense/sparse/hybrid 检索、本地 Qwen3 rerank、多种上下文组织方式、Qdrant Server、异步/批量 API 及模型微批队列均可使用。

## 已确定的边界

- 完全离线运行。
- 中英文混合检索。
- 单部署实例、单知识库，不实现多租户平台。
- 不按业务领域划分索引，但允许按文档结构选择解析和切块方式。
- 没有 UI；提供索引 CLI 和 Python SDK。
- 使用 LangChain 作为组件集成层。
- 使用一个 Qdrant 向量数据库；单进程可用 Local，多进程企业部署使用内网 Server。
- 检索模块不负责最终答案生成。
- 能作为库直接接入公司的问答 AI。
- 原生支持同步、异步与保持输入顺序的批量查询，LangChain `invoke/ainvoke` 均走对应链路。

完整设计见 [docs/design.md](docs/design.md)。
评测格式与指标门槛见 [docs/evaluation.md](docs/evaluation.md)。

## 开发环境

仓库使用 `uv.lock` 锁定所有直接和传递依赖。安装完整本地检索开发环境：

```bash
uv sync --all-extras
source .venv/bin/activate
```

系统没有 pip 时，`uv venv --seed .venv` 会在项目虚拟环境中安装 pip，不修改系统 Python。

依赖按用途分为 `qdrant`、`embedding`、`documents`、`sparse`、`tables` 和 `ocr` extras；核心包只保留配置、LangChain 核心与切块协议所需依赖。

## 验证

```bash
python -m pytest -q
ruff check .
ruff format --check .
mypy src
```

并行存储层压力回归：

```bash
python -m pytest -q tests/test_concurrent_workload.py
```

可跟踪实施计划见 [docs/tasks.md](docs/tasks.md)。

配置、插件和开发说明分别见 [docs/configuration.md](docs/configuration.md)、[docs/plugins.md](docs/plugins.md) 和 [docs/development.md](docs/development.md)。

文件发现、TXT/Markdown 加载与安全规则见 [docs/ingestion.md](docs/ingestion.md)。

## CLI 快速开始

```bash
rag-index init ./my-rag
rag-index preview ./documents --show-content
rag-index --config rag.yaml build ./documents --json
rag-index --config rag.yaml query "如何离线安装？" --profile fast --json
rag-index --config rag.yaml query "ERR-1042" --profile balanced --filter metadata.department=support --json
```

`build` 和 `query` 只读取配置指定的本地模型，不会自动下载。Python 查询接口、Qdrant Local 限制和引用结果见 [docs/querying.md](docs/querying.md)。

Manifest、单文件 sidecar 与组合路由见 [docs/import-modes.md](docs/import-modes.md)。

增量同步、journal、中断恢复、staging alias 和回滚见 [docs/index-lifecycle.md](docs/index-lifecycle.md)。

原生 Python 检索结果和标准 LangChain `BaseRetriever` 接入见 [docs/querying.md](docs/querying.md)。
