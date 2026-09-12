# CorpusCore

[![CI](https://github.com/yutianxiao6/corpus-core/actions/workflows/ci.yml/badge.svg)](https://github.com/yutianxiao6/corpus-core/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)

CorpusCore 是一个面向私有知识库的离线优先文档索引与混合检索引擎。它既可以作为 Python 库嵌入现有应用，也可以通过可选 HTTP 服务独立部署，或作为标准 Retriever 接入 LangChain。

CorpusCore 负责文档解析、切块、向量化、索引维护、检索、重排和上下文组织；最终答案生成、身份认证与业务权限仍由上层应用负责。

> **项目状态：** 当前版本为 `0.1.0.dev0`。主要链路已有自动化测试覆盖，但首个稳定版本发布前，公共 API 仍可能发生变化。

## 为什么选择 CorpusCore？

- **离线优先：** 模型、文档与索引均保留在你控制的基础设施中。
- **库优先：** 提供原生同步、异步和保持输入顺序的批量 Python API。
- **混合检索：** 支持 dense、sparse、RRF 及归一化加权融合。
- **多格式导入：** 支持文本、Markdown、PDF、Office、结构化数据、HTML 和源代码。
- **可靠索引：** 支持增量同步、恢复日志、staging collection、原子 alias 切换与回滚。
- **灵活部署：** 单进程可使用 Qdrant Local，多进程服务可连接 Qdrant Server。
- **可扩展：** 可通过 Python entry point 扩展 loader、parser、chunker、embedding、vector store、retriever 和 organizer。
- **可观测：** 查询结果保留各阶段分数、耗时、索引版本、警告与来源元数据。

## 支持的内容

| 类别 | 格式 |
| --- | --- |
| 文本与标记 | TXT、Markdown、HTML |
| 文档 | PDF、DOCX |
| 演示文稿 | PPTX、PPTM |
| 表格与结构化数据 | CSV、JSON、JSONL、XLSX、XLSM |
| 源代码 | Python 及常见编程语言文件 |
| 扫描内容 | 通过可选 OCR 依赖解析扫描 PDF |

切块策略可按扩展名、路径、元数据、manifest 或单文件 sidecar 选择。内置 recursive、heading-aware、page-aware、table-row、parent-child、syntax-aware 和实验性 semantic chunking。

## 环境要求

- Python 3.11 或更高版本
- 与 `sentence-transformers` 兼容的本地 embedding 模型
- Qdrant Local，或可访问的 Qdrant Server
- 推荐使用 `uv` 管理环境和锁定依赖
- NVIDIA GPU 与支持 CUDA 的 PyTorch 为可选项

CorpusCore 在建库和查询过程中不会自动下载模型。请提前准备模型，并通过 `embedding.model_path` 指向本地目录。

## 安装

克隆仓库并安装全部可选功能：

```bash
git clone https://github.com/yutianxiao6/corpus-core.git
cd corpus-core
uv sync --all-extras
```

Linux 或 macOS 激活环境：

```bash
source .venv/bin/activate
```

Windows PowerShell 激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

如果只需要部分能力，可以从源码选择安装 extras：

```bash
python -m pip install ".[qdrant,embedding,documents,sparse]"
```

可用 extras 包括 `qdrant`、`embedding`、`documents`、`sparse`、`tables`、`office`、`ocr` 和 `http`。

## 快速开始

创建包含版本化配置和本地数据目录的工作区：

```bash
corpus-index init ./workspace
```

将 embedding 模型放入 `workspace/models/Qwen3-Embedding-0.6B`，或者修改 `workspace/corpus.yaml` 中的模型路径。建库前先检查环境：

```bash
corpus-index --config ./workspace/corpus.yaml doctor
```

只预览解析和切块结果，不写入索引：

```bash
corpus-index --config ./workspace/corpus.yaml preview ./workspace/documents --show-content
```

建立索引并查询：

```bash
corpus-index --config ./workspace/corpus.yaml build ./workspace/documents --json
corpus-index --config ./workspace/corpus.yaml query "退款政策是如何规定的？" --profile balanced --json
```

日常更新使用 `sync`；需要 staging 构建并原子替换索引时使用 `rebuild`：

```bash
corpus-index --config ./workspace/corpus.yaml sync ./workspace/documents --json
corpus-index --config ./workspace/corpus.yaml rebuild ./workspace/documents --json
```

## Python API

```python
from corpuscore import RetrievalEngine

with RetrievalEngine.from_yaml("./workspace/corpus.yaml") as engine:
    engine.sync("./workspace/documents")
    result = engine.retrieve(
        "合同的付款条件是什么？",
        profile="balanced",
        final_k=5,
    )

    for hit in result.hits:
        print(hit.rank, hit.score, hit.chunk.content)
```

异步查询和批量查询使用相同的 profile 与结果类型：

```python
async with RetrievalEngine.from_yaml("./workspace/corpus.yaml") as engine:
    result = await engine.aretrieve("合同的付款条件是什么？", profile="balanced")
    results = await engine.abatch_retrieve(queries, profile="balanced")
```

也可以生成标准 LangChain `BaseRetriever`：

```python
with RetrievalEngine.from_yaml("./workspace/corpus.yaml") as engine:
    retriever = engine.as_langchain_retriever(profile="balanced")
    documents = retriever.invoke("合同的付款条件是什么？")
```

Engine 持有模型和数据库资源。每个应用进程应复用一个 Engine，并通过上下文管理器、`close()` 或 `aclose()` 正确关闭。

## 配置

CorpusCore 使用严格、带版本号的 YAML 配置。未知字段与无效 profile 引用会在启动时直接报错，避免拼写错误被静默忽略。

```yaml
version: 1

runtime:
  offline: true

embedding:
  provider: qwen_sentence_transformers
  model_path: ./models/Qwen3-Embedding-0.6B
  device: auto
  dimension: 1024
  normalize: true

vector_store:
  provider: qdrant
  mode: local
  path: ./data/qdrant

retrieval_profiles:
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
    organizer: flat
```

相对路径以配置文件所在目录为基准。使用支持 CUDA 的 PyTorch 时，可将 `embedding.device` 设为 `cuda`，并根据显存容量调整 batch size。

profiles、路由、Qdrant Server、查询并发和模型校验说明见[配置指南](docs/configuration.md)。

## 文档

| 主题 | 指南 |
| --- | --- |
| 配置与部署 | [配置](docs/configuration.md) |
| Python、异步、批量与 LangChain API | [查询](docs/querying.md) |
| 文件发现与导入 | [数据导入](docs/ingestion.md) |
| Manifest、sidecar 与路由 | [导入模式](docs/import-modes.md) |
| 增量索引与回滚 | [索引生命周期](docs/index-lifecycle.md) |
| 检索效果评测 | [评测](docs/evaluation.md) |
| 索引备份与恢复 | [备份](docs/backup.md) |
| 断网环境交付 | [离线 Bundle](docs/bundle.md) |
| 可选查询服务 | [HTTP 服务](docs/http-service.md) |
| 自定义组件 | [插件](docs/plugins.md) |

## 开发

安装依赖并运行与 CI 相同的质量检查：

```bash
uv sync --all-extras
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q
uv build
```

测试不得依赖网络或自动下载模型。外部组件应使用 fake 或本地 fixture；请勿提交业务文档、模型权重、向量索引、密钥或机器专用配置。

## 参与贡献

欢迎通过 Issue 和 Pull Request 提交可复现的问题、文档改进或范围明确的功能：

1. 从 `main` 创建独立分支。
2. 为行为变更补充或更新测试。
3. 运行上面的完整质量检查。
4. 保持 Pull Request 聚焦，并说明公共 API 或配置变化。

报告安全问题时，请勿在公开 Issue 中包含私有文档、凭证、模型文件或检索正文。

## 项目边界

CorpusCore 是检索组件，不是完整的问答产品。它不提供聊天 UI，不生成最终答案，不管理用户身份，也不实现多租户权限系统。访问控制、Prompt 构建、答案生成与用户体验由接入方负责。

## 许可证

项目当前在 `pyproject.toml` 中声明为 `LicenseRef-Proprietary`，仓库尚未包含公开许可证。正式作为开源软件分发前，需要选择并加入 OSI 认可的许可证。
