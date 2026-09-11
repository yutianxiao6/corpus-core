# Offline RAG Retriever 任务清单

更新日期：2026-09-11

## 说明

- 状态：`[ ]` 未开始，`[~]` 进行中，`[x]` 已完成，`[!]` 阻塞。
- 优先级：P0 为后续全部工作的基础；P1 构成首个可用版本；P2 构成完整通用检索能力；P3 为高级扩展。
- 完成任务时必须同时满足实现、测试、文档和错误处理要求。
- 任务编号保持稳定，后续新增任务不重排已有编号。

## 当前进度

| 优先级 | 已完成 | 进行中 | 未开始 |
|---|---:|---:|---:|
| P0 | 13 | 0 | 0 |
| P1 | 20 | 0 | 0 |
| P2 | 19 | 0 | 0 |
| P3 | 2 | 0 | 8 |

## P0：架构基础

- [x] `P0-001` 初始化独立 Git 仓库。
  - 验收：仓库位于独立目录，默认分支为 `main`，不包含相邻项目。
- [x] `P0-002` 编写详细架构设计文档。
  - 验收：覆盖边界、协议、导入、切块、索引、查询、组织、并发、离线和评测。
- [x] `P0-003` 建立 `src` 布局和 Python 包元数据。
  - 依赖：P0-001。
  - 验收：Python 3.11+ 可导入 `offline_rag`，包元数据与 CLI 入口明确。
- [x] `P0-004` 实现异常层级。
  - 依赖：P0-003。
  - 验收：公共异常继承 `OfflineRagError`，错误具有稳定 code 和安全 message。
- [x] `P0-005` 实现核心文档与 Chunk 数据协议。
  - 依赖：P0-003。
  - 验收：Source、ContentBlock、ParsedDocument、Chunk、ChunkDraft 可校验且不可意外修改。
- [x] `P0-006` 实现检索、索引和评测结果协议。
  - 依赖：P0-005。
  - 验收：候选分数、引用、耗时、失败项、索引规格有稳定类型。
- [x] `P0-007` 实现严格配置 schema 与配置合并。
  - 依赖：P0-004、P0-006。
  - 验收：未知字段报错；支持默认值、YAML、CLI/API 覆盖和配置版本。
- [x] `P0-008` 实现组件 Protocol 与泛型注册表。
  - 依赖：P0-004、P0-006。
  - 验收：loader、parser、chunker、processor、embedding、vector store、strategy、organizer 可注册和解析。
- [x] `P0-009` 实现 Python entry point 插件发现。
  - 依赖：P0-008。
  - 验收：只加载显式启用的已安装插件；重复名和错误插件有清晰错误。
- [x] `P0-010` 建立标准库单元测试基线。
  - 依赖：P0-003。
  - 验收：无 pip 环境可运行核心测试；有 pytest 时可复用测试套件。
- [x] `P0-011` 建立 lint、类型检查和 CI 配置。
  - 依赖：P0-003。
  - 验收：ruff、mypy、pytest 命令明确，CI 不需要联网下载模型。
- [x] `P0-012` 确定依赖与锁定策略。
  - 依赖：P0-003。
  - 验收：区分 core、qdrant、embedding、document、ocr、dev extras，并产生可重复 lock。
- [x] `P0-013` 编写配置、插件和开发者文档骨架。
  - 依赖：P0-007、P0-009。

## P1：最小可用索引链路

- [x] `P1-001` 实现目录和显式文件发现器。
- [x] `P1-002` 实现 glob、ignore、隐藏文件和软链接安全规则。
- [x] `P1-003` 实现 TXT loader 与编码处理。
- [x] `P1-004` 实现 Markdown loader/parser 和标题 blocks。
- [x] `P1-005` 实现基础 Unicode、换行和空白标准化。
- [x] `P1-006` 实现 Recursive chunker。
- [x] `P1-007` 实现 Heading Recursive chunker。
- [x] `P1-008` 实现短块合并、超长块强制拆分和空块删除。
- [x] `P1-009` 实现 embedding_text 标题上下文注入。
- [x] `P1-010` 实现稳定 document/chunk ID 与相邻关系。
- [x] `P1-011` 实现 Qwen3 SentenceTransformers embedding 适配器。
- [x] `P1-012` 校验 query instruction 与 document 编码分离。
- [x] `P1-013` 实现索引指纹计算与兼容性检查。
- [x] `P1-014` 实现 Qdrant Local VectorStorePort。
- [x] `P1-015` 实现 dense similarity 查询策略。
- [x] `P1-016` 实现 FlatOrganizer。
- [x] `P1-017` 实现 ContextOrganizer 和引用映射。
- [x] `P1-018` 实现 `rag-index preview/build/query/doctor`。
- [x] `P1-019` 完成 TXT/Markdown 端到端离线测试。
- [x] `P1-020` 建立最小检索评测 fixture 和 Recall@K。

## P2：可靠导入与完整检索

- [x] `P2-001` 实现 PDF 文本解析并保留页码。
- [x] `P2-002` 实现 DOCX 标题、段落和表格解析。
- [x] `P2-003` 实现 HTML 正文与结构解析。
- [x] `P2-004` 实现 CSV/JSON/JSONL 结构解析。
- [x] `P2-005` 实现 Manifest 与 sidecar 配置。
- [x] `P2-006` 实现按扩展名、路径和 metadata 的 routing。
- [x] `P2-007` 实现 Page Aware、Paragraph Packing 和 Table Row chunker。
- [x] `P2-008` 实现 Parent Child chunker 与 parent 存储。
- [x] `P2-009` 实现内容去重和近似重复检测。
- [x] `P2-010` 实现导入状态机、journal 和中断恢复。
- [x] `P2-011` 实现幂等增量新增、修改和删除同步。
- [x] `P2-012` 实现 staging collection、alias 切换和回滚。
- [x] `P2-013` 实现 sparse vector 与 hybrid 检索。
- [x] `P2-014` 实现 RRF 和归一化加权融合。
- [x] `P2-015` 实现 Qwen3 本地 reranker。
- [x] `P2-016` 实现 threshold、MMR、neighbor expansion 和多样性控制。
- [x] `P2-017` 实现 Grouped、MergeNeighbors、Parent、Diverse、Debug organizer。
- [x] `P2-018` 实现 Query Profile 和逐调用覆盖。
- [x] `P2-019` 实现 LangChain `BaseRetriever` 适配器。

## P3：并发、交付与高级扩展

- [x] `P3-001` 实现 Qdrant Server VectorStorePort。
- [x] `P3-002` 实现同步、异步和批量查询一致性。
- [ ] `P3-003` 实现 embedding/reranker 并发队列和微批处理。
- [ ] `P3-004` 实现查询与导入并行压力测试。
- [ ] `P3-005` 实现完整评测工具和回归门槛。
- [ ] `P3-006` 实现索引备份、恢复与恢复验证。
- [ ] `P3-007` 生成离线 wheelhouse、模型 bundle 和 checksum。
- [ ] `P3-008` 实现 OCR、扫描 PDF、Excel 和 PowerPoint extras。
- [ ] `P3-009` 实现代码 AST 和实验性 Semantic chunker。
- [ ] `P3-010` 提供可选 HTTP 查询服务参考适配器。

## 决策检查点

- [x] `D-001` 确认同一 Qdrant 中是否默认启用 sparse vectors（显式配置启用，默认 dense-only）。
- [ ] `D-002` 确认首版强制文档格式和 OCR 范围。
- [ ] `D-003` 确认最低 CPU、内存、GPU 支持范围。
- [ ] `D-004` 确认目标 chunk 数与并发规模。
- [ ] `D-005` 确认 Windows 是否属于首版正式支持平台。
- [ ] `D-006` 确认 Qdrant Server 是否随交付包提供。
