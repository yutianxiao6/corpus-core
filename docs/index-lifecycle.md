# 索引生命周期

导入状态写入本地 SQLite WAL journal。Journal 保存 job、逐 source 阶段事件、当前有效 source/chunk 清单，以及每个保留 collection 对应的 source 快照。进程启动时，未正常结束的 `running` job 会被标记为 `interrupted`，后续操作可安全重试。

`sync` 对稳定 `source_id` 和内容 hash 做四类规划：

- `new`：生成并写入新 chunks。
- `modified`：先完整写入新版本，成功后删除旧 chunks；失败则保留旧版本。
- `unchanged`：不解析、不向量化，记录为 skipped。
- `missing`：按配置删除向量和 journal 状态。

完整重建写入唯一 staging collection。只有导入无失败、point 数与报告 chunk 数一致时才原子切换 active alias。上一 collection 不删除，可用 `rag-index activate <physical-collection>` 回滚；alias 与 journal 快照会一起切换。

```bash
rag-index --config rag.yaml rebuild ./documents --json
rag-index --config rag.yaml sync ./documents --json
rag-index --config rag.yaml activate documents_active__<version>
```
