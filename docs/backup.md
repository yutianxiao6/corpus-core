# 索引备份与恢复

Local 模式会在持有 Qdrant 客户端锁时创建 gzip tar 归档，包含完整 Qdrant 目录、collection 名称、索引指纹和目录 SHA-256。恢复会先校验 manifest 与目录 checksum，再将目录原子替换；旧目录会保留为 `*.before-restore-*`，便于人工回退。

```bash
corpus-index --config corpus.yaml backup ./backups/index.tar.gz --json
corpus-index --config corpus.yaml restore ./backups/index.tar.gz --yes --json
```

`restore` 是覆盖操作，必须显式传入 `--yes`。应用进程必须停止，或使用独立恢复目录完成替换后再启动；不要在正在使用同一 Qdrant Local 目录时恢复。

Server 模式不会把服务端 snapshot 数据下载到客户端，而是生成包含 collection 和 snapshot 名称的 JSON descriptor：

```bash
corpus-index --config corpus-server.yaml backup ./backups/server-snapshot.json --json
corpus-index --config corpus-server.yaml restore ./backups/server-snapshot.json --yes --json
```

descriptor 只能在同一 Qdrant Server 的可见 snapshot 存储上恢复。生产环境应同时备份 Qdrant Server 的 snapshot 存储和本项目 ingestion journal；恢复后运行 `corpus-index doctor` 及固定评测集，确认 collection 指纹和召回质量。
