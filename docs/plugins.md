# 插件开发

扩展组件必须作为已安装 Python 包发布，并通过 entry point 暴露。运行时只加载配置中显式启用的插件。

```toml
[project.entry-points."offline_rag.chunkers"]
company_manual = "company_plugin:CompanyManualChunker"
```

启用方式：

```yaml
enabled_plugins:
  - chunkers:company_manual
```

支持的 kind 与 entry point group：

| kind | group |
|---|---|
| `source_providers` | `offline_rag.source_providers` |
| `loaders` | `offline_rag.loaders` |
| `parsers` | `offline_rag.parsers` |
| `chunkers` | `offline_rag.chunkers` |
| `processors` | `offline_rag.processors` |
| `embeddings` | `offline_rag.embeddings` |
| `vector_stores` | `offline_rag.vector_stores` |
| `retrieval_strategies` | `offline_rag.retrieval_strategies` |
| `organizers` | `offline_rag.organizers` |

同名插件、缺失插件或加载失败都会抛出不包含文档正文的 `RegistryError`。插件属于受信任代码，部署方必须在离线交付前审查包来源和 checksum。
