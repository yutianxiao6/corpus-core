# 插件开发

扩展组件必须作为已安装 Python 包发布，并通过 entry point 暴露。运行时只加载配置中显式启用的插件。

```toml
[project.entry-points."corpuscore.chunkers"]
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
| `source_providers` | `corpuscore.source_providers` |
| `loaders` | `corpuscore.loaders` |
| `parsers` | `corpuscore.parsers` |
| `chunkers` | `corpuscore.chunkers` |
| `processors` | `corpuscore.processors` |
| `embeddings` | `corpuscore.embeddings` |
| `vector_stores` | `corpuscore.vector_stores` |
| `retrieval_strategies` | `corpuscore.retrieval_strategies` |
| `organizers` | `corpuscore.organizers` |

同名插件、缺失插件或加载失败都会抛出不包含文档正文的 `RegistryError`。插件属于受信任代码，部署方必须在离线交付前审查包来源和 checksum。
