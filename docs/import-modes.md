# Manifest、sidecar 与路由

目录扫描之外，可以用 Manifest 为每个文件附加 metadata 或指定 chunk profile：

```yaml
version: 1
namespace: company-knowledge
documents:
  - path: contracts/main.pdf
    profile: parent_child
    metadata:
      category: contract
  - path: faq.md
    profile: markdown_heading
```

将 Manifest 文件作为 `preview` 或 `build` 的唯一输入即可。路径和 glob 相对 Manifest 所在目录解析，同一个 source 重复出现会报错。

单文件 sidecar 命名为 `filename.ext.corpus.yaml`：

```yaml
version: 1
profile: table_rows
metadata:
  department: finance
```

Profile 选择优先级为 Manifest、sidecar、routing、格式默认值。Manifest 和 sidecar 使用严格 schema，不能修改 embedding 或向量索引规格。

Routing 可以同时匹配扩展名、文件名、规范化相对路径和 metadata；同一条规则中配置的条件必须全部满足，第一条匹配规则生效。
