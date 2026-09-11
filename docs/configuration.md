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

## 插件白名单

`enabled_plugins` 只接受 `<kind>:<entry-point-name>`，例如：

```yaml
enabled_plugins:
  - chunkers:company_manual
```

配置不能引用 Python 文件路径或任意 import string。
