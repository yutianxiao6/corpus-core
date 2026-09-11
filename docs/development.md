# 开发指南

## 环境

```bash
uv sync --all-extras
source .venv/bin/activate
```

核心质量门：

```bash
pytest -q
python -m unittest discover -s tests -q
ruff check .
ruff format --check .
mypy src
```

测试不能访问网络或下载模型。外部组件使用 fake 或本地 fixture；模型与大体积文档不得提交到仓库。

## 完成定义

任务只有在实现、边界错误、单元测试、类型检查和对应文档同时完成后，才能在 [任务清单](tasks.md) 中标记为完成。公共协议和配置不兼容变化必须按语义化版本升级。
