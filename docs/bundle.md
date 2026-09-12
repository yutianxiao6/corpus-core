# 离线交付 bundle

`uv build` 生成 wheel 和 sdist 后，可用标准库脚本组装 wheelhouse、模型目录和校验清单：

```bash
uv build
python scripts/build_offline_bundle.py ./corpuscore-bundle \
  --wheel dist/corpuscore-0.1.0.dev0-py3-none-any.whl \
  --wheel dist/corpuscore-0.1.0.dev0.tar.gz \
  --model ./models/Qwen3-Embedding-0.6B \
  --model ./models/Qwen3-Reranker-0.6B
```

脚本只复制调用方明确提供的本地文件，不解析依赖、不联网下载。输出结构为 `wheelhouse/`、`models/`、`manifest.json` 和 `checksums.sha256`。部署前可在目标机执行：

```bash
sha256sum -c checksums.sha256
python -m pip install --no-index --find-links wheelhouse corpuscore
```

`--force` 才会替换已有非空输出目录。模型 bundle 必须与配置中的模型路径、revision 和索引 fingerprint 一起交付；升级时重新生成清单并在断网环境验证安装、`doctor`、固定评测集和索引恢复。
