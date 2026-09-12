# 离线检索评测

评测集是 JSON 数组或 JSONL 文件，每条记录提供一个 query 及至少一个人工确认的相关 chunk：

```json
[
  {"query": "退款条件是什么？", "relevant_chunk_ids": ["chunk-refund-1"]},
  {"query": "ERR-1042 如何处理？", "relevant_chunk_ids": ["chunk-error-1", "chunk-runbook-2"]}
]
```

使用 CLI 在指定 profile 上运行：

```bash
corpus-index --config corpus.yaml evaluate eval.jsonl \
  --profile balanced --k 8 \
  --min-recall 0.90 --min-mrr 0.75 --json
```

报告包含 `recall_at_k`、`hit_rate_at_k`、`precision_at_k`、`mrr_at_k` 和二值相关性的 `ndcg_at_k`。任一 `--min-*` 门槛未达到时命令返回退出码 2，并在 stderr 输出 `evaluation_regression`，适合接入离线 CI。指标门槛应由部署方针对自己的中英混合语料、切块方式、embedding、fusion 和 reranker 校准；模型或索引指纹变化后必须重跑同一评测集。

Python API：

```python
from corpuscore.evaluation import RetrievalEvaluator, assert_thresholds, load_examples

examples = load_examples("eval.jsonl")
evaluator = RetrievalEvaluator(
    lambda query, k: engine.retrieve(query, profile="balanced", final_k=k).hits
)
report = evaluator.evaluate(examples, k=8)
assert_thresholds(report, {"recall_at_k": 0.90, "mrr_at_k": 0.75})
print(report.to_dict())
```
