# 可选 HTTP 查询适配器

HTTP 层是对 Python 检索接口的薄封装，不复制检索逻辑，也不负责答案生成。安装可选依赖：

```bash
pip install 'offline-rag-retriever[http]'
```

仅监听本机时可以直接启动：

```bash
rag-index --config rag.yaml serve --host 127.0.0.1 --port 8000
```

绑定内网地址必须通过环境变量启用 Bearer token：

```bash
export OFFLINE_RAG_HTTP_TOKEN='replace-with-a-long-random-secret'
rag-index --config rag.yaml serve \
  --host 0.0.0.0 \
  --port 8000 \
  --bearer-token-env OFFLINE_RAG_HTTP_TOKEN
```

除 `/health/live` 外，请求需要 `Authorization: Bearer ...`。适配器默认关闭 Swagger/ReDoc、CORS 和 access log，不会记录查询正文；生产环境仍应在内网反向代理完成 TLS、IP 白名单、限流、token 轮换及审计。

## 接口

- `GET /health/live`：进程存活检查，不要求鉴权。
- `GET /health/ready`：engine 生命周期已启动检查。
- `POST /v1/query`：单条异步检索。
- `POST /v1/query:batch`：保持输入顺序的批量异步检索，默认最多 32 条。

单条请求示例：

```json
{
  "query": "如何离线部署？",
  "profile": "balanced",
  "filters": {"department": "engineering"},
  "final_k": 6,
  "organizer": "context"
}
```

响应包含命中正文、各阶段分数、来源、页码、metadata、context、citations、groups、warnings、index version、embedding fingerprint 和耗时。客户端提供的合法 `X-Request-ID` 会原样返回，否则服务生成新的 ID。并发队列饱和映射为 HTTP 429；模型、索引或 Qdrant 不可用映射为 503；未知内部异常只返回通用 500，不泄露路径和堆栈。

## 嵌入现有 ASGI 服务

公司可以直接复用应用工厂：

```python
from offline_rag.config import load_config
from offline_rag.http_service import create_app

app = create_app(
    load_config("rag.yaml"),
    bearer_token_env="OFFLINE_RAG_HTTP_TOKEN",
    maximum_batch_size=32,
)
```

应用生命周期内只创建一个 `OfflineRagEngine`，关闭时异步释放模型队列、journal 和 Qdrant client。多进程部署必须使用 Qdrant Server；每个 worker 会各自加载 embedding/reranker 模型，因此 worker 数要按显存和内存规划。Qdrant Local 只能用于单进程。
