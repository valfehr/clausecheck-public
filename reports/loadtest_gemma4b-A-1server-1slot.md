# Load test `gemma4b-A-1server-1slot`

Endpoint http://127.0.0.1:8081; 32 requests per level; ~1.5k-token prompts, max 200 output tokens, streamed.

| concurrency | TTFT p50 | TTFT p95 | per-request tok/s (p50) | throughput tok/s | req/s | backends |
|---|---|---|---|---|---|---|
| 1 | 0.36s | 0.39s | 88.9 | 62.5 | 0.86 | direct:32 |
| 4 | 3.70s | 4.37s | 87.2 | 61.4 | 0.87 | direct:32 |
| 16 | 15.97s | 18.37s | 87.2 | 61.8 | 0.87 | direct:32 |
