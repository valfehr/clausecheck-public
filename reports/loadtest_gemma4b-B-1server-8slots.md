# Load test `gemma4b-B-1server-8slots`

Endpoint http://127.0.0.1:8081; 32 requests per level; ~1.5k-token prompts, max 200 output tokens, streamed.

| concurrency | TTFT p50 | TTFT p95 | per-request tok/s (p50) | throughput tok/s | req/s | backends |
|---|---|---|---|---|---|---|
| 1 | 0.36s | 0.41s | 88.5 | 61.9 | 0.85 | direct:32 |
| 4 | 0.58s | 0.81s | 22.4 | 78.7 | 1.14 | direct:32 |
| 16 | 4.68s | 6.68s | 19.4 | 119.0 | 1.72 | direct:32 |
