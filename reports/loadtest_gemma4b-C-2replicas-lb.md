# Load test `gemma4b-C-2replicas-lb`

Endpoint http://127.0.0.1:8080; 32 requests per level; ~1.5k-token prompts, max 200 output tokens, streamed.

| concurrency | TTFT p50 | TTFT p95 | per-request tok/s (p50) | throughput tok/s | req/s | backends |
|---|---|---|---|---|---|---|
| 1 | 0.38s | 0.41s | 88.8 | 62.0 | 0.86 | 8081:32 |
| 4 | 0.68s | 0.98s | 25.0 | 77.7 | 1.08 | 8081:15, 8082:17 |
| 16 | 4.45s | 7.40s | 20.8 | 111.5 | 1.52 | 8081:15, 8082:17 |
