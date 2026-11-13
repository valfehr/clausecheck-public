# Component cards

Each card answers the five questions. **Local** = what runs today on the laptop. **AWS** = target
(see [aws-topology.md](aws-topology.md), costs in [aws-costs.md](aws-costs.md)). Numbers come from PLAN.md → Results.
Updated 2026-11-12 (after Phase 6).

## Built and measured

| Component | Where it runs (local → AWS) | Triggered by | Identity | When it fails | How I measured it helped |
|---|---|---|---|---|---|
| **Parser / segmenter** (`segment.py`) | Python in-process → inside the worker | Graph node | n/a | Malformed layout → empty or contradictory verdicts are rejected and retried once, then flagged | 📏 95% of lawyer-labelled clauses kept in one segment (900 chars, 2.2x excess); 3 ms per contract |
| **Context router** (`graph.py`) | Python → worker | Graph start | n/a | Token estimate off → 20% safety margin below the context window | 📏 full contract and RAG tie on quality; full contract passes on long contracts where RAG misses by one case |
| **Judge** (`judge.py`, `judge-v1.1+sc3`) | Claude Haiku API → Bedrock | Graph node per rule | API key in `.env` → IAM `bedrock:InvokeModel` | Unverifiable citation → retry once → "needs a closer look"; disagreement across re-asks → flagged | 📏 holdout 12/12 reds (11/12 before a label correction), 0 critical misses, 100% citations; stability 3.0% (3 runs, test) |
| **Model adapter + cost ledger** (`llm.py`) | Python → same code, Bedrock client | Every model call | as above | SDK retries 429/5xx; **hard budget cap** raises before an unaffordable call; lock on ledger writes | every call logged with $; 📏 total project spend $11.20 of a $15 cap; disk cache = $0 re-runs |
| **Prompt caching** | Anthropic API → Bedrock (check model support) | Rule 1 writes, rules 2–8 read | n/a | Parallel first calls would each pay the write → rule 1 first | 📏 70/70 cache hits; −69% per contract ($0.006 vs $0.017) |
| **Retriever + index** (`search.py`) | Elasticsearch local → OpenSearch, or Postgres full-text + pgvector at low volume | RAG branch only (beyond the context window, local models, cross-contract search) | SigV4 / IAM; tenant filter | Retry, then BM25-only, marked degraded | 📏 hybrid hit@10 100% on real spans; reranker rejected (−6 points, +500 ms) |
| **Embeddings** (`embed.py`) | bge-small on GPU/CPU → worker CPU or a Bedrock embedding model | Indexing, queries | n/a | Concurrent GPU calls crashed CUDA → one call at a time (lock) | 📏 16 segments/s CPU on real clauses |
| **Party metadata** (`parties.py`) | Haiku at ingestion → same, stored in `contract_meta` | Contract upload | as judge | Client name not matched → prompt unchanged, never a guess | 📏 no measurable gain for Claude; built, not enabled in the judge prompt |
| **Review graph** (LangGraph) | Python + SQLite checkpoints → ECS Fargate worker + Postgres | Job (SQS on AWS) | Worker task role | Crash → resume from checkpoint; finished parallel rules aren't re-run | 📏 parity 42/42 at $0; crash + resume re-ran only the crashed rule (3 calls) |
| **Verdict store** | SQLite `data/clausecheck.db` → RDS Postgres | Graph end; re-open | DB role per service | Key = contract hash + prompt + model version; a change in any part re-runs | 📏 42/42 second-pass reviews served with 0 model calls |
| **Human review step** | Graph `interrupt()` → same, UI in Phase 7 | Verdicts ready | Reviewer identity (JWT) | Pause persists in the checkpoint; nothing auto-signed | feedback rows stored per decision (routing measured in Phase 7) |
| **Drafting agent v2** (`drafting_v2.py`) | Haiku drafter + Sonnet verifiers → Bedrock | Red/yellow verdict, on request | as judge | Rejected after 3 rounds → no redline, suggestions only | 📏 24/24 flawed redlines caught; 2/2 accepted confirmed by Opus; yield 2/7 (too strict) |
| **MCP server** (`mcp_server.py`) | stdio process → streamable-HTTP behind the API (auth) | MCP client tool call | Local user → OAuth/JWT on AWS | Expected refusals as `ToolError` (readable); unexpected errors hidden | 📏 4 tools tested through a real stdio client; path escape refused before any file access |
| **Local model server** | llama.cpp (CUDA) → EC2 GPU / SageMaker | Adapter `local:NAME` | Network-internal | Overflow spills to RAM silently on Windows → check headroom | 📏 batching 1.9x throughput; local models blocked by the gates (Phases 4–5) |
| **Eval runner** (`run_eval.py` + compare scripts) | scripts → CI (GitHub Actions / CodeBuild) | Manual / PR | Budget-capped key; holdout needs `--release` | Fails the build on a gate regression (Phase 7) | it *is* the measurement |

## Built in Phase 7 (local)

| Component | Where it runs (local → AWS) | Triggered by | Identity | When it fails | How I measured it helped |
|---|---|---|---|---|---|
| **API** (`api.py`, FastAPI) | uvicorn + thread pool → ECS Fargate behind ALB (or API Gateway + Lambda for a demo) + SQS | HTTP | API key per user, hashed; tenant on every query (→ Cognito/SSO JWT) | Idempotency key returns the same job; other tenants get 404; job failures recorded, API never crashes | 📏 tests at $0: pause/resume, idempotent retry with 0 extra calls, cross-tenant 404; real review 5 s end to end |
| **Suggest a fix** (API → `drafting_v2.py`) | same | Lawyer clicks on a red/yellow verdict | as API | No verified fix → "no confident fix found" + suggestions; unverified fix can't be accepted | 📏 one real fix verified in round 1 for $0.02; lawyer edit ratio recorded per decision |
| **Tracing** (`telemetry.py`) | OpenTelemetry → local JSONL (→ Langfuse / X-Ray) | Every run, node, model call | n/a locally (→ service key) | Never blocks; metadata only, no document text | 📏 per-rule cost and cache hits; lost re-ask spans found and fixed (regression test) |
| **Feedback pipeline** (`feedback.py`) | SQLite + script → RDS + scheduled task | Nightly | Reviewer identity | Uncorroborated changes → audit queue; untrusted reviewers' approvals ignored | 📏 simulated: rubber-stamper and sloppy reviewer caught; 0 wrong labels reached evals; nothing in the holdout |
| **Injection detector** (`injection.py`) | in the graph → same | Every review | n/a | Flags the whole review with the reason; never blocks | 📏 5/5 attacks caught, 0 false alarms on 510 contracts; judge itself 0/40 changed |
| **CI** (`.github/workflows`) | GitHub Actions | Every push / PR; paid eval by hand | Repo secret for the paid eval only | Red build; on the public copy, protected `main` blocks the merge | 📏 25 s per run; a deliberately broken commit failed 2 tests |
| **Public export** (`export_public.py`) | local script → public GitHub repo | By hand | No-reply git identity | Privacy scan refuses to commit on personal wording, emails or keys | 📏 first run blocked 212 hits (all third-party contract text) → scoped scan → clean |

## Planned (Phase 8)

| Component | Where it runs (local → AWS) | Triggered by | Identity | When it fails | How it will be measured |
|---|---|---|---|---|---|
| **Web UI** | local page served by the API → S3 + CloudFront | User | API key locally → Cognito / SSO JWT | Static; CDN retries | lawyer actions per review, time to validate a contract |
| **Intake** | script → Lambda on S3 event | Upload | One bucket + one queue | Retries ×2 → failure destination | n/a |
