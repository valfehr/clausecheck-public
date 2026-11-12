# ClauseCheck — target AWS topology

Status: **design on paper** (Phase 0). Becomes real in Phase 8. Every box has a component card in
[components.md](components.md).

## Diagram

```
                ┌──────────────────────────── Client's AWS account (VPC) ─────────────────────────────┐
 Lawyer         │                                                                                     │
 browser ──HTTPS──► CloudFront ──► S3 (React static build)                                            │
    │           │                                                                                     │
    │  API calls (JWT from Cognito / client SSO)                                                      │
    └──────────────► ALB ──► API service (FastAPI, ECS Fargate, 2+ tasks, private subnets)            │
                │              │  ① POST /documents: presigned S3 URL; PDF goes straight to S3        │
                │              │  ② POST /reviews: write job row, send message                         │
                │              ▼                                                                      │
                │   S3 (contracts, KMS-encrypted) ──event──► Lambda "intake" (validate, virus-scan hook,│
                │              │                               enqueue parse job)                      │
                │              ▼                                                                      │
                │            SQS (review jobs, DLQ after 3 failures)                                  │
                │              ▼                                                                      │
                │   Worker service (LangGraph, ECS Fargate, autoscaled on queue depth)                │
                │      ├─► OpenSearch (BM25 + k-NN vectors, per-tenant index / doc-level security)    │
                │      ├─► Model endpoint ─┬─ Amazon Bedrock (Claude) via VPC endpoint (PrivateLink)  │
                │      │                   └─ or vLLM on EC2 GPU / SageMaker endpoint (open-weights)  │
                │      └─► RDS Postgres (jobs, verdicts, citations, feedback, LangGraph checkpoints)  │
                │                                                                                     │
                │   Batch side (not on the request path):                                             │
                │      Glue job (nightly): re-parse / re-embed corpus, export feedback → eval sets    │
                │      Step Functions / CodeBuild: eval suite run before a model or prompt is promoted│
                │                                                                                     │
                │   Cross-cutting: Secrets Manager · KMS · CloudWatch + X-Ray/OTel → Langfuse/Datadog │
                └─────────────────────────────────────────────────────────────────────────────────────┘
```

## One request, end to end ("check this NDA against the 8 golden rules")

| # | Step | Runs where | Triggered by | Identity | On failure |
|---|---|---|---|---|---|
| 1 | Browser loads the app | CloudFront → S3 | User | Public static assets | CDN retry |
| 2 | User signs in | Cognito / client IdP (SAML/OIDC) | User | User identity → JWT | Login error |
| 3 | Upload PDF | Browser → **S3 directly** (presigned URL from API) | User | Presigned URL scoped to one key, 5-min expiry | Re-upload; API never proxies large files |
| 4 | Intake | **Lambda** (S3 `ObjectCreated` event) | S3 event | Lambda execution role: read that bucket, send to SQS | Lambda retries ×2, then on-failure destination |
| 5 | Start review | API on **ECS Fargate** (`POST /reviews`) | HTTP | User JWT validated; task role writes RDS + SQS | 4xx/5xx to client; idempotency key prevents duplicate jobs |
| 6 | Run the graph | **Worker on ECS Fargate** (LangGraph) | SQS message | Worker task role (least privilege: OpenSearch read, Bedrock invoke, RDS write) | Checkpoint per node in Postgres → resume from last node; message returns to queue; DLQ after 3 |
| 6b | Choose context | Worker (pure Python) | Graph node | n/a | Full contract whenever it fits the model's context window (20% margin); RAG only beyond it (decision 2026-11-12). Verdict store hit (same contract hash + prompt + model version) → skip to step 11 with no model calls |
| 7 | Retrieve (RAG branch only) | OpenSearch (managed), or Postgres full-text + pgvector at low volume | Graph node | Worker role via IAM/SigV4; filtered to the user's tenant/docs | Retry, then fall back to BM25-only; node marked degraded |
| 8 | Judge per rule (parallel) | **Bedrock** (or vLLM endpoint) | Graph node | Worker role `bedrock:InvokeModel` on specific model ARNs | SDK retries on 429/5xx; fallback model; rule marked "needs human" rather than guessing |
| 9 | Verify citations | Worker (pure Python) | Graph node | n/a | Unverifiable citation → reject verdict → retry once → "needs human" |
| 10 | Persist + notify | RDS Postgres | Graph end | Worker role | Transaction; job state = failed with reason |
| 11 | Show results | API → browser (poll or SSE) | Browser | User JWT; row-level check user ↔ document | Partial results shown per rule as they complete |

## Where Lambda fits, and where it doesn't

**Lambda fits:** event glue that is short, stateless and spiky. S3 upload → validate → enqueue; webhook
receivers; scheduled small jobs; a thin API for low-traffic admin endpoints.

**Lambda doesn't fit:**
- The **LangGraph run**. A review can take minutes, fans out to many model calls, and needs checkpoints and
  resume. Lambda's 15-minute ceiling, cold starts and per-invocation model make that awkward. An
  ECS/Fargate worker pulling from SQS is simpler and cheaper to reason about.
- **GPU inference.** Lambda has no GPUs.
- **Holding streaming connections** (SSE/websockets) for long jobs.

You *could* run the graph on Step Functions + Lambda (one Lambda per node). That's a legitimate design when
you want AWS-native orchestration and visual audit trails. The trade-off is that state passing goes through
Step Functions payload limits (256 KB), so documents and context are passed by S3 reference.

## Where Glue fits

Glue is **batch ETL** (managed Spark/Python shell). Here: nightly re-parse/re-embed of the corpus when the
chunker or embedding model changes; exporting lawyer feedback from Postgres into versioned eval/training sets
in S3. **Never on the request path.** Startup latency is measured in tens of seconds to minutes, and it isn't
built for per-request work.

## How the model is called: three options

| | Direct vendor API (Anthropic / OpenAI) | **Amazon Bedrock** | Azure OpenAI |
|---|---|---|---|
| Network | Public internet egress (or vendor private link where offered) | VPC endpoint (PrivateLink); traffic stays on AWS | Azure private endpoint; cross-cloud from AWS |
| Identity | API key in Secrets Manager | **IAM role**, no long-lived keys | Entra ID / API key |
| Data terms | Vendor's terms (ZDR by agreement) | AWS terms; data not used for training; regional | Microsoft terms; regional |
| Quotas | Vendor org limits | Per-account, per-region quotas; provisioned throughput option | Per-deployment TPM quotas |
| Model choice | One vendor | Claude + Llama, Mistral, etc. through one API | OpenAI models (+ some others) |
| Pick when | Prototyping; newest features first | Client is on AWS and wants IAM, VPC, CloudTrail audit | Client is on Azure / Microsoft-standardised |

In this project: dev uses the direct Anthropic API (cheapest setup). The adapter in `src/clausecheck/llm.py`
gets a Bedrock implementation in Phase 8, and that is the only code that changes.

## Added after Phase 6 (2026-11-12)

- **MCP server** (`check_contract`, `find_clause`, `explain_verdict`, `list_rules`): locally stdio; on AWS a
  streamable-HTTP endpoint behind the API with the same user auth, so external assistants get the user's own access,
  never more. Expected refusals return readable `ToolError` messages.
- **Human review** is a LangGraph interrupt: the job state is "awaiting review" in Postgres, the UI resumes it.
- **Verdict store** in Postgres keyed by contract hash + prompt version + model version (same contract, same answer).
- **Drafting agent**: Haiku drafts, Sonnet verifies (a different, stronger model), only on request for red/yellow verdicts.
- **Cost picture:** [aws-costs.md](aws-costs.md): demo ~$100/mo, small HA production ~$300–400/mo, model calls ~$9 per
  1,000 contracts; at this scale infrastructure (NAT, load balancer, search cluster, database) dominates.

## Open questions to settle by Phase 8
- Multi-tenant isolation: index per tenant vs document-level security in OpenSearch
- Bedrock on-demand vs provisioned throughput at the expected volume (needs Phase 1 $/doc numbers)
- Where PII redaction runs (before indexing? before the model call?)
