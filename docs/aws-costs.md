# ClauseCheck on AWS — monthly cost estimate

Status: **estimate on paper** (2026-11-12), for the topology in [aws-topology.md](aws-topology.md).
Prices are on-demand, us-east-1, as I last knew them, rounded. **Check every line in the AWS Pricing
Calculator before quoting a number**: prices change and differ by region. Model costs come from this
project's measurements, not from list prices alone.

## Is the context switch doable on AWS?

The rule (decision 2026-11-12): full contract whenever it fits the model's context window, RAG only when it
doesn't, and always for local models. Nothing about it is local-only:
- **Count tokens at intake** (Lambda or the worker): Bedrock offers token counting for Claude models (check
  support for the exact model), or use the Phase 0 calibration (real tokens ≈ 1.35 × chars/4). The count only
  has to tell "fits with margin" from "doesn't fit".
- **Full contract → Bedrock with prompt caching** (supported for Claude models on Bedrock; check the model and
  region). Same call pattern as today: rule 1 writes the cache, rules 2–8 read it.
- **Long contract → RAG**: BM25 + vectors in OpenSearch, or in Postgres (full-text search + pgvector) at low
  volume. Embeddings either run on the worker CPU (bge-small is small enough) or via a Bedrock embedding model.
- The router is one `if` in the graph; the rest is the same code with a different client in the adapter.

## Model cost (measured, Haiku 5.5 at $0.10 / $0.50 per M tokens; Bedrock list price assumed equal)

| | per contract (8 rules) | 1k contracts/mo | 10k contracts/mo |
|---|---|---|---|
| Full contract, cached (median contract) | ~$0.006 (+~40% with re-asks ≈ $0.008) | ~$8 | ~$80 |
| Full contract, cached, long contract (~56k tokens) | ~$0.015 (+re-asks ≈ $0.02) | — | — |
| RAG-10 (only for contracts that don't fit the window; none in CUAD) | ~$0.004 (+re-asks ≈ $0.006) | — | — |
| **Blended** (CUAD: median 11.2k tokens, p90 ~41k, max ~114k: everything fits) | **~$0.009** | **~$9** | **~$90** |

## Infrastructure, three sizes

| Component | Demo (single AZ, minimal) | Small production (HA) | Notes |
|---|---|---|---|
| Model calls (Bedrock, 1k contracts/mo) | ~$8 | ~$8 | scales linearly with volume |
| API: ECS Fargate | ~$18 (1 task, 0.5 vCPU / 1 GB) | ~$36 (2 tasks) | ~$0.04 per vCPU-hour + ~$0.0045 per GB-hour |
| Worker: ECS Fargate | ~$5 (scales to 0 when the queue is empty) | ~$36 (1 always on, autoscale) | |
| Load balancer (ALB) | ~$18 | ~$20 | ~$16/mo base + usage; API Gateway + Lambda instead for the demo saves it |
| Postgres (RDS) | ~$15 (db.t4g.micro, 20 GB) | ~$60 (t4g.small, Multi-AZ) | jobs, verdicts, feedback, checkpoints |
| Search for RAG | **$0**: Postgres full-text + pgvector in the same RDS | ~$55–80 (OpenSearch, 2 small nodes) | a single t3.small.search node is ~$26/mo; OpenSearch Serverless has a higher floor (~$175/mo) |
| Networking: NAT gateway | ~$33 | ~$66 (one per AZ) | **the classic surprise**: private subnets need NAT or VPC endpoints (~$7/mo each, per AZ) for ECR, Bedrock, logs, secrets |
| S3 + CloudFront (frontend, contracts) | ~$1 | ~$3 | pennies at this volume |
| SQS, Lambda (intake), EventBridge | ~$0 | ~$1 | within free tier or near it |
| Cognito | ~$0 | ~$0–10 | free tier covers a small user base |
| Secrets Manager, KMS | ~$3 | ~$5 | $0.40/secret, $1/key |
| CloudWatch logs + metrics | ~$3 | ~$15 | log ingest ~$0.50/GB |
| Tracing (Langfuse cloud) | $0 (hobby tier) | ~$0–60 | or self-host on the same Fargate cluster |
| **Total** | **~$100/mo** | **~$300–400/mo** | |

At 10k contracts/month the model line grows to ~$80; the rest barely moves. **Infrastructure, not tokens,
dominates the bill at this scale.**

## Local model option (for comparison)

| | monthly |
|---|---|
| g5.xlarge (1× A10G 24 GB) always on | ~$730 |
| Same, scheduled 8 h × 22 workdays | ~$180 |
| Fine-tune run (Llama-8B QLoRA, 1–2 h) | ~$1–3 one-off |

One GPU running all month costs about 90x the Haiku bill at 1k contracts/month. Local only wins when data
can't leave the environment or the GPU is kept busy at very high volume (Phase 4/5).

## Ways to make the demo nearly free

- **Deploy only for demo days**: IaC (CDK/Terraform) up in ~15 min, down after; pay for hours, not months
- **No NAT** in a demo: public subnets with tight security groups (dev only; production stays private)
- **No OpenSearch**: Postgres full-text + pgvector covers RAG for long contracts at this volume
- **API Gateway + Lambda for the thin API** instead of ALB + always-on Fargate
- **Aurora Serverless v2 / RDS stop** when idle (an RDS instance can be stopped for up to 7 days at a time)
- Rough floor with all of the above: **~$10–25/mo** plus model calls

## Takeaways

- At low volume the bill is fixed infrastructure (NAT, load balancer, search cluster, database), not tokens
- Prompt caching made the full-contract path cheap enough that RAG's token savings rarely pay for a search cluster;
  RAG is kept only where it's needed (documents beyond the context window, local models, cross-contract search)
- The hidden costs are networking (NAT/VPC endpoints) and always-on managed search; the visible cost (the model)
  is the smallest line until roughly 10k+ contracts a month
