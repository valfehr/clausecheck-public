# ClauseCheck

Contract review against a set of "golden rules" (governing law, assignment, liability cap, audit rights, termination
for convenience, insurance, IP ownership, change of control). For each rule, a model returns **green / yellow / red**
with citations to the exact contract passages, and a lawyer validates every verdict.

The project is built to **measure** design choices rather than assume them: long context vs RAG, API vs local models,
fine-tuning, workflows vs agents, verification of AI-drafted redlines, prompt injection. Every claim below comes from
an eval with pass marks fixed before the run. Not legal advice; first-pass verdicts for a lawyer to check.

## Results (Claude Haiku 5.5 unless stated)

| Question | Measured answer |
|---|---|
| Release gate (holdout, contract types never used for tuning) | red recall 12/12, 0 critical misses (red rated green), 100% citations verified (after a blind label audit corrected one label; 11/12 before) |
| Cost per contract, 8 rules, with prompt caching | ~$0.006 (−69% vs no caching), ~7 s |
| Stability (same contract, 3 runs) | 7.6% of verdicts changed → re-asking uncertain verdicts 3x → 3.0%, disagreements flagged for a human |
| Full contract vs RAG (hybrid BM25 + vectors, top-10) | same quality on test, holdout and 47k–114k-token contracts; RAG ~3x cheaper per call uncached, but with caching full contract is cheaper for short contracts → full contract whenever it fits the context window |
| Agent that gathers its own evidence vs fixed top-10 search | same accuracy, 2.1–2.6x the cost → not adopted |
| Local models (Llama-3.1-8B, Gemma-3-4B, 8 GB GPU) | blocked by the gates; Gemma-4B distilled from Claude labels (QLoRA): red recall 24% → 90%, still 1 critical miss and 27% verdict flips |
| Routing every local "green" to Claude for a re-check | 0 critical misses on test and holdout |
| Drafting agent (Haiku drafts, Sonnet verifies, playbook) | verifier rejects 24/24 deliberately flawed redlines; same-model drafting+checking let 3/7 bad redlines through |
| Prompt injection (5 attack types planted in real contracts) | 0/40 verdicts changed; a detector flags such contracts for a human (5/5 attacks, 0 false alarms on 510 contracts) |

## How it works

```
contract → segment into clauses (exact character offsets)
        → full contract if it fits the context window, else hybrid search top-10
        → judge per rule (structured output, segment-ID citations, verbatim-quote check)
        → re-ask uncertain verdicts 3x, flag disagreements
        → pause for the lawyer (approve / change / wrong passage), feedback stored and routed
        → optional: "suggest a fix" (drafter + independent verifier), lawyer accepts / edits / rejects
```

- `src/clausecheck/`: judge, segmenter, search (Elasticsearch), LangGraph review workflow (checkpoints, human-review
  pause, verdict store), FastAPI service (per-tenant access), MCP server, drafting agent, feedback router, tracing
  (OpenTelemetry GenAI attributes, metadata only), injection detector, local-model adapter (llama.cpp)
- `scripts/`: every experiment and eval (each docstring says what it measures and what it costs)
- `data/evals/`: eval suites (severity, hard cases, retrieval, long contracts), split by contract type
- `reports/`: generated results
- `docs/`: target AWS architecture, cost estimate, component cards

## Running it

```bash
uv sync                                   # add --no-default-groups --group dev to skip the ML stack
uv run pytest -q tests                    # $0: model calls are stubbed
uv run uvicorn clausecheck.api:app --port 8090
```

Model calls need `ANTHROPIC_API_KEY` in `.env`; a hard budget cap in code refuses any call that could exceed it.
RAG and indexing need a local Elasticsearch 9 node; local models need llama.cpp.

## Data

Built on **CUAD v1** (Contract Understanding Atticus Dataset) by The Atticus Project, licensed **CC BY 4.0**. The eval
suites contain CUAD contract text, some of it modified on purpose (clauses rewritten to a target label, deleted, or
with planted test content); they are evaluation data, not real agreements.
