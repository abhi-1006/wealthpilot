# WealthPilot — Architecture & Milestone Review

**Domain:** FinTech / Digital Lending — SME Loan Underwriting & Credit Research Assistant
**Program:** Applied AI Professional Certification Program, IIT Hyderabad

## System overview

WealthPilot is an AI copilot for a digital lender's underwriting team. It parses raw SME loan
applications into validated structured data, computes risk signals (DSCR, bureau/bank-statement
lookups), retrieves and cites the actual credit policy when answering underwriting questions,
routes applications through a human-approval gate, runs a review committee (an LLM Analyst plus
deterministic Risk and Compliance Reviewers), and serves policy Q&A and a sample underwriting flow
through a guardrailed HTTP API. The single most important thing it has to get right: **never let a
lending decision reference a protected attribute, and never fabricate a policy citation** —
both are compliance failures, not just quality issues, in a regulated-lending domain.

## Components & data flow

| File | Milestone | What it does | Failure mode if it breaks |
|---|---|---|---|
| `m1.py` | M1 | Parses raw loan applications into validated `LoanApplication`/`FinancialSnapshot` Pydantic objects via a provider-agnostic LiteLLM client, with a repair loop for malformed model output. Scores itself against each record's `ground_truth` block. | A silently-wrong parse (e.g. a field defaulting to `None` due to a key-name mismatch) propagates a wrong number into every downstream risk calculation. |
| `m2.py` | M2 | A tool-calling risk-analysis agent: DSCR calculator, interest calculator, credit-bureau lookup, bank-statement lookup, wired into a bounded LiteLLM tool-calling loop. | A miscalculated DSCR or a stale bureau lookup feeds a wrong risk signal into every later stage. |
| `m3.py` | M3 | Persistent memory (working/episodic/semantic/procedural) via Supermemory, plus lossy-summarization with a tested (not assumed) fact-survival check. | Losing applicant history context across a long conversation, or silently dropping a fact during context compression. |
| `m4.py` | M4 | RAG over the policy corpus: Gemini embeddings (disk-cached) indexed into Qdrant Cloud + BM25 hybrid search + RRF fusion + cross-encoder reranking + answer generation prompted to cite its sources, with prompt-injection defenses. Scored two ways: precision@k/recall@k/MRR against 10 exact-identifier retrieval questions, and the 20-item guardrail golden set. | A fabricated policy citation — a compliance failure in this domain, not just a wrong answer. |
| `m5.py` | M5 | LangGraph workflow: intake → verify → risk-score → conditional routing (auto-decline / request-more-info loop / human-approval), with a **real `interrupt()`** human gate and a **durable `SqliteSaver`** checkpointer, verified by resuming a paused run from the on-disk checkpoint with a fresh connection and graph object (a simulated restart within one process). | An approval that isn't durably paused could be lost or double-executed on a redeploy. |
| `m6.py` + `wealthpilot_mcp_server.py` | M6 | Multi-agent underwriting committee: an LLM Analyst writes the memo; the Risk Reviewer and Compliance Reviewer are deterministic code checks (judgment from the model, facts from code). Enforced per-agent write-scopes, a supervisor constrained to a `legal_routes()` guard, a bounded revision loop, and an escalation path. The intake step fetches bureau data through a real MCP server. | A critic that can edit the producer's output stops being an independent check — which the enforced write-scopes prevent. |
| `m7.py` | M7 | Observability (Langfuse tracing, per-agent latency dashboard, honestly-measured tracing overhead) and reliability hardening (retry with backoff and a fallback path, plus a separately tested circuit breaker) applied to the bureau-lookup function under seeded fault injection, plus systematic evaluation: DeepEval's `HallucinationMetric` on the risk agent's summary against its real tool-result context, a retrieval-recall check against M4's golden set, and `ToolCorrectnessMetric` against the fault-injection harness. | Without hardening, a flaky bureau call crashes the step that depends on it instead of degrading gracefully; without tracing, a bad decision can't be traced back to which step produced it; without the hallucination check, a fluent but ungrounded summary goes undetected. |
| `m8.py` | M8 | Two layers. **Layer 1**: the 20-item policy-RAG golden evaluation (deterministic scoring + a Groq-backed LLM judge) plus a guardrails layer (schema validation, injection detection, protected-attribute scan), served behind FastAPI `/decision`. **Layer 2**: a full planner → tool/retrieval/direct → final-answer assistant (WealthPilot's own domain, not the lab's example), scored by deterministic checks *and* three independent LLM judges (Ragas, DeepEval, TruLens) orchestrated through `langfuse.run_experiment()`, served behind `/chat`, with a public ngrok tunnel for Postman testing. | An unguardrailed endpoint could return a policy-noncompliant answer, or accept a prompt-injection attempt, with no safety net; relying on a single judge framework risks one framework's blind spot going unnoticed. |
| `frontend.html` | Demo | A chat UI served by `m8.py`'s FastAPI app (`GET /`). Free-text questions route through `/chat` (tool computation, policy retrieval, or a direct reply, with source documents shown when retrieval was used); "underwrite application N" routes through `/underwrite/{index}` for the structured M1→M2 decision card. | A guardrail rejection (HTTP 422) not surfaced clearly in the UI could look like a silent failure instead of the safety layer working as intended. |

**How the pieces connect:** the milestones are separate, runnable components, not one chained
pipeline. M5's workflow and M6's committee both reuse M1 parsing and M2 risk scoring. M4's
retrieval powers policy Q&A, and M7 and M8 import it for evaluation. M8 serves policy Q&A and a
sample underwriting flow (M1 parse, M2 DSCR and a policy-threshold rule) over FastAPI. Wiring M4
retrieval into the M5/M6 decision path is a natural next step that this version doesn't do.

## Evaluation results summary

- **M1**: 10/10 records parsed successfully; 9/10 on missing-field accuracy against
  `ground_truth` (one record's model output inferred a value where the answer key expected the
  field left blank — a genuine minor miss, not hidden).
- **M4 retrieval metrics** (precision@k/recall@k/MRR, 10 exact-identifier questions against the
  real Qdrant + Gemini-embedding index): naive-dense MRR 0.742 → hybrid (RRF) 0.817 → hybrid +
  cross-encoder rerank **1.000**; recall@k is a perfect 1.0 across all three variants. This
  matches the lab's own expected pattern almost exactly.
- **M4/M8 policy-RAG golden eval (20 items)**: **19/20 passed** on M4's citation and
  forbidden-content checks (best run). M8 runs the same checks plus the guardrail layer; its last
  full run scored 17/20 with 18/20 guardrails clean: one answer named "single mother" while
  refusing to use it (flagged by the stricter protected-attribute scan), one model call returned
  an empty answer (caught by schema validation), and one answer varied between runs. Outputs are
  not deterministic, so results range 17–19/20. This started at 6/20 and two real bugs were found and fixed during
  verification, not by loosening the check: (1) the auto-generated eval set's `must_cite` field
  was being built with `list("a title string")`, which explodes a string into individual
  characters instead of wrapping it — corrupting 12 of the 20 items' citation targets regardless
  of answer quality; fixed at the source (`datagen/domains/base.py`) and the existing golden set
  was repaired. (2) the citation checker only recognized titles spelled out in words, not the
  model's own bracket-citation style (`[4]`/`【4】`) — fixed by resolving cited chunk ids back to
  their source document. The remaining failure in M4's best run is a genuine, minor gap: the model's answer
  correctly refuses to let "single mother" status influence a decision, but doesn't attach a
  citation to that particular refusal.
- **M7 evaluation additions**: hallucination score 0.00 on DeepEval's scale (0 = no hallucination
  detected — the risk agent's summary is fully grounded in its real tool-call context), retrieval
  recall 1.000 against M4's golden set, tool-correctness 1.00 on both the healthy- and
  failing-dependency cases.
- **M8 Layer 2 (3-judge Langfuse experiment, a separate 20-item set; judge scores on the 5
  final-answer items, one TruLens score skipped after a rate limit)**: `ragas_faithfulness` 1.000,
  `deepeval_geval_correctness` 0.960, `trulens_groundedness` 1.000, `trulens_context_relevance`
  1.000, all four deterministic scorers (tool match, retrieval keyword hit, route match, answer
  keyword) 1.000, `guardrail_passed` 1.000 clean across all 20 items. `ragas_context_precision`
  and `ragas_tool_call_accuracy` sit at 0.600 — the softer, harder-to-satisfy judge metrics, worth
  a closer look but not alarming given every deterministic and other-judge signal on the same
  items is clean. The experiment run itself hit real Gemini free-tier rate limits mid-run and
  recovered via the retry/backoff logic — the reliability engineering proving itself under actual
  conditions, not just a clean happy-path run.
- **This is reported honestly, not tuned to look better.** Every fix above corrected an actual
  bug in the scoring or data pipeline; none of them changed how the model answers.

## Guardrails & safety

**Covered:** Pydantic schema validation on every response, regex-based prompt-injection detection
run on both user input and retrieved context, a protected-attribute term scan on generated
answers, per-agent write-scope enforcement (M6) so a critic cannot edit the artifact it's
supposed to be independently checking, and a bounded revision loop everywhere a loop-back edge
exists (never an unguarded loop).

**Explicitly out of scope for this version, stated honestly:** the protected-attribute scan is a
keyword list, not a trained classifier — it will miss paraphrased bias and can false-positive on
legitimate mentions. The prompt-injection regex is a textbook-pattern catcher, not a hardened
filter. The intake-record → mock-API-table ID linkage is a known, documented gap (the two data
sources use different ID schemes and aren't naturally joined), so every application currently
uses the same placeholder bureau record (score 374) rather than its own.

## Known limitations

- Golden-set results vary between runs (17–19/20) because model outputs aren't deterministic.
  The recurring miss is a correct refusal without an attached citation — see the results above.
- The components aren't chained into one pipeline yet; M4 retrieval isn't part of the M5/M6
  decision path.
- `ragas_context_precision` and `ragas_tool_call_accuracy` (M8 Layer 2) sit at 0.600 — lower than
  every other signal on the same items, worth a closer look in a future pass but not blocking.
- M3 and M4 use different vector/memory backends by design (Supermemory for conversational
  memory, Qdrant Cloud for the policy corpus) — this mirrors the actual lab content more closely
  than the summary problem-statement doc's wording, which conflates the two.
- Fault injection in M7 is applied to the bureau-lookup dependency specifically; other external
  calls (Groq itself) rely on LiteLLM's own retry rather than a bespoke circuit breaker.
- Gemini chat models hit an account-level `limit: 0` quota wall earlier in this project (confirmed
  via direct API calls across two different models); a fresh Google Cloud project's *embedding*
  calls did not hit this, so Gemini is used for embeddings throughout but Groq remains the
  generation LLM everywhere except M8 Layer 2's planner/judges, which use Gemini directly per the
  lab's own design.

## Deployment notes

- Secrets (`GROQ_API_KEY`, `SUPERMEMORY_API_KEY`, `GOOGLE_API_KEY`, `QDRANT_URL`,
  `QDRANT_API_KEY`, `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`, `NGROK_AUTHTOKEN`) live in `.env`,
  excluded from git via `.gitignore`.
- Rate limits: Groq's tight 8000 TPM cap and Gemini's 15 RPM free-tier cap are both handled with
  paced `time.sleep` calls and retry/backoff rather than assuming an unlimited budget; the M8
  Layer 2 experiment run genuinely hit Gemini's rate limit mid-run and recovered via this logic.
- `m8.py`'s FastAPI service is proven both locally (`/health`, `/decision`, `/chat`, a real
  injection attempt correctly rejected with a 422) and publicly, via a live ngrok tunnel tested
  in a browser through the public URL.
- Rollback: changes are tracked in git and can be reverted individually.

