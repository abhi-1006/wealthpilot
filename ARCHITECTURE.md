# WealthPilot — Architecture & Milestone Review

**Domain:** FinTech / Digital Lending — SME Loan Underwriting & Credit Research Assistant
**Program:** Applied AI Professional Certification Program, IIT Hyderabad

## System overview

WealthPilot is an AI copilot for a digital lender's underwriting team. It parses raw SME loan
applications into validated structured data, computes risk signals (DSCR, bureau/bank-statement
lookups), retrieves and cites the actual credit policy when answering underwriting questions,
routes applications through a human-approval gate, coordinates a three-agent review committee
(Analyst, Risk Reviewer, Compliance Reviewer), and packages the whole pipeline behind a
guardrailed HTTP API. The single most important thing it has to get right: **never let a
lending decision reference a protected attribute, and never fabricate a policy citation** —
both are compliance failures, not just quality issues, in a regulated-lending domain.

## Components & data flow

| File | Milestone | What it does | Failure mode if it breaks |
|---|---|---|---|
| `m1.py` | M1 | Parses raw loan applications into validated `LoanApplication`/`FinancialSnapshot` Pydantic objects via a provider-agnostic LiteLLM client, with a repair loop for malformed model output. Scores itself against each record's `ground_truth` block. | A silently-wrong parse (e.g. a field defaulting to `None` due to a key-name mismatch) propagates a wrong number into every downstream risk calculation. |
| `m2.py` | M2 | A tool-calling risk-analysis agent: DSCR calculator, interest calculator, credit-bureau lookup, bank-statement lookup, wired into a bounded LiteLLM tool-calling loop. | A miscalculated DSCR or a stale bureau lookup feeds a wrong risk signal into every later stage. |
| `m3.py` | M3 | Persistent memory (working/episodic/semantic/procedural) via Supermemory, plus lossy-summarization with a tested (not assumed) fact-survival check. | Losing applicant history context across a long conversation, or silently dropping a fact during context compression. |
| `m4.py` | M4 | Production RAG over the policy corpus: local embeddings + BM25 hybrid search + RRF fusion + cross-encoder reranking + grounded, cited answer generation with prompt-injection defenses. Scored against a 20-item golden eval set. | A fabricated policy citation — a compliance failure in this domain, not just a wrong answer. |
| `m5.py` | M5 | LangGraph workflow: intake → verify → risk-score → conditional routing (auto-decline / request-more-info loop / human-approval), with a **real `interrupt()`** human gate and a **durable `SqliteSaver`** checkpointer verified across an actual process restart. | An approval that isn't durably paused could be lost or double-executed on a redeploy. |
| `m6.py` + `wealthpilot_mcp_server.py` | M6 | Multi-agent underwriting committee (Analyst produces, Risk Reviewer + Compliance Reviewer independently critique) with enforced per-agent write-scopes, a supervisor constrained to a `legal_routes()` guard, a bounded revision loop, and an escalation path. Data access to the bureau API goes through a real MCP server. | A critic that can see the producer's full reasoning (or edit its output) stops being an independent check — exactly what the write-scoping and read-scoping prevent. |
| `m7.py` | M7 | Observability (Langfuse tracing, per-agent cost/latency dashboard, honestly-measured tracing overhead) and reliability hardening (retry with backoff, fallback path, circuit breaker) applied to the real bureau-lookup dependency. | Without hardening, a flaky bureau API call fails the whole pipeline instead of degrading gracefully; without tracing, a bad decision can't be traced back to which step produced it. |
| `m8.py` | M8 | The full 20-item golden evaluation (deterministic scoring + one Groq-backed LLM judge), an independent guardrails layer (schema validation, prompt-injection detection, protected-attribute scan), and a FastAPI `/health` + `/decision` deployment package, verified with a real HTTP round-trip. | An unguardrailed endpoint could return a policy-noncompliant answer, or accept a prompt-injection attempt, with no safety net. |

**Data flow, end to end:** raw application → M1 parse → M2 risk signals → M4 policy retrieval
(for any policy question) → M5 orchestrated routing with a human gate → M6 committee review for
final sign-off → M7 traces and hardens every hop → M8 exposes the whole thing as a guardrailed
API, scored against a golden set.

## Evaluation results summary

- **M1**: 10/10 records parsed successfully; missing-field accuracy checked against
  `ground_truth` (catches silent parsing errors that a pure validation-success count would miss).
- **M4**: 4/6 on an early spot-check, improving to a real 20-item run in M8.
- **M8 golden eval (20 items)**: 7/20 passed overall; 18/20 guardrails clean. The failures cluster
  in the auto-generated `factual`/`multi_hop` categories, not the hand-written adversarial cases
  (which mostly passed) — consistent with a real retrieval-quality gap on some question
  phrasings, and with occasionally imperfectly-worded auto-generated eval questions (documented
  in M4 already). **This is reported honestly, not tuned to look better** — a low score on
  auto-generated cases with clean guardrails on all cases is a more useful signal than a
  suspiciously perfect number.

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
sources use different ID schemes and aren't naturally joined) — tools resolve it pragmatically
rather than solving the underlying data-modeling problem.

## Known limitations

- RAG retrieval has real, measured gaps on some question phrasings (M4/M8 eval results above) —
  not perfect recall, and reported as such.
- M3 and M4/M7 use different vector/memory backends by design (Supermemory for conversational
  memory, a local embedding index for the policy corpus) — this mirrors the actual lab content
  more closely than the summary problem-statement doc's wording, which conflates the two.
- Fault injection in M7 is applied to the bureau-lookup dependency specifically; other external
  calls (Groq itself) rely on LiteLLM's own retry rather than a bespoke circuit breaker.
- M8 uses one LLM-judge framework (Groq via LiteLLM) rather than the reference lab's three
  (Ragas/DeepEval/TruLens), a deliberate scope decision given the submission deadline — same
  "judge alongside a deterministic metric" principle, fewer new dependencies.

## Deployment notes

- Secrets (`GROQ_API_KEY`, `SUPERMEMORY_API_KEY`) live in `.env` files excluded from git via
  `.gitignore`; `.env.example` files are committed as templates with empty values.
- Rate limits: this Groq account has a tight 8000 TPM cap discovered during development;
  every multi-call loop in this project paces itself accordingly (`time.sleep` between calls)
  rather than assuming an unlimited budget.
- No public tunnel (ngrok) is wired up for this submission — `m8.py`'s FastAPI service is proven
  with a real local HTTP round-trip; exposing it publicly is a config addition, not a code change.
- Rollback plan: every milestone is its own git commit, so any regression can be isolated to the
  commit that introduced it rather than debugged against the whole system at once.
