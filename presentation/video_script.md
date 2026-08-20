# WealthPilot Demo Video Script (target: ~9:30, under the 10-min cap)

Read this loosely, not word-for-word — sound like you know it, not like you're reading.
Time budget in [brackets]. Have the PDF open in one window, the frontend
(http://127.0.0.1:8000, run `python3 -c "import uvicorn; from m8 import api_app; uvicorn.run(api_app, host='127.0.0.1', port=8000)"`
first) in a browser tab ready to switch to.

---

## [0:00–0:20] Slide 1 — Title

"Hi, I'm Abhiram, and this is WealthPilot — an SME loan underwriting and credit
research assistant, my capstone for the Applied AI Professional Certification
Program at IIT Hyderabad. I'll walk through what it does, how it's built across
all 8 milestones, and then show it running live."

## [0:20–1:00] Slide 2 — The Problem

"The scenario: a digital lender's credit-ops team manually reviews SME loan
applications — reading financials, checking policy, one application at a time.
Three things make this hard to automate carelessly: it's slow, a fabricated or
misquoted policy citation is a compliance failure, not just a wrong answer, and
similar applicants with different demographics have to reach the same decision,
provably. WealthPilot automates the first pass of that — parse, verify,
retrieve policy, decide — while keeping a human in the loop for every real
decision."

## [1:00–1:45] Slide 3 — Architecture

"The system is eight milestones wired into one pipeline. A raw application goes
through intake, risk scoring, policy retrieval, an orchestrated workflow with a
human approval gate, and a multi-agent committee review. Underneath all of
that: persistent memory, observability and reliability hardening, and an
evaluation-and-guardrails layer that packages the whole thing as an API. I'll
go through each one with what I actually built and verified — not just what
the spec asked for."

## [1:45–2:45] Slide 4 — M1 + M2

"M1 parses a raw loan application into a validated object — business name,
sector, requested amount, financials — using a provider-agnostic LLM client,
so I can swap Groq for another provider in one line. If the model's output
doesn't validate, a repair loop sends the error back and asks it to fix it. I
scored this against a ground-truth answer key that ships with the synthetic
data — not just 'did it not crash,' but 'did it get the actual values right.'
Result: 10 out of 10 records parsed correctly.

M2 is a tool-calling risk agent — the model decides which tool to call: a DSCR
calculator, an interest calculator, and real lookups against mock credit-bureau
and bank-statement data. I verified the DSCR the agent computed — 1.64 —
against the same number computed by hand from one of my eval cases."

## [2:45–3:45] Slide 5 — M3 + M4

"M3 is persistent memory — four kinds: working, episodic, semantic, procedural
— backed by Supermemory, a real hosted memory service, not a mock. Because
summarization is lossy, I didn't just assume it works — I planted a fact,
forced a summary, and tested that the fact survived. It does, every run.

M4 is the production RAG layer over the credit policy corpus — local
embeddings plus BM25 keyword search, fused together, then reranked with a
cross-encoder before the model ever sees a chunk. Every answer has to cite its
source. If I ask what the minimum DSCR is, it doesn't just answer — it cites
the exact policy clause. And because a fabricated citation is a compliance
risk here, not just a UX problem, I built in a defense against prompt
injection too — anything retrieved from the corpus is treated as untrusted
data, never as instructions."

## [3:45–4:45] Slide 6 — M5

"M5 is where it gets interesting. This is a LangGraph workflow — intake,
verify, risk-score, then route: auto-decline if the DSCR is well below the
policy floor, loop back for more info if fields are missing, or pause for
human approval if it looks fine on paper — because policy says loans like this
still need a human sign-off.

The pause is a real `interrupt()` call — the graph actually stops and waits,
it doesn't just label a decision and move on. And the checkpoint is durable —
I proved this by pausing the graph, closing the database connection, opening a
brand new connection and a brand new graph object — simulating an actual
process restart — and resuming from exactly where it left off. That's the
actual milestone requirement, not just 'a workflow that runs.'"

## [4:45–5:45] Slide 7 — M6

"M6 is a multi-agent underwriting committee. An Analyst writes the
underwriting memo. Two independent critics review it — a Risk Reviewer that
checks the DSCR figure in the memo actually matches what was computed, and a
Compliance Reviewer that scans for any protected-attribute language. Neither
critic can edit the memo it's reviewing — that's an enforced write-scope, not
just a convention. A supervisor routes between them as a plain function,
constrained to only the routes that are actually legal given the current
state. And the bureau data access goes through a real MCP server — a separate
process, talked to over the actual protocol, not a bare function call."

## [5:45–6:45] Slide 8 — M7 + M8

"M7 is observability and reliability. Every node is traced, tagged by run, and
I built a real per-agent cost and latency dashboard from that trace data. For
reliability, I ran a seeded fault-injection test against the bureau lookup —
simulated a 30% failure rate — and measured a 45% success rate with no
protection. After adding retries, a fallback, and a circuit breaker: 100%.
And the circuit breaker actually stops hammering a dead dependency — in one
test, it let through exactly 3 real failures before short-circuiting the rest.

M8 is the evaluation and guardrails layer — a 20-item golden eval set, scored
both deterministically and by an LLM judge, plus a guardrails layer that runs
independent of answer quality — schema validation, prompt-injection detection,
a protected-attribute scan. It's all packaged behind a FastAPI service."

## [6:45–7:30] Slide 9 — Honest Results

"I want to be upfront about the numbers rather than cherry-pick them. Overall,
7 out of 20 golden eval items passed end-to-end. That's not a great headline
number, and I'm not going to pretend otherwise — but 18 of 20 passed
guardrails clean, and the failures cluster specifically in auto-generated
questions with retrieval gaps, not in the hand-written adversarial cases —
the bias-probe and prompt-injection cases mostly passed. A low score with
clean guardrails and a known, specific cause is a more honest and more useful
signal than a suspiciously perfect one would have been."

## [7:30–9:00] Live Demo

*(Switch to the browser tab showing http://127.0.0.1:8000)*

"Let me show it running. This is the frontend, wired directly to the FastAPI
backend. I'll ask a policy question—"

*(Type: "What is the minimum DSCR for SME term loans?", click Ask, wait for
the grounded, cited answer)*

"—and that's a real retrieval and generation call, citing the actual policy
document. Now let's run an actual underwriting decision—"

*(Click "Underwrite sample application #1")*

"—this is a real application from the synthetic dataset, parsed by M1,
scored by M2, with the decision logic from M5 and M6 applied. DSCR 0.75,
below the policy floor — declined, with the reason stated."

*(Optional: try a second sample application showing a different decision)*

## [9:00–9:30] Slide 10 — Closing

"That's WealthPilot — all 8 milestones, each one built to the actual
technical requirement, not just the surface description, and verified with
real output rather than assumed. Code's at
github.com/abhi-1006/wealthpilot. Thank you."

---

## If you're asked a question live (quick answers ready)

- **"Why not use LangChain/Qdrant/[the exact lab tool] for X?"** — "I made a
  documented scope call given the timeline — [local embeddings instead of
  Qdrant Cloud / one LLM judge instead of three frameworks] — same underlying
  concept, fewer new external dependencies to set up. It's in
  ARCHITECTURE.md."
- **"Why is the eval score only 7/20?"** — "It's a real, honest number. The
  failures are concentrated in auto-generated eval questions with retrieval
  gaps — I checked, and the hand-written adversarial cases (bias probes,
  injection attempts) mostly passed, which is the part that actually matters
  for a lending compliance system."
- **"How does M5's human gate actually work?"** — "It's LangGraph's
  `interrupt()` — the graph genuinely pauses execution and waits for a
  `Command(resume=...)` call. I proved it survives a process restart by
  literally closing the database connection and reopening a fresh one before
  resuming."
