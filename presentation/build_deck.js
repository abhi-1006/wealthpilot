const pptxgen = require("pptxgenjs");

// ---- Palette (ties to the frontend demo's coral accent) ----
const NAVY = "1E2761";
const NAVY_DARK = "141B4D";
const ICE = "CADCFC";
const CORAL = "DA7756";
const WHITE = "FFFFFF";
const INK = "22262E";
const MUTED = "6B7280";
const CARD_BG = "F4F6FB";

function newDeck() {
  const p = new pptxgen();
  p.layout = "LAYOUT_WIDE"; // 13.3 x 7.5
  return p;
}

function titleSlide(p) {
  const s = p.addSlide();
  s.background = { color: NAVY_DARK };
  s.addShape("rect", { x: 0, y: 0, w: 13.33, h: 7.5, fill: { color: NAVY_DARK } });

  // Coral circle motif, top-left
  s.addShape("ellipse", { x: 0.9, y: 0.9, w: 0.7, h: 0.7, fill: { color: CORAL } });
  s.addText("W", { x: 0.9, y: 0.9, w: 0.7, h: 0.7, align: "center", valign: "middle",
    fontFace: "Calibri", fontSize: 28, bold: true, color: WHITE, margin: 0 });

  s.addText("WealthPilot", { x: 0.9, y: 2.7, w: 11.5, h: 1.3,
    fontFace: "Cambria", fontSize: 54, bold: true, color: WHITE, margin: 0 });
  s.addText("SME Loan Underwriting & Credit Research Assistant", { x: 0.9, y: 3.75, w: 11.5, h: 0.7,
    fontFace: "Calibri", fontSize: 22, color: ICE, margin: 0 });

  s.addText("Applied AI Professional Certification Program  —  IIT Hyderabad", {
    x: 0.9, y: 6.35, w: 9, h: 0.5, fontFace: "Calibri", fontSize: 14, color: "9AA4C4", margin: 0 });
  s.addText("Abhiram", { x: 0.9, y: 6.75, w: 9, h: 0.4, fontFace: "Calibri", fontSize: 12, color: "6E77A0", margin: 0 });
}

function sectionHeader(s, kicker, title) {
  s.addText(kicker.toUpperCase(), { x: 0.7, y: 0.45, w: 11.9, h: 0.35,
    fontFace: "Calibri", fontSize: 12, bold: true, color: CORAL, charSpacing: 2, margin: 0 });
  s.addText(title, { x: 0.7, y: 0.78, w: 11.9, h: 0.7,
    fontFace: "Cambria", fontSize: 30, bold: true, color: NAVY, margin: 0 });
}

function problemSlide(p) {
  const s = p.addSlide();
  s.background = { color: WHITE };
  sectionHeader(s, "The Problem", "Underwriting SME loans is slow, manual, and risky to get wrong");

  const items = [
    { n: "1", h: "Manual review is slow", d: "Credit-ops teams read applications and financial statements by hand, application by application." },
    { n: "2", h: "Policy citations must be exact", d: "A fabricated or misquoted policy citation isn't a UX bug here — it's a compliance failure." },
    { n: "3", h: "Bias risk is real and regulated", d: "Similar financials with different applicant demographics must reach the same decision, provably." },
  ];
  const colW = 3.75, gap = 0.35, startX = 0.7, y = 2.2;
  items.forEach((it, i) => {
    const x = startX + i * (colW + gap);
    s.addShape("roundRect", { x, y, w: colW, h: 3.3, rectRadius: 0.12, fill: { color: CARD_BG }, line: { type: "none" } });
    s.addShape("ellipse", { x: x + 0.35, y: y + 0.35, w: 0.6, h: 0.6, fill: { color: NAVY } });
    s.addText(it.n, { x: x + 0.35, y: y + 0.35, w: 0.6, h: 0.6, align: "center", valign: "middle",
      fontFace: "Calibri", fontSize: 20, bold: true, color: WHITE, margin: 0 });
    s.addText(it.h, { x: x + 0.35, y: y + 1.15, w: colW - 0.7, h: 0.7,
      fontFace: "Calibri", fontSize: 16, bold: true, color: NAVY, margin: 0 });
    s.addText(it.d, { x: x + 0.35, y: y + 1.85, w: colW - 0.7, h: 1.3,
      fontFace: "Calibri", fontSize: 12.5, color: MUTED, margin: 0, valign: "top" });
  });

  s.addText("WealthPilot automates the first pass — parse, verify, retrieve policy, decide — while keeping a human in the loop for every real decision.",
    { x: 0.7, y: 5.85, w: 11.9, h: 0.8, fontFace: "Calibri", fontSize: 15, italic: true, color: NAVY, margin: 0 });
}

function architectureSlide(p) {
  const s = p.addSlide();
  s.background = { color: WHITE };
  sectionHeader(s, "System Architecture", "Eight milestones, one pipeline");

  const stages = [
    { m: "M1", t: "Intake", d: "Parse & validate" },
    { m: "M2", t: "Risk Signals", d: "Tools: DSCR, bureau" },
    { m: "M4", t: "Policy RAG", d: "Grounded, cited" },
    { m: "M5", t: "Workflow", d: "Human-gated routing" },
    { m: "M6", t: "Committee", d: "Multi-agent review" },
  ];
  const boxW = 2.05, boxH = 1.5, gapX = 0.28, startX = 0.7, y = 2.3;
  stages.forEach((st, i) => {
    const x = startX + i * (boxW + gapX);
    s.addShape("roundRect", { x, y, w: boxW, h: boxH, rectRadius: 0.1, fill: { color: NAVY }, line: { type: "none" } });
    s.addText(st.m, { x, y: y + 0.15, w: boxW, h: 0.35, align: "center", fontFace: "Calibri", fontSize: 13, bold: true, color: CORAL, margin: 0 });
    s.addText(st.t, { x, y: y + 0.5, w: boxW, h: 0.4, align: "center", fontFace: "Calibri", fontSize: 15, bold: true, color: WHITE, margin: 0 });
    s.addText(st.d, { x: x + 0.1, y: y + 0.95, w: boxW - 0.2, h: 0.5, align: "center", fontFace: "Calibri", fontSize: 10.5, color: ICE, margin: 0 });
    if (i < stages.length - 1) {
      s.addText("→", { x: x + boxW, y: y, w: gapX, h: boxH, align: "center", valign: "middle",
        fontFace: "Calibri", fontSize: 20, bold: true, color: CORAL, margin: 0 });
    }
  });

  // Supporting layers
  const supports = [
    { m: "M3", t: "Persistent Memory", d: "Working / episodic / semantic / procedural, via Supermemory" },
    { m: "M7", t: "Observability + Reliability", d: "Tracing, retries, fallback, circuit breaker" },
    { m: "M8", t: "Eval + Guardrails + API", d: "Golden eval, safety layer, FastAPI deployment" },
  ];
  const sColW = 3.75, sGap = 0.35, sY = 4.6;
  supports.forEach((sp, i) => {
    const x = startX + i * (sColW + sGap);
    s.addShape("roundRect", { x, y: sY, w: sColW, h: 1.7, rectRadius: 0.1, fill: { color: CARD_BG }, line: { color: ICE, width: 1 } });
    s.addText(`${sp.m} — ${sp.t}`, { x: x + 0.25, y: sY + 0.2, w: sColW - 0.5, h: 0.4,
      fontFace: "Calibri", fontSize: 13.5, bold: true, color: NAVY, margin: 0 });
    s.addText(sp.d, { x: x + 0.25, y: sY + 0.65, w: sColW - 0.5, h: 0.9,
      fontFace: "Calibri", fontSize: 11.5, color: MUTED, margin: 0 });
  });

  s.addText("Runs underneath and alongside every stage above", { x: 0.7, y: 4.32, w: 11.9, h: 0.3,
    fontFace: "Calibri", fontSize: 10.5, italic: true, color: MUTED, margin: 0 });
}

function twoColMilestoneSlide(p, kicker, title, left, right) {
  const s = p.addSlide();
  s.background = { color: WHITE };
  sectionHeader(s, kicker, title);

  [left, right].forEach((col, i) => {
    const x = 0.7 + i * 6.1;
    const w = 5.75;
    s.addShape("roundRect", { x, y: 2.15, w, h: 4.5, rectRadius: 0.12, fill: { color: CARD_BG }, line: { type: "none" } });
    s.addShape("roundRect", { x: x + 0.3, y: 2.45, w: 1.0, h: 0.42, rectRadius: 0.08, fill: { color: NAVY }, line: { type: "none" } });
    s.addText(col.tag, { x: x + 0.3, y: 2.45, w: 1.0, h: 0.42, align: "center", valign: "middle",
      fontFace: "Calibri", fontSize: 14, bold: true, color: WHITE, margin: 0 });
    s.addText(col.h, { x: x + 0.3, y: 3.0, w: w - 0.6, h: 0.6,
      fontFace: "Calibri", fontSize: 18, bold: true, color: NAVY, margin: 0 });
    s.addText(col.bullets.map((b, idx) => ({
      text: b, options: { bullet: { code: "2022", indent: 14 }, color: INK, fontSize: 13, breakLine: idx < col.bullets.length - 1, paraSpaceAfter: 8 }
    })), { x: x + 0.3, y: 3.65, w: w - 0.6, h: 2.1, fontFace: "Calibri", margin: 0, valign: "top" });
    s.addShape("roundRect", { x: x + 0.3, y: 5.85, w: w - 0.6, h: 0.6, rectRadius: 0.08, fill: { color: WHITE }, line: { color: CORAL, width: 1.25 } });
    s.addText(col.stat, { x: x + 0.45, y: 5.85, w: w - 0.9, h: 0.6, valign: "middle",
      fontFace: "Calibri", fontSize: 12.5, bold: true, color: CORAL, margin: 0 });
  });
}

function orchestrationSlide(p) {
  const s = p.addSlide();
  s.background = { color: WHITE };
  sectionHeader(s, "M5 — Orchestration", "A real human-in-the-loop gate, not a simulated one");

  const nodes = ["intake", "verify", "risk\nscoring", "human\napproval"];
  const boxW = 1.9, boxH = 1.1, gapX = 0.55, startX = 0.9, y = 2.15;
  nodes.forEach((n, i) => {
    const x = startX + i * (boxW + gapX);
    const isGate = n.includes("approval");
    s.addShape("roundRect", { x, y, w: boxW, h: boxH, rectRadius: 0.1,
      fill: { color: isGate ? CORAL : NAVY }, line: { type: "none" } });
    s.addText(n, { x, y, w: boxW, h: boxH, align: "center", valign: "middle",
      fontFace: "Calibri", fontSize: 13, bold: true, color: WHITE, margin: 0 });
    if (i < nodes.length - 1) {
      s.addText("→", { x: x + boxW, y, w: gapX, h: boxH, align: "center", valign: "middle",
        fontFace: "Calibri", fontSize: 18, bold: true, color: MUTED, margin: 0 });
    }
  });
  s.addText("interrupt() pauses here — real pause, real resume", { x: startX + 3 * (boxW + gapX) - 1.1, y: y + boxH + 0.1, w: 3.4, h: 0.4,
    fontFace: "Calibri", fontSize: 11, italic: true, color: CORAL, margin: 0 });

  const claims = [
    "Typed state with a control/audit field split — not a single overloaded dict",
    "A real interrupt() call pauses the graph and waits — not a node that just labels a decision",
    "A durable SqliteSaver checkpoint — verified surviving an actual process restart, not just an in-memory demo",
    "Bounded revision loop + escalation path — no unguarded loops",
  ];
  s.addText(claims.map((c, idx) => ({
    text: c, options: { bullet: { code: "2022", indent: 16 }, color: INK, fontSize: 14.5, breakLine: idx < claims.length - 1, paraSpaceAfter: 10 }
  })), { x: 0.9, y: 4.0, w: 11.3, h: 2.6, fontFace: "Calibri", margin: 0, valign: "top" });
}

function committeeSlide(p) {
  const s = p.addSlide();
  s.background = { color: WHITE };
  sectionHeader(s, "M6 — Multi-Agent Committee", "Analyst produces, two independent critics decide if it ships");

  // Center supervisor + 3 agents (star)
  const cx = 3.4, cy = 3.9;
  s.addShape("ellipse", { x: cx - 0.65, y: cy - 0.55, w: 1.3, h: 1.1, fill: { color: NAVY }, line: { type: "none" } });
  s.addText("Supervisor", { x: cx - 0.65, y: cy - 0.55, w: 1.3, h: 1.1, align: "center", valign: "middle",
    fontFace: "Calibri", fontSize: 12, bold: true, color: WHITE, margin: 0 });

  const agents = [
    { t: "Analyst", d: "Writes the memo", ax: cx, ay: 1.9 },
    { t: "Risk\nReviewer", d: "Checks the DSCR figure", ax: cx - 1.9, ay: 5.2 },
    { t: "Compliance\nReviewer", d: "Scans for bias terms", ax: cx + 1.9, ay: 5.2 },
  ];
  agents.forEach(a => {
    s.addShape("line", { x: cx, y: cy, w: a.ax - cx, h: a.ay - cy, line: { color: ICE, width: 2 } });
    s.addShape("ellipse", { x: a.ax - 0.55, y: a.ay - 0.5, w: 1.1, h: 1.0, fill: { color: CORAL }, line: { type: "none" } });
    s.addText(a.t, { x: a.ax - 0.55, y: a.ay - 0.5, w: 1.1, h: 1.0, align: "center", valign: "middle",
      fontFace: "Calibri", fontSize: 11, bold: true, color: WHITE, margin: 0 });
  });

  const right = [
    "Enforced per-agent write scopes — a critic can't edit the memo it's checking",
    "Supervisor routes as a pure function, constrained to a legal_routes() guard",
    "Bounded revision loop + escalation — same discipline as M5",
    "Bureau/bank data served over a real MCP server, not a bare function call",
  ];
  s.addText(right.map((c, idx) => ({
    text: c, options: { bullet: { code: "2022", indent: 16 }, color: INK, fontSize: 14, breakLine: idx < right.length - 1, paraSpaceAfter: 12 }
  })), { x: 6.9, y: 2.3, w: 5.7, h: 3.8, fontFace: "Calibri", margin: 0, valign: "top" });
}

function statsSlide(p) {
  const s = p.addSlide();
  s.background = { color: WHITE };
  sectionHeader(s, "M7 + M8 — Reliability, Eval & Guardrails", "Hardening the pipeline, then proving it");

  // Big stat callout
  s.addShape("roundRect", { x: 0.7, y: 2.15, w: 3.9, h: 2.15, rectRadius: 0.12, fill: { color: NAVY }, line: { type: "none" } });
  s.addText("45%  →  100%", { x: 0.9, y: 2.4, w: 3.5, h: 0.9, fontFace: "Calibri", fontSize: 32, bold: true, color: CORAL, margin: 0 });
  s.addText("Bureau-lookup success rate, before vs. after retry + fallback hardening (seeded fault injection)",
    { x: 0.9, y: 3.35, w: 3.5, h: 0.85, fontFace: "Calibri", fontSize: 12, color: ICE, margin: 0 });

  s.addShape("roundRect", { x: 4.85, y: 2.15, w: 3.9, h: 2.15, rectRadius: 0.12, fill: { color: CARD_BG }, line: { type: "none" } });
  s.addText("3", { x: 5.05, y: 2.35, w: 3.5, h: 0.9, fontFace: "Calibri", fontSize: 44, bold: true, color: NAVY, margin: 0 });
  s.addText("Real failures the circuit breaker allowed before short-circuiting the rest — never hammered a dead dependency",
    { x: 5.05, y: 3.35, w: 3.5, h: 0.85, fontFace: "Calibri", fontSize: 12, color: MUTED, margin: 0 });

  s.addShape("roundRect", { x: 9.0, y: 2.15, w: 3.6, h: 2.15, rectRadius: 0.12, fill: { color: CARD_BG }, line: { type: "none" } });
  s.addText("20", { x: 9.2, y: 2.35, w: 3.2, h: 0.9, fontFace: "Calibri", fontSize: 44, bold: true, color: NAVY, margin: 0 });
  s.addText("golden eval items — deterministic checks + an LLM judge, scored honestly",
    { x: 9.2, y: 3.35, w: 3.2, h: 0.85, fontFace: "Calibri", fontSize: 12, color: MUTED, margin: 0 });

  const bullets = [
    "Observability: every node traced, tagged by run, with a real per-agent cost/latency dashboard",
    "Guardrails run independent of answer quality — schema validation, prompt-injection detection, protected-attribute scan",
    "18/20 golden-set items passed guardrails clean; an injection attempt against the live API correctly returns HTTP 422",
    "FastAPI /health + /decision deployment package, verified with a real HTTP round-trip",
  ];
  s.addText(bullets.map((c, idx) => ({
    text: c, options: { bullet: { code: "2022", indent: 16 }, color: INK, fontSize: 13.5, breakLine: idx < bullets.length - 1, paraSpaceAfter: 9 }
  })), { x: 0.7, y: 4.75, w: 11.9, h: 2.2, fontFace: "Calibri", margin: 0, valign: "top" });
}

function honestResultsSlide(p) {
  const s = p.addSlide();
  s.background = { color: WHITE };
  sectionHeader(s, "Results", "What actually worked — and what didn't");

  const rows = [
    ["M1 Intake", "10/10 records parsed; missing-field accuracy checked against ground truth"],
    ["M4 RAG", "Grounded, cited answers; hybrid search + reranking measurably beats naive retrieval"],
    ["M5 Workflow", "Real interrupt → process restart → resume, verified, not simulated"],
    ["M7 Reliability", "Bureau lookup: 45% → 100% success under seeded fault injection"],
    ["M8 Eval", "7/20 golden-set items passed end-to-end; 18/20 guardrails clean"],
  ];
  let y = 2.2;
  rows.forEach(([label, val]) => {
    s.addShape("roundRect", { x: 0.7, y, w: 2.6, h: 0.62, rectRadius: 0.08, fill: { color: NAVY }, line: { type: "none" } });
    s.addText(label, { x: 0.7, y, w: 2.6, h: 0.62, align: "center", valign: "middle",
      fontFace: "Calibri", fontSize: 12.5, bold: true, color: WHITE, margin: 0 });
    s.addText(val, { x: 3.5, y, w: 9.1, h: 0.62, valign: "middle",
      fontFace: "Calibri", fontSize: 13, color: INK, margin: 0 });
    y += 0.74;
  });

  s.addShape("roundRect", { x: 0.7, y: y + 0.15, w: 11.9, h: 1.35, rectRadius: 0.1, fill: { color: CARD_BG }, line: { color: CORAL, width: 1 } });
  s.addText("Honest limitation:", { x: 1.0, y: y + 0.3, w: 3, h: 0.4, fontFace: "Calibri", fontSize: 13, bold: true, color: CORAL, margin: 0 });
  s.addText("The 7/20 eval score is reported as-is, not tuned to look better. Failures cluster in auto-generated questions with retrieval gaps — the hand-written adversarial cases (bias probes, injection attempts) mostly passed. A low score with clean guardrails is a more useful signal than a suspiciously perfect one.",
    { x: 1.0, y: y + 0.62, w: 11.3, h: 0.85, fontFace: "Calibri", fontSize: 12, italic: true, color: INK, margin: 0 });
}

function closingSlide(p) {
  const s = p.addSlide();
  s.background = { color: NAVY_DARK };
  s.addShape("ellipse", { x: 0.9, y: 0.8, w: 0.6, h: 0.6, fill: { color: CORAL } });
  s.addText("W", { x: 0.9, y: 0.8, w: 0.6, h: 0.6, align: "center", valign: "middle",
    fontFace: "Calibri", fontSize: 22, bold: true, color: WHITE, margin: 0 });

  s.addText("Thank you", { x: 0.9, y: 2.6, w: 11, h: 1.0,
    fontFace: "Cambria", fontSize: 44, bold: true, color: WHITE, margin: 0 });
  s.addText("WealthPilot — all 8 milestones, built and independently verified.",
    { x: 0.9, y: 3.6, w: 11, h: 0.6, fontFace: "Calibri", fontSize: 17, color: ICE, margin: 0 });

  s.addShape("roundRect", { x: 0.9, y: 4.6, w: 6.2, h: 0.7, rectRadius: 0.1, fill: { color: "263073" }, line: { type: "none" } });
  s.addText("github.com/abhi-1006/wealthpilot", { x: 1.1, y: 4.6, w: 5.8, h: 0.7, valign: "middle",
    fontFace: "Calibri", fontSize: 15, bold: true, color: CORAL, margin: 0 });

  s.addText("Questions?", { x: 0.9, y: 6.4, w: 6, h: 0.5, fontFace: "Calibri", fontSize: 14, color: "9AA4C4", margin: 0 });
}

// ---- Build ----
const p = newDeck();
titleSlide(p);
problemSlide(p);
architectureSlide(p);
twoColMilestoneSlide(p, "M1 + M2 — Intake & Risk Agent", "From a raw application to real risk signals",
  { tag: "M1", h: "Structured Intake", stat: "10/10 records parsed successfully", bullets: [
    "Provider-agnostic LLM client (LiteLLM) — swap providers in one line",
    "Raw record → validated Pydantic LoanApplication / FinancialSnapshot",
    "Repair loop retries on malformed output instead of failing outright",
    "Scored against each record's ground_truth for real accuracy, not just “did it validate”",
  ]},
  { tag: "M2", h: "Tool-Enabled Agent", stat: "DSCR, bureau & bank data — all real tool calls", bullets: [
    "Bounded tool-calling loop: DSCR calculator, interest calculator",
    "Credit-bureau and bank-statement lookups as real callable tools",
    "The model chooses which tool to call and when — not a hardcoded script",
    "Verified against a known eval case: computed DSCR matched by hand",
  ]}
);
twoColMilestoneSlide(p, "M3 + M4 — Memory & Grounded Retrieval", "Remembering context, answering only from real policy",
  { tag: "M3", h: "Persistent Memory", stat: "Real Supermemory recall, semantic search verified", bullets: [
    "Four memory kinds: working, episodic, semantic, procedural",
    "Backed by Supermemory — a real hosted memory store, not a mock",
    "Long conversations get compressed — lossy, so it's tested, not assumed",
    "Confirmed: a planted fact survives summarization, every run",
  ]},
  { tag: "M4", h: "Production RAG", stat: "Hybrid search + reranking, cited answers", bullets: [
    "Local embeddings + BM25, fused with Reciprocal Rank Fusion",
    "Cross-encoder reranking before the model ever sees a chunk",
    "Every answer cites its source — fabricated policy = compliance failure here",
    "Prompt-injection defense: retrieved content is treated as untrusted data",
  ]}
);
orchestrationSlide(p);
committeeSlide(p);
statsSlide(p);
honestResultsSlide(p);
closingSlide(p);

p.writeFile({ fileName: "WealthPilot_Presentation.pptx" }).then(() => {
  console.log("Deck written.");
});
