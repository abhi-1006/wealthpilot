"""WealthPilot serving layer: the assistant, its guardrails and the FastAPI app.

Kept separate from m8.py so the deployed server only loads what it serves.
m8.py imports everything here and adds the evaluation layers (golden set,
Ragas / DeepEval / TruLens judges, Langfuse experiment) on top.

Routes: /health, / (frontend.html), /chat, /underwrite/{index}, /decision.
Public-demo protections: per-visitor and global rate limits, and cached
underwriting results, so free-tier API quotas can't be drained.

Run locally:  uvicorn api:api_app --port 8000
"""

import inspect
import json
import os
import random
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types as genai_types
from langfuse import Langfuse
from pydantic import BaseModel, Field, field_validator

from m2 import dscr_calculator, interest_calculator
from m4 import CHUNKS, answer, retrieve

#
# Guardrails, independent of answer quality
#

class DecisionResponse(BaseModel):
    """Schema every answer must satisfy before it's allowed out of the pipeline."""
    question: str
    answer: str = Field(min_length=1, max_length=3000)

    @field_validator("answer")
    @classmethod
    def not_placeholder(cls, v):
        if v.strip().lower() in {"", "n/a", "todo", "..."}:
            raise ValueError("answer looks like a placeholder, not a real answer")
        return v


INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) instructions",
    r"disregard (the |your )?(system|previous) (prompt|instructions)",
    r"reveal (your |the )?system prompt",
    r"you are now in (developer|debug|dan) mode",
    r"override (your |the )?(safety|guardrail) (settings|rules)",
]
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

PROTECTED_ATTRIBUTE_TERMS = [
    "religion", "caste", "gender", "marital status", "disability",
    "single mother", "community", "region of origin",
]


def detect_prompt_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(text or ""))


def guardrail_check(question: str, answer_text: str) -> dict:
    flags = []
    try:
        DecisionResponse(question=question, answer=answer_text)
    except Exception as e:
        flags.append(f"schema_violation: {e}")

    if detect_prompt_injection(question):
        flags.append("prompt_injection_in_question")

    lower = answer_text.lower()
    hit_terms = [t for t in PROTECTED_ATTRIBUTE_TERMS if t in lower]
    if hit_terms:
        flags.append(f"protected_attribute_referenced: {hit_terms}")

    return {"passed": len(flags) == 0, "flags": flags}


class AgentResponse(BaseModel):
    query: str
    route: Literal["tool", "retrieval", "direct"]
    final_answer: str = Field(min_length=1, max_length=2000)
    retrieved_doc_ids: List[str] = Field(default_factory=list)

    @field_validator("final_answer")
    @classmethod
    def answer_not_placeholder(cls, v):
        if v.strip().lower() in {"", "n/a", "todo", "..."}:
            raise ValueError("final_answer looks like a placeholder, not a real answer")
        return v


def assistant_guardrail_check(trace: dict) -> dict:
    flags = []
    try:
        AgentResponse(query=trace["query"], route=trace["route"],
                      final_answer=trace["final_answer"],
                      retrieved_doc_ids=trace.get("retrieved_doc_ids", []))
    except Exception as e:
        flags.append(f"schema_violation: {e}")

    if detect_prompt_injection(trace["query"]):
        flags.append("prompt_injection_in_user_input")
    for ctx in trace.get("retrieved_contexts", []):
        if detect_prompt_injection(ctx):
            flags.append("prompt_injection_in_retrieved_content")

    return {"passed": len(flags) == 0, "flags": flags}


#
# Gemini client with shared rate-limit pacing
#

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", GEMINI_API_KEY)
os.environ.setdefault("GOOGLE_API_KEY", GEMINI_API_KEY)

MODEL_NAME = "gemini-flash-lite-latest"
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

genai_client = genai.Client(api_key=GEMINI_API_KEY)

# Every Gemini call (the assistant here, plus m8's judges) shares one 15 RPM
# free-tier quota, so they share one pacing gate. The lock keeps concurrent
# requests from slipping through the gap together.
_MIN_GAP_SECONDS = 4.5
_last_call_ts = [0.0]
_pace_lock = threading.Lock()
_RETRY_DELAY_RE = re.compile(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'")


def _pace():
    with _pace_lock:
        elapsed = time.time() - _last_call_ts[0]
        if elapsed < _MIN_GAP_SECONDS:
            time.sleep(_MIN_GAP_SECONDS - elapsed)
        _last_call_ts[0] = time.time()


def _suggested_wait(exc: Exception, fallback: float) -> float:
    m = _RETRY_DELAY_RE.search(str(exc))
    return float(m.group(1)) + 1.0 if m else fallback


#
# Tracing: tokens, latency and estimated cost for every model call
#

# Turned on only when Langfuse keys are set; without them every call below is a no-op.
langfuse = Langfuse(
    base_url=os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST"),
    tracing_enabled=bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")),
)

# USD per 1M tokens, Gemini API Standard paid tier (ai.google.dev pricing, checked
# 2026-11-06). The demo runs on the free tier, so these are what each call *would*
# cost on a paid plan, not actual spend. MODEL_NAME is a moving "latest" alias, so
# the price is looked up from the model version each response reports.
PRICES_PER_1M = {
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
}


def estimate_cost(model_version: str, input_tokens: int, output_tokens: int) -> Optional[dict]:
    """Estimated USD cost of one call, or None for a model missing from PRICES_PER_1M."""
    for prefix in sorted(PRICES_PER_1M, key=len, reverse=True):
        if (model_version or "").startswith(prefix):
            in_price, out_price = PRICES_PER_1M[prefix]
            cost_in = input_tokens * in_price / 1_000_000
            cost_out = output_tokens * out_price / 1_000_000
            return {"input": cost_in, "output": cost_out, "total": cost_in + cost_out}
    return None


def call_llm(prompt: str, system: Optional[str] = None,
             json_schema: Optional[dict] = None, max_retries: int = 4,
             name: str = "gemini-call") -> str:
    config_kwargs = {"temperature": 0.2}
    if system:
        config_kwargs["system_instruction"] = system
    if json_schema:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = json_schema

    for attempt in range(max_retries):
        _pace()  # waiting for the rate-limit gate happens outside the traced span
        error = None
        with langfuse.start_as_current_observation(
            name=name, as_type="generation", model=MODEL_NAME, input=prompt,
            model_parameters={"temperature": 0.2}, metadata={"attempt": attempt + 1},
        ) as gen:
            try:
                started = time.perf_counter()
                resp = genai_client.models.generate_content(
                    model=MODEL_NAME, contents=prompt,
                    config=genai_types.GenerateContentConfig(**config_kwargs),
                )
                seconds = time.perf_counter() - started
            except Exception as e:
                gen.update(level="ERROR", status_message=str(e)[:500])
                error = e
            else:
                usage = resp.usage_metadata
                input_tokens = (usage.prompt_token_count or 0) if usage else 0
                # Gemini bills "thinking" tokens as output tokens.
                output_tokens = ((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else 0
                gen.update(
                    model=resp.model_version or MODEL_NAME,
                    output=resp.text,
                    usage_details={"input": input_tokens, "output": output_tokens},
                    cost_details=estimate_cost(resp.model_version, input_tokens, output_tokens),
                    metadata={
                        "tokens_per_second": round(output_tokens / seconds, 1) if seconds > 0 else None,
                        "cost_basis": "estimated at paid-tier list price; the demo runs on the free tier",
                    },
                )
                return resp.text

        if attempt == max_retries - 1:
            raise error
        wait = _suggested_wait(error, fallback=(2 ** attempt) + random.random())
        print(f"  [retry {attempt+1}/{max_retries}] waiting {wait:.1f}s: {error}")
        time.sleep(wait)


#
# The assistant: planner -> tool / retrieval / direct -> final answer
#

WP_TOOLS = {"dscr_calculator": dscr_calculator, "interest_calculator": interest_calculator}

PLANNER_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": ["tool", "retrieval", "direct"]},
        "tool_name": {"type": "string", "enum": ["dscr_calculator", "interest_calculator", "none"]},
        "tool_args": {
            "type": "object",
            "properties": {
                "ebitda_inr": {"type": "number"},
                "existing_debt_inr": {"type": "number"},
                "principal_inr": {"type": "number"},
                "annual_rate_percent": {"type": "number"},
                "tenor_months": {"type": "number"},
            },
        },
        "reasoning": {"type": "string"},
    },
    "required": ["route", "tool_name", "tool_args", "reasoning"],
}

PLANNER_SYSTEM = """You are the planning module of the WealthPilot underwriting assistant.
Given a user query, decide ONE route:
- "tool": the user wants a DSCR or interest computed from stated numbers. Set tool_name to
  "dscr_calculator" (needs ebitda_inr, existing_debt_inr) or "interest_calculator" (needs
  principal_inr, annual_rate_percent, tenor_months), filling only the relevant args.
- "retrieval": the user is asking about a written credit policy (DSCR floors, leverage
  limits, risk grading, fair lending, reason codes, sector notes). Set tool_name to "none".
- "direct": small talk, thanks, or anything needing neither a tool nor a policy lookup.
  Set tool_name to "none" and tool_args to {}."""


def plan(query: str) -> dict:
    raw = call_llm(query, system=PLANNER_SYSTEM, json_schema=PLANNER_SCHEMA, name="planner")
    return json.loads(raw)


ANSWER_SYSTEM = """You are the WealthPilot underwriting assistant. Answer the user's question
using ONLY the tool result or retrieved policy text given to you as context. Be concise
(2-4 sentences). If the context doesn't contain the answer, say so plainly instead of guessing."""


def synthesize_answer(query: str, context: str) -> str:
    prompt = f"User question: {query}\n\nContext:\n{context}\n\nAnswer:"
    return call_llm(prompt, system=ANSWER_SYSTEM, name="final-answer")


def run_agent(query: str) -> dict:
    plan_result = plan(query)
    route = plan_result["route"]

    tool_call, tool_result, retrieved_ids = None, None, []

    if route == "tool":
        tool_name = plan_result["tool_name"]
        tool_args = plan_result.get("tool_args") or {}
        tool_call = {"name": tool_name, "args": tool_args}
        if tool_name in WP_TOOLS:
            accepted = set(inspect.signature(WP_TOOLS[tool_name]).parameters)
            call_args = {k: v for k, v in tool_args.items() if k in accepted and v not in (None, "")}
            with langfuse.start_as_current_observation(name=tool_name, as_type="tool", input=call_args) as tool_span:
                try:
                    tool_result = WP_TOOLS[tool_name](**call_args)
                except TypeError as e:
                    tool_result = {"error": f"bad arguments from planner: {e}"}
                tool_span.update(output=tool_result)
        context = json.dumps(tool_result, indent=2) if not isinstance(tool_result, (int, float)) \
            else json.dumps({"result": tool_result})

    elif route == "retrieval":
        with langfuse.start_as_current_observation(name="policy-retrieval", as_type="retriever", input=query) as retriever:
            retrieved_ids = retrieve(query, k=3)
            retriever.update(output=[CHUNKS[cid]["source"] for cid in retrieved_ids])
        context = "\n\n".join(CHUNKS[cid]["text"] for cid in retrieved_ids)

    else:
        context = "(no tool or retrieval needed)"

    final_answer = synthesize_answer(query, context)

    return {
        "query": query,
        "route": route,
        "plan_reasoning": plan_result.get("reasoning", ""),
        "tool_call": tool_call,
        "tool_result": tool_result,
        "retrieved_contexts": [CHUNKS[cid]["text"] for cid in retrieved_ids],
        "retrieved_doc_ids": [CHUNKS[cid]["source"] for cid in retrieved_ids],
        "final_answer": final_answer,
    }


#
# Rate limiting for the public demo
#

class RateLimiter:
    """Sliding-window limits, kept in memory.

    This works because the service runs as a single instance (Cloud Run
    --max-instances 1). With more instances, the counts would need a shared
    store such as Redis.
    """

    def __init__(self, per_visitor_per_min: int, per_visitor_per_day: int, global_per_day: int):
        self.limits = [("minute", 60, per_visitor_per_min), ("day", 86400, per_visitor_per_day)]
        self.global_per_day = global_per_day
        self.visitor_hits: dict[str, deque] = {}
        self.global_hits: deque = deque()
        self.lock = threading.Lock()

    def check(self, visitor: str) -> None:
        now = time.time()
        with self.lock:
            while self.global_hits and now - self.global_hits[0] > 86400:
                self.global_hits.popleft()
            if len(self.global_hits) >= self.global_per_day:
                raise HTTPException(429, "The demo has reached its daily limit. Please try again tomorrow.")

            hits = self.visitor_hits.setdefault(visitor, deque())
            while hits and now - hits[0] > 86400:
                hits.popleft()
            for label, window, limit in self.limits:
                if sum(1 for t in hits if now - t <= window) >= limit:
                    raise HTTPException(429, f"Too many requests this {label}. Please wait and try again.")

            hits.append(now)
            self.global_hits.append(now)


limiter = RateLimiter(
    per_visitor_per_min=int(os.getenv("RATE_PER_MIN", "5")),
    per_visitor_per_day=int(os.getenv("RATE_PER_DAY", "40")),
    global_per_day=int(os.getenv("RATE_GLOBAL_PER_DAY", "300")),
)


def visitor_id(request: Request) -> str:
    # Behind Cloud Run the client address arrives in X-Forwarded-For. A visitor
    # can fake that header, which is why the global daily cap exists too.
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


#
# FastAPI app
#

class DecisionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class DecisionAPIResponse(BaseModel):
    question: str
    answer: str


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)


class ChatResponse(BaseModel):
    route: str
    final_answer: str
    retrieved_doc_ids: List[str]


class UnderwriteResponse(BaseModel):
    application_id: str
    business_name: str
    sector: str
    requested_amount_inr: int
    dscr: float | None
    bureau_score: str | None
    decision: str
    reasons: list[str]


api_app = FastAPI(title="WealthPilot Underwriting Policy Assistant", version="1.0")

RECORDS_PATH = "capstone-data-toolkit/data/wealthpilot/intake/records.jsonl"


@api_app.get("/health")
def health():
    return {"status": "ok"}


@api_app.get("/", response_class=HTMLResponse)
def frontend():
    return Path("frontend.html").read_text(encoding="utf-8")


# The sample applications never change, so each one is parsed by the LLM once
# per server start and the result is reused for later visitors.
_underwrite_cache: dict[int, UnderwriteResponse] = {}


def _underwrite_record(index: int) -> UnderwriteResponse:
    from m1 import parse_invoice
    from m2 import credit_bureau_lookup

    with open(RECORDS_PATH) as f:
        record = [json.loads(line) for line in f][index]

    app, _ = parse_invoice(record)
    if app is None:
        raise HTTPException(status_code=500, detail="parsing failed for this record")

    fin = app.declared_financials
    dscr = dscr_calculator(fin.ebitda_inr, fin.existing_debt_inr) if fin.ebitda_inr and fin.existing_debt_inr else None
    bureau = credit_bureau_lookup("ASH-A-00000")

    if dscr is None:
        decision, reasons = "human_review", ["Cannot compute DSCR -- financials incomplete."]
    elif dscr < 1.0:
        decision, reasons = "declined", [f"DSCR {dscr} is well below the 1.25x policy floor."]
    elif dscr < 1.25:
        decision, reasons = "borderline", [f"DSCR {dscr} is close to the 1.25x floor."]
    else:
        decision, reasons = "pending_human_signoff", [f"DSCR {dscr} meets the 1.25x floor; requires human sign-off per policy."]

    return UnderwriteResponse(
        application_id=app.application_id, business_name=app.business_name, sector=app.sector,
        requested_amount_inr=app.requested_amount_inr, dscr=dscr,
        bureau_score=bureau.get("bureau_score") if bureau else None,
        decision=decision, reasons=reasons,
    )


@api_app.get("/underwrite/{index}", response_model=UnderwriteResponse)
def underwrite(index: int, request: Request):
    with open(RECORDS_PATH) as f:
        count = sum(1 for _ in f)
    if index < 0 or index >= count:
        raise HTTPException(status_code=404, detail=f"index must be 0-{count - 1}")
    if index not in _underwrite_cache:  # only uncached results cost model calls
        limiter.check(visitor_id(request))
        _underwrite_cache[index] = _underwrite_record(index)
    return _underwrite_cache[index]


@api_app.post("/decision", response_model=DecisionAPIResponse)
def decision(req: DecisionRequest, request: Request):
    limiter.check(visitor_id(request))
    ans = answer(req.question)
    guard = guardrail_check(req.question, ans)
    if not guard["passed"]:
        raise HTTPException(status_code=422,
                             detail={"message": "guardrail check failed", "flags": guard["flags"]})
    return DecisionAPIResponse(question=req.question, answer=ans)


@api_app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    # Reject injection attempts before spending any model calls on them.
    if detect_prompt_injection(req.query):
        raise HTTPException(status_code=422, detail={"message": "guardrail check failed",
                                                     "flags": ["prompt_injection_in_user_input"]})
    limiter.check(visitor_id(request))
    try:
        # One Langfuse trace per request: planner, tool or retrieval step, and final
        # answer appear as child observations with their own tokens, latency and cost.
        with langfuse.start_as_current_observation(name="chat-request", as_type="agent", input={"query": req.query}) as span:
            trace = run_agent(req.query)
            guard = assistant_guardrail_check(trace)
            span.update(output={"route": trace["route"], "final_answer": trace["final_answer"],
                                "sources": trace["retrieved_doc_ids"]},
                        metadata={"guardrail_passed": guard["passed"], "guardrail_flags": guard["flags"]})
    finally:
        # Cloud Run only gives the container CPU while a request is open, so send the
        # trace now instead of leaving it to a background thread.
        langfuse.flush()
    if not guard["passed"]:
        raise HTTPException(status_code=422,
                             detail={"message": "guardrail check failed", "flags": guard["flags"]})
    return ChatResponse(route=trace["route"], final_answer=trace["final_answer"],
                         retrieved_doc_ids=trace["retrieved_doc_ids"])
