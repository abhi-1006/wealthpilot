"""M3 -- Persistent memory, backed by Supermemory.

Four kinds of memory:
  working    -- the live conversation buffer, transient, in-process only
  episodic   -- "what happened" -- specific timestamped events
  semantic   -- "what is stably true" -- durable facts about an applicant
  procedural -- "how to do this task" -- reusable steps
"""

import os
import time

import litellm
import supermemory
from dotenv import load_dotenv

load_dotenv()

mem = supermemory.Supermemory(api_key=os.getenv("SUPERMEMORY_API_KEY"))
CONTAINER_TAG = "wealthpilot-m3-demo"


def remember(text: str, kind: str, **extra) -> None:
    """Write one memory of a given kind (episodic / semantic / procedural)."""
    mem.add(content=text, container_tag=CONTAINER_TAG, metadata={"type": kind, **extra})


def recall(query: str, k: int = 3) -> list[tuple[str, str, float]]:
    """Recall memories by meaning. Tries the extracted-memories layer first
    (readable text via .memory), falls back to raw documents if empty."""
    out = []
    for r in mem.search.memories(q=query, container_tag=CONTAINER_TAG, limit=k).results:
        if getattr(r, "memory", None):
            out.append(((r.metadata or {}).get("type", "?"), r.memory, float(r.similarity or 0.0)))
    if not out:
        for r in mem.search.documents(q=query, container_tags=[CONTAINER_TAG], limit=k).results:
            text = " ".join(c.content for c in (r.chunks or []) if c and c.content).strip()
            if text:
                out.append(((r.metadata or {}).get("type", "?"), text, float(r.score or 0.0)))
    return out


def token_estimate(text: str) -> int:
    """Cheap stand-in for a real tokenizer -- ~4 chars/token."""
    return max(1, len(text) // 4)



# Summarization -- lossy, so we TEST that a planted fact survives compression
# rather than assuming it does.


RECENT_KEEP = 4
TOKEN_BUDGET = 220


def summarize_turns(turns: list[dict], prev_summary: str = "") -> str:
    convo = "\n".join(f'{t["role"]}: {t["content"]}' for t in turns)
    system = (
        "You maintain a running summary of a conversation. Update the "
        "existing summary with the new turns, preserving concrete facts "
        "(names, numbers, application ids). Return only the updated "
        "summary, under 100 words."
    )
    user = f"Existing summary:\n{prev_summary or '(none)'}\n\nNew turns:\n{convo}\n\nUpdated summary:"
    resp = litellm.completion(
        model="groq/openai/gpt-oss-20b",
        temperature=0,
        num_retries=5,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content.strip()


if __name__ == "__main__":
    # --- Working memory: just the live buffer, nothing persisted ---
    working_memory = [
        {"role": "user", "content": "I'm reviewing loan application ASH-L-4324 for Ravi Foodwala."},
        {"role": "assistant", "content": "Sure -- want me to check bureau history and DSCR first?"},
    ]
    used = sum(token_estimate(m["content"]) for m in working_memory)
    print(f"Working memory: {len(working_memory)} turns, ~{used} tokens in the buffer.")

    # --- Episodic / semantic / procedural: written to Supermemory ---
    remember(
        "On 2026-08-19, application ASH-L-4324 (Ravi Foodwala, Retail Trade) "
        "was submitted requesting 1,000,000 INR for a food cart business.",
        "episodic", application_id="ASH-L-4324",
    )
    remember(
        "Ravi Foodwala is a Retail Trade business. Sector DSCR floor is the "
        "standard 1.25x. Bureau score at intake was 500.",
        "semantic", application_id="ASH-L-4324",
    )
    remember(
        "Underwriting procedure: (1) parse application via M1, (2) compute "
        "DSCR and pull bureau/bank data via M2 tools, (3) check policy via "
        "M4 RAG, (4) route the decision via M5 workflow.",
        "procedural",
    )

    print("Waiting for Supermemory to extract the memories (usually a few seconds)...")
    for _ in range(20):
        if mem.search.memories(q="Ravi Foodwala", container_tag=CONTAINER_TAG, limit=1).results:
            break
        time.sleep(3)

    print("\nEpisodic recall  :", recall("what application did we receive for Ravi Foodwala?", k=2))
    print("Semantic recall  :", recall("what is Ravi Foodwala's sector and bureau score?", k=2))
    print("Procedural recall:", recall("what is the underwriting procedure?", k=2))

    # --- Force a summary, then TEST an early fact survived the compression ---
    history = [
        {"role": "user", "content": "Application ASH-L-4324 requests 1,000,000 INR, business name Ravi Foodwala."},
        {"role": "assistant", "content": "Noted -- Ravi Foodwala, 1,000,000 INR requested."},
    ]
    for i in range(8):
        history.append({"role": "user", "content": f"Explain underwriting concept {i} in detail."})
        history.append({"role": "assistant", "content": "Detailed explanation. " * 12})

    total_tokens = sum(token_estimate(m["content"]) for m in history)
    assert total_tokens > TOKEN_BUDGET, "budget was not exceeded -- test is meaningless if this fails"
    old_turns = history[:-RECENT_KEEP]
    summary = summarize_turns(old_turns)

    assert "4324" in summary or "ravi foodwala" in summary.lower(), (
        "Planted fact (application ASH-L-4324 / Ravi Foodwala) was LOST during "
        "summarization -- this is exactly the failure mode the test exists to catch."
    )
    print("\nPASS -- summarization fired and the planted fact survived compression.")
    print("Rolling summary:", summary)

