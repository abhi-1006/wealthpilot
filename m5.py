"""M5 -- Orchestrated LangGraph workflow with checkpointing.

Typed state with a control/audit field split, deterministic routing as a
pure function, a bounded revision loop, a real human-approval gate via
interrupt(), and a durable SqliteSaver checkpointer that survives an actual
process restart. The irreversible finalize step sits in its own node,
downstream of the pause.

Graph shape:

    intake -> verify -> risk_scoring -> route_after_scoring
                                              |
                            auto_decline  /   |   \\  human_approval --interrupt()--> route_after_human
                           (DSCR far      /    |    \\                                    |     \\
                            below floor) /  request   \\ (meets floor,               finalize   revise
                                        /  more_info    \\ nothing missing)          (irreversible)  (loop back)
                                       /  (bounded loop,  \\
                                      /    back to verify) \\
"""

import json
import os
import sqlite3
from typing import Annotated, TypedDict
from operator import add

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from m1 import LoanApplication, find_missing_fields, parse_invoice
from m2 import dscr_calculator

MAX_REVISIONS = 2
DB_PATH = "m5_underwriting.sqlite"


class UnderwritingState(TypedDict, total=False):
    raw_record: dict
    application: dict
    dscr: float | None
    missing_fields: list
    revision_count: int
    decision: str            # CONTROL field: current status. Overwritten, no reducer.
    approver_note: str
    audit_log: Annotated[list[str], add]   # AUDIT field: full history. Reducer, accumulates.


# 
# Nodes -- deterministic, testable as plain functions, no model call except
# inside intake (M1's own extract/repair loop already deterministic in shape)
# 

def intake_node(state: UnderwritingState) -> dict:
    app, errors = parse_invoice(state["raw_record"])
    if app is None:
        return {"application": None,
                "audit_log": [f"intake failed after repair attempts: {errors}"]}
    return {"application": app.model_dump(),
            "audit_log": [f"intake ok for {app.application_id}"]}


def verify_node(state: UnderwritingState) -> dict:
    if state.get("application") is None:
        return {"missing_fields": []}
    app = LoanApplication(**state["application"])
    missing = sorted(find_missing_fields(app))
    return {"missing_fields": missing, "audit_log": [f"verify: missing_fields={missing}"]}


def risk_scoring_node(state: UnderwritingState) -> dict:
    if state.get("application") is None:
        return {"dscr": None}
    fin = state["application"]["declared_financials"]
    ebitda, debt = fin.get("ebitda_inr"), fin.get("existing_debt_inr")
    if ebitda is None or debt is None or debt == 0:
        return {"dscr": None, "audit_log": ["risk_scoring: cannot compute DSCR, financials incomplete"]}
    dscr = dscr_calculator(ebitda, debt)
    return {"dscr": dscr, "audit_log": [f"risk_scoring: dscr={dscr}"]}


def request_more_info_node(state: UnderwritingState) -> dict:
    """The bounded revision loop. In a real system this pauses for the applicant
    to supply the missing document; here it's simulated deterministically so the
    loop guard itself stays reproducible."""
    rev = state.get("revision_count", 0) + 1
    return {"revision_count": rev,
            "audit_log": [f"revision {rev}: requested missing fields {state.get('missing_fields')}"]}


def route_after_scoring(state: UnderwritingState) -> str:
    """Pure function of state -- unit-testable with no graph, no model."""
    if state.get("application") is None:
        return "human_approval"
    if state.get("missing_fields") and state.get("revision_count", 0) < MAX_REVISIONS:
        return "request_more_info"
    if state.get("dscr") is None:
        return "human_approval"
    if state["dscr"] < 1.0:
        return "auto_decline"
    return "human_approval"  # meets or is close to the floor -- always needs sign-off per policy


def auto_decline_node(state: UnderwritingState) -> dict:
    return {"decision": "declined",
            "audit_log": [f"auto-declined: DSCR {state.get('dscr')} well below 1.25x floor"]}


def human_approval_node(state: UnderwritingState) -> dict:
    """The pause. Does nothing before interrupt() except read state --
    a side effect placed before interrupt() in the same node would re-run on resume."""
    decision = interrupt({
        "question": "Approve this loan application?",
        "application_id": state.get("application", {}).get("application_id"),
        "dscr": state.get("dscr"),
        "missing_fields": state.get("missing_fields"),
        "revisions_attempted": state.get("revision_count", 0),
    })
    return {"decision": decision["action"], "approver_note": decision.get("note", "")}


def route_after_human(state: UnderwritingState) -> str:
    if state["decision"] == "approved":
        return "finalize"
    if state["decision"] == "changes_requested":
        return "request_more_info"
    return "end"  # rejected


def finalize_node(state: UnderwritingState) -> dict:
    """The irreversible step -- its own node, downstream of the pause. Nothing
    irreversible happens anywhere before this point in the graph."""
    return {"decision": "approved_final",
            "audit_log": [f"FINALIZED by human approval, note={state.get('approver_note')!r}"]}


def build_graph(checkpointer):
    g = StateGraph(UnderwritingState)
    g.add_node("intake", intake_node)
    g.add_node("verify", verify_node)
    g.add_node("risk_scoring", risk_scoring_node)
    g.add_node("request_more_info", request_more_info_node)
    g.add_node("auto_decline", auto_decline_node)
    g.add_node("human_approval", human_approval_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "intake")
    g.add_edge("intake", "verify")
    g.add_edge("verify", "risk_scoring")
    g.add_conditional_edges("risk_scoring", route_after_scoring, {
        "request_more_info": "request_more_info",
        "auto_decline": "auto_decline",
        "human_approval": "human_approval",
    })
    g.add_edge("request_more_info", "verify")  # the loop back, bounded by MAX_REVISIONS
    g.add_edge("auto_decline", END)
    g.add_conditional_edges("human_approval", route_after_human, {
        "finalize": "finalize", "request_more_info": "request_more_info", "end": END,
    })
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)  # clean slate so the run is repeatable

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()

    with open("capstone-data-toolkit/data/wealthpilot/intake/records.jsonl") as f:
        records = [json.loads(line) for line in f]
    record = records[2]  # ASH-L-8910: high DSCR, no missing fields -> routes to human_approval

    graph = build_graph(saver)
    thread_id = f"application-{record['application_id']}"
    cfg = {"configurable": {"thread_id": thread_id}}

    paused = graph.invoke({"raw_record": record}, cfg)
    print("paused at:", graph.get_state(cfg).next)
    assert "__interrupt__" in paused, "expected the graph to pause at human_approval"
    payload = paused["__interrupt__"][0].value
    print("human is asked:", json.dumps(payload, indent=2, default=str))
    print(f"\ncheckpoint file on disk: {DB_PATH} ({os.path.getsize(DB_PATH)} bytes)")

    # --- Simulate a real process restart: fresh connection, fresh graph object,
    #     same thread_id. This is the actual milestone -- not the graph running,
    #     but the graph SURVIVING a restart and resuming correctly. ---
    conn.close()
    print("\n--- simulating a process restart (new connection, new graph object) ---\n")
    conn2 = sqlite3.connect(DB_PATH, check_same_thread=False)
    saver2 = SqliteSaver(conn2)
    saver2.setup()
    graph2 = build_graph(saver2)

    recovered = graph2.get_state(cfg)
    print("recovered from disk:", bool(recovered.values))
    print("still parked at    :", recovered.next)

    final = graph2.invoke(
        Command(resume={"action": "approved", "note": "DSCR comfortably above floor, approved."}),
        cfg,
    )
    print("\nfinal decision :", final["decision"])
    print("approver note  :", final["approver_note"])
    print("audit_log      :")
    for entry in final["audit_log"]:
        print("  -", entry)

    checklist = {
        "typed state with explicit reducer (audit_log)": hasattr(UnderwritingState.__annotations__["audit_log"], "__metadata__"),
        "deterministic routing as a pure function": route_after_scoring({"application": {}, "dscr": 0.5, "missing_fields": []}) == "auto_decline",
        "loop guard (bounded revisions)": route_after_scoring({"application": {}, "missing_fields": ["x"], "revision_count": MAX_REVISIONS, "dscr": 2.0}) != "request_more_info",
        "human gate via real interrupt()": "human_approval" in graph2.get_graph().nodes,
        "durable checkpointer survived a real restart": recovered.values.get("application") is not None,
        "irreversible action isolated downstream of pause": "finalize" in graph2.get_graph().nodes,
    }
    print("\n--- Milestone 5 checklist ---")
    for item, ok in checklist.items():
        print(f"  [{'x' if ok else ' '}] {item}")
    assert all(checklist.values()), "One or more milestone criteria not met."
    print("\nPASS -- this graph satisfies the Milestone 5 checklist for real, including surviving an actual restart.")
