"""M5 -- Orchestrated LangGraph workflow with checkpointing.

Wires M1 (intake), M2 (risk-signal tools), and M4 (policy RAG) together into
one workflow graph:

    intake -> document verification -> risk scoring -> conditional routing
                                                              |
                                            auto_decline / borderline_flag / human_review

Checkpointing uses LangGraph's in-memory MemorySaver -- deliberate scope call
given the deadline: demonstrates the checkpoint/resume mechanism itself
(state is saved after every node, a run can be resumed from any point via
its thread_id) without standing up a persistent checkpoint DB.
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from m1 import parse_invoice, find_missing_fields
from m2 import dscr_calculator, credit_bureau_lookup


class UnderwritingState(TypedDict, total=False):
    raw_record: dict
    application: dict          # serialized LoanApplication (model_dump)
    parse_errors: list
    missing_fields: list
    dscr: float
    decision: str
    route: str
    reasons: list[str]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def intake_node(state: UnderwritingState) -> dict:
    """M1: parse the raw record into a validated LoanApplication."""
    app, errors = parse_invoice(state["raw_record"])
    if app is None:
        return {"application": None, "parse_errors": errors, "route": "human_review",
                "reasons": ["Failed structured parsing after repair attempts -- needs manual review."]}
    return {"application": app.model_dump(), "parse_errors": errors}


def verify_node(state: UnderwritingState) -> dict:
    """Which fields, if any, are missing -- feeds the routing decision."""
    if state.get("application") is None:
        return {}
    from m1 import LoanApplication
    app = LoanApplication(**state["application"])
    missing = find_missing_fields(app)
    return {"missing_fields": sorted(missing)}


def risk_scoring_node(state: UnderwritingState) -> dict:
    """M2: compute DSCR from the parsed financials."""
    if state.get("application") is None:
        return {}
    fin = state["application"]["declared_financials"]
    ebitda = fin.get("ebitda_inr")
    debt = fin.get("existing_debt_inr")
    if ebitda is None or debt is None or debt == 0:
        return {"dscr": None}
    return {"dscr": dscr_calculator(ebitda, debt)}


def route_decision(state: UnderwritingState) -> str:
    """Conditional edge: decide which branch to take next."""
    if state.get("application") is None:
        return "human_review"
    if state.get("dscr") is None or state.get("missing_fields"):
        return "borderline_flag"
    if state["dscr"] < 1.0:
        return "auto_decline"
    if state["dscr"] < 1.25:
        return "borderline_flag"
    return "human_review"  # meets DSCR floor on paper -- still needs sign-off per policy


def auto_decline_node(state: UnderwritingState) -> dict:
    return {"decision": "declined",
            "reasons": [f"DSCR {state.get('dscr')} is well below the 1.25x policy floor."]}


def borderline_flag_node(state: UnderwritingState) -> dict:
    reasons = []
    if state.get("missing_fields"):
        reasons.append(f"Missing fields: {state['missing_fields']}")
    if state.get("dscr") is not None:
        reasons.append(f"DSCR {state['dscr']} is close to the 1.25x floor -- borderline.")
    return {"decision": "borderline", "reasons": reasons or ["Borderline case -- needs closer review."]}


def human_review_node(state: UnderwritingState) -> dict:
    return {"decision": "pending_human_signoff",
            "reasons": ["Meets policy criteria on paper -- routed for required human sign-off."]}


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph():
    g = StateGraph(UnderwritingState)
    g.add_node("intake", intake_node)
    g.add_node("verify", verify_node)
    g.add_node("risk_scoring", risk_scoring_node)
    g.add_node("auto_decline", auto_decline_node)
    g.add_node("borderline_flag", borderline_flag_node)
    g.add_node("human_review", human_review_node)

    g.add_edge(START, "intake")
    g.add_edge("intake", "verify")
    g.add_edge("verify", "risk_scoring")
    g.add_conditional_edges("risk_scoring", route_decision, {
        "auto_decline": "auto_decline",
        "borderline_flag": "borderline_flag",
        "human_review": "human_review",
    })
    g.add_edge("auto_decline", END)
    g.add_edge("borderline_flag", END)
    g.add_edge("human_review", END)

    checkpointer = MemorySaver()
    return g.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    import json

    graph = build_graph()

    with open("capstone-data-toolkit/data/wealthpilot/intake/records.jsonl") as f:
        records = [json.loads(line) for line in f]

    for i, record in enumerate(records[:3]):
        config = {"configurable": {"thread_id": f"application-{i}"}}
        result = graph.invoke({"raw_record": record}, config=config)
        print(f"\n--- {record['application_id']} ---")
        print("route/decision:", result.get("decision"))
        print("dscr:", result.get("dscr"), "| missing_fields:", result.get("missing_fields"))
        print("reasons:", result.get("reasons"))

        # Prove checkpointing actually works: read the saved state back out
        # via the SAME thread_id, independent of the `result` variable above.
        saved = graph.get_state(config)
        assert saved.values.get("decision") == result.get("decision"), \
            "checkpointed state doesn't match the run's own result"
    print("\nPASS -- checkpointed state matches each run's result, verified via get_state().")
