import csv
import json
import os

from dotenv import load_dotenv
from litellm import completion

from m1 import LoanApplication, FinancialSnapshot

load_dotenv()

MOCK_DIR = "capstone-data-toolkit/data/wealthpilot/mock_api"


def dscr_calculator(ebitda_inr: float, existing_debt_inr: float) -> float:
    dscr = round(ebitda_inr/existing_debt_inr,2)
    return dscr

def interest_calculator(principal_inr: float, annual_rate_percent: float, tenor_months: float) -> float:
    total_interest = principal_inr * (annual_rate_percent / 100) * (tenor_months / 12)
    return total_interest

def credit_bureau_lookup(applicant_id: str) -> dict | None:
    with open(f"{MOCK_DIR}/bureau_reports.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["applicant_id"] == applicant_id:
                return row          # <-- INSIDE the loop, INSIDE the if

    return None                     # <-- AFTER the loop, NOT inside it


def bank_statement_lookup(applicant_id: str) -> list[dict]:
    """Bank statements are monthly -- one applicant has MANY rows, not one."""
    statements = []
    with open(f"{MOCK_DIR}/bank_statements.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["applicant_id"] == applicant_id:
                statements.append(row)
    return statements


# ---------------------------------------------------------------------------
# M2 core: wire the four tools above into a single tool-calling agent.
# Uses LiteLLM's own tool-calling support directly (OpenAI-compatible
# function schema) rather than LangChain -- same provider-agnostic principle
# as M1, no new dependency needed.
# ---------------------------------------------------------------------------

TOOLS_BY_NAME = {
    "dscr_calculator": dscr_calculator,
    "interest_calculator": interest_calculator,
    "credit_bureau_lookup": credit_bureau_lookup,
    "bank_statement_lookup": bank_statement_lookup,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "dscr_calculator",
            "description": "Compute the Debt-Service Coverage Ratio (EBITDA / existing annual debt service). Policy floor is 1.25x.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ebitda_inr": {"type": "number", "description": "Annual EBITDA in INR"},
                    "existing_debt_inr": {"type": "number", "description": "Annual existing debt service in INR"},
                },
                "required": ["ebitda_inr", "existing_debt_inr"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "interest_calculator",
            "description": "Compute total simple interest owed over a loan's tenor, given principal, annual rate, and tenor in months.",
            "parameters": {
                "type": "object",
                "properties": {
                    "principal_inr": {"type": "number"},
                    "annual_rate_percent": {"type": "number"},
                    "tenor_months": {"type": "number"},
                },
                "required": ["principal_inr", "annual_rate_percent", "tenor_months"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "credit_bureau_lookup",
            "description": "Look up an applicant's credit bureau report (score, delinquencies, trade lines) by applicant_id, e.g. 'ASH-A-00042'.",
            "parameters": {
                "type": "object",
                "properties": {"applicant_id": {"type": "string"}},
                "required": ["applicant_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bank_statement_lookup",
            "description": "Look up an applicant's monthly bank statement history by applicant_id, e.g. 'ASH-A-00042'. Returns a list, one entry per month.",
            "parameters": {
                "type": "object",
                "properties": {"applicant_id": {"type": "string"}},
                "required": ["applicant_id"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are a risk-analysis agent for Ashva Capital, an SME digital lender. "
    "Use the available tools to compute financial risk signals for a loan "
    "application. Show your reasoning, then give a short plain-text summary "
    "of the risk signals you found. You do not make the final lending "
    "decision -- that is a later stage's job."
)


def run_risk_agent(user_request: str, max_iterations: int = 6) -> str:
    """Bounded tool-calling loop: ask model -> run any requested tools ->
    feed results back -> ask again -> repeat until a final text answer,
    or max_iterations is hit (never loop forever on a confused model)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_request},
    ]

    for _ in range(max_iterations):
        resp = completion(
            model="groq/openai/gpt-oss-20b",
            messages=messages,
            tools=TOOLS_SCHEMA,
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump())

        if not msg.tool_calls:
            return msg.content

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn = TOOLS_BY_NAME[fn_name]
            try:
                fn_args = json.loads(tc.function.arguments)
                result = fn(**fn_args)
            except Exception as e:
                result = f"ERROR calling {fn_name}: {e}"
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })

    return "Stopped: reached max_iterations without a final answer."


if __name__ == "__main__":
    answer = run_risk_agent(
        "Applicant ASH-A-00000 wants to confirm their credit bureau profile "
        "and check the DSCR for a loan with EBITDA 3,120,000 INR and "
        "existing annual debt service of 1,900,000 INR. Also compute total "
        "interest on a 3,000,000 INR loan at 12% annual rate over 24 months."
    )
    print(answer)

