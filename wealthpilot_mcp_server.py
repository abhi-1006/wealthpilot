"""FastMCP server exposing WealthPilot's mock credit-bureau/bank-statement APIs,
plus one resource (the fair-lending policy), matching the Day 3 Session 2 lab's
Lab B pattern: tools the model chooses to call, a resource the application loads.

Run standalone with: python wealthpilot_mcp_server.py
"""
import csv
import sys
from pathlib import Path

from fastmcp import FastMCP

PROJECT_ROOT = Path(__file__).resolve().parent
MOCK_DIR = PROJECT_ROOT / "capstone-data-toolkit/data/wealthpilot/mock_api"
FAIR_LENDING_PATH = PROJECT_ROOT / "capstone-data-toolkit/data/wealthpilot/corpus/markdown/fair-lending.md"

mcp = FastMCP("wealthpilot-underwriting")


@mcp.tool()
def credit_bureau_lookup(applicant_id: str) -> dict | None:
    """Look up an applicant's credit bureau report by applicant_id, e.g. 'ASH-A-00042'.

    Returns bureau_score, enquiries_last_6m, active_trade_lines, dpd_30_last_12m,
    dpd_90_ever, written_off_amount_inr -- or null if the applicant_id is not found.
    """
    with open(MOCK_DIR / "bureau_reports.csv") as f:
        for row in csv.DictReader(f):
            if row["applicant_id"] == applicant_id:
                return row
    return None


@mcp.tool()
def bank_statement_lookup(applicant_id: str) -> list[dict]:
    """Look up an applicant's monthly bank statement history by applicant_id.

    Returns a list, one entry per month (inflow_inr, outflow_inr, closing_balance_inr,
    bounced_instruments, avg_daily_balance_inr). Empty list if none found.
    """
    statements = []
    with open(MOCK_DIR / "bank_statements.csv") as f:
        for row in csv.DictReader(f):
            if row["applicant_id"] == applicant_id:
                statements.append(row)
    return statements


@mcp.resource("wealthpilot://policy/fair-lending")
def fair_lending_policy() -> str:
    """The fair-lending policy every underwriting decision must comply with."""
    return FAIR_LENDING_PATH.read_text(encoding="utf-8")


if __name__ == "__main__":
    print("[wealthpilot_mcp_server] starting on stdio", file=sys.stderr)
    mcp.run(transport="stdio", show_banner=False)

