import os, json, time
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, Field
from litellm import completion
from groq import Groq

load_dotenv()

client = Groq(api_key=os.environ['GROQ_API_KEY'])

class FinancialSnapshot(BaseModel):
    annual_revenue_inr: int | None = None
    ebitda_inr: int | None = None
    existing_debt_inr: int | None = None
    current_assets_inr: int | None = None
    current_liabilities_inr: int | None = None
    months_operating: int | None = None

class LoanApplication(BaseModel):
    application_id:	str
    channel: str
    received_at: str
    raw_narrative: str
    business_name: str
    sector: str
    requested_amount_inr: int
    declared_financials: FinancialSnapshot
    bureau_score: int | None = None


def extract_raw(record: dict) -> str:
    prompt = f"""Normalize this loan application record into JSON with exactly
    these fields: application_id, channel, received_at, raw_narrative,
    business_name, sector, requested_amount_inr, declared_financials
    (annual_revenue_inr, ebitda_inr, existing_debt_inr, current_assets_inr,
    current_liabilities_inr, months_operating), bureau_score.
    Return ONLY valid JSON, no markdown fences, no commentary.

    Record: {record}"""

    resp = completion(
        model="groq/openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


def validate_invoice(raw: str) -> tuple[LoanApplication | None, str | None]:
    # Parse raw as JSON, construct an Invoice, and return (invoice, None) on success
    # or (None, str(e)) on failure
    try:
        data = json.loads(raw)
        app = LoanApplication(**data)
        return app, None
    except (json.JSONDecodeError, ValidationError) as e:
        return None, str(e)


def parse_invoice(record: str, max_retries: int = 2) -> tuple[LoanApplication | None, list[str]]:
    """Returns (parsed Invoice or None, list of attempt errors)."""
    errors = []
    raw = extract_raw(record)
    for attempt in range(max_retries + 1):
        try:
            data = json.loads(raw)
            app = LoanApplication(**data)
            return app, errors

        except (json.JSONDecodeError, ValidationError) as e:
            errors.append(str(e))
            if attempt == max_retries:
                return None, errors
            repair_prompt = f"""The following JSON failed validation with error: {e}

Original text:
{record}

Previous JSON attempt:
{raw}

Return corrected JSON with EXACTLY these fields:
application_id, channel, received_at, raw_narrative,
business_name, sector, requested_amount_inr, declared_financials
(annual_revenue_inr, ebitda_inr, existing_debt_inr, current_assets_inr,
current_liabilities_inr, months_operating), bureau_score.
Return ONLY valid JSON, no markdown fences, no commentary."""
            resp = completion(
                model="groq/openai/gpt-oss-20b",
                messages=[{"role": "user", "content": repair_prompt}],
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
    return None, errors


FINANCIAL_FIELDS = [
    "annual_revenue_inr", "ebitda_inr", "existing_debt_inr",
    "current_assets_inr", "current_liabilities_inr", "months_operating",
]

def find_missing_fields(app: LoanApplication) -> set[str]:
    """Which fields did OUR parser end up leaving as None?"""
    missing = set()
    if app.bureau_score is None:
        missing.add("bureau_score")
    for field_name in FINANCIAL_FIELDS:
        if getattr(app.declared_financials, field_name) is None:
            missing.add(field_name)
    return missing


if __name__ == "__main__":
    with open("capstone-data-toolkit/data/wealthpilot/intake/records.jsonl") as f:
        records = [json.loads(line) for line in f]

    results = []
    for record in records:
        invoice, errs = parse_invoice(record)
        entry = {"id": record["application_id"], "app": invoice, "errors": errs}

        if invoice is not None:
            expected_missing = set(record["ground_truth"]["missing_fields"])
            actual_missing = find_missing_fields(invoice)
            entry["missing_fields_correct"] = (actual_missing == expected_missing)
            entry["expected_missing"] = expected_missing
            entry["actual_missing"] = actual_missing

        results.append(entry)
        time.sleep(2)  # stay under Groq's free-tier tokens-per-minute limit

    success = sum(1 for r in results if r["app"] is not None)
    print(f"Parsed {success}/{len(results)} records successfully.")

    accurate = sum(1 for r in results if r.get("missing_fields_correct"))
    print(f"Missing-field accuracy vs ground_truth: {accurate}/{success}")

    for r in results:
        if r.get("app") is not None and not r.get("missing_fields_correct"):
            print(f"  MISMATCH {r['id']}: expected missing={r['expected_missing']}, "
                  f"parser found missing={r['actual_missing']}")
