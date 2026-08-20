"""Output writers.

Corpora are emitted in the formats your RAG pipeline will actually meet in
M4: markdown for clean-parse baselines, PDF for the messy real case, JSONL
for structured records, CSV for mock-API seed tables.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_markdown(path: Path, title: str, body: str) -> Path:
    ensure_dir(path.parent)
    path.write_text(f"# {title}\n\n{body.strip()}\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return path


def write_json(path: Path, obj: Any) -> Path:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return path


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


_STYLES = getSampleStyleSheet()
_BODY = ParagraphStyle(
    "CorpusBody", parent=_STYLES["BodyText"], fontSize=10, leading=14, spaceAfter=6
)
_H1 = ParagraphStyle(
    "CorpusH1", parent=_STYLES["Heading1"], fontSize=15, spaceAfter=10
)
_H2 = ParagraphStyle("CorpusH2", parent=_STYLES["Heading2"], fontSize=12, spaceAfter=8)


def write_pdf(path: Path, title: str, body_markdown: str) -> Path:
    """Render a markdown-ish body to PDF.

    Intentionally a *lossy* renderer. Real policy PDFs lose their heading
    structure too, which is exactly the parsing problem M4 asks you to solve.
    If your chunker only works on the clean markdown copy, it is not ready.
    """
    ensure_dir(path.parent)
    doc = SimpleDocTemplate(
        str(path),
        pagesize=LETTER,
        topMargin=0.9 * inch,
        bottomMargin=0.9 * inch,
        leftMargin=1.0 * inch,
        rightMargin=1.0 * inch,
        title=title,
    )
    flow: list[Any] = [Paragraph(_esc(title), _H1), Spacer(1, 8)]

    for block in body_markdown.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block == "---":
            flow.append(PageBreak())
        elif block.startswith("###"):
            flow.append(Paragraph(_esc(block.lstrip("# ").strip()), _H2))
        elif block.startswith("##"):
            flow.append(Paragraph(_esc(block.lstrip("# ").strip()), _H2))
        elif block.startswith("#"):
            flow.append(Paragraph(_esc(block.lstrip("# ").strip()), _H1))
        else:
            flow.append(Paragraph(_esc(block).replace("\n", "<br/>"), _BODY))

    doc.build(flow)
    return path


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("**", "")
    )


def write_manifest(
    out_dir: Path,
    *,
    domain: str,
    seed: int,
    provider: str,
    model: str,
    assets: dict[str, Any],
    public_sources: list[dict[str, str]],
) -> Path:
    """Provenance record.

    M8 asks you to defend your evaluation. "Where did this data come from and
    under what licence" is the first question a reviewer will ask, so the
    generator answers it for you rather than leaving it to memory.
    """
    manifest = {
        "domain": domain,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "generator": {"provider": provider, "model": model},
        "assets": assets,
        "public_sources_referenced": public_sources,
        "synthetic": True,
        "contains_real_personal_data": False,
        "notice": (
            "All records in this corpus are synthetic. Any resemblance to a "
            "real person, company, contract, or patient is coincidental. "
            "Public datasets listed under public_sources_referenced were used "
            "for schema and distribution grounding only; consult each source's "
            "own licence before redistributing anything derived from it."
        ),
    }
    return write_json(out_dir / "manifest.json", manifest)


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
