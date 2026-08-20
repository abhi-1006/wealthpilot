#!/usr/bin/env python3
"""Capstone data generator CLI.

Examples
--------
    # See what would be produced without spending a single API call
    python generate.py --domain shopsense --dry-run

    # Generate everything for one capstone
    python generate.py --domain careflow

    # Just the parts you need right now
    python generate.py --domain lexops --only corpus,eval

    # Mock API tables only -- no LLM, no API key, runs offline in seconds
    python generate.py --domain plantguard --only tables

    # Smaller corpus while you iterate
    python generate.py --domain wealthpilot --corpus-docs 8 --intake 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from datagen.config import settings
from datagen.domains import REGISTRY

_STAGES = ("corpus", "intake", "tables", "eval")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate.py",
        description="Generate capstone datasets for the Applied AI programme.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--domain",
        required=True,
        choices=sorted(REGISTRY) + ["all"],
        help="Which capstone to generate data for.",
    )
    p.add_argument(
        "--only",
        default=",".join(_STAGES),
        help=f"Comma-separated subset of stages: {', '.join(_STAGES)}",
    )
    p.add_argument("--seed", type=int, help="Override the reproducibility seed.")
    p.add_argument("--corpus-docs", type=int, help="Number of corpus documents.")
    p.add_argument("--intake", type=int, help="Number of intake records.")
    p.add_argument("--eval-items", type=int, help="Size of the golden eval set.")
    p.add_argument("--provider", choices=("gemini", "openrouter", "ollama", "groq"))
    p.add_argument("--out", type=Path, help="Output directory.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generation plan and exit without calling any API.",
    )
    return p


def apply_overrides(args: argparse.Namespace) -> None:
    if args.seed is not None:
        settings.seed = args.seed
    if args.corpus_docs is not None:
        settings.corpus_docs = args.corpus_docs
    if args.intake is not None:
        settings.intake_records = args.intake
    if args.eval_items is not None:
        settings.eval_items = args.eval_items
    if args.provider:
        settings.provider = args.provider
    if args.out:
        settings.output_dir = args.out


def preflight(stages: set[str]) -> None:
    """Fail fast on a missing key rather than 40 documents in."""
    if not (stages & {"corpus", "intake", "eval"}):
        return  # tables-only needs no provider
    if settings.provider == "gemini" and not settings.gemini_api_key:
        sys.exit(
            "No Gemini key. Set DATAGEN_GEMINI_API_KEY in .env, or run with\n"
            "  --provider ollama   (local, no key)\n"
            "  --only tables       (no LLM needed)"
        )
    if settings.provider == "openrouter" and not settings.openrouter_api_key:
        sys.exit("No OpenRouter key. Set DATAGEN_OPENROUTER_API_KEY in .env.")
    if settings.provider == "groq" and not settings.groq_api_key:
        sys.exit("No Groq key. Set DATAGEN_GROQ_API_KEY in .env.")


def run_domain(key: str, stages: set[str], dry_run: bool) -> None:
    spec = REGISTRY[key]()
    docs = spec.doc_specs()[: settings.corpus_docs]

    print(f"\n{'=' * 68}\n{spec.name}\n{'=' * 68}")
    print(f"  provider   : {settings.provider}")
    print(f"  seed       : {settings.seed}")
    print(f"  stages     : {', '.join(sorted(stages))}")
    print(f"  corpus     : {len(docs)} documents (markdown + PDF)")
    print(f"  intake     : {settings.intake_records} records")
    print(f"  eval       : {settings.eval_items} cases")
    print(f"  tables     : {', '.join(spec.seed_tables().keys())}")
    print("\n  public datasets this domain is grounded against:")
    for src in spec.public_sources:
        print(f"    - {src['name']}  [{src['licence']}]")

    if dry_run:
        approx_calls = (
            (len(docs) if "corpus" in stages else 0)
            + (settings.intake_records // 20 if "intake" in stages else 0)
            + (1 if "eval" in stages else 0)
        )
        print(f"\n  DRY RUN -- would make roughly {approx_calls} LLM calls.")
        return

    out_root = settings.output_dir
    started = time.time()
    out_dir = out_root / spec.key
    out_dir.mkdir(parents=True, exist_ok=True)

    if "corpus" in stages:
        print("\n  [1/4] corpus ...", flush=True)
        spec.build_corpus(out_dir)
    if "intake" in stages:
        print("  [2/4] intake ...", flush=True)
        spec.build_intake(out_dir)
    if "tables" in stages:
        print("  [3/4] mock API tables ...", flush=True)
        spec.build_seed_tables(out_dir)
    if "eval" in stages:
        print("  [4/4] golden eval set ...", flush=True)
        spec.build_eval_set(out_dir, [d.title for d in docs])

    print(f"\n  done in {time.time() - started:.0f}s -> {out_dir.resolve()}")


def main() -> None:
    args = build_parser().parse_args()
    apply_overrides(args)

    stages = {s.strip() for s in args.only.split(",") if s.strip()}
    unknown = stages - set(_STAGES)
    if unknown:
        sys.exit(f"Unknown stage(s): {', '.join(sorted(unknown))}")

    if not args.dry_run:
        preflight(stages)

    keys = sorted(REGISTRY) if args.domain == "all" else [args.domain]
    for key in keys:
        run_domain(key, stages, args.dry_run)


if __name__ == "__main__":
    main()
