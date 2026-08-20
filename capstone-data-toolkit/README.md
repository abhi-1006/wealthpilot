# Capstone Data Toolkit

Generates the four data assets every capstone in the Applied AI Professional
Certification Program needs, for all five problem statements.

| Asset | Feeds | What it is |
|---|---|---|
| **A. RAG corpus** | M3, M4 | Policy/playbook/manual documents, in markdown *and* PDF |
| **B. Intake records** | M1 | Raw messy inputs to parse into Pydantic objects |
| **C. Mock API tables** | M2, M6 | Relational seed data behind your tools and MCP server |
| **D. Golden eval set** | M8 | 20 graded cases, adversarial ones included |

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env          # add your Gemini key
python generate.py --domain shopsense --dry-run    # see the plan, spend nothing
python generate.py --domain shopsense              # generate everything
```

Domains: `careflow`, `lexops`, `wealthpilot`, `shopsense`, `plantguard`, `all`.

**No API key yet?** The mock API tables need no LLM at all:

```bash
python generate.py --domain plantguard --only tables
```

That gets you a working M2 in minutes while you sort out credentials.

## Useful invocations

```bash
# Iterate fast on a small corpus before committing to a full run
python generate.py --domain lexops --corpus-docs 4 --intake 20 --eval-items 6

# Rate-limited on Gemini? Switch provider, keep everything else
python generate.py --domain careflow --provider ollama

# Regenerate one stage only
python generate.py --domain wealthpilot --only eval
```

## Output layout

```
data/<domain>/
  corpus/
    markdown/*.md        clean-parse baseline
    pdf/*.pdf            the messy real case -- chunk this too
    index.json
  intake/records.jsonl   each record carries a ground_truth block
  mock_api/*.csv|json    seed tables for your tools
  eval/golden_set.json   20 cases with categories and expected routes
  manifest.json          provenance, seed, licences
```

## Three design decisions worth knowing

**Corpus is emitted as both markdown and PDF, deliberately.** The PDF
renderer is lossy — it flattens heading structure the way real policy PDFs do.
If your M4 chunker only performs well on the markdown copy, it is not ready
for the PDF, and the PDF is what production looks like.

**Every intake record carries a `ground_truth` block.** That is your M1
parsing accuracy metric for free, and your M5 routing labels. Do not throw it
away when you load the records.

**The eval set mixes model-written and hand-written cases.** A model asked to
attack a corpus it just wrote produces polite attacks. The `-EV-9xx` cases in
each domain are hand-written to actually break things: prompt injection,
authority-limit probes, false premises, and impossible sensor readings. The
WealthPilot set includes *matched bias pairs* — identical financials, one
varying demographic proxy — where any divergence in outcome is a fair-lending
failure rather than a judgement call.

## Reproducibility

Same seed ⇒ byte-identical output, including all dates (generation is anchored
to a frozen reference clock, not the wall clock). This is what lets teams be
graded against a common rubric. LLM-generated stages vary with provider
sampling; the tables and eval scaffolding do not.

```bash
python generate.py --domain shopsense --only tables --seed 42
```

## This is the starting point, not the finish line

The generator produces the *policy layer* — the documents nobody publishes in
machine-readable form. For volume and realism, layer real public data on top.
Each domain's `manifest.json` lists the datasets it was grounded against, with
licences. The headline ones:

- **LexOps** → CUAD (510 real contracts, 41 clause types, 13k expert
  annotations, CC BY 4.0). Use it as your primary contract corpus; the
  annotations double as free retrieval ground truth.
- **PlantGuard** → AI4I 2020 (CC BY 4.0) and NASA C-MAPSS. Never generate
  sensor time series with an LLM — the output looks plausible and is
  statistically wrong.
- **ShopSense** → Bitext retail/e-commerce intents (CDLA-Sharing 1.0) and
  Olist's nine-table order database.
- **CareFlow** → Synthea, which produces synthetic FHIR patient records that
  are free of privacy restrictions by construction.
- **WealthPilot** → SBA 7(a) public loan data; RBI's MSME Master Direction for
  the India context.

See the *Capstone Data Sourcing Guide* for the full treatment of each.

## Ground rules

No real personal data goes into any capstone, ever — not patient records, not
applicant files, not customer PII. Generated content is synthetic by
construction and every run writes a `manifest.json` asserting so. When you
layer in a public dataset, check its licence: several listed above are
non-commercial or share-alike, which is fine for coursework and not fine for
whatever you build next.
