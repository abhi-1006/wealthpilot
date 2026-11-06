# WealthPilot

SME Loan Underwriting & Credit Research Assistant. IIT Hyderabad Applied AI capstone.
See [ARCHITECTURE.md](ARCHITECTURE.md) for design, results, and limitations.

**Live demo:** https://wealthpilot-972504936573.us-central1.run.app
Ask a credit-policy question, calculate a DSCR, or underwrite a sample application. It runs on synthetic data for a fictional lender. The first request after a quiet period can take up to a minute while the server starts.

## Setup

```bash
pip3 install -r requirements.txt
```

Add `.env` with:

```
GROQ_API_KEY=
SUPERMEMORY_API_KEY=
GOOGLE_API_KEY=
QDRANT_URL=
QDRANT_API_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_BASE_URL=https://cloud.langfuse.com
NGROK_AUTHTOKEN=
```

`NGROK_AUTHTOKEN` is optional — `m8.py` skips the public tunnel and just runs locally without it.

## Run

```bash
python3 m1.py   # intake parsing
python3 m2.py   # tool-calling risk agent
python3 m3.py   # persistent memory
python3 m4.py   # RAG + eval
python3 m5.py   # workflow + human gate
python3 m6.py   # multi-agent committee + MCP
python3 m7.py   # observability + reliability
python3 m8.py   # eval + guardrails + API
```

Run from the project root — later milestones import from earlier ones.

To run only the demo server locally:

```bash
python3 -m uvicorn api:api_app --port 8000
```

## Deploy

The demo runs on Google Cloud Run. `Dockerfile` builds a slim image from `requirements-serve.txt` (no evaluation libraries, CPU-only PyTorch), and `.gcloudignore` uploads only the files it needs. API keys live in Secret Manager, not in the image.

```bash
gcloud run deploy wealthpilot --source . --region us-central1 --allow-unauthenticated \
  --memory 2Gi --cpu 1 --min-instances 0 --max-instances 1 --concurrency 4 --cpu-boost --timeout 300 \
  --set-secrets "GROQ_API_KEY=GROQ_API_KEY:latest,GOOGLE_API_KEY=GOOGLE_API_KEY:latest,QDRANT_URL=QDRANT_URL:latest,QDRANT_API_KEY=QDRANT_API_KEY:latest"
```

## Layout

- `m1.py`–`m8.py` — one file per milestone
- `api.py` — the assistant, guardrails, rate limits and FastAPI app that the demo serves
- `wealthpilot_mcp_server.py` — MCP server for M6
- `frontend.html` — demo UI, served by `api.py`
- `capstone-data-toolkit/` — synthetic data generator

