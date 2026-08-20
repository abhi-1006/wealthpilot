# WealthPilot

SME Loan Underwriting & Credit Research Assistant. IIT Hyderabad Applied AI capstone.
See [ARCHITECTURE.md](ARCHITECTURE.md) for design, results, and limitations.

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

## Layout

- `m1.py`–`m8.py` — one file per milestone
- `wealthpilot_mcp_server.py` — MCP server for M6
- `frontend.html` — demo UI, served by M8's FastAPI app
- `capstone-data-toolkit/` — synthetic data generator
