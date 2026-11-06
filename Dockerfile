FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf

WORKDIR /app

# CPU-only torch is about a fifth the size of the default CUDA build.
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Download the reranker model at build time, so a cold start doesn't wait on it.
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY api.py m1.py m2.py m4.py frontend.html ./
COPY capstone-data-toolkit/data/wealthpilot ./capstone-data-toolkit/data/wealthpilot

# Run as a non-root user; .embcache is the only place the app writes.
RUN useradd --create-home appuser && mkdir .embcache && chown appuser .embcache
USER appuser

# Cloud Run sets PORT (8080 by default).
CMD ["sh", "-c", "exec uvicorn api:api_app --host 0.0.0.0 --port ${PORT:-8080}"]
