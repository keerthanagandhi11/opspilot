# Production image. Works on Hugging Face Spaces (port 7860), Render, Fly.io, or any Docker host.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    TRUST_PROXY=true \
    FASTEMBED_CACHE_PATH=/app/.fastembed_cache

# Hugging Face Spaces runs containers as uid 1000
RUN useradd -m -u 1000 app
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY backend ./backend
COPY scripts/generate_samples.py ./scripts/

# Hugging Face deploys leave out the binary sample PDFs, so rebuild them if missing
RUN if ! ls backend/app/sample_data/*.pdf >/dev/null 2>&1; then \
      python scripts/generate_samples.py; \
    fi

# Bake the embedding model into the image so cold starts don't download it
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')" \
    && chown -R app:app /app

USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"7860\")}/api/health')"

CMD ["sh", "-c", "uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
