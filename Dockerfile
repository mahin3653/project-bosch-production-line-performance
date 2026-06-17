# Bosch Production Line Performance — prediction API
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgomp1 is required by LightGBM / XGBoost at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install only the serving dependencies (not the notebook/EDA stack)
COPY requirements-serve.txt ./
RUN pip install -r requirements-serve.txt

# Single-file application (config + features + serving) and trained artifacts
COPY rafel.py ./
COPY artifacts/ ./artifacts/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "bosch:app", "--host", "0.0.0.0", "--port", "8000"]
