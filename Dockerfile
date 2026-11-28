# syntax=docker/dockerfile:1
FROM python:3.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FANTASY_DB_PATH=/data/fantasy_basketball.db \
    FANTASY_MODEL_DIR=/models

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --no-cache-dir \
    "torch==2.15.0" --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip check

COPY fantasy_basketball ./fantasy_basketball

ADD --checksum=sha256:4cda7eb4c7395d7223358af196456c39afedb236f98fc0155a5bdaef49a1f191 \
    https://github.com/Ibrahim7811/fantasy-basketball-analytics/releases/download/models-v1/32_players_3_features_v1.zip \
    /tmp/models.zip

RUN python -c "from fantasy_basketball.artifacts import install_models; install_models('/tmp/models.zip', '/models')" \
    && chmod -R a+rX /models \
    && rm /tmp/models.zip

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

CMD ["sh", "-c", "exec python -m uvicorn fantasy_basketball.api:app --host 0.0.0.0 --port ${PORT:-8000}"]