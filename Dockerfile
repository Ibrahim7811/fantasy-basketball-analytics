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

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "fantasy_basketball.api:app", "--host", "0.0.0.0", "--port", "8000"]