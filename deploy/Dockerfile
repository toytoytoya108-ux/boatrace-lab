FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 sqlite3 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY boatlab ./boatlab
COPY web ./web
COPY scripts ./scripts
RUN pip install --no-cache-dir -e .

ENV BOATLAB_ROOT=/app \
    BOATLAB_DATA_DIR=/app/data \
    PYTHONUNBUFFERED=1

EXPOSE 8080
CMD ["uvicorn", "boatlab.api.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
