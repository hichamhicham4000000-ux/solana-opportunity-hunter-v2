FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /install /usr/local

COPY run.py .
COPY bot/ bot/
COPY config/ config/
COPY database/ database/
COPY data/ data/
COPY analysis/ analysis/
COPY monitoring/ monitoring/
COPY smart_money/ smart_money/
COPY utils/ utils/

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN useradd --create-home --shell /bin/bash botuser && \
    mkdir -p /app/data && \
    chown -R botuser:botuser /app

USER botuser

CMD ["python", "run.py"]
