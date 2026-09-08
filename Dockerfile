FROM python:3.12-slim

WORKDIR /app

# System deps needed by faiss/pymupdf wheels at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/uploads indexes/faiss

ENV PYTHONUNBUFFERED=1 \
    PORT=5000

EXPOSE 5000

CMD ["python", "run.py"]
