FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# OCR and PDF runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY . /app
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Default runtime paths; can be overridden via env
ENV UPLOAD_FOLDER=/var/lib/smartdoc/uploads \
    DATABASE_URL=postgresql://smartdoc:smartdoc@db:5432/smartdoc

RUN mkdir -p /var/lib/smartdoc/uploads

EXPOSE 5000

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "app.py"]
