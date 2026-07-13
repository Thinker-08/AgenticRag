FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

ARG INSTALL_ML=1
RUN pip install -e '.[pdf,stores,obs]' \
    && if [ "$INSTALL_ML" = "1" ]; then pip install -e '.[ml]'; fi

COPY config ./config

EXPOSE 8000

CMD ["uvicorn", "agrag.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
