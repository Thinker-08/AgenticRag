# agrag API image. Full-mode extras baked in: pdf (PyMuPDF/pdfplumber/OCR), stores (qdrant/redis),
# obs (otel/langfuse), and — by default — ml (torch + FlagEmbedding), because config/full.yaml
# selects BGE-M3/BGE-reranker/NLI providers that need it. Build slim (no torch) with:
#   docker compose build --build-arg INSTALL_ML=0
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps for messy-PDF ingestion: OCR (tesseract), pdftoppm/pdfinfo (poppler), and the
# shared libs image parsing / opencv pulls in (libgl, glib).
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only what the build needs first so the dependency layer caches across source edits.
COPY pyproject.toml README.md ./
COPY src ./src

ARG INSTALL_ML=1
RUN pip install -e '.[pdf,stores,obs]' \
    && if [ "$INSTALL_ML" = "1" ]; then pip install -e '.[ml]'; fi
# For CUDA, prefer a torch wheel matched to your driver before the ml extra, e.g.
#   pip install --index-url https://download.pytorch.org/whl/cu121 torch

COPY config ./config

EXPOSE 8000

CMD ["uvicorn", "agrag.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
