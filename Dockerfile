# =============================================================================
# Audiobook Pipeline — Flask App
# Base: CUDA 12.1 + cuDNN 8 on Ubuntu 22.04
# =============================================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# ─── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-venv \
        python3-pip \
        ffmpeg \
        calibre \
        git \
        curl \
        build-essential \
        libsndfile1 \
        libgl1-mesa-glx \
        libglib2.0-0 \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# ─── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ─── Python deps (cached layer) ───────────────────────────────────────────────
COPY requirements.txt .

RUN pip install --upgrade pip

# PyTorch — try CUDA first, fall back to CPU
RUN pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
        --index-url https://download.pytorch.org/whl/cu121 \
    || pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
        --index-url https://download.pytorch.org/whl/cpu

RUN pip install -r requirements.txt

# Gunicorn for production serving
RUN pip install gunicorn==21.2.0

# Pre-download NLTK data into the image
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# ─── Application code ─────────────────────────────────────────────────────────
COPY . .

# Persistent directories (will be bind-mounted in production)
RUN mkdir -p uploads outputs temp

# ─── Healthcheck ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -sf http://localhost:5000/ || exit 1

EXPOSE 5000

# ─── Entrypoint ───────────────────────────────────────────────────────────────
# 1 worker + 4 threads is the right shape for a thread-heavy Flask app.
# --timeout 600 covers long TTS + video jobs.
# --preload runs app-level init (Kokoro health-check) once before forking.
CMD ["gunicorn", \
     "--workers", "1", \
     "--threads", "4", \
     "--bind", "0.0.0.0:5000", \
     "--timeout", "600", \
     "--keep-alive", "5", \
     "--log-level", "info", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:app"]