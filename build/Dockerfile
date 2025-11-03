FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app \
    PORT=5000 \
    WORKERS=2 \
    THREADS=2 \
    TORCH_HOME=/app/.cache/torch \
    XDG_CACHE_HOME=/app/.cache

# Install runtime/build dependencies required for Pillow, facenet-pytorch, etc.
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    libjpeg-dev \
    zlib1g-dev \
  && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r app && useradd -r -g app -d $APP_HOME -s /sbin/nologin app \
  && mkdir -p $APP_HOME \
  && chown app:app $APP_HOME

WORKDIR $APP_HOME

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt ./

# Upgrade pip and install Python dependencies
RUN python -m pip install --upgrade pip setuptools wheel \
  && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create cache directory and download facenet model during build (as root)
RUN mkdir -p $APP_HOME/.cache/torch/checkpoints \
  && python -c "from facenet_pytorch import InceptionResnetV1; InceptionResnetV1(pretrained='vggface2')" \
  && chown -R app:app $APP_HOME

USER app

EXPOSE 5000

# Healthcheck (requires curl which is installed above)
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

# Run Gunicorn. Adjust WORKERS and THREADS via env vars if needed.
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "src.main:app", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "--threads", "2"]