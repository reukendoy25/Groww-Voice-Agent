FROM python:3.11-slim

WORKDIR /app

# System deps for faster-whisper and audio
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first (smaller image for non-GPU deploys)
RUN pip install --no-cache-dir torch==2.2.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install remaining deps
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.30.0 \
    websockets==12.0 \
    httpx==0.27.0 \
    sentence-transformers==3.0.1 \
    scikit-learn==1.5.0 \
    numpy==1.26.4 \
    faiss-cpu==1.8.0 \
    langchain==0.2.5 \
    langchain-community==0.2.5 \
    langchain-core==0.2.10 \
    vaderSentiment==3.3.2 \
    sqlalchemy==2.0.30 \
    python-dotenv==1.0.1 \
    pydantic==2.7.3 \
    transformers==4.41.2 \
    accelerate==0.30.0

# NOTE: In Docker (CPU mode), faster-whisper uses 'small' model, TTS is disabled.
# For full GPU pipeline, run on Colab (see notebooks/).

COPY app/ ./app/
COPY data/ ./data/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
