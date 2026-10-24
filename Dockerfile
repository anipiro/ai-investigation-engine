# Hugging Face Spaces (Docker SDK) image. Also runs anywhere: docker run -p 7860:7860 --env-file .env <image>
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MODEL_CACHE_DIR=/app/models HF_HOME=/app/models/hf

# Spaces run the container as uid 1000.
RUN useradd -m -u 1000 app
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the ONNX embedding + re-ranking models into the image so a cold start does not download them.
RUN python -c "from fastembed import TextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/app/models'); TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2', cache_dir='/app/models')" \
    && chown -R app:app /app

COPY --chown=app:app engine ./engine
COPY --chown=app:app main.py .

USER app
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--timeout-keep-alive", "75"]
