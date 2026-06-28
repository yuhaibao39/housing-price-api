# ============================================================================
# Stage 1: Build / train the model
# ============================================================================
FROM python:3.12-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app source and training data
COPY app/ app/
COPY data/ data/

# Train the model and save it to models/
RUN python -c "\
import logging; \
logging.basicConfig(level=logging.INFO); \
from app.model import train_model; \
train_model(); \
"

# ============================================================================
# Stage 2: Runtime image
# ============================================================================
FROM python:3.12-slim

RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy only runtime dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache

# Copy app code and pre-trained model from builder
COPY app/ app/
COPY --from=builder /build/models/ models/

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
