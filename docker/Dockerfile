# Quickscene Docker Image
# Production-ready container for video timestamp-retrieval system

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY config/ ./config/
COPY main.py .

# Create data directories
RUN mkdir -p data/{videos,transcripts,chunks,embeddings,index} logs

# Set environment variables
ENV PYTHONPATH=/app
ENV QUICKSCENE_CONFIG=/app/config/production.yaml

# Expose port for potential API server
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import app; print('Quickscene is healthy')" || exit 1

# Default command
CMD ["python", "main.py", "--help"]
