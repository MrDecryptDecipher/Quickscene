# 🚀 Quickscene Deployment Guide

> **Complete guide for deploying Quickscene in production environments**

## 📋 **Overview**

This guide covers deploying Quickscene for production use, including Docker deployment, cloud deployment, performance optimization, and monitoring setup.

## 🐳 **Docker Deployment**

### **Quick Start with Docker Compose**

The fastest way to deploy Quickscene is using Docker Compose:

```bash
# Clone the repository
git clone <repository-url>
cd Quickscene

# Build and start services
docker-compose up --build -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f quickscene
```

### **Docker Compose Configuration**

```yaml
# docker-compose.yml
version: '3.8'

services:
  quickscene:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./config.yaml:/app/config.yaml
    environment:
      - QUICKSCENE_HOST=0.0.0.0
      - QUICKSCENE_PORT=8000
      - QUICKSCENE_DEBUG=false
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - quickscene
    restart: unless-stopped

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
    restart: unless-stopped

volumes:
  grafana-storage:
```

### **Dockerfile**

```dockerfile
# Dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/videos data/transcripts data/chunks data/embeddings data/index data/analytics logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Start application
CMD ["python3", "api_server.py"]
```

## ☁️ **Cloud Deployment**

### **AWS Deployment**

#### **1. EC2 Instance Setup**

```bash
# Launch EC2 instance (recommended: t3.large or larger)
# Ubuntu 22.04 LTS, 4GB+ RAM, 20GB+ storage

# Connect to instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Clone and deploy
git clone <repository-url>
cd Quickscene
docker-compose up -d
```

#### **2. ECS Deployment**

```json
{
  "family": "quickscene",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "quickscene",
      "image": "your-account.dkr.ecr.region.amazonaws.com/quickscene:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "QUICKSCENE_HOST", "value": "0.0.0.0"},
        {"name": "QUICKSCENE_PORT", "value": "8000"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/quickscene",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/api/v1/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

### **Google Cloud Platform**

#### **Cloud Run Deployment**

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/quickscene

# Deploy to Cloud Run
gcloud run deploy quickscene \
  --image gcr.io/PROJECT_ID/quickscene \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10
```

### **Azure Container Instances**

```bash
# Create resource group
az group create --name quickscene-rg --location eastus

# Deploy container
az container create \
  --resource-group quickscene-rg \
  --name quickscene \
  --image your-registry/quickscene:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --environment-variables QUICKSCENE_HOST=0.0.0.0 QUICKSCENE_PORT=8000
```

## 🔧 **Production Configuration**

### **Environment Variables**

```bash
# Server Configuration
export QUICKSCENE_HOST=0.0.0.0
export QUICKSCENE_PORT=8000
export QUICKSCENE_DEBUG=false
export QUICKSCENE_WORKERS=4

# Performance Tuning
export QUICKSCENE_MAX_WORKERS=8
export QUICKSCENE_BATCH_SIZE=64
export QUICKSCENE_CACHE_TTL=3600

# Security
export QUICKSCENE_API_KEY=your-secure-api-key
export QUICKSCENE_CORS_ORIGINS=https://yourdomain.com
export QUICKSCENE_RATE_LIMIT=1000

# Monitoring
export QUICKSCENE_METRICS_ENABLED=true
export QUICKSCENE_LOG_LEVEL=INFO
```

### **Production Config File**

```yaml
# config/production.yaml
# Production configuration for Quickscene

# ASR Configuration
asr_mode: "whisper"
whisper_model: "base"  # Use 'base' for better accuracy in production

# Embedding Configuration
embedding_model: "all-MiniLM-L6-v2"
embedding_dimension: 384
chunk_duration_sec: 15
chunk_overlap_sec: 2

# Performance Configuration
max_workers: 8
batch_size: 64
default_top_k: 5
similarity_threshold: 0.3

# Caching
enable_caching: true
cache_ttl_hours: 24

# Monitoring
enable_monitoring: true
log_level: "INFO"
metrics_enabled: true

# Security
api_key_required: true
cors_enabled: true
rate_limit_per_minute: 1000

# File Paths
video_path: "/app/data/videos/"
transcript_path: "/app/data/transcripts/"
chunks_path: "/app/data/chunks/"
embedding_path: "/app/data/embeddings/"
index_path: "/app/data/index/"
log_file: "/app/logs/quickscene.log"
```

## 🔒 **Security Configuration**

### **Nginx Reverse Proxy**

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream quickscene {
        server quickscene:8000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        # SSL Configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;

        # Security Headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

        location /api/ {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://quickscene;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 5s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        location / {
            proxy_pass http://quickscene;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

### **API Key Authentication**

```python
# Add to api_server.py
from fastapi import HTTPException, Depends, Header
import os

async def verify_api_key(x_api_key: str = Header(None)):
    expected_key = os.getenv("QUICKSCENE_API_KEY")
    if expected_key and x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

# Protect endpoints
@app.post("/api/v1/query", dependencies=[Depends(verify_api_key)])
async def query_endpoint(request: QueryRequest):
    # ... endpoint logic
```

## 📊 **Monitoring & Observability**

### **Prometheus Configuration**

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'quickscene'
    static_configs:
      - targets: ['quickscene:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
```

### **Grafana Dashboard**

```json
{
  "dashboard": {
    "title": "Quickscene Performance",
    "panels": [
      {
        "title": "Query Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, quickscene_query_duration_seconds_bucket)",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, quickscene_query_duration_seconds_bucket)",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "title": "Queries per Second",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(quickscene_queries_total[5m])",
            "legendFormat": "QPS"
          }
        ]
      }
    ]
  }
}
```

## 🔄 **CI/CD Pipeline**

### **GitHub Actions**

```yaml
# .github/workflows/deploy.yml
name: Deploy Quickscene

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          python -m pytest tests/
      - name: Run benchmark
        run: |
          python benchmark.py --quick

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build and push Docker image
        run: |
          docker build -t quickscene:${{ github.sha }} .
          docker tag quickscene:${{ github.sha }} your-registry/quickscene:latest
          docker push your-registry/quickscene:latest
      - name: Deploy to production
        run: |
          # Deploy to your cloud provider
          kubectl set image deployment/quickscene quickscene=your-registry/quickscene:latest
```

## 📈 **Performance Optimization**

### **Hardware Recommendations**

| Deployment Size | CPU | RAM | Storage | Expected QPS |
|----------------|-----|-----|---------|--------------|
| Small (1-10 videos) | 2 cores | 4GB | 20GB | 50-100 |
| Medium (10-100 videos) | 4 cores | 8GB | 50GB | 100-500 |
| Large (100+ videos) | 8 cores | 16GB | 100GB | 500-1000 |

### **Optimization Tips**

1. **Use SSD storage** for faster index loading
2. **Enable caching** for frequently accessed data
3. **Tune batch sizes** based on available memory
4. **Use CDN** for static assets
5. **Implement connection pooling** for database connections

## 🚨 **Troubleshooting**

### **Common Issues**

**1. High Memory Usage**
```bash
# Reduce batch size
export QUICKSCENE_BATCH_SIZE=16

# Monitor memory usage
docker stats quickscene
```

**2. Slow Query Performance**
```bash
# Rebuild index
docker exec quickscene python -m app.production_indexer --rebuild

# Check system status
curl http://localhost:8000/api/v1/status
```

**3. Container Startup Issues**
```bash
# Check logs
docker logs quickscene

# Verify health check
curl http://localhost:8000/api/v1/health
```

## 📞 **Support**

- **Documentation**: [API Docs](API.md)
- **Monitoring**: Access Grafana at `http://your-domain:3000`
- **Metrics**: Access Prometheus at `http://your-domain:9090`
- **Health Check**: `GET /api/v1/health`

---

**Last Updated**: July 9, 2025  
**Version**: 1.0.0
