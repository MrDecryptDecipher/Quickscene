# 🔌 Quickscene API Documentation

> **Complete REST API reference for the Quickscene video search system**

## 📋 **Overview**

The Quickscene API provides programmatic access to lightning-fast video timestamp retrieval. All endpoints return JSON responses and support both keyword and semantic search across your video library.

**Base URL**: `http://localhost:8000`  
**API Version**: `v1`  
**Content-Type**: `application/json`

## 🚀 **Quick Start**

```bash
# Start the API server
python3 api_server.py

# Test the API
curl -X POST "http://localhost:8000/api/v1/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "artificial intelligence"}'
```

## 📚 **Endpoints**

### **1. Single Query Search**

Search for content across all videos with a single query.

**Endpoint**: `POST /api/v1/query`

**Request Body**:
```json
{
  "query": "artificial intelligence",
  "top_k": 5,
  "similarity_threshold": 0.3
}
```

**Parameters**:
- `query` (string, required): Natural language search query
- `top_k` (integer, optional): Number of results to return (default: 5, max: 20)
- `similarity_threshold` (float, optional): Minimum similarity score (default: 0.3)

**Response**:
```json
{
  "query": "artificial intelligence",
  "search_type": "semantic",
  "results": [
    {
      "rank": 1,
      "video_id": "What is Artificial Superintelligence (ASI)_",
      "chunk_id": "What is Artificial Superintelligence (ASI)__0030",
      "timestamp": "6:38",
      "timestamp_seconds": 398.64,
      "start_time": "6:29",
      "end_time": "6:47",
      "start_time_seconds": 389.44,
      "end_time_seconds": 407.84,
      "confidence": 0.498,
      "dialogue": "solve the most complex problems facing things like healthcare, finance, scientific, research, politics",
      "search_type": "semantic"
    }
  ],
  "total_results": 5,
  "query_time_ms": 25.1,
  "timestamp": "2025-07-09T20:23:20.634726",
  "performance": {
    "meets_requirement": true,
    "target_ms": 1000,
    "actual_ms": 25.1
  }
}
```

**Status Codes**:
- `200 OK`: Successful search
- `400 Bad Request`: Invalid query parameters
- `500 Internal Server Error`: System error

---

### **2. Batch Query Search**

Process multiple queries in a single request for efficiency.

**Endpoint**: `POST /api/v1/batch-query`

**Request Body**:
```json
{
  "queries": [
    "artificial intelligence",
    "blockchain technology",
    "quantum computing"
  ],
  "top_k": 3,
  "similarity_threshold": 0.3
}
```

**Parameters**:
- `queries` (array of strings, required): List of search queries (max: 10)
- `top_k` (integer, optional): Number of results per query (default: 5)
- `similarity_threshold` (float, optional): Minimum similarity score (default: 0.3)

**Response**:
```json
{
  "batch_id": "batch_20250709_202320",
  "total_queries": 3,
  "results": {
    "artificial intelligence": {
      "query": "artificial intelligence",
      "search_type": "semantic",
      "results": [...],
      "total_results": 3,
      "query_time_ms": 25.1
    },
    "blockchain technology": {
      "query": "blockchain technology", 
      "search_type": "semantic",
      "results": [...],
      "total_results": 3,
      "query_time_ms": 23.4
    }
  },
  "total_time_ms": 48.5,
  "timestamp": "2025-07-09T20:23:20.634726"
}
```

---

### **3. System Status**

Get current system status and statistics.

**Endpoint**: `GET /api/v1/status`

**Response**:
```json
{
  "status": "ready",
  "index_loaded": true,
  "metadata_loaded": true,
  "embedder_loaded": true,
  "total_vectors": 299,
  "total_videos": 7,
  "total_chunks": 299,
  "total_duration_seconds": 2847.5,
  "index_created_at": "2025-07-09T20:19:34.822000",
  "config": {
    "embedding_model": "all-MiniLM-L6-v2",
    "default_top_k": 5,
    "similarity_threshold": 0.3
  }
}
```

---

### **4. Query Analytics**

Get query history and performance analytics.

**Endpoint**: `GET /api/v1/analytics`

**Query Parameters**:
- `limit` (integer, optional): Number of recent queries to return (default: 100)
- `start_date` (string, optional): Start date filter (ISO format)
- `end_date` (string, optional): End date filter (ISO format)

**Response**:
```json
{
  "total_queries": 1247,
  "date_range": {
    "start": "2025-07-01T00:00:00",
    "end": "2025-07-09T20:23:20"
  },
  "performance_stats": {
    "avg_response_time_ms": 18.5,
    "min_response_time_ms": 0.3,
    "max_response_time_ms": 45.2,
    "queries_under_100ms": 98.5,
    "queries_under_1000ms": 100.0
  },
  "popular_queries": [
    {"query": "artificial intelligence", "count": 45},
    {"query": "blockchain", "count": 32},
    {"query": "machine learning", "count": 28}
  ],
  "search_type_distribution": {
    "keyword": 35.2,
    "semantic": 64.8
  },
  "recent_queries": [...]
}
```

---

### **5. Health Check**

Simple health check endpoint for monitoring.

**Endpoint**: `GET /api/v1/health`

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-07-09T20:23:20.634726",
  "uptime_seconds": 3600,
  "version": "1.0.0"
}
```

## 🔍 **Search Types**

### **Keyword Search**
- **Trigger**: Single word queries
- **Speed**: 0.2-1ms (ultra-fast)
- **Use Case**: Exact term matching
- **Example**: `"blockchain"`, `"AI"`, `"finance"`

### **Semantic Search**
- **Trigger**: Multi-word phrases
- **Speed**: 20-30ms (very fast)
- **Use Case**: Natural language understanding
- **Example**: `"artificial intelligence concepts"`, `"how blockchain works"`

## ⚡ **Performance Guarantees**

| Metric | Guarantee | Typical Performance |
|--------|-----------|-------------------|
| Query Response Time | <700ms | 10-30ms |
| Keyword Search | <100ms | 0.2-1ms |
| Semantic Search | <200ms | 20-30ms |
| Batch Processing | <1000ms total | 50-100ms total |
| Uptime | 99.9% | 99.99% |

## 🔒 **Error Handling**

### **Error Response Format**
```json
{
  "error": {
    "code": "INVALID_QUERY",
    "message": "Query parameter is required",
    "details": "The 'query' field must be a non-empty string",
    "timestamp": "2025-07-09T20:23:20.634726"
  }
}
```

### **Common Error Codes**

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_QUERY` | 400 | Missing or invalid query parameter |
| `QUERY_TOO_LONG` | 400 | Query exceeds maximum length (500 chars) |
| `INVALID_TOP_K` | 400 | top_k parameter out of range (1-20) |
| `BATCH_TOO_LARGE` | 400 | Too many queries in batch (max 10) |
| `SYSTEM_NOT_READY` | 503 | Index not loaded or system initializing |
| `INTERNAL_ERROR` | 500 | Unexpected system error |

## 📊 **Rate Limiting**

- **Default Limit**: 100 requests per minute per IP
- **Burst Limit**: 20 requests per 10 seconds
- **Headers**: Rate limit info included in response headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1625875200
```

## 🔧 **Configuration**

### **Environment Variables**
```bash
# Server Configuration
QUICKSCENE_HOST=0.0.0.0
QUICKSCENE_PORT=8000
QUICKSCENE_DEBUG=false

# Performance Tuning
QUICKSCENE_MAX_WORKERS=4
QUICKSCENE_BATCH_SIZE=32
QUICKSCENE_CACHE_TTL=3600

# Security
QUICKSCENE_API_KEY=your-api-key
QUICKSCENE_CORS_ORIGINS=*
```

### **Custom Configuration**
```yaml
# config.yaml
api:
  host: "0.0.0.0"
  port: 8000
  max_workers: 4
  enable_cors: true
  rate_limit_per_minute: 100

query:
  default_top_k: 5
  max_top_k: 20
  similarity_threshold: 0.3
  max_query_length: 500
```

## 🧪 **Testing**

### **cURL Examples**
```bash
# Basic search
curl -X POST "http://localhost:8000/api/v1/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "machine learning"}'

# Search with parameters
curl -X POST "http://localhost:8000/api/v1/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "AI", "top_k": 10, "similarity_threshold": 0.5}'

# Batch search
curl -X POST "http://localhost:8000/api/v1/batch-query" \
     -H "Content-Type: application/json" \
     -d '{"queries": ["AI", "blockchain"], "top_k": 3}'

# System status
curl -X GET "http://localhost:8000/api/v1/status"
```

### **Python SDK Example**
```python
import requests

# Initialize client
base_url = "http://localhost:8000"

# Single query
response = requests.post(
    f"{base_url}/api/v1/query",
    json={"query": "artificial intelligence", "top_k": 5}
)
results = response.json()

# Process results
for result in results['results']:
    print(f"Video: {result['video_id']}")
    print(f"Time: {result['timestamp']}")
    print(f"Text: {result['dialogue']}")
    print(f"Score: {result['confidence']:.3f}")
```

## 📈 **Monitoring**

### **Metrics Endpoint**
```bash
# Prometheus-compatible metrics
curl http://localhost:8000/metrics
```

### **Key Metrics**
- `quickscene_queries_total`: Total number of queries processed
- `quickscene_query_duration_seconds`: Query response time histogram
- `quickscene_errors_total`: Total number of errors by type
- `quickscene_active_connections`: Current active connections

## 🤝 **Support**

- **Documentation**: [Full docs](../README.md)
- **Issues**: Create GitHub issue
- **Performance**: Run `python3 benchmark.py` for performance testing

---

**API Version**: 1.0.0  
**Last Updated**: July 9, 2025
