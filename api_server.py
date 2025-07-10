#!/usr/bin/env python3
"""
Production FastAPI Server for Quickscene

High-performance REST API with comprehensive error handling, rate limiting,
monitoring, and analytics for the video search system.
"""

import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
import uuid

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, validator
import uvicorn

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.config import get_config
from app.production_query_handler import ProductionQueryHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
query_handler = None
config = None
analytics_data = {
    'queries': [],
    'performance_stats': {
        'total_queries': 0,
        'avg_response_time_ms': 0,
        'error_count': 0
    }
}

# Pydantic Models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of results to return")
    similarity_threshold: Optional[float] = Field(0.3, ge=0.0, le=1.0, description="Minimum similarity score")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()

class BatchQueryRequest(BaseModel):
    queries: List[str] = Field(..., min_items=1, max_items=10, description="List of search queries")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of results per query")
    similarity_threshold: Optional[float] = Field(0.3, ge=0.0, le=1.0, description="Minimum similarity score")
    
    @validator('queries')
    def validate_queries(cls, v):
        cleaned_queries = []
        for query in v:
            if not query.strip():
                raise ValueError('All queries must be non-empty')
            cleaned_queries.append(query.strip())
        return cleaned_queries

class ErrorResponse(BaseModel):
    error: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    uptime_seconds: float
    version: str = "1.0.0"

class StatusResponse(BaseModel):
    status: str
    index_loaded: bool
    metadata_loaded: bool
    embedder_loaded: bool
    total_vectors: int
    total_videos: int
    total_chunks: int
    total_duration_seconds: float
    index_created_at: str
    config: Dict[str, Any]

# Initialize FastAPI app
app = FastAPI(
    title="Quickscene API",
    description="Lightning-fast video timestamp retrieval system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="web/static"), name="static")
templates = Jinja2Templates(directory="web/templates")

# Startup time for uptime calculation
startup_time = time.time()

# Rate limiting (simple in-memory implementation)
rate_limit_storage = {}

async def rate_limit_check(request: Request):
    """Simple rate limiting middleware"""
    client_ip = request.client.host
    current_time = time.time()
    
    # Clean old entries
    cutoff_time = current_time - 60  # 1 minute window
    rate_limit_storage[client_ip] = [
        timestamp for timestamp in rate_limit_storage.get(client_ip, [])
        if timestamp > cutoff_time
    ]
    
    # Check rate limit (100 requests per minute)
    if len(rate_limit_storage.get(client_ip, [])) >= 100:
        raise HTTPException(
            status_code=429,
            detail={
                "error": {
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": "Too many requests",
                    "details": "Rate limit of 100 requests per minute exceeded",
                    "timestamp": datetime.now().isoformat()
                }
            }
        )
    
    # Add current request
    if client_ip not in rate_limit_storage:
        rate_limit_storage[client_ip] = []
    rate_limit_storage[client_ip].append(current_time)

async def log_analytics(query_data: Dict[str, Any]):
    """Log query analytics in background"""
    try:
        analytics_data['queries'].append({
            **query_data,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 1000 queries
        if len(analytics_data['queries']) > 1000:
            analytics_data['queries'] = analytics_data['queries'][-1000:]
        
        # Update performance stats
        analytics_data['performance_stats']['total_queries'] += 1
        
        # Calculate average response time
        recent_queries = analytics_data['queries'][-100:]  # Last 100 queries
        if recent_queries:
            avg_time = sum(q.get('response_time_ms', 0) for q in recent_queries) / len(recent_queries)
            analytics_data['performance_stats']['avg_response_time_ms'] = round(avg_time, 2)
        
        # Save to file periodically
        if analytics_data['performance_stats']['total_queries'] % 10 == 0:
            analytics_file = Path("data/analytics/query_history.json")
            analytics_file.parent.mkdir(exist_ok=True)
            with open(analytics_file, 'w') as f:
                json.dump(analytics_data, f, indent=2)
                
    except Exception as e:
        logger.error(f"Failed to log analytics: {e}")

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    analytics_data['performance_stats']['error_count'] += 1
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": "HTTP_ERROR",
                "message": str(exc.detail),
                "status_code": exc.status_code,
                "timestamp": datetime.now().isoformat()
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    analytics_data['performance_stats']['error_count'] += 1
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": str(exc) if config and config.config.get('debug', False) else "Internal server error",
                "timestamp": datetime.now().isoformat()
            }
        }
    )

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    global query_handler, config
    
    try:
        logger.info("Starting Quickscene API server...")
        
        # Load configuration
        config = get_config()
        logger.info("Configuration loaded successfully")
        
        # Initialize query handler
        query_handler = ProductionQueryHandler()
        logger.info("Query handler initialized successfully")
        
        # Create analytics directory
        Path("data/analytics").mkdir(parents=True, exist_ok=True)
        
        logger.info("Quickscene API server started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

@app.get("/", include_in_schema=False)
async def root(request: Request):
    """Serve the web interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api", include_in_schema=False)
async def api_info():
    """API endpoint with basic info"""
    return {
        "name": "Quickscene API",
        "version": "1.0.0",
        "description": "Lightning-fast video timestamp retrieval",
        "docs": "/docs",
        "health": "/api/v1/health",
        "status": "/api/v1/status"
    }

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - startup_time
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        uptime_seconds=round(uptime, 2)
    )

@app.get("/api/v1/status", response_model=StatusResponse)
async def system_status():
    """Get system status and statistics"""
    try:
        if not query_handler:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": {
                        "code": "SYSTEM_NOT_READY",
                        "message": "Query handler not initialized",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            )
        
        status_data = query_handler.get_system_status()
        
        return StatusResponse(
            status=status_data.get('status', 'unknown'),
            index_loaded=status_data.get('index_loaded', False),
            metadata_loaded=status_data.get('metadata_loaded', False),
            embedder_loaded=status_data.get('embedder_loaded', False),
            total_vectors=status_data.get('total_vectors', 0),
            total_videos=status_data.get('total_videos', 0),
            total_chunks=status_data.get('total_chunks', 0),
            total_duration_seconds=status_data.get('total_duration_seconds', 0),
            index_created_at=status_data.get('index_created_at', 'unknown'),
            config=status_data.get('config', {})
        )
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "STATUS_ERROR",
                    "message": "Failed to get system status",
                    "details": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            }
        )

@app.post("/api/v1/query", dependencies=[Depends(rate_limit_check)])
async def query_videos(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    req: Request
):
    """Search for content across all videos"""
    start_time = time.time()
    
    try:
        if not query_handler:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": {
                        "code": "SYSTEM_NOT_READY",
                        "message": "Query handler not initialized",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            )
        
        # Execute query
        result = query_handler.query(
            request.query,
            top_k=request.top_k
        )
        
        response_time_ms = (time.time() - start_time) * 1000
        
        # Log analytics in background
        background_tasks.add_task(
            log_analytics,
            {
                'query': request.query,
                'top_k': request.top_k,
                'response_time_ms': response_time_ms,
                'results_count': len(result.get('results', [])),
                'client_ip': req.client.host,
                'search_type': result.get('search_type', 'unknown')
            }
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "QUERY_ERROR",
                    "message": "Query execution failed",
                    "details": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            }
        )

@app.post("/api/v1/batch-query", dependencies=[Depends(rate_limit_check)])
async def batch_query_videos(
    request: BatchQueryRequest,
    background_tasks: BackgroundTasks,
    req: Request
):
    """Process multiple queries in a single request"""
    start_time = time.time()
    
    try:
        if not query_handler:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": {
                        "code": "SYSTEM_NOT_READY",
                        "message": "Query handler not initialized",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            )
        
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        results = {}
        
        # Process each query
        for query in request.queries:
            try:
                result = query_handler.query(
                    query,
                    top_k=request.top_k
                )
                results[query] = result
                
            except Exception as e:
                logger.error(f"Batch query failed for '{query}': {e}")
                results[query] = {
                    "error": {
                        "code": "QUERY_ERROR",
                        "message": f"Query failed: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    }
                }
        
        total_time_ms = (time.time() - start_time) * 1000
        
        # Log analytics in background
        background_tasks.add_task(
            log_analytics,
            {
                'batch_id': batch_id,
                'queries': request.queries,
                'query_count': len(request.queries),
                'response_time_ms': total_time_ms,
                'client_ip': req.client.host,
                'type': 'batch'
            }
        )
        
        return {
            "batch_id": batch_id,
            "total_queries": len(request.queries),
            "results": results,
            "total_time_ms": round(total_time_ms, 2),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "BATCH_QUERY_ERROR",
                    "message": "Batch query execution failed",
                    "details": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            }
        )

@app.get("/api/v1/analytics")
async def get_analytics(
    limit: int = 100,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Get query analytics and performance metrics"""
    try:
        # Filter queries by date range if provided
        filtered_queries = analytics_data['queries']

        if start_date or end_date:
            filtered_queries = []
            for query in analytics_data['queries']:
                query_time = datetime.fromisoformat(query['timestamp'].replace('Z', '+00:00'))

                if start_date:
                    start_dt = datetime.fromisoformat(start_date)
                    if query_time < start_dt:
                        continue

                if end_date:
                    end_dt = datetime.fromisoformat(end_date)
                    if query_time > end_dt:
                        continue

                filtered_queries.append(query)

        # Limit results
        recent_queries = filtered_queries[-limit:] if limit else filtered_queries

        # Calculate performance stats
        if recent_queries:
            response_times = [q.get('response_time_ms', 0) for q in recent_queries if 'response_time_ms' in q]

            if response_times:
                perf_stats = {
                    'avg_response_time_ms': round(sum(response_times) / len(response_times), 2),
                    'min_response_time_ms': round(min(response_times), 2),
                    'max_response_time_ms': round(max(response_times), 2),
                    'queries_under_100ms': round((sum(1 for t in response_times if t < 100) / len(response_times)) * 100, 1),
                    'queries_under_1000ms': round((sum(1 for t in response_times if t < 1000) / len(response_times)) * 100, 1)
                }
            else:
                perf_stats = analytics_data['performance_stats']
        else:
            perf_stats = analytics_data['performance_stats']

        # Popular queries
        query_counts = {}
        search_types = {'keyword': 0, 'semantic': 0}

        for query in recent_queries:
            if 'query' in query:
                query_text = query['query']
                query_counts[query_text] = query_counts.get(query_text, 0) + 1

            if 'search_type' in query:
                search_type = query['search_type']
                if search_type in search_types:
                    search_types[search_type] += 1

        popular_queries = sorted(
            [{'query': q, 'count': c} for q, c in query_counts.items()],
            key=lambda x: x['count'],
            reverse=True
        )[:10]

        # Search type distribution
        total_searches = sum(search_types.values())
        search_type_dist = {}
        if total_searches > 0:
            for search_type, count in search_types.items():
                search_type_dist[search_type] = round((count / total_searches) * 100, 1)

        return {
            'total_queries': len(analytics_data['queries']),
            'date_range': {
                'start': recent_queries[0]['timestamp'] if recent_queries else None,
                'end': recent_queries[-1]['timestamp'] if recent_queries else None
            },
            'performance_stats': perf_stats,
            'popular_queries': popular_queries,
            'search_type_distribution': search_type_dist,
            'recent_queries': recent_queries[-20:]  # Last 20 queries
        }

    except Exception as e:
        logger.error(f"Analytics retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "ANALYTICS_ERROR",
                    "message": "Failed to retrieve analytics",
                    "details": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            }
        )

@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus-compatible metrics endpoint"""
    from fastapi.responses import Response

    try:
        total_queries = analytics_data['performance_stats']['total_queries']
        avg_response_time = analytics_data['performance_stats']['avg_response_time_ms'] / 1000  # Convert to seconds
        error_count = analytics_data['performance_stats']['error_count']

        metrics = f"""# HELP quickscene_queries_total Total number of queries processed
# TYPE quickscene_queries_total counter
quickscene_queries_total {total_queries}

# HELP quickscene_query_duration_seconds Average query response time in seconds
# TYPE quickscene_query_duration_seconds gauge
quickscene_query_duration_seconds {avg_response_time}

# HELP quickscene_errors_total Total number of errors
# TYPE quickscene_errors_total counter
quickscene_errors_total {error_count}

# HELP quickscene_uptime_seconds Server uptime in seconds
# TYPE quickscene_uptime_seconds gauge
quickscene_uptime_seconds {time.time() - startup_time}
"""

        return Response(content=metrics, media_type="text/plain")

    except Exception as e:
        logger.error(f"Metrics generation failed: {e}")
        return Response(content="# Error generating metrics", media_type="text/plain")

if __name__ == "__main__":
    import os

    host = os.getenv("QUICKSCENE_HOST", "0.0.0.0")
    port = int(os.getenv("QUICKSCENE_PORT", "8000"))
    debug = os.getenv("QUICKSCENE_DEBUG", "false").lower() == "true"
    workers = int(os.getenv("QUICKSCENE_WORKERS", "1"))

    logger.info(f"Starting Quickscene API server on {host}:{port}")

    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=debug,
        workers=workers if not debug else 1,
        log_level="info"
    )
