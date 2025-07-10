"""
Quickscene: Video Timestamp-Retrieval System

A production-ready backend system that accepts natural language queries
and returns accurate video timestamps using transcript-based search.

Core modules:
- transcriber: Video-to-transcript conversion with timestamps
- chunker: Transcript segmentation into searchable chunks  
- embedder: Text-to-vector embedding generation
- indexer: FAISS index construction and management
- query_handler: Natural language query processing and response
"""

__version__ = "1.0.0"
__author__ = "Quickscene Development Team"

# Core module imports - lazy loading to handle missing dependencies
__all__ = [
    "Transcriber",
    "Chunker",
    "Embedder",
    "Indexer",
    "QueryHandler"
]

def __getattr__(name):
    """Lazy import to handle missing dependencies gracefully."""
    if name == "Transcriber":
        from .transcriber import Transcriber
        return Transcriber
    elif name == "Chunker":
        from .chunker import Chunker
        return Chunker
    elif name == "Embedder":
        from .embedder import Embedder
        return Embedder
    elif name == "Indexer":
        from .indexer import Indexer
        return Indexer
    elif name == "QueryHandler":
        from .query_handler import QueryHandler
        return QueryHandler
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
