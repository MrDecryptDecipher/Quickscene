#!/usr/bin/env python3
"""
Production Query Handler

Handles natural language queries against the unified FAISS index.
Searches across ALL videos simultaneously and returns best matches.
Follows PRD specifications for sub-1 second query response.
"""

import json
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import faiss
from sentence_transformers import SentenceTransformer

from .config import get_config

def seconds_to_video_timestamp(seconds: float, include_milliseconds: bool = False) -> str:
    """
    Convert seconds to standard video timestamp format.
    
    Args:
        seconds: Time in seconds (e.g., 575.1)
        include_milliseconds: Whether to include milliseconds
        
    Returns:
        Formatted timestamp:
        - Under 1 hour: MM:SS (e.g., "9:35")
        - Over 1 hour: HH:MM:SS (e.g., "1:05:22")
        - With milliseconds: HH:MM:SS.mmm (e.g., "1:05:22.500")
    """
    total_seconds = int(seconds)
    milliseconds = int((seconds - total_seconds) * 1000)
    
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    
    if hours > 0:
        # Format: HH:MM:SS or HH:MM:SS.mmm
        if include_milliseconds and milliseconds > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
        else:
            return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        # Format: MM:SS or MM:SS.mmm
        if include_milliseconds and milliseconds > 0:
            return f"{minutes}:{secs:02d}.{milliseconds:03d}"
        else:
            return f"{minutes}:{secs:02d}"

class ProductionQueryHandler:
    """
    Production query handler that searches across all videos.
    Implements the query_handler.py module as specified in PRD.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize production query handler with config"""
        self.config = get_config(config_path)
        self.logger = logging.getLogger(__name__)
        self.index = None
        self.metadata = None
        self.embedder = None
        self._load_index_and_model()
    
    def _load_index_and_model(self):
        """Load FAISS index, metadata, and embedding model"""
        try:
            # Load FAISS index and metadata
            if not self.config.faiss_index_path.exists():
                raise FileNotFoundError(f"FAISS index not found: {self.config.faiss_index_path}")
            
            if not self.config.metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {self.config.metadata_path}")
            
            self.index = faiss.read_index(str(self.config.faiss_index_path))
            
            with open(self.config.metadata_path, 'r') as f:
                metadata_file = json.load(f)
            
            self.metadata = metadata_file.get('metadata', [])
            
            # Load embedding model
            self.embedder = SentenceTransformer(self.config.embedding_model)
            
            self.logger.info(f"Loaded index with {self.index.ntotal} vectors and {len(self.metadata)} metadata entries")
            
        except Exception as e:
            self.logger.error(f"Failed to load index and model: {e}")
            raise
    
    def _embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for query text.
        
        Args:
            query: Natural language query
            
        Returns:
            Query embedding as numpy array
        """
        embedding = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.astype(np.float32)
    
    def _is_keyword_search(self, query: str) -> bool:
        """Check if query is a single keyword search"""
        return len(query.strip().split()) == 1
    
    def _get_keyword_variations(self, keyword: str) -> List[str]:
        """Get variations of a keyword for better matching"""
        keyword_lower = keyword.lower()
        variations = [keyword_lower]
        
        # Add common variations
        if keyword_lower == "superintelligence":
            variations.extend(["super-intelligence", "super intelligence"])
        elif keyword_lower == "super-intelligence":
            variations.extend(["superintelligence", "super intelligence"])
        elif keyword_lower == "ai":
            variations.extend(["artificial intelligence", "a.i.", "a i"])
        elif keyword_lower == "machine":
            variations.extend(["machines", "machinery"])
        
        return variations
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search across all chunks.
        
        Args:
            query: Keyword query
            top_k: Number of results to return
            
        Returns:
            List of matching results
        """
        keyword_variations = self._get_keyword_variations(query)
        matches = []
        
        for i, metadata in enumerate(self.metadata):
            text_lower = metadata['text'].lower()
            
            # Check for any variation match
            best_score = 0
            best_variation = None
            
            for variation in keyword_variations:
                if variation in text_lower:
                    # Calculate relevance score
                    count = text_lower.count(variation)
                    position = text_lower.find(variation)
                    score = count * 0.5 + (1.0 - position / len(text_lower)) * 0.5
                    
                    # Bonus for exact match
                    if variation == query.lower():
                        score += 0.5
                    
                    if score > best_score:
                        best_score = score
                        best_variation = variation
            
            if best_variation:
                # Calculate exact timestamp
                exact_timestamp = (metadata['start_time'] + metadata['end_time']) / 2
                
                match = {
                    'rank': 0,  # Will be set later
                    'video_id': metadata['video_id'],
                    'chunk_id': metadata['chunk_id'],
                    'timestamp': seconds_to_video_timestamp(exact_timestamp),
                    'timestamp_seconds': exact_timestamp,
                    'start_time': seconds_to_video_timestamp(metadata['start_time']),
                    'end_time': seconds_to_video_timestamp(metadata['end_time']),
                    'start_time_seconds': metadata['start_time'],
                    'end_time_seconds': metadata['end_time'],
                    'confidence': best_score,
                    'dialogue': metadata['text'],
                    'matched_term': best_variation,
                    'search_type': 'keyword'
                }
                matches.append(match)
        
        # Sort by relevance score and return top_k
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Set ranks
        for i, match in enumerate(matches[:top_k]):
            match['rank'] = i + 1
        
        return matches[:top_k]
    
    def _semantic_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Perform semantic search using FAISS index.
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            
        Returns:
            List of matching results
        """
        # Generate query embedding
        query_embedding = self._embed_query(query)
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0 and idx < len(self.metadata):
                metadata = self.metadata[idx]
                
                # Calculate exact timestamp
                exact_timestamp = (metadata['start_time'] + metadata['end_time']) / 2
                
                result = {
                    'rank': i + 1,
                    'video_id': metadata['video_id'],
                    'chunk_id': metadata['chunk_id'],
                    'timestamp': seconds_to_video_timestamp(exact_timestamp),
                    'timestamp_seconds': exact_timestamp,
                    'start_time': seconds_to_video_timestamp(metadata['start_time']),
                    'end_time': seconds_to_video_timestamp(metadata['end_time']),
                    'start_time_seconds': metadata['start_time'],
                    'end_time_seconds': metadata['end_time'],
                    'confidence': float(score),
                    'dialogue': metadata['text'],
                    'search_type': 'semantic'
                }
                results.append(result)
        
        return results
    
    def query(self, query_text: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Process natural language query and return results.
        
        Args:
            query_text: Natural language query
            top_k: Number of results to return (defaults to config)
            
        Returns:
            Query results with metadata
        """
        if top_k is None:
            top_k = self.config.default_top_k
        
        start_time = time.time()
        
        # Determine search type
        is_keyword = self._is_keyword_search(query_text)
        
        try:
            if is_keyword:
                # Use keyword search for single words
                results = self._keyword_search(query_text, top_k)
            else:
                # Use semantic search for phrases
                results = self._semantic_search(query_text, top_k)
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in results 
                if result['confidence'] >= self.config.similarity_threshold
            ]
            
            query_time_ms = (time.time() - start_time) * 1000
            
            return {
                'query': query_text,
                'search_type': 'keyword' if is_keyword else 'semantic',
                'results': filtered_results,
                'total_results': len(filtered_results),
                'query_time_ms': query_time_ms,
                'timestamp': datetime.now().isoformat(),
                'performance': {
                    'meets_requirement': query_time_ms < 1000,  # <1 second requirement
                    'target_ms': 1000,
                    'actual_ms': query_time_ms
                }
            }
            
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            return {
                'query': query_text,
                'error': str(e),
                'results': [],
                'total_results': 0,
                'query_time_ms': (time.time() - start_time) * 1000,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and statistics"""
        try:
            # Load metadata file for additional info
            with open(self.config.metadata_path, 'r') as f:
                metadata_file = json.load(f)
            
            video_ids = set(item['video_id'] for item in self.metadata)
            total_duration = sum(item['duration'] for item in self.metadata)
            
            return {
                'status': 'ready',
                'index_loaded': self.index is not None,
                'metadata_loaded': self.metadata is not None,
                'embedder_loaded': self.embedder is not None,
                'total_vectors': self.index.ntotal if self.index else 0,
                'total_videos': len(video_ids),
                'total_chunks': len(self.metadata),
                'total_duration_seconds': total_duration,
                'index_created_at': metadata_file.get('created_at', 'unknown'),
                'config': {
                    'embedding_model': self.config.embedding_model,
                    'default_top_k': self.config.default_top_k,
                    'similarity_threshold': self.config.similarity_threshold
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

def main():
    """CLI interface for production query handler"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Production Query Handler")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--query", help="Execute single query")
    parser.add_argument("--top-k", type=int, help="Number of results to return")
    parser.add_argument("--interactive", action="store_true", help="Interactive query mode")
    
    args = parser.parse_args()
    
    try:
        handler = ProductionQueryHandler(args.config)
        
        if args.status:
            status = handler.get_system_status()
            print(json.dumps(status, indent=2))
            return
        
        if args.query:
            result = handler.query(args.query, args.top_k)
            print(json.dumps(result, indent=2))
            return
        
        if args.interactive:
            print("🎮 Interactive Query Mode")
            print("Enter queries (or 'quit' to exit):")
            
            while True:
                try:
                    query = input("\n🔍 Query: ").strip()
                    if query.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    if not query:
                        continue
                    
                    result = handler.query(query, args.top_k)
                    
                    print(f"⚡ Response time: {result['query_time_ms']:.1f}ms")
                    print(f"🔍 Search type: {result.get('search_type', 'unknown')}")
                    
                    if result['total_results'] > 0:
                        print(f"\n📋 Results ({result['total_results']}):")
                        for res in result['results']:
                            print(f"   {res['rank']}. 🎬 {res['video_id']}")
                            print(f"      ⏰ {res['timestamp']}")
                            print(f"      💬 \"{res['dialogue'][:100]}...\"")
                            print(f"      🎯 Confidence: {res['confidence']:.3f}")
                            print()
                    else:
                        print("❌ No results found")
                
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
            
            print("\n👋 Goodbye!")
        
        else:
            print("Use --query, --status, or --interactive")
    
    except Exception as e:
        print(f"❌ Failed to initialize query handler: {e}")

if __name__ == "__main__":
    main()
