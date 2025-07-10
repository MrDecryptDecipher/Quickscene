"""
Query Handler Module for Quickscene System

Processes natural language queries and returns ranked video timestamps.
Integrates embedding generation, similarity search, and result formatting.
"""

import time
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from .embedder import Embedder
from .indexer import Indexer
from .utils.config_loader import ConfigLoader
from .utils.file_manager import FileManager
from .utils.validators import Validators


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


class QueryHandler:
    """
    Natural language query processing and search service.
    
    Features:
    - Natural language query processing
    - Real-time embedding generation
    - FAISS similarity search
    - Result ranking and filtering
    - Sub-1 second response time optimization
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the query handler with configuration.
        
        Args:
            config: Configuration dictionary containing query settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.validators = Validators()
        
        # Extract query configuration
        self.query_config = config.get('query', {})
        self.max_results = self.query_config.get('max_results', 5)
        self.similarity_threshold = self.query_config.get('similarity_threshold', 0.3)
        self.response_timeout = self.query_config.get('response_timeout_sec', 1.0)
        
        # Initialize components
        self.embedder = Embedder(config)
        self.indexer = Indexer(config)
        
        # Verify index is available
        if self.indexer.index is None:
            self.logger.warning("No search index available. Run indexing first.")
        
        self.logger.info(f"Query handler initialized: max_results={self.max_results}, "
                        f"threshold={self.similarity_threshold}, timeout={self.response_timeout}s")
    
    def process_query(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Process a natural language query and return ranked results.
        
        Args:
            query: Natural language query string
            top_k: Number of results to return (overrides config)
            
        Returns:
            List of search results with video timestamps
            
        Raises:
            ValueError: If query is invalid
            RuntimeError: If query processing fails or times out
        """
        start_time = time.time()
        
        # Validate query
        is_valid, errors = self.validators.validate_query(query)
        if not is_valid:
            raise ValueError(f"Invalid query: {errors}")
        
        # Check if index is available
        if self.indexer.index is None:
            raise RuntimeError("Search index not available. Please build index first.")
        
        # Set number of results
        k = top_k if top_k is not None else self.max_results
        
        self.logger.info(f"Processing query: '{query[:50]}...' (top-{k})")
        
        try:
            # Step 1: Generate query embedding
            embedding_start = time.time()
            query_embedding = self.embedder.embed_query(query)
            embedding_time = time.time() - embedding_start
            
            # Step 2: Search index
            search_start = time.time()
            raw_results = self.indexer.search(query_embedding, k * 2)  # Get more for filtering
            search_time = time.time() - search_start
            
            # Step 3: Filter and rank results
            filter_start = time.time()
            filtered_results = self._filter_and_rank_results(raw_results, query, k)
            filter_time = time.time() - filter_start
            
            # Step 4: Format results
            format_start = time.time()
            formatted_results = self._format_results(filtered_results)
            format_time = time.time() - format_start
            
            # Check total response time
            total_time = time.time() - start_time
            
            if total_time > self.response_timeout:
                self.logger.warning(f"Query processing exceeded timeout: {total_time:.3f}s > {self.response_timeout}s")
            
            # Log performance metrics
            self.logger.info(f"Query completed in {total_time:.3f}s: "
                           f"embed={embedding_time:.3f}s, search={search_time:.3f}s, "
                           f"filter={filter_time:.3f}s, format={format_time:.3f}s")
            
            # Validate results
            is_valid, errors = self.validators.validate_search_results(formatted_results)
            if not is_valid:
                self.logger.warning(f"Generated results are invalid: {errors}")
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            raise RuntimeError(f"Query processing failed: {e}")
    
    def _filter_and_rank_results(self, raw_results: List[Dict[str, Any]], 
                                query: str, k: int) -> List[Dict[str, Any]]:
        """
        Filter and rank search results based on relevance and quality.
        
        Args:
            raw_results: Raw search results from index
            query: Original query string
            k: Number of results to return
            
        Returns:
            Filtered and ranked results
        """
        if not raw_results:
            return []
        
        # Filter by similarity threshold
        filtered_results = [
            result for result in raw_results 
            if result['confidence'] >= self.similarity_threshold
        ]
        
        # If no results meet threshold, return top results anyway (but log warning)
        if not filtered_results and raw_results:
            self.logger.warning(f"No results above threshold {self.similarity_threshold}, "
                              f"returning top results anyway")
            filtered_results = raw_results[:k]
        
        # Additional ranking factors
        for result in filtered_results:
            result['ranking_score'] = self._calculate_ranking_score(result, query)
        
        # Sort by ranking score (descending)
        filtered_results.sort(key=lambda x: x['ranking_score'], reverse=True)
        
        # Return top k results
        return filtered_results[:k]
    
    def _calculate_ranking_score(self, result: Dict[str, Any], query: str) -> float:
        """
        Calculate a ranking score for a search result.
        
        Args:
            result: Search result dictionary
            query: Original query string
            
        Returns:
            Ranking score (higher is better)
        """
        # Start with similarity confidence
        score = result['confidence']
        
        # Boost score for longer text matches (more context)
        text_length = len(result['text'])
        if text_length > 100:
            score *= 1.1
        elif text_length < 50:
            score *= 0.9
        
        # Boost score for exact word matches
        query_words = set(query.lower().split())
        result_words = set(result['text'].lower().split())
        word_overlap = len(query_words.intersection(result_words))
        if word_overlap > 0:
            overlap_ratio = word_overlap / len(query_words)
            score *= (1.0 + overlap_ratio * 0.2)  # Up to 20% boost
        
        # Slight penalty for very short or very long chunks
        duration = result['end_time'] - result['start_time']
        if duration < 5 or duration > 30:
            score *= 0.95
        
        return score
    
    def _format_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format search results for output.
        
        Args:
            results: Filtered search results
            
        Returns:
            Formatted results
        """
        formatted_results = []
        
        for i, result in enumerate(results):
            # Calculate exact timestamp (middle of chunk for best relevance)
            exact_timestamp_seconds = (result['start_time'] + result['end_time']) / 2

            # Convert to proper video timestamp format
            video_timestamp = seconds_to_video_timestamp(exact_timestamp_seconds)
            start_timestamp = seconds_to_video_timestamp(result['start_time'])
            end_timestamp = seconds_to_video_timestamp(result['end_time'])

            formatted_result = {
                'rank': i + 1,
                'video_id': result['video_id'],
                'timestamp': video_timestamp,  # Proper video format (MM:SS or HH:MM:SS)
                'timestamp_seconds': round(exact_timestamp_seconds, 1),  # Keep for reference
                'start_time': start_timestamp,  # Video format
                'end_time': end_timestamp,      # Video format
                'start_time_seconds': round(result['start_time'], 1),  # Keep for reference
                'end_time_seconds': round(result['end_time'], 1),      # Keep for reference
                'duration': round(result['end_time'] - result['start_time'], 1),
                'confidence': round(result['confidence'], 3),
                'dialogue': result['text'].strip(),  # Exact spoken words
                'chunk_id': result.get('chunk_id', ''),
                'created_at': datetime.now().isoformat()
            }
            
            # Add ranking score if available
            if 'ranking_score' in result:
                formatted_result['ranking_score'] = round(result['ranking_score'], 3)
            
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def batch_query(self, queries: List[str], top_k: Optional[int] = None) -> List[List[Dict[str, Any]]]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of query strings
            top_k: Number of results per query
            
        Returns:
            List of result lists (one per query)
        """
        if not queries:
            return []
        
        self.logger.info(f"Processing batch of {len(queries)} queries")
        
        all_results = []
        failed_queries = []
        
        for i, query in enumerate(queries, 1):
            try:
                self.logger.debug(f"Processing query {i}/{len(queries)}: '{query[:30]}...'")
                results = self.process_query(query, top_k)
                all_results.append(results)
                
            except Exception as e:
                self.logger.error(f"Failed to process query {i}: {e}")
                failed_queries.append(query)
                all_results.append([])  # Empty results for failed query
        
        # Log summary
        success_count = len(queries) - len(failed_queries)
        self.logger.info(f"Batch query complete: {success_count}/{len(queries)} successful")
        
        if failed_queries:
            self.logger.warning(f"Failed queries: {failed_queries}")
        
        return all_results
    
    def get_similar_chunks(self, video_id: str, start_time: float, 
                          end_time: float, k: int = 3) -> List[Dict[str, Any]]:
        """
        Find chunks similar to a specific video segment.
        
        Args:
            video_id: Video identifier
            start_time: Start time of the segment
            end_time: End time of the segment
            k: Number of similar chunks to return
            
        Returns:
            List of similar chunks
        """
        # Find the chunk that contains this time segment
        target_chunk = None
        
        for metadata in self.indexer.metadata:
            if (metadata['video_id'] == video_id and 
                metadata['start_time'] <= start_time and 
                metadata['end_time'] >= end_time):
                target_chunk = metadata
                break
        
        if not target_chunk:
            self.logger.warning(f"No chunk found for {video_id} at {start_time}-{end_time}")
            return []
        
        # Use the chunk text as a query
        query = target_chunk['text']
        
        try:
            results = self.process_query(query, k + 1)  # +1 to exclude self
            
            # Filter out the original chunk
            filtered_results = [
                result for result in results 
                if not (result['video_id'] == video_id and 
                       result['start_time'] == target_chunk['start_time'])
            ]
            
            return filtered_results[:k]
            
        except Exception as e:
            self.logger.error(f"Failed to find similar chunks: {e}")
            return []
    
    def get_query_suggestions(self, partial_query: str, max_suggestions: int = 5) -> List[str]:
        """
        Get query suggestions based on partial input.
        
        Args:
            partial_query: Partial query string
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List of suggested queries
        """
        # Simple implementation - could be enhanced with more sophisticated NLP
        suggestions = []
        
        if len(partial_query) < 3:
            return suggestions
        
        # Search for chunks containing the partial query
        try:
            results = self.process_query(partial_query, max_suggestions * 2)
            
            # Extract unique phrases from results
            phrases = set()
            for result in results:
                text = result['text'].lower()
                words = text.split()
                
                # Find phrases containing the partial query
                partial_lower = partial_query.lower()
                for i in range(len(words) - 2):
                    phrase = ' '.join(words[i:i+3])
                    if partial_lower in phrase and len(phrase) > len(partial_query):
                        phrases.add(phrase)
            
            suggestions = list(phrases)[:max_suggestions]
            
        except Exception as e:
            self.logger.debug(f"Failed to generate suggestions: {e}")
        
        return suggestions
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get status information about the query system.
        
        Returns:
            Dictionary containing system status
        """
        index_stats = self.indexer.get_index_stats()
        embedder_info = self.embedder.get_model_info()
        
        status = {
            'query_handler': {
                'max_results': self.max_results,
                'similarity_threshold': self.similarity_threshold,
                'response_timeout': self.response_timeout
            },
            'index': index_stats,
            'embedder': embedder_info,
            'ready_for_queries': index_stats['index_exists'] and index_stats['total_vectors'] > 0
        }
        
        return status
    
    def warm_up(self) -> None:
        """
        Warm up the query system with a test query to improve first-query performance.
        """
        if self.indexer.index is None:
            self.logger.warning("Cannot warm up: no index available")
            return
        
        try:
            self.logger.info("Warming up query system...")
            test_query = "test query for system warmup"
            self.process_query(test_query, top_k=1)
            self.logger.info("Query system warmed up successfully")
            
        except Exception as e:
            self.logger.warning(f"Warm up failed: {e}")
    
    def clear_caches(self) -> None:
        """Clear all caches to free memory."""
        self.embedder.clear_cache()
        self.logger.info("Query handler caches cleared")
