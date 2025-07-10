"""
Data Validators for Quickscene System

Provides validation functions for data formats, structures, and constraints.
Ensures data integrity throughout the processing pipeline.
"""

import re
import logging
import numpy as np
from typing import Dict, Any, List, Union, Optional, Tuple
from pathlib import Path


class Validators:
    """
    Comprehensive validation utilities for Quickscene system.
    
    Features:
    - Transcript format validation
    - Chunk structure validation
    - Embedding format validation
    - Configuration parameter validation
    - File format validation
    """
    
    def __init__(self):
        """Initialize the validators."""
        self.logger = logging.getLogger(__name__)
    
    def validate_transcript(self, transcript: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate transcript structure and content.
        
        Args:
            transcript: Transcript dictionary to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required top-level fields
        required_fields = ['video_id', 'duration', 'segments']
        for field in required_fields:
            if field not in transcript:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return False, errors
        
        # Validate video_id
        if not isinstance(transcript['video_id'], str) or not transcript['video_id'].strip():
            errors.append("video_id must be a non-empty string")
        
        # Validate duration
        if not isinstance(transcript['duration'], (int, float)) or transcript['duration'] <= 0:
            errors.append("duration must be a positive number")
        
        # Validate segments
        if not isinstance(transcript['segments'], list):
            errors.append("segments must be a list")
        else:
            segment_errors = self._validate_segments(transcript['segments'])
            errors.extend(segment_errors)
        
        # Check for temporal consistency
        if not errors:
            temporal_errors = self._validate_temporal_consistency(transcript['segments'])
            errors.extend(temporal_errors)
        
        is_valid = len(errors) == 0
        if is_valid:
            self.logger.debug(f"Transcript validation passed for {transcript['video_id']}")
        else:
            self.logger.warning(f"Transcript validation failed: {errors}")
        
        return is_valid, errors
    
    def _validate_segments(self, segments: List[Dict[str, Any]]) -> List[str]:
        """
        Validate individual transcript segments.
        
        Args:
            segments: List of segment dictionaries
            
        Returns:
            List of validation errors
        """
        errors = []
        
        if not segments:
            errors.append("segments list cannot be empty")
            return errors
        
        required_segment_fields = ['start', 'end', 'text']
        
        for i, segment in enumerate(segments):
            if not isinstance(segment, dict):
                errors.append(f"Segment {i} must be a dictionary")
                continue
            
            # Check required fields
            for field in required_segment_fields:
                if field not in segment:
                    errors.append(f"Segment {i} missing required field: {field}")
            
            # Validate start time
            if 'start' in segment:
                if not isinstance(segment['start'], (int, float)) or segment['start'] < 0:
                    errors.append(f"Segment {i} start time must be a non-negative number")
            
            # Validate end time
            if 'end' in segment:
                if not isinstance(segment['end'], (int, float)) or segment['end'] < 0:
                    errors.append(f"Segment {i} end time must be a non-negative number")
            
            # Validate text
            if 'text' in segment:
                if not isinstance(segment['text'], str):
                    errors.append(f"Segment {i} text must be a string")
                elif not segment['text'].strip():
                    errors.append(f"Segment {i} text cannot be empty")
            
            # Validate start < end
            if 'start' in segment and 'end' in segment:
                if segment['start'] >= segment['end']:
                    errors.append(f"Segment {i} start time must be less than end time")
        
        return errors
    
    def _validate_temporal_consistency(self, segments: List[Dict[str, Any]]) -> List[str]:
        """
        Validate temporal consistency across segments.
        
        Args:
            segments: List of segment dictionaries
            
        Returns:
            List of validation errors
        """
        errors = []
        
        if len(segments) < 2:
            return errors
        
        # Check for temporal ordering and overlaps
        for i in range(len(segments) - 1):
            current_segment = segments[i]
            next_segment = segments[i + 1]
            
            if 'end' not in current_segment or 'start' not in next_segment:
                continue
            
            # Check for proper ordering
            if current_segment['end'] > next_segment['start']:
                errors.append(f"Segments {i} and {i+1} have temporal overlap")
        
        return errors
    
    def validate_chunks(self, chunks: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate chunk structure and content.
        
        Args:
            chunks: List of chunk dictionaries to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not isinstance(chunks, list):
            errors.append("chunks must be a list")
            return False, errors
        
        if not chunks:
            errors.append("chunks list cannot be empty")
            return False, errors
        
        required_chunk_fields = ['video_id', 'chunk_id', 'start_time', 'end_time', 'text']
        
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                errors.append(f"Chunk {i} must be a dictionary")
                continue
            
            # Check required fields
            for field in required_chunk_fields:
                if field not in chunk:
                    errors.append(f"Chunk {i} missing required field: {field}")
            
            # Validate video_id
            if 'video_id' in chunk:
                if not isinstance(chunk['video_id'], str) or not chunk['video_id'].strip():
                    errors.append(f"Chunk {i} video_id must be a non-empty string")
            
            # Validate chunk_id
            if 'chunk_id' in chunk:
                if not isinstance(chunk['chunk_id'], (str, int)):
                    errors.append(f"Chunk {i} chunk_id must be a string or integer")
            
            # Validate timing
            if 'start_time' in chunk and 'end_time' in chunk:
                start_time = chunk['start_time']
                end_time = chunk['end_time']
                
                if not isinstance(start_time, (int, float)) or start_time < 0:
                    errors.append(f"Chunk {i} start_time must be a non-negative number")
                
                if not isinstance(end_time, (int, float)) or end_time < 0:
                    errors.append(f"Chunk {i} end_time must be a non-negative number")
                
                if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
                    if start_time >= end_time:
                        errors.append(f"Chunk {i} start_time must be less than end_time")
                    
                    duration = end_time - start_time
                    if duration > 30:  # Maximum reasonable chunk duration
                        errors.append(f"Chunk {i} duration ({duration:.1f}s) exceeds maximum (30s)")
            
            # Validate text
            if 'text' in chunk:
                if not isinstance(chunk['text'], str):
                    errors.append(f"Chunk {i} text must be a string")
                elif not chunk['text'].strip():
                    errors.append(f"Chunk {i} text cannot be empty")
        
        is_valid = len(errors) == 0
        if is_valid:
            self.logger.debug(f"Chunks validation passed for {len(chunks)} chunks")
        else:
            self.logger.warning(f"Chunks validation failed: {errors}")
        
        return is_valid, errors
    
    def validate_embeddings(self, embeddings: np.ndarray, expected_dim: int = 384) -> Tuple[bool, List[str]]:
        """
        Validate embedding array format and dimensions.
        
        Args:
            embeddings: NumPy array of embeddings
            expected_dim: Expected embedding dimension
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check if it's a NumPy array
        if not isinstance(embeddings, np.ndarray):
            errors.append("embeddings must be a NumPy array")
            return False, errors
        
        # Check dimensions
        if embeddings.ndim != 2:
            errors.append(f"embeddings must be 2-dimensional, got {embeddings.ndim}")
        
        # Check shape
        if embeddings.ndim == 2:
            num_embeddings, embedding_dim = embeddings.shape
            
            if embedding_dim != expected_dim:
                errors.append(f"embedding dimension must be {expected_dim}, got {embedding_dim}")
            
            if num_embeddings == 0:
                errors.append("embeddings array cannot be empty")
        
        # Check data type
        if embeddings.dtype != np.float32:
            errors.append(f"embeddings must be float32, got {embeddings.dtype}")
        
        # Check for invalid values
        if embeddings.size > 0:
            if np.any(np.isnan(embeddings)):
                errors.append("embeddings contain NaN values")
            
            if np.any(np.isinf(embeddings)):
                errors.append("embeddings contain infinite values")
        
        is_valid = len(errors) == 0
        if is_valid:
            self.logger.debug(f"Embeddings validation passed: shape {embeddings.shape}")
        else:
            self.logger.warning(f"Embeddings validation failed: {errors}")
        
        return is_valid, errors
    
    def validate_query(self, query: str) -> Tuple[bool, List[str]]:
        """
        Validate search query format and content.
        
        Args:
            query: Search query string
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check if query is a string
        if not isinstance(query, str):
            errors.append("query must be a string")
            return False, errors
        
        # Check if query is not empty
        if not query.strip():
            errors.append("query cannot be empty")
            return False, errors
        
        # Check query length
        if len(query) > 1000:
            errors.append("query is too long (maximum 1000 characters)")
        
        if len(query.strip()) < 3:
            errors.append("query is too short (minimum 3 characters)")
        
        # Check for potentially problematic characters
        if re.search(r'[^\w\s\-\.\,\?\!\:\;\(\)\'\"]+', query):
            errors.append("query contains unsupported special characters")
        
        is_valid = len(errors) == 0
        if is_valid:
            self.logger.debug(f"Query validation passed: '{query[:50]}...'")
        else:
            self.logger.warning(f"Query validation failed: {errors}")
        
        return is_valid, errors
    
    def validate_video_file(self, file_path: Union[str, Path]) -> Tuple[bool, List[str]]:
        """
        Validate video file existence and format.
        
        Args:
            file_path: Path to video file
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            errors.append(f"Video file does not exist: {file_path}")
            return False, errors
        
        # Check if it's a file (not directory)
        if not file_path.is_file():
            errors.append(f"Path is not a file: {file_path}")
            return False, errors
        
        # Check file extension
        supported_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        if file_path.suffix.lower() not in supported_extensions:
            errors.append(f"Unsupported video format: {file_path.suffix}")
        
        # Check file size
        try:
            file_size = file_path.stat().st_size
            if file_size == 0:
                errors.append("Video file is empty")
            elif file_size > 10 * 1024 * 1024 * 1024:  # 10GB limit
                errors.append("Video file is too large (>10GB)")
        except OSError as e:
            errors.append(f"Cannot access video file: {e}")
        
        is_valid = len(errors) == 0
        if is_valid:
            self.logger.debug(f"Video file validation passed: {file_path}")
        else:
            self.logger.warning(f"Video file validation failed: {errors}")
        
        return is_valid, errors
    
    def validate_search_results(self, results: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate search results format and content.
        
        Args:
            results: List of search result dictionaries
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not isinstance(results, list):
            errors.append("results must be a list")
            return False, errors
        
        required_result_fields = ['video_id', 'start_time', 'end_time', 'confidence', 'text']
        
        for i, result in enumerate(results):
            if not isinstance(result, dict):
                errors.append(f"Result {i} must be a dictionary")
                continue
            
            # Check required fields
            for field in required_result_fields:
                if field not in result:
                    errors.append(f"Result {i} missing required field: {field}")
            
            # Validate confidence score
            if 'confidence' in result:
                confidence = result['confidence']
                if not isinstance(confidence, (int, float)):
                    errors.append(f"Result {i} confidence must be a number")
                elif not 0 <= confidence <= 1:
                    errors.append(f"Result {i} confidence must be between 0 and 1")
            
            # Validate timing
            if 'start_time' in result and 'end_time' in result:
                start_time = result['start_time']
                end_time = result['end_time']
                
                if not isinstance(start_time, (int, float)) or start_time < 0:
                    errors.append(f"Result {i} start_time must be a non-negative number")
                
                if not isinstance(end_time, (int, float)) or end_time < 0:
                    errors.append(f"Result {i} end_time must be a non-negative number")
                
                if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
                    if start_time >= end_time:
                        errors.append(f"Result {i} start_time must be less than end_time")
        
        is_valid = len(errors) == 0
        if is_valid:
            self.logger.debug(f"Search results validation passed for {len(results)} results")
        else:
            self.logger.warning(f"Search results validation failed: {errors}")
        
        return is_valid, errors
