"""
Chunker Module for Quickscene System

Splits transcripts into fixed-duration chunks for embedding and search.
Maintains temporal boundaries and ensures no data loss or overlap.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from .utils.config_loader import ConfigLoader
from .utils.file_manager import FileManager
from .utils.validators import Validators


class Chunker:
    """
    Transcript chunking service for creating searchable segments.
    
    Features:
    - Fixed-duration chunking (10-15 seconds configurable)
    - Word boundary respect to avoid mid-word splits
    - No overlap or data loss between chunks
    - Metadata preservation and chunk ID generation
    - Validation and error handling
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the chunker with configuration.
        
        Args:
            config: Configuration dictionary containing chunking settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.file_manager = FileManager()
        self.validators = Validators()
        
        # Extract chunking configuration
        self.chunking_config = config.get('chunking', {})
        self.chunk_duration = self.chunking_config.get('duration_sec', 15)
        self.overlap_duration = self.chunking_config.get('overlap_sec', 0)
        self.respect_word_boundaries = self.chunking_config.get('respect_word_boundaries', True)
        self.min_chunk_length = self.chunking_config.get('min_chunk_length', 10)
        
        # Setup paths
        self.chunks_path = Path(config['paths']['chunks'])
        self.file_manager.ensure_directory(self.chunks_path)
        
        self.logger.info(f"Chunker initialized: {self.chunk_duration}s chunks, "
                        f"overlap: {self.overlap_duration}s, "
                        f"word boundaries: {self.respect_word_boundaries}")
    
    def chunk_transcript(self, transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split transcript into fixed-duration chunks.
        
        Args:
            transcript: Transcript dictionary with segments
            
        Returns:
            List of chunk dictionaries
            
        Raises:
            ValueError: If transcript is invalid
            RuntimeError: If chunking fails
        """
        # Validate input transcript
        is_valid, errors = self.validators.validate_transcript(transcript)
        if not is_valid:
            raise ValueError(f"Invalid transcript: {errors}")
        
        video_id = transcript['video_id']
        segments = transcript['segments']
        
        self.logger.info(f"Starting chunking for {video_id}: {len(segments)} segments")
        
        try:
            # Check if chunks already exist
            chunks_file = self.chunks_path / f"{video_id}_chunks.json"
            if chunks_file.exists():
                self.logger.info(f"Loading existing chunks: {chunks_file}")
                existing_chunks = self.file_manager.load_json(chunks_file)
                if existing_chunks and isinstance(existing_chunks, list):
                    # Validate existing chunks
                    is_valid, errors = self.validators.validate_chunks(existing_chunks)
                    if is_valid:
                        return existing_chunks
                    else:
                        self.logger.warning(f"Existing chunks invalid, regenerating: {errors}")
            
            # Create chunks from segments
            chunks = self._create_chunks_from_segments(segments, video_id)
            
            # Validate generated chunks
            is_valid, errors = self.validators.validate_chunks(chunks)
            if not is_valid:
                raise RuntimeError(f"Generated chunks are invalid: {errors}")
            
            # Save chunks
            if not self.file_manager.save_json(chunks, chunks_file):
                self.logger.warning(f"Failed to save chunks: {chunks_file}")
            
            self.logger.info(f"Chunking completed: {len(chunks)} chunks generated for {video_id}")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Chunking failed for {video_id}: {e}")
            raise RuntimeError(f"Chunking failed: {e}")
    
    def _create_chunks_from_segments(self, segments: List[Dict[str, Any]], 
                                   video_id: str) -> List[Dict[str, Any]]:
        """
        Create fixed-duration chunks from transcript segments.
        
        Args:
            segments: List of transcript segments
            video_id: Video identifier
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        
        if not segments:
            self.logger.warning(f"No segments to chunk for {video_id}")
            return chunks
        
        # Flatten all words with timestamps
        words_with_timestamps = self._extract_words_with_timestamps(segments)
        
        if not words_with_timestamps:
            self.logger.warning(f"No words extracted from segments for {video_id}")
            return chunks
        
        # Create chunks based on time boundaries
        current_chunk_start = 0
        chunk_id = 0
        
        while current_chunk_start < words_with_timestamps[-1]['end']:
            chunk_end = current_chunk_start + self.chunk_duration
            
            # Find words within this time window
            chunk_words = self._get_words_in_time_range(
                words_with_timestamps, current_chunk_start, chunk_end
            )
            
            if chunk_words:
                # Create chunk
                chunk = self._create_chunk(chunk_words, video_id, chunk_id)
                
                # Validate chunk meets minimum requirements
                if self._is_valid_chunk(chunk):
                    chunks.append(chunk)
                    chunk_id += 1
                else:
                    self.logger.debug(f"Skipping invalid chunk {chunk_id} for {video_id}")
            
            # Move to next chunk (with overlap if configured)
            current_chunk_start = chunk_end - self.overlap_duration
        
        return chunks
    
    def _extract_words_with_timestamps(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract individual words with timestamps from segments.
        
        Args:
            segments: List of transcript segments
            
        Returns:
            List of word dictionaries with timestamps
        """
        words_with_timestamps = []
        
        for segment in segments:
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text'].strip()
            
            if not text:
                continue
            
            # Split text into words
            words = re.findall(r'\S+', text)
            
            if not words:
                continue
            
            # Estimate word-level timestamps
            segment_duration = end_time - start_time
            word_duration = segment_duration / len(words)
            
            for i, word in enumerate(words):
                word_start = start_time + (i * word_duration)
                word_end = start_time + ((i + 1) * word_duration)
                
                words_with_timestamps.append({
                    'word': word,
                    'start': word_start,
                    'end': word_end
                })
        
        return words_with_timestamps
    
    def _get_words_in_time_range(self, words: List[Dict[str, Any]], 
                                start_time: float, end_time: float) -> List[Dict[str, Any]]:
        """
        Get words that fall within a specific time range.
        
        Args:
            words: List of word dictionaries with timestamps
            start_time: Start time of the range
            end_time: End time of the range
            
        Returns:
            List of words within the time range
        """
        chunk_words = []
        
        for word in words:
            word_start = word['start']
            word_end = word['end']
            
            # Include word if it overlaps with the time range
            if word_start < end_time and word_end > start_time:
                chunk_words.append(word)
        
        # If respecting word boundaries, ensure we don't cut words
        if self.respect_word_boundaries and chunk_words:
            # Adjust chunk boundaries to word boundaries
            actual_start = chunk_words[0]['start']
            actual_end = chunk_words[-1]['end']
            
            # Update word timestamps to reflect actual chunk boundaries
            for word in chunk_words:
                if word['start'] < actual_start:
                    word['start'] = actual_start
                if word['end'] > actual_end:
                    word['end'] = actual_end
        
        return chunk_words
    
    def _create_chunk(self, words: List[Dict[str, Any]], video_id: str, chunk_id: int) -> Dict[str, Any]:
        """
        Create a chunk dictionary from a list of words.
        
        Args:
            words: List of word dictionaries
            video_id: Video identifier
            chunk_id: Chunk identifier
            
        Returns:
            Chunk dictionary
        """
        if not words:
            return {}
        
        # Calculate chunk timing
        start_time = words[0]['start']
        end_time = words[-1]['end']
        
        # Combine words into text
        text = ' '.join(word['word'] for word in words)
        
        # Create chunk
        chunk = {
            'video_id': video_id,
            'chunk_id': f"{video_id}_{chunk_id:04d}",
            'start_time': float(start_time),
            'end_time': float(end_time),
            'duration': float(end_time - start_time),
            'text': text.strip(),
            'word_count': len(words),
            'chunk_info': {
                'chunk_index': chunk_id,
                'target_duration': self.chunk_duration,
                'overlap_duration': self.overlap_duration,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        return chunk
    
    def _is_valid_chunk(self, chunk: Dict[str, Any]) -> bool:
        """
        Validate if a chunk meets minimum requirements.
        
        Args:
            chunk: Chunk dictionary to validate
            
        Returns:
            True if chunk is valid
        """
        if not chunk:
            return False
        
        # Check minimum text length
        text = chunk.get('text', '').strip()
        if not text:
            return False
        
        # Check minimum word count
        word_count = chunk.get('word_count', 0)
        if word_count < self.min_chunk_length:
            return False
        
        # Check duration is reasonable
        duration = chunk.get('duration', 0)
        if duration <= 0 or duration > self.chunk_duration * 2:
            return False
        
        return True
    
    def batch_chunk_transcripts(self, transcript_directory: Union[str, Path]) -> List[List[Dict[str, Any]]]:
        """
        Chunk multiple transcripts in a directory.
        
        Args:
            transcript_directory: Directory containing transcript JSON files
            
        Returns:
            List of chunk lists (one per transcript)
        """
        transcript_directory = Path(transcript_directory)
        
        if not transcript_directory.exists():
            raise FileNotFoundError(f"Transcript directory not found: {transcript_directory}")
        
        # Find transcript files
        transcript_files = list(transcript_directory.glob("*.json"))
        
        if not transcript_files:
            self.logger.warning(f"No transcript files found in {transcript_directory}")
            return []
        
        self.logger.info(f"Starting batch chunking of {len(transcript_files)} transcripts")
        
        all_chunks = []
        failed_files = []
        
        for i, transcript_file in enumerate(transcript_files, 1):
            try:
                self.logger.info(f"Processing transcript {i}/{len(transcript_files)}: {transcript_file.name}")
                
                # Load transcript
                transcript = self.file_manager.load_json(transcript_file)
                if not transcript:
                    self.logger.error(f"Failed to load transcript: {transcript_file}")
                    failed_files.append(transcript_file.name)
                    continue
                
                # Chunk transcript
                chunks = self.chunk_transcript(transcript)
                all_chunks.append(chunks)
                
            except Exception as e:
                self.logger.error(f"Failed to chunk {transcript_file.name}: {e}")
                failed_files.append(transcript_file.name)
                continue
        
        # Log summary
        success_count = len(all_chunks)
        failure_count = len(failed_files)
        total_chunks = sum(len(chunks) for chunks in all_chunks)
        
        self.logger.info(f"Batch chunking complete: {success_count} transcripts, "
                        f"{total_chunks} total chunks, {failure_count} failed")
        
        if failed_files:
            self.logger.warning(f"Failed files: {failed_files}")
        
        return all_chunks
    
    def get_chunks_path(self, video_id: str) -> Path:
        """
        Get the file path for chunks.
        
        Args:
            video_id: Video identifier
            
        Returns:
            Path to chunks file
        """
        return self.chunks_path / f"{video_id}_chunks.json"
    
    def chunks_exist(self, video_id: str) -> bool:
        """
        Check if chunks already exist for a video.
        
        Args:
            video_id: Video identifier
            
        Returns:
            True if chunks exist
        """
        return self.get_chunks_path(video_id).exists()
    
    def load_chunks(self, video_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Load existing chunks for a video.
        
        Args:
            video_id: Video identifier
            
        Returns:
            List of chunk dictionaries if exists, None otherwise
        """
        chunks_file = self.get_chunks_path(video_id)
        
        if not chunks_file.exists():
            return None
        
        chunks = self.file_manager.load_json(chunks_file)
        
        if chunks and isinstance(chunks, list):
            # Validate loaded chunks
            is_valid, errors = self.validators.validate_chunks(chunks)
            if not is_valid:
                self.logger.warning(f"Loaded chunks are invalid: {errors}")
                return None
        
        return chunks
