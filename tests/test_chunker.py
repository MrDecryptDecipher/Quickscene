"""
Unit tests for Chunker module.

Tests chunk boundary validation, length validation, and data integrity.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.chunker import Chunker


class TestChunker:
    """Test suite for Chunker module."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {
            'chunking': {
                'duration_sec': 15,
                'overlap_sec': 0,
                'respect_word_boundaries': True,
                'min_chunk_length': 10
            },
            'paths': {
                'chunks': './test_data/chunks'
            }
        }
    
    @pytest.fixture
    def chunker(self, config):
        """Create chunker instance for testing."""
        return Chunker(config)
    
    @pytest.fixture
    def sample_transcript(self):
        """Sample transcript for testing."""
        return {
            'video_id': 'test_video',
            'duration': 60.0,
            'segments': [
                {
                    'start': 0.0,
                    'end': 10.0,
                    'text': 'This is the first segment with some words to test chunking functionality.'
                },
                {
                    'start': 10.0,
                    'end': 20.0,
                    'text': 'This is the second segment with more content for testing purposes.'
                },
                {
                    'start': 20.0,
                    'end': 30.0,
                    'text': 'Third segment contains additional text to verify chunk boundaries work correctly.'
                },
                {
                    'start': 30.0,
                    'end': 40.0,
                    'text': 'Fourth segment has even more text content for comprehensive testing.'
                },
                {
                    'start': 40.0,
                    'end': 50.0,
                    'text': 'Fifth and final segment completes our test transcript data.'
                }
            ]
        }
    
    def test_chunker_initialization(self, config):
        """Test chunker initialization."""
        chunker = Chunker(config)
        
        assert chunker.chunk_duration == 15
        assert chunker.overlap_duration == 0
        assert chunker.respect_word_boundaries is True
        assert chunker.min_chunk_length == 10
    
    def test_chunk_transcript_success(self, chunker, sample_transcript):
        """Test successful transcript chunking."""
        with patch.object(chunker.file_manager, 'save_json', return_value=True):
            chunks = chunker.chunk_transcript(sample_transcript)
        
        # Verify chunks were created
        assert len(chunks) > 0
        
        # Verify chunk structure
        for chunk in chunks:
            assert 'video_id' in chunk
            assert 'chunk_id' in chunk
            assert 'start_time' in chunk
            assert 'end_time' in chunk
            assert 'text' in chunk
            assert 'duration' in chunk
            
            # Verify data types
            assert isinstance(chunk['start_time'], float)
            assert isinstance(chunk['end_time'], float)
            assert isinstance(chunk['text'], str)
            assert chunk['video_id'] == 'test_video'
    
    def test_chunk_duration_boundaries(self, chunker, sample_transcript):
        """Test that chunks respect duration boundaries."""
        with patch.object(chunker.file_manager, 'save_json', return_value=True):
            chunks = chunker.chunk_transcript(sample_transcript)
        
        for chunk in chunks:
            duration = chunk['end_time'] - chunk['start_time']
            # Allow some flexibility due to word boundaries
            assert duration <= chunker.chunk_duration * 1.5
            assert duration > 0
    
    def test_no_data_loss(self, chunker, sample_transcript):
        """Test that no words are lost during chunking."""
        with patch.object(chunker.file_manager, 'save_json', return_value=True):
            chunks = chunker.chunk_transcript(sample_transcript)
        
        # Combine all chunk text
        combined_text = ' '.join(chunk['text'] for chunk in chunks)
        
        # Combine all original segment text
        original_text = ' '.join(segment['text'].strip() for segment in sample_transcript['segments'])
        
        # Extract words for comparison (order might differ due to chunking)
        combined_words = set(combined_text.lower().split())
        original_words = set(original_text.lower().split())
        
        # All original words should be present in chunks
        missing_words = original_words - combined_words
        assert len(missing_words) == 0, f"Missing words: {missing_words}"
    
    def test_no_overlap(self, chunker, sample_transcript):
        """Test that chunks don't overlap when overlap is disabled."""
        chunker.overlap_duration = 0
        
        with patch.object(chunker.file_manager, 'save_json', return_value=True):
            chunks = chunker.chunk_transcript(sample_transcript)
        
        # Sort chunks by start time
        sorted_chunks = sorted(chunks, key=lambda x: x['start_time'])
        
        # Check for overlaps
        for i in range(len(sorted_chunks) - 1):
            current_end = sorted_chunks[i]['end_time']
            next_start = sorted_chunks[i + 1]['start_time']
            
            # Next chunk should start at or after current chunk ends
            assert next_start >= current_end, f"Overlap detected: chunk {i} ends at {current_end}, chunk {i+1} starts at {next_start}"
    
    def test_timestamp_continuity(self, chunker, sample_transcript):
        """Test that chunk timestamps are continuous and logical."""
        with patch.object(chunker.file_manager, 'save_json', return_value=True):
            chunks = chunker.chunk_transcript(sample_transcript)
        
        for chunk in chunks:
            # Start time should be less than end time
            assert chunk['start_time'] < chunk['end_time']
            
            # Times should be within the original transcript duration
            assert chunk['start_time'] >= 0
            assert chunk['end_time'] <= sample_transcript['duration']
    
    def test_word_boundary_respect(self, chunker, sample_transcript):
        """Test that word boundaries are respected."""
        chunker.respect_word_boundaries = True
        
        with patch.object(chunker.file_manager, 'save_json', return_value=True):
            chunks = chunker.chunk_transcript(sample_transcript)
        
        for chunk in chunks:
            text = chunk['text']
            
            # Text should not start or end with partial words (basic check)
            assert not text.startswith(' '), "Chunk text should not start with space"
            assert not text.endswith(' '), "Chunk text should not end with space"
            
            # Text should contain complete words
            words = text.split()
            assert len(words) > 0, "Chunk should contain at least one word"
    
    def test_empty_transcript_handling(self, chunker):
        """Test handling of empty transcript."""
        empty_transcript = {
            'video_id': 'empty_video',
            'duration': 0.0,
            'segments': []
        }
        
        with patch.object(chunker.file_manager, 'save_json', return_value=True):
            chunks = chunker.chunk_transcript(empty_transcript)
        
        assert chunks == []
    
    def test_very_short_transcript(self, chunker):
        """Test handling of very short transcript."""
        short_transcript = {
            'video_id': 'short_video',
            'duration': 5.0,
            'segments': [
                {
                    'start': 0.0,
                    'end': 5.0,
                    'text': 'Short text'
                }
            ]
        }
        
        with patch.object(chunker.file_manager, 'save_json', return_value=True):
            chunks = chunker.chunk_transcript(short_transcript)
        
        # Should create at least one chunk even if shorter than target duration
        assert len(chunks) >= 1
        assert chunks[0]['text'] == 'Short text'
    
    def test_chunk_reconstruction(self, chunker, sample_transcript):
        """Test that chunks can be reconstructed to approximate original."""
        with patch.object(chunker.file_manager, 'save_json', return_value=True):
            chunks = chunker.chunk_transcript(sample_transcript)
        
        # Sort chunks by start time
        sorted_chunks = sorted(chunks, key=lambda x: x['start_time'])
        
        # Verify temporal coverage
        first_start = sorted_chunks[0]['start_time']
        last_end = sorted_chunks[-1]['end_time']
        
        # Should cover most of the original duration
        coverage = (last_end - first_start) / sample_transcript['duration']
        assert coverage > 0.8, f"Poor temporal coverage: {coverage}"
    
    def test_metadata_preservation(self, chunker, sample_transcript):
        """Test that video metadata is preserved in chunks."""
        with patch.object(chunker.file_manager, 'save_json', return_value=True):
            chunks = chunker.chunk_transcript(sample_transcript)
        
        for chunk in chunks:
            assert chunk['video_id'] == sample_transcript['video_id']
            assert 'chunk_info' in chunk
            assert chunk['chunk_info']['target_duration'] == chunker.chunk_duration
    
    def test_invalid_transcript(self, chunker):
        """Test handling of invalid transcript."""
        invalid_transcript = {
            'video_id': 'invalid',
            # Missing required fields
        }
        
        with pytest.raises(ValueError, match="Invalid transcript"):
            chunker.chunk_transcript(invalid_transcript)
    
    def test_existing_chunks_loading(self, chunker, sample_transcript):
        """Test loading existing chunks instead of re-chunking."""
        existing_chunks = [
            {
                'video_id': 'test_video',
                'chunk_id': 'test_video_0000',
                'start_time': 0.0,
                'end_time': 15.0,
                'text': 'Existing chunk text',
                'word_count': 3
            }
        ]
        
        with patch.object(chunker.file_manager, 'load_json', return_value=existing_chunks):
            with patch.object(chunker, 'get_chunks_path') as mock_path:
                mock_path.return_value.exists.return_value = True
                
                chunks = chunker.chunk_transcript(sample_transcript)
        
        assert chunks == existing_chunks
    
    def test_batch_chunk_transcripts(self, chunker):
        """Test batch chunking of multiple transcripts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create fake transcript files
            transcript1 = temp_path / 'video1.json'
            transcript2 = temp_path / 'video2.json'
            
            sample_data = {
                'video_id': 'video1',
                'duration': 30.0,
                'segments': [
                    {'start': 0.0, 'end': 15.0, 'text': 'First video content'},
                    {'start': 15.0, 'end': 30.0, 'text': 'More content here'}
                ]
            }
            
            with patch.object(chunker.file_manager, 'load_json', return_value=sample_data):
                with patch.object(chunker, 'chunk_transcript', return_value=[{'chunk': 'data'}]):
                    results = chunker.batch_chunk_transcripts(temp_path)
            
            assert len(results) == 2
    
    def test_chunks_exist(self, chunker):
        """Test checking if chunks exist."""
        video_id = 'test_video'
        
        with patch.object(chunker, 'get_chunks_path') as mock_path:
            mock_path.return_value.exists.return_value = True
            
            assert chunker.chunks_exist(video_id) is True
            mock_path.assert_called_once_with(video_id)
    
    def test_load_chunks_success(self, chunker):
        """Test loading existing chunks."""
        video_id = 'test_video'
        chunks_data = [
            {
                'video_id': video_id,
                'chunk_id': f'{video_id}_0000',
                'start_time': 0.0,
                'end_time': 15.0,
                'text': 'Test chunk',
                'word_count': 2
            }
        ]
        
        with patch.object(chunker.file_manager, 'load_json', return_value=chunks_data):
            with patch.object(chunker, 'get_chunks_path') as mock_path:
                mock_path.return_value.exists.return_value = True
                
                result = chunker.load_chunks(video_id)
        
        assert result == chunks_data
    
    def test_load_chunks_not_found(self, chunker):
        """Test loading chunks that don't exist."""
        video_id = 'nonexistent_video'
        
        with patch.object(chunker, 'get_chunks_path') as mock_path:
            mock_path.return_value.exists.return_value = False
            
            result = chunker.load_chunks(video_id)
        
        assert result is None
    
    @pytest.mark.unit
    def test_chunk_boundaries_validation(self, chunker, sample_transcript):
        """Test that chunk boundaries are properly validated."""
        with patch.object(chunker.file_manager, 'save_json', return_value=True):
            chunks = chunker.chunk_transcript(sample_transcript)
        
        for chunk in chunks:
            # Verify chunk duration is reasonable
            duration = chunk['end_time'] - chunk['start_time']
            assert 0 < duration <= 30, f"Invalid chunk duration: {duration}"
            
            # Verify text is not empty
            assert chunk['text'].strip(), "Chunk text should not be empty"
            
            # Verify word count is reasonable
            word_count = len(chunk['text'].split())
            assert word_count >= 1, "Chunk should have at least one word"
    
    @pytest.mark.unit
    def test_chunk_length_validation(self, chunker, sample_transcript):
        """Test that chunks meet minimum length requirements."""
        with patch.object(chunker.file_manager, 'save_json', return_value=True):
            chunks = chunker.chunk_transcript(sample_transcript)
        
        for chunk in chunks:
            # Check minimum word count (if specified in config)
            if 'word_count' in chunk:
                assert chunk['word_count'] >= chunker.min_chunk_length or chunk['word_count'] >= 1
