"""
Unit tests for Transcriber module.

Tests ASR functionality, transcript format validation, and error handling.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.transcriber import Transcriber
from app.utils.config_loader import ConfigLoader


class TestTranscriber:
    """Test suite for Transcriber module."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {
            'asr': {
                'mode': 'whisper',
                'whisper_model': 'base',
                'language': 'en',
                'device': 'cpu'
            },
            'paths': {
                'transcripts': './test_data/transcripts'
            },
            'performance': {
                'max_concurrent_transcriptions': 2
            }
        }
    
    @pytest.fixture
    def transcriber(self, config):
        """Create transcriber instance for testing."""
        with patch('app.transcriber.whisper.load_model') as mock_whisper:
            mock_model = Mock()
            mock_whisper.return_value = mock_model
            return Transcriber(config)
    
    @pytest.fixture
    def sample_video_file(self):
        """Create a temporary video file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(b'fake video content')
            yield Path(f.name)
        Path(f.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def sample_whisper_result(self):
        """Sample Whisper transcription result."""
        return {
            'duration': 120.5,
            'language': 'en',
            'segments': [
                {
                    'start': 0.0,
                    'end': 5.2,
                    'text': ' This is the first segment of the video.'
                },
                {
                    'start': 5.2,
                    'end': 10.8,
                    'text': ' This is the second segment with more content.'
                },
                {
                    'start': 10.8,
                    'end': 15.0,
                    'text': ' Final segment of the sample video.'
                }
            ]
        }
    
    def test_transcriber_initialization(self, config):
        """Test transcriber initialization with Whisper."""
        with patch('app.transcriber.whisper.load_model') as mock_whisper:
            mock_model = Mock()
            mock_whisper.return_value = mock_model
            
            transcriber = Transcriber(config)
            
            assert transcriber.asr_mode == 'whisper'
            assert transcriber.device == 'cpu'
            mock_whisper.assert_called_once_with('base', device='cpu')
    
    def test_transcriber_initialization_assemblyai(self, config):
        """Test transcriber initialization with AssemblyAI."""
        config['asr']['mode'] = 'assemblyai'
        
        with patch('app.transcriber.aai') as mock_aai:
            mock_transcriber = Mock()
            mock_aai.Transcriber.return_value = mock_transcriber
            
            transcriber = Transcriber(config)
            
            assert transcriber.asr_mode == 'assemblyai'
            assert transcriber.model == mock_transcriber
    
    def test_transcriber_invalid_mode(self, config):
        """Test transcriber initialization with invalid ASR mode."""
        config['asr']['mode'] = 'invalid_mode'
        
        with pytest.raises(ValueError, match="Unsupported ASR mode"):
            Transcriber(config)
    
    def test_transcribe_whisper_success(self, transcriber, sample_video_file, sample_whisper_result):
        """Test successful Whisper transcription."""
        # Mock Whisper model
        transcriber.model.transcribe.return_value = sample_whisper_result
        
        # Mock file operations
        with patch.object(transcriber.file_manager, 'save_json', return_value=True):
            result = transcriber.transcribe(sample_video_file)
        
        # Verify result structure
        assert result['video_id'] == sample_video_file.stem
        assert result['duration'] == 120.5
        assert result['language'] == 'en'
        assert len(result['segments']) == 3
        
        # Verify segments
        assert result['segments'][0]['start'] == 0.0
        assert result['segments'][0]['end'] == 5.2
        assert result['segments'][0]['text'] == 'This is the first segment of the video.'
        
        # Verify transcription info
        assert result['transcription_info']['asr_provider'] == 'whisper'
        assert result['transcription_info']['model'] == 'base'
    
    def test_transcribe_invalid_video_file(self, transcriber):
        """Test transcription with invalid video file."""
        invalid_file = Path('/nonexistent/file.mp4')
        
        with pytest.raises(ValueError, match="Invalid video file"):
            transcriber.transcribe(invalid_file)
    
    def test_transcribe_existing_transcript(self, transcriber, sample_video_file):
        """Test loading existing transcript instead of re-transcribing."""
        existing_transcript = {
            'video_id': sample_video_file.stem,
            'duration': 60.0,
            'segments': []
        }
        
        # Mock existing transcript file
        with patch.object(transcriber.file_manager, 'load_json', return_value=existing_transcript):
            with patch.object(transcriber, 'get_transcript_path') as mock_path:
                mock_path.return_value.exists.return_value = True
                
                result = transcriber.transcribe(sample_video_file)
        
        assert result == existing_transcript
    
    def test_transcribe_whisper_failure(self, transcriber, sample_video_file):
        """Test handling of Whisper transcription failure."""
        # Mock Whisper failure
        transcriber.model.transcribe.side_effect = Exception("Whisper failed")
        
        with pytest.raises(RuntimeError, match="Transcription failed"):
            transcriber.transcribe(sample_video_file)
    
    def test_batch_transcribe_success(self, transcriber, sample_whisper_result):
        """Test successful batch transcription."""
        # Create temporary directory with video files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create fake video files
            video1 = temp_path / 'video1.mp4'
            video2 = temp_path / 'video2.mp4'
            video1.write_bytes(b'fake content 1')
            video2.write_bytes(b'fake content 2')
            
            # Mock transcription
            transcriber.model.transcribe.return_value = sample_whisper_result
            
            with patch.object(transcriber.file_manager, 'save_json', return_value=True):
                results = transcriber.batch_transcribe(temp_path)
            
            assert len(results) == 2
            assert results[0]['video_id'] == 'video1'
            assert results[1]['video_id'] == 'video2'
    
    def test_batch_transcribe_empty_directory(self, transcriber):
        """Test batch transcription with empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = transcriber.batch_transcribe(temp_dir)
            assert results == []
    
    def test_batch_transcribe_nonexistent_directory(self, transcriber):
        """Test batch transcription with nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            transcriber.batch_transcribe('/nonexistent/directory')
    
    def test_batch_transcribe_partial_failure(self, transcriber, sample_whisper_result):
        """Test batch transcription with some failures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create fake video files
            video1 = temp_path / 'video1.mp4'
            video2 = temp_path / 'video2.mp4'
            video1.write_bytes(b'fake content 1')
            video2.write_bytes(b'fake content 2')
            
            # Mock transcription - first succeeds, second fails
            def mock_transcribe(video_path):
                if 'video1' in str(video_path):
                    return sample_whisper_result
                else:
                    raise Exception("Transcription failed")
            
            with patch.object(transcriber, 'transcribe', side_effect=mock_transcribe):
                results = transcriber.batch_transcribe(temp_path)
            
            # Should have one successful result
            assert len(results) == 1
            assert results[0]['video_id'] == 'video1'
    
    def test_transcript_exists(self, transcriber):
        """Test checking if transcript exists."""
        video_id = 'test_video'
        
        with patch.object(transcriber, 'get_transcript_path') as mock_path:
            mock_path.return_value.exists.return_value = True
            
            assert transcriber.transcript_exists(video_id) is True
            mock_path.assert_called_once_with(video_id)
    
    def test_load_transcript_success(self, transcriber):
        """Test loading existing transcript."""
        video_id = 'test_video'
        transcript_data = {
            'video_id': video_id,
            'duration': 60.0,
            'segments': []
        }
        
        with patch.object(transcriber.file_manager, 'load_json', return_value=transcript_data):
            with patch.object(transcriber, 'get_transcript_path') as mock_path:
                mock_path.return_value.exists.return_value = True
                
                result = transcriber.load_transcript(video_id)
        
        assert result == transcript_data
    
    def test_load_transcript_not_found(self, transcriber):
        """Test loading transcript that doesn't exist."""
        video_id = 'nonexistent_video'
        
        with patch.object(transcriber, 'get_transcript_path') as mock_path:
            mock_path.return_value.exists.return_value = False
            
            result = transcriber.load_transcript(video_id)
        
        assert result is None
    
    def test_load_transcript_invalid(self, transcriber):
        """Test loading invalid transcript."""
        video_id = 'test_video'
        invalid_transcript = {'invalid': 'data'}
        
        with patch.object(transcriber.file_manager, 'load_json', return_value=invalid_transcript):
            with patch.object(transcriber, 'get_transcript_path') as mock_path:
                mock_path.return_value.exists.return_value = True
                
                result = transcriber.load_transcript(video_id)
        
        assert result is None
    
    def test_get_transcript_path(self, transcriber):
        """Test getting transcript file path."""
        video_id = 'test_video'
        expected_path = transcriber.transcript_path / f"{video_id}.json"
        
        result = transcriber.get_transcript_path(video_id)
        
        assert result == expected_path
    
    @pytest.mark.unit
    def test_whisper_model_loading(self, config):
        """Test that Whisper model loads correctly."""
        with patch('app.transcriber.whisper.load_model') as mock_whisper:
            mock_model = Mock()
            mock_whisper.return_value = mock_model
            
            transcriber = Transcriber(config)
            
            # Verify model was loaded with correct parameters
            mock_whisper.assert_called_once_with('base', device='cpu')
            assert transcriber.model == mock_model
    
    @pytest.mark.unit
    def test_transcript_format_validation(self, transcriber, sample_video_file, sample_whisper_result):
        """Test that generated transcripts have correct format."""
        transcriber.model.transcribe.return_value = sample_whisper_result
        
        with patch.object(transcriber.file_manager, 'save_json', return_value=True):
            result = transcriber.transcribe(sample_video_file)
        
        # Check required fields
        required_fields = ['video_id', 'duration', 'segments', 'transcription_info']
        for field in required_fields:
            assert field in result
        
        # Check segments structure
        for segment in result['segments']:
            assert 'start' in segment
            assert 'end' in segment
            assert 'text' in segment
            assert isinstance(segment['start'], float)
            assert isinstance(segment['end'], float)
            assert isinstance(segment['text'], str)
    
    @pytest.mark.unit
    def test_timestamp_accuracy(self, transcriber, sample_video_file, sample_whisper_result):
        """Test that timestamps are preserved accurately."""
        transcriber.model.transcribe.return_value = sample_whisper_result
        
        with patch.object(transcriber.file_manager, 'save_json', return_value=True):
            result = transcriber.transcribe(sample_video_file)
        
        # Verify timestamps match input
        assert result['segments'][0]['start'] == 0.0
        assert result['segments'][0]['end'] == 5.2
        assert result['segments'][1]['start'] == 5.2
        assert result['segments'][1]['end'] == 10.8
