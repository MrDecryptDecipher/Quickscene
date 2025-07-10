"""
Transcriber Module for Quickscene System

Handles video-to-transcript conversion using OpenAI Whisper or AssemblyAI.
Generates timestamped transcripts in JSON format for downstream processing.
"""

import os
import logging
import whisper
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from .utils.config_loader import ConfigLoader
from .utils.file_manager import FileManager
from .utils.validators import Validators


class Transcriber:
    """
    Video transcription service using OpenAI Whisper or AssemblyAI.
    
    Features:
    - Multiple ASR provider support (Whisper, AssemblyAI)
    - Accurate timestamp generation
    - Batch processing capabilities
    - Error handling and recovery
    - Progress tracking and logging
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the transcriber with configuration.
        
        Args:
            config: Configuration dictionary containing ASR settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.file_manager = FileManager()
        self.validators = Validators()
        
        # Extract ASR configuration
        self.asr_config = config.get('asr', {})
        self.asr_mode = self.asr_config.get('mode', 'whisper')
        self.device = self.asr_config.get('device', 'cpu')
        
        # Initialize ASR model
        self.model = None
        self._initialize_model()
        
        # Setup paths
        self.transcript_path = Path(config['paths']['transcripts'])
        self.file_manager.ensure_directory(self.transcript_path)
    
    def _initialize_model(self) -> None:
        """Initialize the ASR model based on configuration."""
        try:
            if self.asr_mode == 'whisper':
                self._initialize_whisper()
            elif self.asr_mode == 'assemblyai':
                self._initialize_assemblyai()
            else:
                raise ValueError(f"Unsupported ASR mode: {self.asr_mode}")
                
            self.logger.info(f"ASR model initialized: {self.asr_mode}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ASR model: {e}")
            raise
    
    def _initialize_whisper(self) -> None:
        """Initialize OpenAI Whisper model."""
        model_name = self.asr_config.get('whisper_model', 'base')
        
        try:
            self.model = whisper.load_model(model_name, device=self.device)
            self.logger.info(f"Whisper model '{model_name}' loaded on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model '{model_name}': {e}")
            raise
    
    def _initialize_assemblyai(self) -> None:
        """Initialize AssemblyAI client."""
        try:
            import assemblyai as aai
            
            # Get API key from environment or config
            api_key = os.getenv('ASSEMBLYAI_API_KEY') or self.asr_config.get('api_key')
            if not api_key:
                raise ValueError("AssemblyAI API key not found in environment or config")
            
            aai.settings.api_key = api_key
            self.model = aai.Transcriber()
            self.logger.info("AssemblyAI client initialized")
            
        except ImportError:
            raise ImportError("AssemblyAI package not installed. Run: pip install assemblyai")
        except Exception as e:
            self.logger.error(f"Failed to initialize AssemblyAI: {e}")
            raise
    
    def transcribe(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Transcribe a single video file to timestamped transcript.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing transcript with timestamps
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video file is invalid
            RuntimeError: If transcription fails
        """
        video_path = Path(video_path)
        
        # Validate video file
        is_valid, errors = self.validators.validate_video_file(video_path)
        if not is_valid:
            raise ValueError(f"Invalid video file: {errors}")
        
        self.logger.info(f"Starting transcription: {video_path.name}")
        start_time = datetime.now()
        
        try:
            # Generate video ID from filename
            video_id = video_path.stem
            
            # Check if transcript already exists
            transcript_file = self.transcript_path / f"{video_id}.json"
            if transcript_file.exists():
                self.logger.info(f"Loading existing transcript: {transcript_file}")
                existing_transcript = self.file_manager.load_json(transcript_file)
                if existing_transcript:
                    return existing_transcript
            
            # Perform transcription based on ASR mode
            if self.asr_mode == 'whisper':
                transcript = self._transcribe_with_whisper(video_path, video_id)
            elif self.asr_mode == 'assemblyai':
                transcript = self._transcribe_with_assemblyai(video_path, video_id)
            else:
                raise ValueError(f"Unsupported ASR mode: {self.asr_mode}")
            
            # Validate transcript
            is_valid, errors = self.validators.validate_transcript(transcript)
            if not is_valid:
                raise ValueError(f"Generated transcript is invalid: {errors}")
            
            # Save transcript
            if not self.file_manager.save_json(transcript, transcript_file):
                self.logger.warning(f"Failed to save transcript: {transcript_file}")
            
            # Log completion
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Transcription completed in {duration:.1f}s: {video_path.name}")
            
            return transcript
            
        except Exception as e:
            self.logger.error(f"Transcription failed for {video_path.name}: {e}")
            raise RuntimeError(f"Transcription failed: {e}")
    
    def _transcribe_with_whisper(self, video_path: Path, video_id: str) -> Dict[str, Any]:
        """
        Transcribe video using OpenAI Whisper.
        
        Args:
            video_path: Path to video file
            video_id: Unique identifier for the video
            
        Returns:
            Transcript dictionary with Whisper results
        """
        try:
            # Whisper transcription options
            options = {
                'language': self.asr_config.get('language', 'en'),
                'task': 'transcribe',
                'verbose': False
            }
            
            # Perform transcription
            self.logger.debug(f"Running Whisper transcription on {video_path}")
            result = self.model.transcribe(str(video_path), **options)
            
            # Extract segments with timestamps
            segments = []
            for segment in result.get('segments', []):
                segments.append({
                    'start': float(segment['start']),
                    'end': float(segment['end']),
                    'text': segment['text'].strip()
                })
            
            # Build transcript structure
            transcript = {
                'video_id': video_id,
                'video_path': str(video_path),
                'duration': float(result.get('duration', 0)),
                'language': result.get('language', 'en'),
                'segments': segments,
                'transcription_info': {
                    'asr_provider': 'whisper',
                    'model': self.asr_config.get('whisper_model', 'base'),
                    'timestamp': datetime.now().isoformat(),
                    'device': self.device
                }
            }
            
            self.logger.debug(f"Whisper generated {len(segments)} segments")
            return transcript
            
        except Exception as e:
            self.logger.error(f"Whisper transcription failed: {e}")
            raise
    
    def _transcribe_with_assemblyai(self, video_path: Path, video_id: str) -> Dict[str, Any]:
        """
        Transcribe video using AssemblyAI.
        
        Args:
            video_path: Path to video file
            video_id: Unique identifier for the video
            
        Returns:
            Transcript dictionary with AssemblyAI results
        """
        try:
            # AssemblyAI transcription config
            config = {
                'speaker_labels': False,
                'auto_chapters': False,
                'auto_highlights': False
            }
            
            # Upload and transcribe
            self.logger.debug(f"Running AssemblyAI transcription on {video_path}")
            transcript_result = self.model.transcribe(str(video_path), config=config)
            
            if transcript_result.status == 'error':
                raise RuntimeError(f"AssemblyAI transcription failed: {transcript_result.error}")
            
            # Extract segments with timestamps
            segments = []
            if hasattr(transcript_result, 'words') and transcript_result.words:
                # Group words into segments (approximate 10-second segments)
                current_segment = {'start': 0, 'end': 0, 'text': ''}
                segment_duration = 10000  # 10 seconds in milliseconds
                
                for word in transcript_result.words:
                    if word.start - current_segment['start'] > segment_duration and current_segment['text']:
                        # Finish current segment
                        current_segment['start'] = current_segment['start'] / 1000.0  # Convert to seconds
                        current_segment['end'] = current_segment['end'] / 1000.0
                        segments.append(current_segment)
                        
                        # Start new segment
                        current_segment = {
                            'start': word.start,
                            'end': word.end,
                            'text': word.text
                        }
                    else:
                        # Add to current segment
                        if not current_segment['text']:
                            current_segment['start'] = word.start
                        current_segment['end'] = word.end
                        current_segment['text'] += ' ' + word.text if current_segment['text'] else word.text
                
                # Add final segment
                if current_segment['text']:
                    current_segment['start'] = current_segment['start'] / 1000.0
                    current_segment['end'] = current_segment['end'] / 1000.0
                    segments.append(current_segment)
            
            # Build transcript structure
            transcript = {
                'video_id': video_id,
                'video_path': str(video_path),
                'duration': float(transcript_result.audio_duration or 0),
                'language': 'en',  # AssemblyAI default
                'segments': segments,
                'transcription_info': {
                    'asr_provider': 'assemblyai',
                    'model': 'assemblyai-default',
                    'timestamp': datetime.now().isoformat(),
                    'confidence': float(transcript_result.confidence or 0)
                }
            }
            
            self.logger.debug(f"AssemblyAI generated {len(segments)} segments")
            return transcript
            
        except Exception as e:
            self.logger.error(f"AssemblyAI transcription failed: {e}")
            raise
    
    def batch_transcribe(self, video_directory: Union[str, Path], 
                        max_concurrent: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Transcribe multiple videos in a directory.
        
        Args:
            video_directory: Directory containing video files
            max_concurrent: Maximum concurrent transcriptions (defaults to config)
            
        Returns:
            List of transcript dictionaries
        """
        video_directory = Path(video_directory)
        
        if not video_directory.exists():
            raise FileNotFoundError(f"Video directory not found: {video_directory}")
        
        # Find video files
        video_files = self.file_manager.find_video_files(video_directory)
        
        if not video_files:
            self.logger.warning(f"No video files found in {video_directory}")
            return []
        
        self.logger.info(f"Starting batch transcription of {len(video_files)} videos")
        
        # Set concurrency limit
        if max_concurrent is None:
            max_concurrent = self.config.get('performance', {}).get('max_concurrent_transcriptions', 2)
        
        transcripts = []
        failed_files = []
        
        # Process videos sequentially for now (can be parallelized later)
        for i, video_file in enumerate(video_files, 1):
            try:
                self.logger.info(f"Processing video {i}/{len(video_files)}: {video_file.name}")
                transcript = self.transcribe(video_file)
                transcripts.append(transcript)
                
            except Exception as e:
                self.logger.error(f"Failed to transcribe {video_file.name}: {e}")
                failed_files.append(video_file.name)
                continue
        
        # Log summary
        success_count = len(transcripts)
        failure_count = len(failed_files)
        self.logger.info(f"Batch transcription complete: {success_count} successful, {failure_count} failed")
        
        if failed_files:
            self.logger.warning(f"Failed files: {failed_files}")
        
        return transcripts
    
    def get_transcript_path(self, video_id: str) -> Path:
        """
        Get the file path for a transcript.
        
        Args:
            video_id: Video identifier
            
        Returns:
            Path to transcript file
        """
        return self.transcript_path / f"{video_id}.json"
    
    def transcript_exists(self, video_id: str) -> bool:
        """
        Check if transcript already exists for a video.
        
        Args:
            video_id: Video identifier
            
        Returns:
            True if transcript exists
        """
        return self.get_transcript_path(video_id).exists()
    
    def load_transcript(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Load existing transcript for a video.
        
        Args:
            video_id: Video identifier
            
        Returns:
            Transcript dictionary if exists, None otherwise
        """
        transcript_file = self.get_transcript_path(video_id)
        
        if not transcript_file.exists():
            return None
        
        transcript = self.file_manager.load_json(transcript_file)
        
        if transcript:
            # Validate loaded transcript
            is_valid, errors = self.validators.validate_transcript(transcript)
            if not is_valid:
                self.logger.warning(f"Loaded transcript is invalid: {errors}")
                return None
        
        return transcript
