#!/usr/bin/env python3
"""
Production Video Transcription Module

Processes ALL videos in the video directory using OpenAI Whisper.
Outputs structured JSON with timestamps for downstream processing.
Follows PRD specifications for complete offline processing.
"""

import whisper
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import concurrent.futures
from tqdm import tqdm

from .config import get_config

class ProductionTranscriber:
    """
    Production transcriber that processes all videos in the video directory.
    Implements the transcriber.py module as specified in PRD.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize production transcriber with config"""
        self.config = get_config(config_path)
        self.logger = logging.getLogger(__name__)
        self.model = None
        
    def _load_model(self):
        """Load Whisper model (lazy loading)"""
        if self.model is None:
            self.logger.info(f"Loading Whisper model: {self.config.whisper_model}")
            self.model = whisper.load_model(self.config.whisper_model)
            self.logger.info("Whisper model loaded successfully")
    
    def transcribe_video(self, video_path: Path) -> Dict[str, Any]:
        """
        Transcribe a single video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Transcript dictionary with metadata and segments
        """
        self._load_model()
        
        video_id = self.config.get_video_id(video_path)
        transcript_file = self.config.get_transcript_file(video_id)
        
        # Check if transcript already exists and caching is enabled
        if self.config.enable_caching and transcript_file.exists():
            self.logger.info(f"Using cached transcript for {video_id}")
            with open(transcript_file, 'r') as f:
                return json.load(f)
        
        self.logger.info(f"Transcribing video: {video_path.name}")
        start_time = time.time()
        
        try:
            # Transcribe using Whisper
            result = self.model.transcribe(
                str(video_path),
                language="en",
                task="transcribe",
                verbose=False
            )
            
            # Create structured transcript
            transcript = {
                'video_id': video_id,
                'video_path': str(video_path),
                'duration': result.get('duration', 0),
                'language': result.get('language', 'en'),
                'segments': [],
                'metadata': {
                    'transcribed_at': datetime.now().isoformat(),
                    'whisper_model': self.config.whisper_model,
                    'processing_time_seconds': 0
                }
            }
            
            # Process segments
            for segment in result.get('segments', []):
                transcript['segments'].append({
                    'id': segment.get('id', 0),
                    'start': float(segment['start']),
                    'end': float(segment['end']),
                    'text': segment['text'].strip(),
                    'confidence': segment.get('avg_logprob', 0.0)
                })
            
            processing_time = time.time() - start_time
            transcript['metadata']['processing_time_seconds'] = processing_time
            
            # Save transcript to file
            with open(transcript_file, 'w') as f:
                json.dump(transcript, f, indent=2)
            
            self.logger.info(f"Transcribed {video_id}: {len(transcript['segments'])} segments in {processing_time:.1f}s")
            return transcript
            
        except Exception as e:
            self.logger.error(f"Failed to transcribe {video_path}: {e}")
            raise
    
    def transcribe_all_videos(self, max_workers: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """
        Transcribe all videos in the video directory.
        
        Args:
            max_workers: Number of parallel workers (defaults to config)
            
        Returns:
            Dictionary mapping video_id to transcript data
        """
        video_files = self.config.get_video_files()
        
        if not video_files:
            self.logger.warning(f"No video files found in {self.config.video_path}")
            return {}
        
        self.logger.info(f"Found {len(video_files)} videos to transcribe")
        
        # Use single-threaded processing for Whisper (GPU memory constraints)
        transcripts = {}
        
        for video_path in tqdm(video_files, desc="Transcribing videos"):
            try:
                video_id = self.config.get_video_id(video_path)
                transcript = self.transcribe_video(video_path)
                transcripts[video_id] = transcript
                
            except Exception as e:
                self.logger.error(f"Failed to process {video_path}: {e}")
                continue
        
        self.logger.info(f"Successfully transcribed {len(transcripts)}/{len(video_files)} videos")
        return transcripts
    
    def get_transcript_summary(self) -> Dict[str, Any]:
        """Get summary of all transcribed videos"""
        video_files = self.config.get_video_files()
        transcripts = {}
        total_duration = 0
        total_segments = 0
        
        for video_path in video_files:
            video_id = self.config.get_video_id(video_path)
            transcript_file = self.config.get_transcript_file(video_id)
            
            if transcript_file.exists():
                with open(transcript_file, 'r') as f:
                    transcript = json.load(f)
                    transcripts[video_id] = {
                        'duration': transcript.get('duration', 0),
                        'segments': len(transcript.get('segments', [])),
                        'language': transcript.get('language', 'unknown'),
                        'transcribed_at': transcript.get('metadata', {}).get('transcribed_at', 'unknown')
                    }
                    total_duration += transcript.get('duration', 0)
                    total_segments += len(transcript.get('segments', []))
        
        return {
            'total_videos': len(video_files),
            'transcribed_videos': len(transcripts),
            'total_duration_seconds': total_duration,
            'total_segments': total_segments,
            'transcripts': transcripts
        }
    
    def validate_transcripts(self) -> bool:
        """Validate all transcript files"""
        video_files = self.config.get_video_files()
        valid_count = 0
        
        for video_path in video_files:
            video_id = self.config.get_video_id(video_path)
            transcript_file = self.config.get_transcript_file(video_id)
            
            if not transcript_file.exists():
                self.logger.warning(f"Missing transcript for {video_id}")
                continue
            
            try:
                with open(transcript_file, 'r') as f:
                    transcript = json.load(f)
                
                # Validate structure
                required_keys = ['video_id', 'segments', 'metadata']
                if not all(key in transcript for key in required_keys):
                    self.logger.error(f"Invalid transcript structure for {video_id}")
                    continue
                
                # Validate segments
                segments = transcript.get('segments', [])
                if not segments:
                    self.logger.warning(f"No segments in transcript for {video_id}")
                    continue
                
                # Check segment structure
                for segment in segments:
                    if not all(key in segment for key in ['start', 'end', 'text']):
                        self.logger.error(f"Invalid segment structure in {video_id}")
                        break
                else:
                    valid_count += 1
                    
            except Exception as e:
                self.logger.error(f"Failed to validate transcript for {video_id}: {e}")
                continue
        
        self.logger.info(f"Validated {valid_count}/{len(video_files)} transcripts")
        return valid_count == len(video_files)

def main():
    """CLI interface for production transcriber"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Production Video Transcriber")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--summary", action="store_true", help="Show transcript summary")
    parser.add_argument("--validate", action="store_true", help="Validate transcripts")
    parser.add_argument("--force", action="store_true", help="Force re-transcription")
    
    args = parser.parse_args()
    
    transcriber = ProductionTranscriber(args.config)
    
    if args.summary:
        summary = transcriber.get_transcript_summary()
        print(json.dumps(summary, indent=2))
        return
    
    if args.validate:
        is_valid = transcriber.validate_transcripts()
        print(f"Transcripts valid: {is_valid}")
        return
    
    if args.force:
        # Clear cache to force re-transcription
        for video_path in transcriber.config.get_video_files():
            video_id = transcriber.config.get_video_id(video_path)
            transcript_file = transcriber.config.get_transcript_file(video_id)
            if transcript_file.exists():
                transcript_file.unlink()
    
    # Transcribe all videos
    transcripts = transcriber.transcribe_all_videos()
    print(f"Transcribed {len(transcripts)} videos successfully")

if __name__ == "__main__":
    main()
