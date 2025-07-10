#!/usr/bin/env python3
"""
Production Configuration Management
Loads and validates configuration from config.yaml
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, List
import logging

class Config:
    """Production configuration manager"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
        self._ensure_directories()
        self._setup_logging()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def _validate_config(self):
        """Validate required configuration keys"""
        required_keys = [
            'asr_mode', 'embedding_model', 'chunk_duration_sec',
            'video_path', 'transcript_path', 'chunks_path', 
            'embedding_path', 'index_path', 'faiss_index_path'
        ]
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.video_path, self.transcript_path, self.chunks_path,
            self.embedding_path, self.index_path, Path("logs")
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.get('log_level', 'INFO'))
        log_file = self.config.get('log_file', './logs/quickscene.log')
        
        # Ensure logs directory exists
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    # Property accessors for easy access
    @property
    def asr_mode(self) -> str:
        return self.config['asr_mode']
    
    @property
    def whisper_model(self) -> str:
        return self.config.get('whisper_model', 'tiny')
    
    @property
    def embedding_model(self) -> str:
        return self.config['embedding_model']
    
    @property
    def embedding_dimension(self) -> int:
        return self.config.get('embedding_dimension', 384)
    
    @property
    def chunk_duration_sec(self) -> int:
        return self.config['chunk_duration_sec']
    
    @property
    def chunk_overlap_sec(self) -> int:
        return self.config.get('chunk_overlap_sec', 2)
    
    @property
    def video_path(self) -> Path:
        return Path(self.config['video_path'])
    
    @property
    def transcript_path(self) -> Path:
        return Path(self.config['transcript_path'])
    
    @property
    def chunks_path(self) -> Path:
        return Path(self.config['chunks_path'])
    
    @property
    def embedding_path(self) -> Path:
        return Path(self.config['embedding_path'])
    
    @property
    def index_path(self) -> Path:
        return Path(self.config['index_path'])
    
    @property
    def faiss_index_path(self) -> Path:
        return Path(self.config['faiss_index_path'])
    
    @property
    def metadata_path(self) -> Path:
        return Path(self.config['metadata_path'])
    
    @property
    def index_type(self) -> str:
        return self.config.get('index_type', 'IndexFlatIP')
    
    @property
    def max_workers(self) -> int:
        return self.config.get('max_workers', 4)
    
    @property
    def batch_size(self) -> int:
        return self.config.get('batch_size', 32)
    
    @property
    def default_top_k(self) -> int:
        return self.config.get('default_top_k', 5)
    
    @property
    def similarity_threshold(self) -> float:
        return self.config.get('similarity_threshold', 0.3)
    
    @property
    def supported_formats(self) -> List[str]:
        return self.config.get('supported_formats', ['.mp4', '.avi', '.mov', '.mkv', '.webm'])
    
    @property
    def max_video_duration_hours(self) -> int:
        return self.config.get('max_video_duration_hours', 5)
    
    @property
    def enable_caching(self) -> bool:
        return self.config.get('enable_caching', True)
    
    @property
    def cache_ttl_hours(self) -> int:
        return self.config.get('cache_ttl_hours', 24)
    
    @property
    def enable_monitoring(self) -> bool:
        return self.config.get('enable_monitoring', True)
    
    def get_video_files(self) -> List[Path]:
        """Get all supported video files from video directory"""
        video_files = []
        for ext in self.supported_formats:
            video_files.extend(self.video_path.glob(f"*{ext}"))
        return sorted(video_files)
    
    def get_video_id(self, video_path: Path) -> str:
        """Get video ID from video file path"""
        return video_path.stem
    
    def get_transcript_file(self, video_id: str) -> Path:
        """Get transcript file path for video ID"""
        return self.transcript_path / f"{video_id}.json"
    
    def get_chunks_file(self, video_id: str) -> Path:
        """Get chunks file path for video ID"""
        return self.chunks_path / f"{video_id}_chunks.json"
    
    def get_embedding_file(self, video_id: str) -> Path:
        """Get embedding file path for video ID"""
        return self.embedding_path / f"{video_id}.npy"
    
    def __str__(self) -> str:
        return f"Config(videos={len(self.get_video_files())}, model={self.embedding_model})"

# Global config instance
config = None

def get_config(config_path: str = "config.yaml") -> Config:
    """Get global configuration instance"""
    global config
    if config is None:
        config = Config(config_path)
    return config
