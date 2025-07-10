"""
Configuration Loader for Quickscene System

Handles YAML configuration loading, validation, and environment-specific settings.
Supports both development and production configurations with parameter validation.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union


class ConfigLoader:
    """
    YAML configuration loader with validation and environment support.
    
    Features:
    - YAML file loading with error handling
    - Configuration validation against required parameters
    - Environment variable override support
    - Default value fallback
    """
    
    def __init__(self):
        """Initialize the configuration loader."""
        self.logger = logging.getLogger(__name__)
        self._config_cache = {}
    
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from YAML file with validation.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Dictionary containing validated configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            ValueError: If required parameters are missing
        """
        config_path = Path(config_path)
        
        # Check cache first
        cache_key = str(config_path.absolute())
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]
        
        # Validate file exists
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load YAML file
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML config: {e}")
        
        if config is None:
            raise ValueError(f"Empty configuration file: {config_path}")
        
        # Validate configuration structure
        validated_config = self._validate_config(config)
        
        # Apply environment variable overrides
        final_config = self._apply_env_overrides(validated_config)
        
        # Cache the result
        self._config_cache[cache_key] = final_config
        
        self.logger.info(f"Configuration loaded successfully from {config_path}")
        return final_config
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration against required parameters.
        
        Args:
            config: Raw configuration dictionary
            
        Returns:
            Validated configuration with defaults applied
            
        Raises:
            ValueError: If required parameters are missing
        """
        # Required top-level sections
        required_sections = ['asr', 'embedding', 'chunking', 'index', 'query', 'paths']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate ASR section
        asr_config = config['asr']
        if 'mode' not in asr_config:
            raise ValueError("Missing required parameter: asr.mode")
        if asr_config['mode'] not in ['whisper', 'assemblyai']:
            raise ValueError("asr.mode must be 'whisper' or 'assemblyai'")
        
        # Validate embedding section
        embedding_config = config['embedding']
        if 'model_name' not in embedding_config:
            raise ValueError("Missing required parameter: embedding.model_name")
        
        # Validate chunking section
        chunking_config = config['chunking']
        if 'duration_sec' not in chunking_config:
            raise ValueError("Missing required parameter: chunking.duration_sec")
        if not isinstance(chunking_config['duration_sec'], (int, float)):
            raise ValueError("chunking.duration_sec must be a number")
        if chunking_config['duration_sec'] < 5 or chunking_config['duration_sec'] > 30:
            raise ValueError("chunking.duration_sec must be between 5 and 30 seconds")
        
        # Validate paths section
        paths_config = config['paths']
        required_paths = ['videos', 'transcripts', 'chunks', 'embeddings', 'index']
        for path_key in required_paths:
            if path_key not in paths_config:
                raise ValueError(f"Missing required path: paths.{path_key}")
        
        # Apply defaults for optional parameters
        config = self._apply_defaults(config)
        
        return config
    
    def _apply_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply default values for optional configuration parameters.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with defaults applied
        """
        # ASR defaults
        asr_defaults = {
            'whisper_model': 'base',
            'language': 'en',
            'device': 'cpu'
        }
        for key, default_value in asr_defaults.items():
            config['asr'].setdefault(key, default_value)
        
        # Embedding defaults
        embedding_defaults = {
            'batch_size': 32,
            'max_seq_length': 512,
            'device': 'cpu'
        }
        for key, default_value in embedding_defaults.items():
            config['embedding'].setdefault(key, default_value)
        
        # Chunking defaults
        chunking_defaults = {
            'overlap_sec': 0,
            'respect_word_boundaries': True,
            'min_chunk_length': 10
        }
        for key, default_value in chunking_defaults.items():
            config['chunking'].setdefault(key, default_value)
        
        # Index defaults
        index_defaults = {
            'type': 'IndexFlatIP',
            'metric': 'METRIC_INNER_PRODUCT'
        }
        for key, default_value in index_defaults.items():
            config['index'].setdefault(key, default_value)
        
        # Query defaults
        query_defaults = {
            'max_results': 5,
            'similarity_threshold': 0.3,
            'response_timeout_sec': 1.0
        }
        for key, default_value in query_defaults.items():
            config['query'].setdefault(key, default_value)
        
        # Performance defaults
        if 'performance' not in config:
            config['performance'] = {}
        performance_defaults = {
            'max_concurrent_transcriptions': 2,
            'embedding_batch_size': 32,
            'index_build_batch_size': 1000
        }
        for key, default_value in performance_defaults.items():
            config['performance'].setdefault(key, default_value)
        
        # Logging defaults
        if 'logging' not in config:
            config['logging'] = {}
        logging_defaults = {
            'level': 'INFO',
            'format': '{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}',
            'file': './logs/quickscene.log'
        }
        for key, default_value in logging_defaults.items():
            config['logging'].setdefault(key, default_value)
        
        return config
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.
        
        Environment variables should be prefixed with QUICKSCENE_ and use
        double underscores to separate nested keys.
        
        Example: QUICKSCENE_ASR__MODE=whisper
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with environment overrides applied
        """
        env_prefix = "QUICKSCENE_"
        
        for env_key, env_value in os.environ.items():
            if not env_key.startswith(env_prefix):
                continue
            
            # Parse nested key structure
            config_key = env_key[len(env_prefix):].lower()
            key_parts = config_key.split('__')
            
            # Navigate to the correct nested dictionary
            current_dict = config
            for part in key_parts[:-1]:
                if part not in current_dict:
                    current_dict[part] = {}
                current_dict = current_dict[part]
            
            # Set the value with type conversion
            final_key = key_parts[-1]
            current_dict[final_key] = self._convert_env_value(env_value)
            
            self.logger.info(f"Applied environment override: {config_key} = {env_value}")
        
        return config
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """
        Convert environment variable string to appropriate type.
        
        Args:
            value: String value from environment variable
            
        Returns:
            Converted value with appropriate type
        """
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer conversion
        try:
            if '.' not in value:
                return int(value)
        except ValueError:
            pass
        
        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string if no conversion possible
        return value
    
    def get_nested_value(self, config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
        """
        Get nested configuration value using dot notation.
        
        Args:
            config: Configuration dictionary
            key_path: Dot-separated path to value (e.g., 'asr.whisper_model')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        current = config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        self._config_cache.clear()
        self.logger.debug("Configuration cache cleared")
