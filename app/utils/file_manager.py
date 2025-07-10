"""
File Manager for Quickscene System

Handles file I/O operations, path management, and data persistence.
Provides safe file operations with error handling and validation.
"""

import os
import json
import pickle
import shutil
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from datetime import datetime


class FileManager:
    """
    Centralized file management for Quickscene system.
    
    Features:
    - Safe file I/O operations with error handling
    - JSON and binary data persistence
    - Directory management and validation
    - File format validation
    - Backup and recovery operations
    """
    
    def __init__(self, base_path: Union[str, Path] = None):
        """
        Initialize the file manager.
        
        Args:
            base_path: Base directory for file operations (defaults to current directory)
        """
        self.logger = logging.getLogger(__name__)
        self.base_path = Path(base_path) if base_path else Path.cwd()
        
        # Supported video formats
        self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        
        # Supported audio formats
        self.audio_extensions = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a'}
    
    def ensure_directory(self, directory_path: Union[str, Path]) -> Path:
        """
        Ensure directory exists, create if necessary.
        
        Args:
            directory_path: Path to directory
            
        Returns:
            Path object for the directory
            
        Raises:
            OSError: If directory creation fails
        """
        dir_path = Path(directory_path)
        
        # Make path absolute if relative
        if not dir_path.is_absolute():
            dir_path = self.base_path / dir_path
        
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Directory ensured: {dir_path}")
            return dir_path
        except OSError as e:
            self.logger.error(f"Failed to create directory {dir_path}: {e}")
            raise
    
    def save_json(self, data: Dict[str, Any], file_path: Union[str, Path], 
                  indent: int = 2, backup: bool = True) -> bool:
        """
        Save data to JSON file with error handling.
        
        Args:
            data: Dictionary to save as JSON
            file_path: Path to save the JSON file
            indent: JSON indentation for readability
            backup: Whether to create backup of existing file
            
        Returns:
            True if successful, False otherwise
        """
        file_path = Path(file_path)
        
        # Make path absolute if relative
        if not file_path.is_absolute():
            file_path = self.base_path / file_path
        
        # Ensure parent directory exists
        self.ensure_directory(file_path.parent)
        
        # Create backup if file exists and backup is requested
        if backup and file_path.exists():
            self._create_backup(file_path)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)
            
            self.logger.debug(f"JSON saved successfully: {file_path}")
            return True
            
        except (IOError, TypeError, ValueError) as e:
            self.logger.error(f"Failed to save JSON to {file_path}: {e}")
            return False
    
    def load_json(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Load data from JSON file with error handling.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Dictionary containing JSON data, None if failed
        """
        file_path = Path(file_path)
        
        # Make path absolute if relative
        if not file_path.is_absolute():
            file_path = self.base_path / file_path
        
        if not file_path.exists():
            self.logger.warning(f"JSON file not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.logger.debug(f"JSON loaded successfully: {file_path}")
            return data
            
        except (IOError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to load JSON from {file_path}: {e}")
            return None
    
    def save_numpy(self, array: np.ndarray, file_path: Union[str, Path], 
                   backup: bool = True) -> bool:
        """
        Save NumPy array to file.
        
        Args:
            array: NumPy array to save
            file_path: Path to save the array
            backup: Whether to create backup of existing file
            
        Returns:
            True if successful, False otherwise
        """
        file_path = Path(file_path)
        
        # Make path absolute if relative
        if not file_path.is_absolute():
            file_path = self.base_path / file_path
        
        # Ensure parent directory exists
        self.ensure_directory(file_path.parent)
        
        # Create backup if file exists and backup is requested
        if backup and file_path.exists():
            self._create_backup(file_path)
        
        try:
            np.save(file_path, array)
            self.logger.debug(f"NumPy array saved successfully: {file_path}")
            return True
            
        except (IOError, ValueError) as e:
            self.logger.error(f"Failed to save NumPy array to {file_path}: {e}")
            return False
    
    def load_numpy(self, file_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Load NumPy array from file.
        
        Args:
            file_path: Path to NumPy file
            
        Returns:
            NumPy array if successful, None otherwise
        """
        file_path = Path(file_path)
        
        # Make path absolute if relative
        if not file_path.is_absolute():
            file_path = self.base_path / file_path
        
        # Add .npy extension if not present
        if file_path.suffix != '.npy':
            file_path = file_path.with_suffix('.npy')
        
        if not file_path.exists():
            self.logger.warning(f"NumPy file not found: {file_path}")
            return None
        
        try:
            array = np.load(file_path)
            self.logger.debug(f"NumPy array loaded successfully: {file_path}")
            return array
            
        except (IOError, ValueError) as e:
            self.logger.error(f"Failed to load NumPy array from {file_path}: {e}")
            return None
    
    def save_pickle(self, data: Any, file_path: Union[str, Path], 
                    backup: bool = True) -> bool:
        """
        Save data using pickle serialization.
        
        Args:
            data: Data to pickle
            file_path: Path to save the pickle file
            backup: Whether to create backup of existing file
            
        Returns:
            True if successful, False otherwise
        """
        file_path = Path(file_path)
        
        # Make path absolute if relative
        if not file_path.is_absolute():
            file_path = self.base_path / file_path
        
        # Ensure parent directory exists
        self.ensure_directory(file_path.parent)
        
        # Create backup if file exists and backup is requested
        if backup and file_path.exists():
            self._create_backup(file_path)
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.logger.debug(f"Pickle saved successfully: {file_path}")
            return True
            
        except (IOError, pickle.PickleError) as e:
            self.logger.error(f"Failed to save pickle to {file_path}: {e}")
            return False
    
    def load_pickle(self, file_path: Union[str, Path]) -> Optional[Any]:
        """
        Load data from pickle file.
        
        Args:
            file_path: Path to pickle file
            
        Returns:
            Unpickled data if successful, None otherwise
        """
        file_path = Path(file_path)
        
        # Make path absolute if relative
        if not file_path.is_absolute():
            file_path = self.base_path / file_path
        
        if not file_path.exists():
            self.logger.warning(f"Pickle file not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            self.logger.debug(f"Pickle loaded successfully: {file_path}")
            return data
            
        except (IOError, pickle.PickleError) as e:
            self.logger.error(f"Failed to load pickle from {file_path}: {e}")
            return None
    
    def find_video_files(self, directory: Union[str, Path]) -> List[Path]:
        """
        Find all video files in a directory.
        
        Args:
            directory: Directory to search
            
        Returns:
            List of video file paths
        """
        directory = Path(directory)
        
        # Make path absolute if relative
        if not directory.is_absolute():
            directory = self.base_path / directory
        
        if not directory.exists():
            self.logger.warning(f"Directory not found: {directory}")
            return []
        
        video_files = []
        for ext in self.video_extensions:
            video_files.extend(directory.glob(f"*{ext}"))
            video_files.extend(directory.glob(f"*{ext.upper()}"))
        
        video_files.sort()
        self.logger.info(f"Found {len(video_files)} video files in {directory}")
        return video_files
    
    def is_video_file(self, file_path: Union[str, Path]) -> bool:
        """
        Check if file is a supported video format.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if file is a supported video format
        """
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.video_extensions
    
    def is_audio_file(self, file_path: Union[str, Path]) -> bool:
        """
        Check if file is a supported audio format.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if file is a supported audio format
        """
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.audio_extensions
    
    def get_file_size(self, file_path: Union[str, Path]) -> Optional[int]:
        """
        Get file size in bytes.
        
        Args:
            file_path: Path to file
            
        Returns:
            File size in bytes, None if file doesn't exist
        """
        file_path = Path(file_path)
        
        # Make path absolute if relative
        if not file_path.is_absolute():
            file_path = self.base_path / file_path
        
        try:
            return file_path.stat().st_size
        except OSError:
            return None
    
    def delete_file(self, file_path: Union[str, Path], backup: bool = True) -> bool:
        """
        Safely delete a file with optional backup.
        
        Args:
            file_path: Path to file to delete
            backup: Whether to create backup before deletion
            
        Returns:
            True if successful, False otherwise
        """
        file_path = Path(file_path)
        
        # Make path absolute if relative
        if not file_path.is_absolute():
            file_path = self.base_path / file_path
        
        if not file_path.exists():
            self.logger.warning(f"File not found for deletion: {file_path}")
            return False
        
        # Create backup if requested
        if backup:
            self._create_backup(file_path)
        
        try:
            file_path.unlink()
            self.logger.info(f"File deleted successfully: {file_path}")
            return True
            
        except OSError as e:
            self.logger.error(f"Failed to delete file {file_path}: {e}")
            return False
    
    def _create_backup(self, file_path: Path) -> bool:
        """
        Create backup of existing file.
        
        Args:
            file_path: Path to file to backup
            
        Returns:
            True if backup created successfully
        """
        if not file_path.exists():
            return False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = file_path.with_suffix(f".{timestamp}.backup{file_path.suffix}")
        
        try:
            shutil.copy2(file_path, backup_path)
            self.logger.debug(f"Backup created: {backup_path}")
            return True
            
        except OSError as e:
            self.logger.warning(f"Failed to create backup for {file_path}: {e}")
            return False
    
    def cleanup_backups(self, directory: Union[str, Path], max_backups: int = 5) -> int:
        """
        Clean up old backup files, keeping only the most recent ones.
        
        Args:
            directory: Directory to clean up
            max_backups: Maximum number of backups to keep per file
            
        Returns:
            Number of backup files deleted
        """
        directory = Path(directory)
        
        # Make path absolute if relative
        if not directory.is_absolute():
            directory = self.base_path / directory
        
        if not directory.exists():
            return 0
        
        # Find all backup files
        backup_files = list(directory.glob("*.backup.*"))
        
        # Group by original filename
        backup_groups = {}
        for backup_file in backup_files:
            # Extract original filename from backup name
            parts = backup_file.name.split('.backup.')
            if len(parts) == 2:
                original_name = parts[0]
                if original_name not in backup_groups:
                    backup_groups[original_name] = []
                backup_groups[original_name].append(backup_file)
        
        deleted_count = 0
        
        # Clean up each group
        for original_name, backups in backup_groups.items():
            # Sort by modification time (newest first)
            backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Delete excess backups
            for backup_file in backups[max_backups:]:
                try:
                    backup_file.unlink()
                    deleted_count += 1
                    self.logger.debug(f"Deleted old backup: {backup_file}")
                except OSError as e:
                    self.logger.warning(f"Failed to delete backup {backup_file}: {e}")
        
        if deleted_count > 0:
            self.logger.info(f"Cleaned up {deleted_count} old backup files")
        
        return deleted_count
