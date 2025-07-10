"""
Embedder Module for Quickscene System

Converts text chunks into vector embeddings using SentenceTransformers.
Provides batch processing and caching for efficient embedding generation.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("SentenceTransformers not installed. Run: pip install sentence-transformers")

from .utils.config_loader import ConfigLoader
from .utils.file_manager import FileManager
from .utils.validators import Validators


class Embedder:
    """
    Text embedding service using SentenceTransformers.
    
    Features:
    - SentenceTransformer model integration
    - Batch processing for efficiency
    - 384-dimensional float32 embeddings
    - Caching and persistence
    - CPU-optimized inference
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the embedder with configuration.
        
        Args:
            config: Configuration dictionary containing embedding settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.file_manager = FileManager()
        self.validators = Validators()
        
        # Extract embedding configuration
        self.embedding_config = config.get('embedding', {})
        self.model_name = self.embedding_config.get('model_name', 'all-MiniLM-L6-v2')
        self.batch_size = self.embedding_config.get('batch_size', 32)
        self.max_seq_length = self.embedding_config.get('max_seq_length', 512)
        self.device = self.embedding_config.get('device', 'cpu')
        
        # Initialize model
        self.model = None
        self._initialize_model()
        
        # Setup paths
        self.embeddings_path = Path(config['paths']['embeddings'])
        self.file_manager.ensure_directory(self.embeddings_path)
        
        # Cache for embeddings
        self._embedding_cache = {}
    
    def _initialize_model(self) -> None:
        """Initialize the SentenceTransformer model."""
        try:
            self.logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Set maximum sequence length
            if hasattr(self.model, 'max_seq_length'):
                self.model.max_seq_length = self.max_seq_length
            
            # Get embedding dimension
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            self.logger.info(f"SentenceTransformer model loaded: {self.model_name}, "
                           f"dimension: {self.embedding_dim}, device: {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load SentenceTransformer model: {e}")
            raise
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generate embeddings for a list of text chunks.
        
        Args:
            chunks: List of chunk dictionaries containing text
            
        Returns:
            NumPy array of embeddings (N, embedding_dim)
            
        Raises:
            ValueError: If chunks are invalid
            RuntimeError: If embedding generation fails
        """
        # Validate input chunks
        is_valid, errors = self.validators.validate_chunks(chunks)
        if not is_valid:
            raise ValueError(f"Invalid chunks: {errors}")
        
        if not chunks:
            return np.array([], dtype=np.float32).reshape(0, self.embedding_dim)
        
        video_id = chunks[0]['video_id']
        self.logger.info(f"Starting embedding generation for {video_id}: {len(chunks)} chunks")
        
        try:
            # Check if embeddings already exist
            embeddings_file = self.embeddings_path / f"{video_id}.npy"
            if embeddings_file.exists():
                self.logger.info(f"Loading existing embeddings: {embeddings_file}")
                existing_embeddings = self.file_manager.load_numpy(embeddings_file)
                if existing_embeddings is not None:
                    # Validate existing embeddings
                    is_valid, errors = self.validators.validate_embeddings(
                        existing_embeddings, self.embedding_dim
                    )
                    if is_valid and len(existing_embeddings) == len(chunks):
                        return existing_embeddings
                    else:
                        self.logger.warning(f"Existing embeddings invalid, regenerating: {errors}")
            
            # Extract text from chunks
            texts = [chunk['text'] for chunk in chunks]
            
            # Generate embeddings in batches
            embeddings = self._generate_embeddings_batch(texts)
            
            # Validate generated embeddings
            is_valid, errors = self.validators.validate_embeddings(embeddings, self.embedding_dim)
            if not is_valid:
                raise RuntimeError(f"Generated embeddings are invalid: {errors}")
            
            # Save embeddings
            if not self.file_manager.save_numpy(embeddings, embeddings_file):
                self.logger.warning(f"Failed to save embeddings: {embeddings_file}")
            
            self.logger.info(f"Embedding generation completed: {embeddings.shape} for {video_id}")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed for {video_id}: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")
    
    def _generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for texts using batch processing.
        
        Args:
            texts: List of text strings
            
        Returns:
            NumPy array of embeddings
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.embedding_dim)
        
        # Preprocess texts
        processed_texts = self._preprocess_texts(texts)
        
        # Generate embeddings in batches
        all_embeddings = []
        
        for i in range(0, len(processed_texts), self.batch_size):
            batch_texts = processed_texts[i:i + self.batch_size]
            
            self.logger.debug(f"Processing batch {i//self.batch_size + 1}/{(len(processed_texts) + self.batch_size - 1)//self.batch_size}")
            
            try:
                # Generate embeddings for batch
                batch_embeddings = self.model.encode(
                    batch_texts,
                    batch_size=len(batch_texts),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True  # Normalize for cosine similarity
                )
                
                # Ensure float32 dtype
                batch_embeddings = batch_embeddings.astype(np.float32)
                
                all_embeddings.append(batch_embeddings)
                
            except Exception as e:
                self.logger.error(f"Failed to generate embeddings for batch: {e}")
                raise
        
        # Concatenate all batches
        if all_embeddings:
            embeddings = np.vstack(all_embeddings)
        else:
            embeddings = np.array([], dtype=np.float32).reshape(0, self.embedding_dim)
        
        return embeddings
    
    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        Preprocess texts before embedding generation.
        
        Args:
            texts: List of raw text strings
            
        Returns:
            List of preprocessed text strings
        """
        processed_texts = []
        
        for text in texts:
            # Basic text cleaning
            processed_text = text.strip()
            
            # Remove excessive whitespace
            processed_text = ' '.join(processed_text.split())
            
            # Truncate if too long (based on approximate token count)
            max_chars = self.max_seq_length * 4  # Rough estimate: 4 chars per token
            if len(processed_text) > max_chars:
                processed_text = processed_text[:max_chars].rsplit(' ', 1)[0]
                self.logger.debug(f"Truncated text to {len(processed_text)} characters")
            
            # Ensure minimum length
            if len(processed_text) < 3:
                processed_text = "empty text"  # Fallback for very short texts
            
            processed_texts.append(processed_text)
        
        return processed_texts
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query string.
        
        Args:
            query: Query string to embed
            
        Returns:
            NumPy array of query embedding (1, embedding_dim)
            
        Raises:
            ValueError: If query is invalid
            RuntimeError: If embedding generation fails
        """
        # Validate query
        is_valid, errors = self.validators.validate_query(query)
        if not is_valid:
            raise ValueError(f"Invalid query: {errors}")
        
        # Check cache first
        cache_key = query.strip().lower()
        if cache_key in self._embedding_cache:
            self.logger.debug("Using cached query embedding")
            return self._embedding_cache[cache_key]
        
        try:
            # Preprocess query
            processed_query = self._preprocess_texts([query])[0]
            
            # Generate embedding
            self.logger.debug(f"Generating embedding for query: '{query[:50]}...'")
            
            query_embedding = self.model.encode(
                [processed_query],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Ensure float32 dtype and correct shape
            query_embedding = query_embedding.astype(np.float32)
            
            # Validate embedding
            is_valid, errors = self.validators.validate_embeddings(query_embedding, self.embedding_dim)
            if not is_valid:
                raise RuntimeError(f"Generated query embedding is invalid: {errors}")
            
            # Cache the result
            self._embedding_cache[cache_key] = query_embedding
            
            # Limit cache size
            if len(self._embedding_cache) > 100:
                # Remove oldest entries
                oldest_key = next(iter(self._embedding_cache))
                del self._embedding_cache[oldest_key]
            
            self.logger.debug(f"Query embedding generated: shape {query_embedding.shape}")
            return query_embedding
            
        except Exception as e:
            self.logger.error(f"Query embedding generation failed: {e}")
            raise RuntimeError(f"Query embedding generation failed: {e}")
    
    def batch_embed_chunks(self, chunks_directory: Union[str, Path]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple chunk files in a directory.
        
        Args:
            chunks_directory: Directory containing chunk JSON files
            
        Returns:
            List of embedding arrays (one per chunk file)
        """
        chunks_directory = Path(chunks_directory)
        
        if not chunks_directory.exists():
            raise FileNotFoundError(f"Chunks directory not found: {chunks_directory}")
        
        # Find chunk files
        chunk_files = list(chunks_directory.glob("*_chunks.json"))
        
        if not chunk_files:
            self.logger.warning(f"No chunk files found in {chunks_directory}")
            return []
        
        self.logger.info(f"Starting batch embedding of {len(chunk_files)} chunk files")
        
        all_embeddings = []
        failed_files = []
        
        for i, chunk_file in enumerate(chunk_files, 1):
            try:
                self.logger.info(f"Processing chunks {i}/{len(chunk_files)}: {chunk_file.name}")
                
                # Load chunks
                chunks = self.file_manager.load_json(chunk_file)
                if not chunks:
                    self.logger.error(f"Failed to load chunks: {chunk_file}")
                    failed_files.append(chunk_file.name)
                    continue
                
                # Generate embeddings
                embeddings = self.embed_chunks(chunks)
                all_embeddings.append(embeddings)
                
            except Exception as e:
                self.logger.error(f"Failed to embed {chunk_file.name}: {e}")
                failed_files.append(chunk_file.name)
                continue
        
        # Log summary
        success_count = len(all_embeddings)
        failure_count = len(failed_files)
        total_embeddings = sum(len(embeddings) for embeddings in all_embeddings)
        
        self.logger.info(f"Batch embedding complete: {success_count} files, "
                        f"{total_embeddings} total embeddings, {failure_count} failed")
        
        if failed_files:
            self.logger.warning(f"Failed files: {failed_files}")
        
        return all_embeddings
    
    def get_embeddings_path(self, video_id: str) -> Path:
        """
        Get the file path for embeddings.
        
        Args:
            video_id: Video identifier
            
        Returns:
            Path to embeddings file
        """
        return self.embeddings_path / f"{video_id}.npy"
    
    def embeddings_exist(self, video_id: str) -> bool:
        """
        Check if embeddings already exist for a video.
        
        Args:
            video_id: Video identifier
            
        Returns:
            True if embeddings exist
        """
        return self.get_embeddings_path(video_id).exists()
    
    def load_embeddings(self, video_id: str) -> Optional[np.ndarray]:
        """
        Load existing embeddings for a video.
        
        Args:
            video_id: Video identifier
            
        Returns:
            NumPy array of embeddings if exists, None otherwise
        """
        embeddings_file = self.get_embeddings_path(video_id)
        
        if not embeddings_file.exists():
            return None
        
        embeddings = self.file_manager.load_numpy(embeddings_file)
        
        if embeddings is not None:
            # Validate loaded embeddings
            is_valid, errors = self.validators.validate_embeddings(embeddings, self.embedding_dim)
            if not is_valid:
                self.logger.warning(f"Loaded embeddings are invalid: {errors}")
                return None
        
        return embeddings
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        self.logger.debug("Embedding cache cleared")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'max_sequence_length': self.max_seq_length,
            'device': self.device,
            'batch_size': self.batch_size
        }
