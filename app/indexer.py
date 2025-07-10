"""
Indexer Module for Quickscene System

Builds and manages FAISS index for fast vector similarity search.
Handles metadata mapping and provides efficient search capabilities.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime

try:
    import faiss
except ImportError:
    raise ImportError("FAISS not installed. Run: pip install faiss-cpu")

from .utils.config_loader import ConfigLoader
from .utils.file_manager import FileManager
from .utils.validators import Validators


class Indexer:
    """
    FAISS-based vector index for fast similarity search.
    
    Features:
    - FAISS IndexFlatIP for exact cosine similarity search
    - Metadata management for chunk-to-video mapping
    - Batch index building and incremental updates
    - Persistent storage and loading
    - CPU-optimized performance
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the indexer with configuration.
        
        Args:
            config: Configuration dictionary containing index settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.file_manager = FileManager()
        self.validators = Validators()
        
        # Extract index configuration
        self.index_config = config.get('index', {})
        self.index_type = self.index_config.get('type', 'IndexFlatIP')
        self.metric = self.index_config.get('metric', 'METRIC_INNER_PRODUCT')
        
        # Setup paths
        self.index_path = Path(config['paths']['index'])
        self.file_manager.ensure_directory(self.index_path)
        
        self.index_file = self.index_path / config['paths'].get('index_file', 'superbryne.index')
        self.metadata_file = self.index_path / config['paths'].get('metadata_file', 'metadata.json')
        
        # Initialize index and metadata
        self.index = None
        self.metadata = []
        self.embedding_dim = None
        
        # Load existing index if available
        self._load_existing_index()
    
    def _load_existing_index(self) -> None:
        """Load existing FAISS index and metadata if available."""
        try:
            if self.index_file.exists() and self.metadata_file.exists():
                self.logger.info(f"Loading existing index: {self.index_file}")
                
                # Load FAISS index
                self.index = faiss.read_index(str(self.index_file))
                self.embedding_dim = self.index.d
                
                # Load metadata
                metadata = self.file_manager.load_json(self.metadata_file)
                if metadata and isinstance(metadata, list):
                    self.metadata = metadata
                    self.logger.info(f"Index loaded: {len(self.metadata)} vectors, "
                                   f"dimension: {self.embedding_dim}")
                else:
                    self.logger.warning("Invalid metadata file, resetting index")
                    self.index = None
                    self.metadata = []
            else:
                self.logger.info("No existing index found, will create new one")
                
        except Exception as e:
            self.logger.warning(f"Failed to load existing index: {e}")
            self.index = None
            self.metadata = []
    
    def build_index(self, all_embeddings: List[np.ndarray], 
                   all_chunks: List[List[Dict[str, Any]]]) -> None:
        """
        Build FAISS index from embeddings and chunk metadata.
        
        Args:
            all_embeddings: List of embedding arrays (one per video)
            all_chunks: List of chunk lists (one per video)
            
        Raises:
            ValueError: If input data is invalid
            RuntimeError: If index building fails
        """
        if not all_embeddings or not all_chunks:
            raise ValueError("No embeddings or chunks provided for index building")
        
        if len(all_embeddings) != len(all_chunks):
            raise ValueError("Number of embedding arrays must match number of chunk lists")
        
        self.logger.info(f"Building index from {len(all_embeddings)} videos")
        
        try:
            # Combine all embeddings and metadata
            combined_embeddings, combined_metadata = self._combine_data(all_embeddings, all_chunks)
            
            if len(combined_embeddings) == 0:
                raise ValueError("No valid embeddings found")
            
            # Validate embeddings
            is_valid, errors = self.validators.validate_embeddings(combined_embeddings)
            if not is_valid:
                raise ValueError(f"Invalid embeddings: {errors}")
            
            # Create FAISS index
            self._create_faiss_index(combined_embeddings)
            
            # Store metadata
            self.metadata = combined_metadata
            
            # Save index and metadata
            self._save_index()
            
            self.logger.info(f"Index built successfully: {len(self.metadata)} vectors, "
                           f"dimension: {self.embedding_dim}")
            
        except Exception as e:
            self.logger.error(f"Index building failed: {e}")
            raise RuntimeError(f"Index building failed: {e}")
    
    def _combine_data(self, all_embeddings: List[np.ndarray], 
                     all_chunks: List[List[Dict[str, Any]]]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Combine embeddings and chunks from multiple videos.
        
        Args:
            all_embeddings: List of embedding arrays
            all_chunks: List of chunk lists
            
        Returns:
            Tuple of (combined_embeddings, combined_metadata)
        """
        combined_embeddings = []
        combined_metadata = []
        
        for embeddings, chunks in zip(all_embeddings, all_chunks):
            # Validate that embeddings and chunks match
            if len(embeddings) != len(chunks):
                self.logger.warning(f"Embedding count ({len(embeddings)}) doesn't match "
                                  f"chunk count ({len(chunks)}) for video, skipping")
                continue
            
            # Add embeddings
            combined_embeddings.append(embeddings)
            
            # Add metadata with global index
            for i, chunk in enumerate(chunks):
                metadata_entry = {
                    'global_index': len(combined_metadata),
                    'video_id': chunk['video_id'],
                    'chunk_id': chunk['chunk_id'],
                    'start_time': chunk['start_time'],
                    'end_time': chunk['end_time'],
                    'text': chunk['text'],
                    'chunk_index': i
                }
                combined_metadata.append(metadata_entry)
        
        # Combine all embeddings
        if combined_embeddings:
            final_embeddings = np.vstack(combined_embeddings)
        else:
            final_embeddings = np.array([], dtype=np.float32).reshape(0, 384)  # Default dimension
        
        return final_embeddings, combined_metadata
    
    def _create_faiss_index(self, embeddings: np.ndarray) -> None:
        """
        Create FAISS index from embeddings.
        
        Args:
            embeddings: NumPy array of embeddings
        """
        self.embedding_dim = embeddings.shape[1]
        
        # Create index based on configuration
        if self.index_type == 'IndexFlatIP':
            # Inner Product index for cosine similarity (with normalized vectors)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == 'IndexFlatL2':
            # L2 distance index
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            # Default to IndexFlatIP
            self.logger.warning(f"Unknown index type {self.index_type}, using IndexFlatIP")
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Add embeddings to index
        self.logger.debug(f"Adding {len(embeddings)} vectors to FAISS index")
        self.index.add(embeddings)
        
        self.logger.info(f"FAISS index created: {self.index.ntotal} vectors")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the index.
        
        Args:
            query_embedding: Query vector to search for
            k: Number of results to return
            
        Returns:
            List of search results with metadata
            
        Raises:
            RuntimeError: If index is not built or search fails
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        if len(self.metadata) == 0:
            raise RuntimeError("No metadata available for search results")
        
        # Validate query embedding
        is_valid, errors = self.validators.validate_embeddings(query_embedding, self.embedding_dim)
        if not is_valid:
            raise ValueError(f"Invalid query embedding: {errors}")
        
        try:
            # Ensure query embedding is 2D
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Perform search
            self.logger.debug(f"Searching index for top-{k} results")
            scores, indices = self.index.search(query_embedding, k)
            
            # Format results
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                score = float(scores[0][i])
                
                # Skip invalid indices
                if idx < 0 or idx >= len(self.metadata):
                    continue
                
                # Get metadata for this result
                metadata = self.metadata[idx]
                
                result = {
                    'video_id': metadata['video_id'],
                    'chunk_id': metadata['chunk_id'],
                    'start_time': metadata['start_time'],
                    'end_time': metadata['end_time'],
                    'text': metadata['text'],
                    'confidence': score,
                    'global_index': metadata['global_index']
                }
                
                results.append(result)
            
            self.logger.debug(f"Search completed: {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise RuntimeError(f"Search failed: {e}")
    
    def add_embeddings(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]) -> None:
        """
        Add new embeddings to existing index (incremental update).
        
        Args:
            embeddings: New embeddings to add
            chunks: Corresponding chunk metadata
            
        Raises:
            ValueError: If input data is invalid
            RuntimeError: If addition fails
        """
        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")
        
        # Validate embeddings
        is_valid, errors = self.validators.validate_embeddings(embeddings, self.embedding_dim)
        if not is_valid:
            raise ValueError(f"Invalid embeddings: {errors}")
        
        try:
            # If no index exists, create one
            if self.index is None:
                self._create_faiss_index(embeddings)
                self.metadata = []
            else:
                # Add to existing index
                self.index.add(embeddings)
            
            # Add metadata
            start_index = len(self.metadata)
            for i, chunk in enumerate(chunks):
                metadata_entry = {
                    'global_index': start_index + i,
                    'video_id': chunk['video_id'],
                    'chunk_id': chunk['chunk_id'],
                    'start_time': chunk['start_time'],
                    'end_time': chunk['end_time'],
                    'text': chunk['text'],
                    'chunk_index': i
                }
                self.metadata.append(metadata_entry)
            
            # Save updated index
            self._save_index()
            
            self.logger.info(f"Added {len(embeddings)} vectors to index. "
                           f"Total: {len(self.metadata)} vectors")
            
        except Exception as e:
            self.logger.error(f"Failed to add embeddings to index: {e}")
            raise RuntimeError(f"Failed to add embeddings: {e}")
    
    def _save_index(self) -> None:
        """Save FAISS index and metadata to disk."""
        try:
            if self.index is not None:
                # Save FAISS index
                faiss.write_index(self.index, str(self.index_file))
                self.logger.debug(f"FAISS index saved: {self.index_file}")
            
            # Save metadata
            if self.metadata:
                metadata_with_info = {
                    'vectors': self.metadata,
                    'index_info': {
                        'total_vectors': len(self.metadata),
                        'embedding_dimension': self.embedding_dim,
                        'index_type': self.index_type,
                        'created_at': datetime.now().isoformat(),
                        'last_updated': datetime.now().isoformat()
                    }
                }
                
                if not self.file_manager.save_json(metadata_with_info, self.metadata_file):
                    self.logger.warning(f"Failed to save metadata: {self.metadata_file}")
                else:
                    self.logger.debug(f"Metadata saved: {self.metadata_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save index: {e}")
            raise
    
    def rebuild_from_embeddings(self) -> None:
        """Rebuild index from existing embedding files."""
        embeddings_dir = Path(self.config['paths']['embeddings'])
        chunks_dir = Path(self.config['paths']['chunks'])
        
        if not embeddings_dir.exists() or not chunks_dir.exists():
            raise FileNotFoundError("Embeddings or chunks directory not found")
        
        # Find all embedding files
        embedding_files = list(embeddings_dir.glob("*.npy"))
        
        if not embedding_files:
            raise FileNotFoundError("No embedding files found")
        
        self.logger.info(f"Rebuilding index from {len(embedding_files)} embedding files")
        
        all_embeddings = []
        all_chunks = []
        
        for embedding_file in embedding_files:
            video_id = embedding_file.stem
            
            # Load embeddings
            embeddings = self.file_manager.load_numpy(embedding_file)
            if embeddings is None:
                self.logger.warning(f"Failed to load embeddings: {embedding_file}")
                continue
            
            # Load corresponding chunks
            chunks_file = chunks_dir / f"{video_id}_chunks.json"
            chunks = self.file_manager.load_json(chunks_file)
            if not chunks:
                self.logger.warning(f"Failed to load chunks: {chunks_file}")
                continue
            
            all_embeddings.append(embeddings)
            all_chunks.append(chunks)
        
        # Build index
        self.build_index(all_embeddings, all_chunks)
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current index.
        
        Returns:
            Dictionary containing index statistics
        """
        stats = {
            'index_exists': self.index is not None,
            'total_vectors': len(self.metadata) if self.metadata else 0,
            'embedding_dimension': self.embedding_dim,
            'index_type': self.index_type,
            'index_file_exists': self.index_file.exists(),
            'metadata_file_exists': self.metadata_file.exists()
        }
        
        if self.index is not None:
            stats['faiss_total'] = self.index.ntotal
            stats['faiss_dimension'] = self.index.d
        
        return stats
    
    def clear_index(self) -> None:
        """Clear the current index and metadata."""
        self.index = None
        self.metadata = []
        self.embedding_dim = None
        
        # Remove files
        if self.index_file.exists():
            self.index_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        
        self.logger.info("Index cleared")
