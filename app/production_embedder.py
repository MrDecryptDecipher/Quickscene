#!/usr/bin/env python3
"""
Production Embedding Generation Module

Generates embeddings for all chunked transcripts using SentenceTransformers.
Outputs embeddings/{video_id}.npy files for indexing.
Follows PRD specifications for complete offline processing.
"""

import json
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from .config import get_config

class ProductionEmbedder:
    """
    Production embedder that generates embeddings for all chunks.
    Implements the embedder.py module as specified in PRD.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize production embedder with config"""
        self.config = get_config(config_path)
        self.logger = logging.getLogger(__name__)
        self.model = None
    
    def _load_model(self):
        """Load SentenceTransformer model (lazy loading)"""
        if self.model is None:
            self.logger.info(f"Loading SentenceTransformer model: {self.config.embedding_model}")
            self.model = SentenceTransformer(self.config.embedding_model)
            self.logger.info("SentenceTransformer model loaded successfully")
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generate embeddings for a list of chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Numpy array of embeddings (n_chunks, embedding_dim)
        """
        self._load_model()
        
        if not chunks:
            return np.array([]).reshape(0, self.config.embedding_dimension)
        
        # Extract text from chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings in batches
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalize for cosine similarity
            show_progress_bar=False
        )
        
        # Ensure correct dtype
        embeddings = embeddings.astype(np.float32)
        
        # Validate embedding dimensions
        expected_dim = self.config.embedding_dimension
        if embeddings.shape[1] != expected_dim:
            self.logger.warning(f"Embedding dimension mismatch: got {embeddings.shape[1]}, expected {expected_dim}")
        
        return embeddings
    
    def embed_video_chunks(self, video_id: str) -> Optional[np.ndarray]:
        """
        Generate embeddings for chunks of a specific video.
        
        Args:
            video_id: Video identifier
            
        Returns:
            Numpy array of embeddings or None if failed
        """
        chunks_file = self.config.get_chunks_file(video_id)
        embedding_file = self.config.get_embedding_file(video_id)
        
        # Check if embeddings already exist and caching is enabled
        if self.config.enable_caching and embedding_file.exists():
            self.logger.info(f"Using cached embeddings for {video_id}")
            return np.load(embedding_file)
        
        # Load chunks
        if not chunks_file.exists():
            self.logger.error(f"Chunks file not found for {video_id}: {chunks_file}")
            return None
        
        try:
            with open(chunks_file, 'r') as f:
                chunks = json.load(f)
            
            if not chunks:
                self.logger.warning(f"No chunks found for {video_id}")
                return np.array([]).reshape(0, self.config.embedding_dimension)
            
            self.logger.info(f"Generating embeddings for {len(chunks)} chunks from {video_id}")
            start_time = time.time()
            
            # Generate embeddings
            embeddings = self.embed_chunks(chunks)
            
            processing_time = time.time() - start_time
            
            # Save embeddings to file
            np.save(embedding_file, embeddings)
            
            self.logger.info(f"Generated embeddings for {video_id}: {embeddings.shape} in {processing_time:.1f}s")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings for {video_id}: {e}")
            return None
    
    def embed_all_chunks(self) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for all video chunks.
        
        Returns:
            Dictionary mapping video_id to embeddings array
        """
        video_files = self.config.get_video_files()
        all_embeddings = {}
        
        if not video_files:
            self.logger.warning(f"No video files found in {self.config.video_path}")
            return {}
        
        self.logger.info(f"Generating embeddings for {len(video_files)} videos")
        
        for video_path in tqdm(video_files, desc="Generating embeddings"):
            video_id = self.config.get_video_id(video_path)
            
            try:
                embeddings = self.embed_video_chunks(video_id)
                if embeddings is not None and embeddings.size > 0:
                    all_embeddings[video_id] = embeddings
                    
            except Exception as e:
                self.logger.error(f"Failed to generate embeddings for {video_id}: {e}")
                continue
        
        self.logger.info(f"Successfully generated embeddings for {len(all_embeddings)}/{len(video_files)} videos")
        return all_embeddings
    
    def get_embeddings_summary(self) -> Dict[str, Any]:
        """Get summary of all generated embeddings"""
        video_files = self.config.get_video_files()
        embeddings_info = {}
        total_embeddings = 0
        total_size_mb = 0
        
        for video_path in video_files:
            video_id = self.config.get_video_id(video_path)
            embedding_file = self.config.get_embedding_file(video_id)
            
            if embedding_file.exists():
                try:
                    embeddings = np.load(embedding_file)
                    
                    # Calculate file size
                    file_size_mb = embedding_file.stat().st_size / (1024 * 1024)
                    
                    embeddings_info[video_id] = {
                        'shape': embeddings.shape,
                        'dtype': str(embeddings.dtype),
                        'size_mb': file_size_mb,
                        'num_embeddings': embeddings.shape[0] if embeddings.ndim > 1 else 0,
                        'embedding_dim': embeddings.shape[1] if embeddings.ndim > 1 else 0
                    }
                    
                    total_embeddings += embeddings.shape[0] if embeddings.ndim > 1 else 0
                    total_size_mb += file_size_mb
                    
                except Exception as e:
                    self.logger.error(f"Failed to read embeddings for {video_id}: {e}")
                    continue
        
        return {
            'total_videos': len(video_files),
            'embedded_videos': len(embeddings_info),
            'total_embeddings': total_embeddings,
            'total_size_mb': total_size_mb,
            'avg_embeddings_per_video': total_embeddings / len(embeddings_info) if embeddings_info else 0,
            'embeddings_info': embeddings_info
        }
    
    def validate_embeddings(self) -> bool:
        """Validate all embedding files"""
        video_files = self.config.get_video_files()
        valid_count = 0
        expected_dim = self.config.embedding_dimension
        
        for video_path in video_files:
            video_id = self.config.get_video_id(video_path)
            embedding_file = self.config.get_embedding_file(video_id)
            
            if not embedding_file.exists():
                self.logger.warning(f"Missing embeddings file for {video_id}")
                continue
            
            try:
                embeddings = np.load(embedding_file)
                
                # Validate shape
                if embeddings.ndim != 2:
                    self.logger.error(f"Invalid embedding shape for {video_id}: {embeddings.shape}")
                    continue
                
                # Validate dimension
                if embeddings.shape[1] != expected_dim:
                    self.logger.error(f"Invalid embedding dimension for {video_id}: {embeddings.shape[1]}, expected {expected_dim}")
                    continue
                
                # Validate dtype
                if embeddings.dtype != np.float32:
                    self.logger.warning(f"Unexpected dtype for {video_id}: {embeddings.dtype}")
                
                # Check for NaN or infinite values
                if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
                    self.logger.error(f"Invalid values (NaN/Inf) in embeddings for {video_id}")
                    continue
                
                valid_count += 1
                
            except Exception as e:
                self.logger.error(f"Failed to validate embeddings for {video_id}: {e}")
                continue
        
        self.logger.info(f"Validated {valid_count}/{len(video_files)} embedding files")
        return valid_count == len(video_files)
    
    def get_combined_embeddings(self) -> tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Get all embeddings combined into a single matrix with metadata.
        
        Returns:
            Tuple of (embeddings_matrix, metadata_list)
        """
        video_files = self.config.get_video_files()
        all_embeddings = []
        all_metadata = []
        
        for video_path in video_files:
            video_id = self.config.get_video_id(video_path)
            embedding_file = self.config.get_embedding_file(video_id)
            chunks_file = self.config.get_chunks_file(video_id)
            
            if not embedding_file.exists() or not chunks_file.exists():
                continue
            
            try:
                # Load embeddings and chunks
                embeddings = np.load(embedding_file)
                with open(chunks_file, 'r') as f:
                    chunks = json.load(f)
                
                # Ensure matching lengths
                if len(embeddings) != len(chunks):
                    self.logger.warning(f"Mismatch between embeddings and chunks for {video_id}")
                    continue
                
                # Add to combined arrays
                all_embeddings.append(embeddings)
                
                # Create metadata for each chunk
                for i, chunk in enumerate(chunks):
                    metadata = {
                        'video_id': video_id,
                        'chunk_id': chunk['chunk_id'],
                        'chunk_index': chunk['chunk_index'],
                        'start_time': chunk['start_time'],
                        'end_time': chunk['end_time'],
                        'text': chunk['text'],
                        'global_index': len(all_metadata)
                    }
                    all_metadata.append(metadata)
                
            except Exception as e:
                self.logger.error(f"Failed to load embeddings/chunks for {video_id}: {e}")
                continue
        
        if all_embeddings:
            combined_embeddings = np.vstack(all_embeddings)
            self.logger.info(f"Combined embeddings shape: {combined_embeddings.shape}")
            return combined_embeddings, all_metadata
        else:
            return np.array([]).reshape(0, self.config.embedding_dimension), []

def main():
    """CLI interface for production embedder"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Production Embedding Generator")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--summary", action="store_true", help="Show embeddings summary")
    parser.add_argument("--validate", action="store_true", help="Validate embeddings")
    parser.add_argument("--force", action="store_true", help="Force re-embedding")
    parser.add_argument("--video-id", help="Process specific video only")
    
    args = parser.parse_args()
    
    embedder = ProductionEmbedder(args.config)
    
    if args.summary:
        summary = embedder.get_embeddings_summary()
        print(json.dumps(summary, indent=2))
        return
    
    if args.validate:
        is_valid = embedder.validate_embeddings()
        print(f"Embeddings valid: {is_valid}")
        return
    
    if args.force:
        # Clear cache to force re-embedding
        for video_path in embedder.config.get_video_files():
            video_id = embedder.config.get_video_id(video_path)
            embedding_file = embedder.config.get_embedding_file(video_id)
            if embedding_file.exists():
                embedding_file.unlink()
    
    # Generate embeddings
    if args.video_id:
        embeddings = embedder.embed_video_chunks(args.video_id)
        if embeddings is not None:
            print(f"Generated embeddings for {args.video_id}: {embeddings.shape}")
    else:
        all_embeddings = embedder.embed_all_chunks()
        total_embeddings = sum(emb.shape[0] for emb in all_embeddings.values())
        print(f"Generated {total_embeddings} embeddings across {len(all_embeddings)} videos")

if __name__ == "__main__":
    main()
