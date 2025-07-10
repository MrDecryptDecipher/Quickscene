#!/usr/bin/env python3
"""
Production FAISS Index Builder

Builds unified FAISS index from all video embeddings.
Creates single searchable index across all videos with metadata.
Follows PRD specifications for complete offline processing.
"""

import json
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import faiss

from .config import get_config

def seconds_to_video_timestamp(seconds: float, include_milliseconds: bool = False) -> str:
    """
    Convert seconds to standard video timestamp format.

    Args:
        seconds: Time in seconds (e.g., 575.1)
        include_milliseconds: Whether to include milliseconds

    Returns:
        Formatted timestamp:
        - Under 1 hour: MM:SS (e.g., "9:35")
        - Over 1 hour: HH:MM:SS (e.g., "1:05:22")
        - With milliseconds: HH:MM:SS.mmm (e.g., "1:05:22.500")
    """
    total_seconds = int(seconds)
    milliseconds = int((seconds - total_seconds) * 1000)

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        # Format: HH:MM:SS or HH:MM:SS.mmm
        if include_milliseconds and milliseconds > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
        else:
            return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        # Format: MM:SS or MM:SS.mmm
        if include_milliseconds and milliseconds > 0:
            return f"{minutes}:{secs:02d}.{milliseconds:03d}"
        else:
            return f"{minutes}:{secs:02d}"

class ProductionIndexer:
    """
    Production indexer that builds unified FAISS index from all embeddings.
    Implements the indexer.py module as specified in PRD.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize production indexer with config"""
        self.config = get_config(config_path)
        self.logger = logging.getLogger(__name__)
        self.index = None
        self.metadata = None
    
    def build_unified_index(self) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
        """
        Build unified FAISS index from all video embeddings.
        
        Returns:
            Tuple of (faiss_index, metadata_list)
        """
        self.logger.info("Building unified FAISS index from all video embeddings")
        start_time = time.time()
        
        # Collect all embeddings and metadata
        all_embeddings = []
        all_metadata = []
        video_files = self.config.get_video_files()
        
        for video_path in video_files:
            video_id = self.config.get_video_id(video_path)
            embedding_file = self.config.get_embedding_file(video_id)
            chunks_file = self.config.get_chunks_file(video_id)
            
            if not embedding_file.exists():
                self.logger.warning(f"Missing embeddings for {video_id}")
                continue
            
            if not chunks_file.exists():
                self.logger.warning(f"Missing chunks for {video_id}")
                continue
            
            try:
                # Load embeddings and chunks
                embeddings = np.load(embedding_file)
                with open(chunks_file, 'r') as f:
                    chunks = json.load(f)
                
                # Validate matching lengths
                if len(embeddings) != len(chunks):
                    self.logger.error(f"Mismatch between embeddings ({len(embeddings)}) and chunks ({len(chunks)}) for {video_id}")
                    continue
                
                # Add embeddings
                all_embeddings.append(embeddings)
                
                # Create metadata for each chunk
                for i, chunk in enumerate(chunks):
                    metadata = {
                        'video_id': video_id,
                        'chunk_id': chunk['chunk_id'],
                        'chunk_index': chunk['chunk_index'],
                        'start_time': chunk['start_time'],
                        'end_time': chunk['end_time'],
                        'duration': chunk['duration'],
                        'text': chunk['text'],
                        'segment_count': chunk.get('segment_count', 0),
                        'global_index': len(all_metadata),
                        'video_path': str(video_path)
                    }
                    all_metadata.append(metadata)
                
                self.logger.info(f"Added {len(embeddings)} embeddings from {video_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to load data for {video_id}: {e}")
                continue
        
        if not all_embeddings:
            raise ValueError("No embeddings found to build index")
        
        # Combine all embeddings
        combined_embeddings = np.vstack(all_embeddings)
        self.logger.info(f"Combined embeddings shape: {combined_embeddings.shape}")
        
        # Build FAISS index
        dimension = combined_embeddings.shape[1]
        
        # Use IndexFlatIP for exact cosine similarity (since embeddings are normalized)
        if self.config.index_type == "IndexFlatIP":
            index = faiss.IndexFlatIP(dimension)
        elif self.config.index_type == "IndexFlatL2":
            index = faiss.IndexFlatL2(dimension)
        else:
            # Default to IndexFlatIP
            index = faiss.IndexFlatIP(dimension)
            self.logger.warning(f"Unknown index type {self.config.index_type}, using IndexFlatIP")
        
        # Add embeddings to index
        index.add(combined_embeddings)
        
        build_time = time.time() - start_time
        self.logger.info(f"Built FAISS index with {index.ntotal} vectors in {build_time:.1f}s")
        
        return index, all_metadata
    
    def save_index(self, index: faiss.Index, metadata: List[Dict[str, Any]]):
        """
        Save FAISS index and metadata to disk.
        
        Args:
            index: FAISS index
            metadata: List of metadata dictionaries
        """
        # Save FAISS index
        faiss.write_index(index, str(self.config.faiss_index_path))
        self.logger.info(f"Saved FAISS index to {self.config.faiss_index_path}")
        
        # Save metadata
        metadata_with_info = {
            'created_at': datetime.now().isoformat(),
            'total_vectors': index.ntotal,
            'dimension': index.d,
            'index_type': type(index).__name__,
            'total_videos': len(set(item['video_id'] for item in metadata)),
            'total_chunks': len(metadata),
            'config': {
                'embedding_model': self.config.embedding_model,
                'chunk_duration_sec': self.config.chunk_duration_sec,
                'chunk_overlap_sec': self.config.chunk_overlap_sec
            },
            'metadata': metadata
        }
        
        with open(self.config.metadata_path, 'w') as f:
            json.dump(metadata_with_info, f, indent=2)
        
        self.logger.info(f"Saved metadata to {self.config.metadata_path}")
    
    def load_index(self) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
        """
        Load FAISS index and metadata from disk.
        
        Returns:
            Tuple of (faiss_index, metadata_list)
        """
        if not self.config.faiss_index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.config.faiss_index_path}")
        
        if not self.config.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.config.metadata_path}")
        
        # Load FAISS index
        index = faiss.read_index(str(self.config.faiss_index_path))
        self.logger.info(f"Loaded FAISS index with {index.ntotal} vectors")
        
        # Load metadata
        with open(self.config.metadata_path, 'r') as f:
            metadata_file = json.load(f)
        
        metadata = metadata_file.get('metadata', [])
        self.logger.info(f"Loaded metadata for {len(metadata)} chunks")
        
        return index, metadata
    
    def build_and_save_index(self) -> bool:
        """
        Build unified index and save to disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if index already exists and caching is enabled
            if (self.config.enable_caching and 
                self.config.faiss_index_path.exists() and 
                self.config.metadata_path.exists()):
                self.logger.info("Index already exists and caching is enabled")
                return True
            
            # Build index
            index, metadata = self.build_unified_index()
            
            # Save to disk
            self.save_index(index, metadata)
            
            # Store in instance
            self.index = index
            self.metadata = metadata
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to build and save index: {e}")
            return False
    
    def get_index_summary(self) -> Dict[str, Any]:
        """Get summary of the current index"""
        if not self.config.faiss_index_path.exists():
            return {"status": "Index not found"}
        
        try:
            index, metadata = self.load_index()
            
            # Calculate statistics
            video_ids = set(item['video_id'] for item in metadata)
            total_duration = sum(item['duration'] for item in metadata)
            
            # Load metadata file for additional info
            with open(self.config.metadata_path, 'r') as f:
                metadata_file = json.load(f)
            
            return {
                "status": "Index exists",
                "total_vectors": index.ntotal,
                "dimension": index.d,
                "index_type": type(index).__name__,
                "total_videos": len(video_ids),
                "total_chunks": len(metadata),
                "total_duration_seconds": total_duration,
                "avg_chunks_per_video": len(metadata) / len(video_ids) if video_ids else 0,
                "created_at": metadata_file.get('created_at', 'unknown'),
                "config": metadata_file.get('config', {}),
                "video_ids": sorted(list(video_ids))
            }
            
        except Exception as e:
            return {"status": f"Error loading index: {e}"}
    
    def validate_index(self) -> bool:
        """Validate the index and metadata consistency"""
        try:
            index, metadata = self.load_index()
            
            # Check if index and metadata sizes match
            if index.ntotal != len(metadata):
                self.logger.error(f"Index size ({index.ntotal}) doesn't match metadata size ({len(metadata)})")
                return False
            
            # Check metadata structure
            required_keys = ['video_id', 'chunk_id', 'start_time', 'end_time', 'text', 'global_index']
            for i, item in enumerate(metadata):
                if not all(key in item for key in required_keys):
                    self.logger.error(f"Invalid metadata structure at index {i}")
                    return False
                
                if item['global_index'] != i:
                    self.logger.error(f"Incorrect global_index at position {i}")
                    return False
            
            # Test search functionality
            if index.ntotal > 0:
                test_vector = np.random.random((1, index.d)).astype(np.float32)
                scores, indices = index.search(test_vector, min(5, index.ntotal))
                
                if len(indices[0]) == 0:
                    self.logger.error("Index search returned no results")
                    return False
            
            self.logger.info("Index validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Index validation failed: {e}")
            return False
    
    def rebuild_index(self) -> bool:
        """Force rebuild of the index"""
        self.logger.info("Force rebuilding index")
        
        # Remove existing index files
        if self.config.faiss_index_path.exists():
            self.config.faiss_index_path.unlink()
        
        if self.config.metadata_path.exists():
            self.config.metadata_path.unlink()
        
        # Build new index
        return self.build_and_save_index()

def main():
    """CLI interface for production indexer"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Production FAISS Index Builder")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--summary", action="store_true", help="Show index summary")
    parser.add_argument("--validate", action="store_true", help="Validate index")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild index")
    
    args = parser.parse_args()
    
    indexer = ProductionIndexer(args.config)
    
    if args.summary:
        summary = indexer.get_index_summary()
        print(json.dumps(summary, indent=2))
        return
    
    if args.validate:
        is_valid = indexer.validate_index()
        print(f"Index valid: {is_valid}")
        return
    
    if args.rebuild:
        success = indexer.rebuild_index()
        print(f"Index rebuild: {'Success' if success else 'Failed'}")
        return
    
    # Build index
    success = indexer.build_and_save_index()
    if success:
        summary = indexer.get_index_summary()
        print("Index built successfully!")
        print(f"Total vectors: {summary.get('total_vectors', 0)}")
        print(f"Total videos: {summary.get('total_videos', 0)}")
        print(f"Total chunks: {summary.get('total_chunks', 0)}")
    else:
        print("Failed to build index")

if __name__ == "__main__":
    main()
