#!/usr/bin/env python3
"""
Production Transcript Chunking Module

Splits transcripts into N-second chunks for embedding.
Processes ALL transcript files and outputs chunks/{video_id}_chunks.json
Follows PRD specifications for complete offline processing.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
from tqdm import tqdm

from .config import get_config

class ProductionChunker:
    """
    Production chunker that processes all transcripts into chunks.
    Implements the chunker.py module as specified in PRD.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize production chunker with config"""
        self.config = get_config(config_path)
        self.logger = logging.getLogger(__name__)
    
    def chunk_transcript(self, transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split transcript into time-based chunks.
        
        Args:
            transcript: Transcript dictionary with segments
            
        Returns:
            List of chunk dictionaries
        """
        video_id = transcript['video_id']
        segments = transcript.get('segments', [])
        
        if not segments:
            self.logger.warning(f"No segments found in transcript for {video_id}")
            return []
        
        chunks = []
        chunk_duration = self.config.chunk_duration_sec
        overlap = self.config.chunk_overlap_sec
        
        # Calculate actual duration from segments if not provided
        duration = transcript.get('duration', 0)
        if duration == 0 and segments:
            duration = max(seg['end'] for seg in segments)

        if duration == 0:
            self.logger.warning(f"No duration found for {video_id}")
            return []

        # Group segments into chunks based on time
        current_chunk_start = 0
        chunk_id = 0

        while current_chunk_start < duration:
            chunk_end = current_chunk_start + chunk_duration
            
            # Find segments that overlap with this chunk
            chunk_segments = []
            chunk_text_parts = []
            
            for segment in segments:
                seg_start = segment['start']
                seg_end = segment['end']
                
                # Check if segment overlaps with chunk time window
                if seg_start < chunk_end and seg_end > current_chunk_start:
                    chunk_segments.append(segment)
                    chunk_text_parts.append(segment['text'].strip())
            
            if chunk_segments:
                # Calculate actual chunk boundaries
                actual_start = min(seg['start'] for seg in chunk_segments)
                actual_end = max(seg['end'] for seg in chunk_segments)
                
                # Create chunk
                chunk = {
                    'chunk_id': f"{video_id}_{chunk_id:04d}",
                    'video_id': video_id,
                    'chunk_index': chunk_id,
                    'start_time': actual_start,
                    'end_time': actual_end,
                    'duration': actual_end - actual_start,
                    'text': ' '.join(chunk_text_parts),
                    'segment_count': len(chunk_segments),
                    'segments': chunk_segments,
                    'metadata': {
                        'target_start': current_chunk_start,
                        'target_end': chunk_end,
                        'created_at': datetime.now().isoformat()
                    }
                }
                
                chunks.append(chunk)
                chunk_id += 1
            
            # Move to next chunk with overlap
            current_chunk_start = chunk_end - overlap
            
            # Prevent infinite loop
            if current_chunk_start >= duration:
                break
        
        self.logger.info(f"Created {len(chunks)} chunks for {video_id}")
        return chunks
    
    def chunk_video_transcript(self, video_id: str) -> List[Dict[str, Any]]:
        """
        Chunk transcript for a specific video.
        
        Args:
            video_id: Video identifier
            
        Returns:
            List of chunks
        """
        transcript_file = self.config.get_transcript_file(video_id)
        chunks_file = self.config.get_chunks_file(video_id)
        
        # Check if chunks already exist and caching is enabled
        if self.config.enable_caching and chunks_file.exists():
            self.logger.info(f"Using cached chunks for {video_id}")
            with open(chunks_file, 'r') as f:
                return json.load(f)
        
        # Load transcript
        if not transcript_file.exists():
            self.logger.error(f"Transcript file not found for {video_id}: {transcript_file}")
            return []
        
        try:
            with open(transcript_file, 'r') as f:
                transcript = json.load(f)
            
            # Create chunks
            chunks = self.chunk_transcript(transcript)
            
            # Save chunks to file
            with open(chunks_file, 'w') as f:
                json.dump(chunks, f, indent=2)
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Failed to chunk transcript for {video_id}: {e}")
            return []
    
    def chunk_all_transcripts(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Chunk all transcript files.
        
        Returns:
            Dictionary mapping video_id to chunks list
        """
        video_files = self.config.get_video_files()
        all_chunks = {}
        
        if not video_files:
            self.logger.warning(f"No video files found in {self.config.video_path}")
            return {}
        
        self.logger.info(f"Chunking transcripts for {len(video_files)} videos")
        
        for video_path in tqdm(video_files, desc="Chunking transcripts"):
            video_id = self.config.get_video_id(video_path)
            
            try:
                chunks = self.chunk_video_transcript(video_id)
                if chunks:
                    all_chunks[video_id] = chunks
                    
            except Exception as e:
                self.logger.error(f"Failed to chunk transcript for {video_id}: {e}")
                continue
        
        self.logger.info(f"Successfully chunked {len(all_chunks)}/{len(video_files)} transcripts")
        return all_chunks
    
    def get_chunks_summary(self) -> Dict[str, Any]:
        """Get summary of all chunked transcripts"""
        video_files = self.config.get_video_files()
        chunks_info = {}
        total_chunks = 0
        total_duration = 0
        
        for video_path in video_files:
            video_id = self.config.get_video_id(video_path)
            chunks_file = self.config.get_chunks_file(video_id)
            
            if chunks_file.exists():
                try:
                    with open(chunks_file, 'r') as f:
                        chunks = json.load(f)
                    
                    chunk_count = len(chunks)
                    chunk_duration = sum(chunk.get('duration', 0) for chunk in chunks)
                    
                    chunks_info[video_id] = {
                        'chunk_count': chunk_count,
                        'total_duration': chunk_duration,
                        'avg_chunk_duration': chunk_duration / chunk_count if chunk_count > 0 else 0,
                        'created_at': chunks[0].get('metadata', {}).get('created_at', 'unknown') if chunks else 'unknown'
                    }
                    
                    total_chunks += chunk_count
                    total_duration += chunk_duration
                    
                except Exception as e:
                    self.logger.error(f"Failed to read chunks for {video_id}: {e}")
                    continue
        
        return {
            'total_videos': len(video_files),
            'chunked_videos': len(chunks_info),
            'total_chunks': total_chunks,
            'total_duration_seconds': total_duration,
            'avg_chunks_per_video': total_chunks / len(chunks_info) if chunks_info else 0,
            'chunks_info': chunks_info
        }
    
    def validate_chunks(self) -> bool:
        """Validate all chunk files"""
        video_files = self.config.get_video_files()
        valid_count = 0
        
        for video_path in video_files:
            video_id = self.config.get_video_id(video_path)
            chunks_file = self.config.get_chunks_file(video_id)
            
            if not chunks_file.exists():
                self.logger.warning(f"Missing chunks file for {video_id}")
                continue
            
            try:
                with open(chunks_file, 'r') as f:
                    chunks = json.load(f)
                
                if not isinstance(chunks, list):
                    self.logger.error(f"Chunks should be a list for {video_id}")
                    continue
                
                # Validate chunk structure
                valid_chunks = 0
                for chunk in chunks:
                    required_keys = ['chunk_id', 'video_id', 'start_time', 'end_time', 'text']
                    if all(key in chunk for key in required_keys):
                        valid_chunks += 1
                    else:
                        self.logger.error(f"Invalid chunk structure in {video_id}")
                        break
                
                if valid_chunks == len(chunks):
                    valid_count += 1
                    
            except Exception as e:
                self.logger.error(f"Failed to validate chunks for {video_id}: {e}")
                continue
        
        self.logger.info(f"Validated {valid_count}/{len(video_files)} chunk files")
        return valid_count == len(video_files)

def main():
    """CLI interface for production chunker"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Production Transcript Chunker")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--summary", action="store_true", help="Show chunks summary")
    parser.add_argument("--validate", action="store_true", help="Validate chunks")
    parser.add_argument("--force", action="store_true", help="Force re-chunking")
    parser.add_argument("--video-id", help="Process specific video only")
    
    args = parser.parse_args()
    
    chunker = ProductionChunker(args.config)
    
    if args.summary:
        summary = chunker.get_chunks_summary()
        print(json.dumps(summary, indent=2))
        return
    
    if args.validate:
        is_valid = chunker.validate_chunks()
        print(f"Chunks valid: {is_valid}")
        return
    
    if args.force:
        # Clear cache to force re-chunking
        for video_path in chunker.config.get_video_files():
            video_id = chunker.config.get_video_id(video_path)
            chunks_file = chunker.config.get_chunks_file(video_id)
            if chunks_file.exists():
                chunks_file.unlink()
    
    # Chunk transcripts
    if args.video_id:
        chunks = chunker.chunk_video_transcript(args.video_id)
        print(f"Created {len(chunks)} chunks for {args.video_id}")
    else:
        all_chunks = chunker.chunk_all_transcripts()
        total_chunks = sum(len(chunks) for chunks in all_chunks.values())
        print(f"Created {total_chunks} chunks across {len(all_chunks)} videos")

if __name__ == "__main__":
    main()
