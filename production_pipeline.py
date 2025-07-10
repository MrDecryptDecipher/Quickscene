#!/usr/bin/env python3
"""
Production Pipeline Controller

Orchestrates the complete offline processing pipeline:
transcriber.transcribe() → chunker.chunk() → embedder.encode() → indexer.build_index()

Follows PRD execution DAG for processing ALL videos.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import argparse

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.config import get_config
from app.production_transcriber import ProductionTranscriber
from app.production_chunker import ProductionChunker
from app.production_embedder import ProductionEmbedder
from app.production_indexer import ProductionIndexer
from app.production_query_handler import ProductionQueryHandler

class ProductionPipeline:
    """
    Complete production pipeline for processing all videos.
    Implements the execution DAG as specified in PRD.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize production pipeline"""
        self.config = get_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.transcriber = ProductionTranscriber(config_path)
        self.chunker = ProductionChunker(config_path)
        self.embedder = ProductionEmbedder(config_path)
        self.indexer = ProductionIndexer(config_path)
    
    def run_full_pipeline(self, force_rebuild: bool = False) -> bool:
        """
        Run the complete production pipeline.
        
        Args:
            force_rebuild: Force rebuild even if cached data exists
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.info("🚀 Starting Production Pipeline")
        self.logger.info("=" * 60)
        
        pipeline_start = time.time()
        
        try:
            # Step 1: Transcription
            self.logger.info("📝 Step 1: Transcribing all videos...")
            step_start = time.time()
            
            if force_rebuild:
                # Clear transcription cache
                for video_path in self.config.get_video_files():
                    video_id = self.config.get_video_id(video_path)
                    transcript_file = self.config.get_transcript_file(video_id)
                    if transcript_file.exists():
                        transcript_file.unlink()
            
            transcripts = self.transcriber.transcribe_all_videos()
            if not transcripts:
                self.logger.error("No videos were transcribed successfully")
                return False
            
            step_time = time.time() - step_start
            self.logger.info(f"✅ Transcription complete: {len(transcripts)} videos in {step_time:.1f}s")
            
            # Step 2: Chunking
            self.logger.info("✂️  Step 2: Chunking all transcripts...")
            step_start = time.time()
            
            if force_rebuild:
                # Clear chunking cache
                for video_path in self.config.get_video_files():
                    video_id = self.config.get_video_id(video_path)
                    chunks_file = self.config.get_chunks_file(video_id)
                    if chunks_file.exists():
                        chunks_file.unlink()
            
            all_chunks = self.chunker.chunk_all_transcripts()
            if not all_chunks:
                self.logger.error("No transcripts were chunked successfully")
                return False
            
            total_chunks = sum(len(chunks) for chunks in all_chunks.values())
            step_time = time.time() - step_start
            self.logger.info(f"✅ Chunking complete: {total_chunks} chunks from {len(all_chunks)} videos in {step_time:.1f}s")
            
            # Step 3: Embedding
            self.logger.info("🧮 Step 3: Generating embeddings for all chunks...")
            step_start = time.time()
            
            if force_rebuild:
                # Clear embedding cache
                for video_path in self.config.get_video_files():
                    video_id = self.config.get_video_id(video_path)
                    embedding_file = self.config.get_embedding_file(video_id)
                    if embedding_file.exists():
                        embedding_file.unlink()
            
            all_embeddings = self.embedder.embed_all_chunks()
            if not all_embeddings:
                self.logger.error("No embeddings were generated successfully")
                return False
            
            total_embeddings = sum(emb.shape[0] for emb in all_embeddings.values())
            step_time = time.time() - step_start
            self.logger.info(f"✅ Embedding complete: {total_embeddings} embeddings from {len(all_embeddings)} videos in {step_time:.1f}s")
            
            # Step 4: Indexing
            self.logger.info("🔍 Step 4: Building unified FAISS index...")
            step_start = time.time()
            
            if force_rebuild:
                # Force rebuild index
                success = self.indexer.rebuild_index()
            else:
                success = self.indexer.build_and_save_index()
            
            if not success:
                self.logger.error("Failed to build FAISS index")
                return False
            
            step_time = time.time() - step_start
            self.logger.info(f"✅ Indexing complete in {step_time:.1f}s")
            
            # Pipeline complete
            total_time = time.time() - pipeline_start
            self.logger.info("=" * 60)
            self.logger.info(f"🎉 Pipeline Complete! Total time: {total_time:.1f}s")
            
            # Show summary
            self._show_pipeline_summary()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return False
    
    def _show_pipeline_summary(self):
        """Show summary of the completed pipeline"""
        try:
            # Get summaries from each component
            transcript_summary = self.transcriber.get_transcript_summary()
            chunks_summary = self.chunker.get_chunks_summary()
            embeddings_summary = self.embedder.get_embeddings_summary()
            index_summary = self.indexer.get_index_summary()
            
            self.logger.info("📊 PIPELINE SUMMARY")
            self.logger.info("-" * 40)
            self.logger.info(f"Videos processed: {transcript_summary['transcribed_videos']}/{transcript_summary['total_videos']}")
            self.logger.info(f"Total duration: {transcript_summary['total_duration_seconds']:.1f}s")
            self.logger.info(f"Total segments: {transcript_summary['total_segments']}")
            self.logger.info(f"Total chunks: {chunks_summary['total_chunks']}")
            self.logger.info(f"Total embeddings: {embeddings_summary['total_embeddings']}")
            self.logger.info(f"Index vectors: {index_summary.get('total_vectors', 0)}")
            self.logger.info(f"Index size: {embeddings_summary['total_size_mb']:.1f} MB")
            
        except Exception as e:
            self.logger.warning(f"Failed to generate summary: {e}")
    
    def validate_pipeline(self) -> bool:
        """Validate the entire pipeline"""
        self.logger.info("🔍 Validating pipeline...")
        
        try:
            # Validate each component
            transcripts_valid = self.transcriber.validate_transcripts()
            chunks_valid = self.chunker.validate_chunks()
            embeddings_valid = self.embedder.validate_embeddings()
            index_valid = self.indexer.validate_index()
            
            all_valid = all([transcripts_valid, chunks_valid, embeddings_valid, index_valid])
            
            self.logger.info(f"Transcripts valid: {transcripts_valid}")
            self.logger.info(f"Chunks valid: {chunks_valid}")
            self.logger.info(f"Embeddings valid: {embeddings_valid}")
            self.logger.info(f"Index valid: {index_valid}")
            self.logger.info(f"Overall pipeline valid: {all_valid}")
            
            return all_valid
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return False
    
    def test_query_system(self) -> bool:
        """Test the query system with sample queries"""
        self.logger.info("🧪 Testing query system...")
        
        try:
            query_handler = ProductionQueryHandler(self.config.config_path)
            
            # Test queries
            test_queries = [
                "artificial intelligence",
                "machine learning",
                "AI",
                "technology",
                "programming"
            ]
            
            all_passed = True
            
            for query in test_queries:
                result = query_handler.query(query, top_k=3)
                
                query_time = result.get('query_time_ms', float('inf'))
                meets_requirement = query_time < 1000  # <1 second
                
                self.logger.info(f"Query '{query}': {query_time:.1f}ms - {'✅' if meets_requirement else '❌'}")
                
                if not meets_requirement:
                    all_passed = False
            
            self.logger.info(f"Query system test: {'✅ PASSED' if all_passed else '❌ FAILED'}")
            return all_passed
            
        except Exception as e:
            self.logger.error(f"Query system test failed: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status"""
        try:
            # Check if all components are ready
            video_files = self.config.get_video_files()
            
            status = {
                'videos_found': len(video_files),
                'transcripts_ready': 0,
                'chunks_ready': 0,
                'embeddings_ready': 0,
                'index_ready': False
            }
            
            # Check transcripts
            for video_path in video_files:
                video_id = self.config.get_video_id(video_path)
                if self.config.get_transcript_file(video_id).exists():
                    status['transcripts_ready'] += 1
            
            # Check chunks
            for video_path in video_files:
                video_id = self.config.get_video_id(video_path)
                if self.config.get_chunks_file(video_id).exists():
                    status['chunks_ready'] += 1
            
            # Check embeddings
            for video_path in video_files:
                video_id = self.config.get_video_id(video_path)
                if self.config.get_embedding_file(video_id).exists():
                    status['embeddings_ready'] += 1
            
            # Check index
            status['index_ready'] = (
                self.config.faiss_index_path.exists() and 
                self.config.metadata_path.exists()
            )
            
            # Calculate readiness percentage
            total_steps = len(video_files) * 3 + 1  # 3 steps per video + 1 index
            completed_steps = (
                status['transcripts_ready'] + 
                status['chunks_ready'] + 
                status['embeddings_ready'] + 
                (1 if status['index_ready'] else 0)
            )
            
            status['readiness_percent'] = (completed_steps / total_steps) * 100 if total_steps > 0 else 0
            status['pipeline_ready'] = status['readiness_percent'] == 100
            
            return status
            
        except Exception as e:
            return {'error': str(e)}

def main():
    """CLI interface for production pipeline"""
    parser = argparse.ArgumentParser(description="Quickscene Production Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--run", action="store_true", help="Run complete pipeline")
    parser.add_argument("--force", action="store_true", help="Force rebuild (ignore cache)")
    parser.add_argument("--validate", action="store_true", help="Validate pipeline")
    parser.add_argument("--test", action="store_true", help="Test query system")
    parser.add_argument("--status", action="store_true", help="Show system status")
    
    args = parser.parse_args()
    
    try:
        pipeline = ProductionPipeline(args.config)
        
        if args.status:
            status = pipeline.get_system_status()
            print(f"📊 System Status:")
            print(f"   Videos found: {status.get('videos_found', 0)}")
            print(f"   Transcripts ready: {status.get('transcripts_ready', 0)}")
            print(f"   Chunks ready: {status.get('chunks_ready', 0)}")
            print(f"   Embeddings ready: {status.get('embeddings_ready', 0)}")
            print(f"   Index ready: {status.get('index_ready', False)}")
            print(f"   Pipeline readiness: {status.get('readiness_percent', 0):.1f}%")
            return
        
        if args.validate:
            is_valid = pipeline.validate_pipeline()
            print(f"Pipeline validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
            return
        
        if args.test:
            test_passed = pipeline.test_query_system()
            print(f"Query system test: {'✅ PASSED' if test_passed else '❌ FAILED'}")
            return
        
        if args.run:
            success = pipeline.run_full_pipeline(force_rebuild=args.force)
            if success:
                print("🎉 Pipeline completed successfully!")
                
                # Run validation and test
                print("\n🔍 Running validation...")
                pipeline.validate_pipeline()
                
                print("\n🧪 Testing query system...")
                pipeline.test_query_system()
            else:
                print("❌ Pipeline failed!")
                sys.exit(1)
        else:
            print("Use --run to execute the pipeline, --status to check status")
    
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
