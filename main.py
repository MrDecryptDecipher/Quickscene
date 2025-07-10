#!/usr/bin/env python3
"""
Quickscene Main Entry Point
Video timestamp-retrieval system CLI interface

Usage:
    python main.py process [--config CONFIG] [--videos VIDEO_DIR]
    python main.py query "search query" [--config CONFIG] [--top-k K]
    python main.py build-index [--config CONFIG]
    python main.py --help
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from app.utils.config_loader import ConfigLoader
    from app.transcriber import Transcriber
    from app.chunker import Chunker
    from app.embedder import Embedder
    from app.indexer import Indexer
    from app.query_handler import QueryHandler
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Please install dependencies by running: ./scripts/setup.sh")
    sys.exit(1)


def setup_logging(config: dict) -> None:
    """Setup logging configuration."""
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_config.get('file', 'quickscene.log'))
        ]
    )


def process_videos(config: dict, video_dir: Optional[str] = None) -> None:
    """Process videos through the complete pipeline."""
    logger = logging.getLogger(__name__)
    logger.info("Starting video processing pipeline...")
    
    # Initialize components
    transcriber = Transcriber(config)
    chunker = Chunker(config)
    embedder = Embedder(config)
    indexer = Indexer(config)
    
    # Get video directory
    videos_path = Path(video_dir) if video_dir else Path(config['paths']['videos'])
    
    if not videos_path.exists():
        logger.error(f"Video directory not found: {videos_path}")
        return
    
    # Process each video
    video_files = list(videos_path.glob("*.mp4")) + list(videos_path.glob("*.avi")) + list(videos_path.glob("*.mov"))
    
    if not video_files:
        logger.warning(f"No video files found in {videos_path}")
        return
    
    logger.info(f"Found {len(video_files)} video files to process")
    
    all_embeddings = []
    all_metadata = []
    
    for video_file in video_files:
        logger.info(f"Processing: {video_file.name}")
        
        try:
            # Step 1: Transcribe
            transcript = transcriber.transcribe(str(video_file))
            
            # Step 2: Chunk
            chunks = chunker.chunk_transcript(transcript)
            
            # Step 3: Embed
            embeddings = embedder.embed_chunks(chunks)
            
            # Store for indexing
            all_embeddings.extend(embeddings)
            all_metadata.extend(chunks)
            
            logger.info(f"✅ Processed {video_file.name}: {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"❌ Failed to process {video_file.name}: {e}")
            continue
    
    # Step 4: Build index
    if all_embeddings:
        logger.info("Building search index...")
        indexer.build_index(all_embeddings, all_metadata)
        logger.info("✅ Pipeline complete!")
    else:
        logger.warning("No embeddings generated - index not built")


def query_videos(config: dict, query: str, top_k: int = 3) -> None:
    """Query the video index."""
    logger = logging.getLogger(__name__)
    logger.info(f"Processing query: '{query}'")
    
    try:
        query_handler = QueryHandler(config)
        results = query_handler.process_query(query, top_k=top_k)
        
        if results:
            print(f"\n🔍 Found {len(results)} results for: '{query}'\n")
            for i, result in enumerate(results, 1):
                print(f"{i}. Video: {result['video_id']}")
                print(f"   Time: {result['start_time']:.1f}s - {result['end_time']:.1f}s")
                print(f"   Confidence: {result['confidence']:.3f}")
                print(f"   Text: {result['text'][:100]}...")
                print()
        else:
            print(f"❌ No results found for: '{query}'")
            
    except Exception as e:
        logger.error(f"Query failed: {e}")
        print(f"❌ Query failed: {e}")


def build_index(config: dict) -> None:
    """Build search index from existing embeddings."""
    logger = logging.getLogger(__name__)
    logger.info("Building search index from existing embeddings...")
    
    try:
        indexer = Indexer(config)
        indexer.rebuild_from_embeddings()
        logger.info("✅ Index built successfully!")
        print("✅ Search index built successfully!")
        
    except Exception as e:
        logger.error(f"Index build failed: {e}")
        print(f"❌ Index build failed: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Quickscene: Video timestamp-retrieval system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py process --videos ./data/videos/
  python main.py query "person talking about AI"
  python main.py build-index
        """
    )
    
    parser.add_argument(
        'command',
        choices=['process', 'query', 'build-index'],
        help='Command to execute'
    )
    
    parser.add_argument(
        'query_text',
        nargs='?',
        help='Query text (required for query command)'
    )
    
    parser.add_argument(
        '--config',
        default='config/default.yaml',
        help='Configuration file path (default: config/default.yaml)'
    )
    
    parser.add_argument(
        '--videos',
        help='Video directory path (overrides config)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Number of results to return (default: 3)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config_loader = ConfigLoader()
        config = config_loader.load_config(args.config)
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        sys.exit(1)
    
    # Setup logging
    setup_logging(config)
    
    # Execute command
    if args.command == 'process':
        process_videos(config, args.videos)
    elif args.command == 'query':
        if not args.query_text:
            print("❌ Query text required for query command")
            parser.print_help()
            sys.exit(1)
        query_videos(config, args.query_text, args.top_k)
    elif args.command == 'build-index':
        build_index(config)


if __name__ == "__main__":
    main()
