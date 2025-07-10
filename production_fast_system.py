#!/usr/bin/env python3
"""
Quickscene Production Fast System

Real implementation with optimized processing for sub-700ms queries.
This processes one video quickly to demonstrate the production system.
"""

import time
import json
import numpy as np
from pathlib import Path
import sys
import os

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

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

def process_single_video_fast():
    """Process one video quickly for demonstration."""
    
    print("🚀 Quickscene Production Fast System")
    print("=" * 50)
    
    # Check for videos
    video_dir = Path("data/videos")
    video_files = list(video_dir.glob("*.mp4"))
    
    if not video_files:
        print("❌ No videos found in data/videos/")
        return
    
    # Pick the shortest video for fast processing
    target_video = video_files[0]  # Use first video
    print(f"📹 Processing: {target_video.name}")
    
    # Step 1: Fast transcription (using tiny model for speed)
    print("\n🎤 Step 1: Fast Transcription...")
    start_time = time.time()
    
    try:
        import whisper
        
        # Use tiny model for speed
        print("   Loading Whisper tiny model...")
        model = whisper.load_model("tiny")
        
        print(f"   Transcribing {target_video.name}...")
        result = model.transcribe(str(target_video), language="en")
        
        # Create transcript structure
        transcript = {
            'video_id': target_video.stem,
            'duration': result.get('duration', 0),
            'language': result.get('language', 'en'),
            'segments': []
        }
        
        for segment in result.get('segments', []):
            transcript['segments'].append({
                'start': float(segment['start']),
                'end': float(segment['end']),
                'text': segment['text'].strip()
            })
        
        transcription_time = time.time() - start_time
        print(f"   ✅ Transcription complete: {transcription_time:.1f}s")
        print(f"   📊 Generated {len(transcript['segments'])} segments")
        
    except Exception as e:
        print(f"   ❌ Transcription failed: {e}")
        return
    
    # Step 2: Fast chunking
    print("\n✂️  Step 2: Fast Chunking...")
    start_time = time.time()
    
    chunks = []
    chunk_duration = 15  # 15 second chunks
    
    for i, segment in enumerate(transcript['segments']):
        chunk = {
            'video_id': transcript['video_id'],
            'chunk_id': f"{transcript['video_id']}_{i:04d}",
            'start_time': segment['start'],
            'end_time': segment['end'],
            'text': segment['text'],
            'duration': segment['end'] - segment['start']
        }
        chunks.append(chunk)
    
    chunking_time = time.time() - start_time
    print(f"   ✅ Chunking complete: {chunking_time:.3f}s")
    print(f"   📊 Generated {len(chunks)} chunks")
    
    # Step 3: Fast embedding
    print("\n🧮 Step 3: Fast Embedding...")
    start_time = time.time()
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Use fast model
        print("   Loading SentenceTransformer model...")
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        print(f"   Generating embeddings for {len(texts)} chunks...")
        embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        embeddings = embeddings.astype(np.float32)
        
        embedding_time = time.time() - start_time
        print(f"   ✅ Embedding complete: {embedding_time:.1f}s")
        print(f"   📊 Generated embeddings: {embeddings.shape}")
        
    except Exception as e:
        print(f"   ❌ Embedding failed: {e}")
        return
    
    # Step 4: Fast indexing
    print("\n🔍 Step 4: Fast Indexing...")
    start_time = time.time()
    
    try:
        import faiss
        
        # Create simple flat index for speed
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        index.add(embeddings)
        
        indexing_time = time.time() - start_time
        print(f"   ✅ Indexing complete: {indexing_time:.3f}s")
        print(f"   📊 Index size: {index.ntotal} vectors")
        
    except Exception as e:
        print(f"   ❌ Indexing failed: {e}")
        return
    
    # Step 5: Query system ready
    print("\n🎯 Step 5: Query System Ready!")
    
    def fast_query(query_text, top_k=3):
        """Ultra-fast query processing."""
        query_start = time.time()

        # Check if this is a keyword search (single word)
        is_keyword_search = len(query_text.strip().split()) == 1
        keyword_lower = query_text.lower().strip()

        results = []

        if is_keyword_search:
            # For keyword searches, scan all chunks for matches (exact and variations)
            keyword_matches = []

            # Create search variations for better matching
            search_terms = [keyword_lower]

            # Add common variations
            if keyword_lower == "superintelligence":
                search_terms.extend(["super-intelligence", "super intelligence"])
            elif keyword_lower == "super-intelligence":
                search_terms.extend(["superintelligence", "super intelligence"])
            elif keyword_lower == "ai":
                search_terms.extend(["artificial intelligence", "a.i.", "a i"])
            elif keyword_lower == "machine":
                search_terms.extend(["machines", "machinery"])

            for i, chunk in enumerate(chunks):
                text_lower = chunk['text'].lower()

                # Check for any search term match
                best_match = None
                best_score = 0

                for term in search_terms:
                    if term in text_lower:
                        # Calculate relevance score based on keyword frequency and position
                        keyword_count = text_lower.count(term)
                        keyword_position = text_lower.find(term)
                        # Higher score for more occurrences and earlier position
                        score = keyword_count * 0.5 + (1.0 - keyword_position / len(text_lower)) * 0.5

                        # Bonus for exact match
                        if term == keyword_lower:
                            score += 0.5

                        if score > best_score:
                            best_score = score
                            best_match = term

                if best_match:
                    # Calculate exact timestamp (middle of the chunk for best relevance)
                    exact_timestamp_seconds = (chunk['start_time'] + chunk['end_time']) / 2

                    # Convert to proper video timestamp format
                    video_timestamp = seconds_to_video_timestamp(exact_timestamp_seconds)
                    start_timestamp = seconds_to_video_timestamp(chunk['start_time'])
                    end_timestamp = seconds_to_video_timestamp(chunk['end_time'])

                    keyword_matches.append({
                        'chunk_index': i,
                        'chunk': chunk,
                        'relevance_score': best_score,
                        'matched_term': best_match,
                        'timestamp': video_timestamp,
                        'timestamp_seconds': exact_timestamp_seconds,
                        'start_time': start_timestamp,
                        'end_time': end_timestamp
                    })

            # Sort by relevance score (highest first)
            keyword_matches.sort(key=lambda x: x['relevance_score'], reverse=True)

            # Format results
            for i, match in enumerate(keyword_matches[:top_k]):
                result = {
                    'rank': i + 1,
                    'video_id': match['chunk']['video_id'],
                    'timestamp': match['timestamp'],
                    'timestamp_seconds': match['timestamp_seconds'],
                    'start_time': match['start_time'],
                    'end_time': match['end_time'],
                    'start_time_seconds': match['chunk']['start_time'],
                    'end_time_seconds': match['chunk']['end_time'],
                    'confidence': match['relevance_score'],
                    'dialogue': match['chunk']['text'],
                    'is_keyword_match': True
                }
                results.append(result)

        else:
            # For semantic searches, use embedding similarity
            query_embedding = embedder.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
            query_embedding = query_embedding.astype(np.float32)

            # Search index
            scores, indices = index.search(query_embedding, top_k)

            # Format results
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx >= 0:  # Valid index
                    chunk = chunks[idx]

                    # Calculate exact timestamp (middle of the chunk for best relevance)
                    exact_timestamp_seconds = (chunk['start_time'] + chunk['end_time']) / 2

                    # Convert to proper video timestamp format
                    video_timestamp = seconds_to_video_timestamp(exact_timestamp_seconds)
                    start_timestamp = seconds_to_video_timestamp(chunk['start_time'])
                    end_timestamp = seconds_to_video_timestamp(chunk['end_time'])

                    result = {
                        'rank': i + 1,
                        'video_id': chunk['video_id'],
                        'timestamp': video_timestamp,
                        'timestamp_seconds': exact_timestamp_seconds,
                        'start_time': start_timestamp,
                        'end_time': end_timestamp,
                        'start_time_seconds': chunk['start_time'],
                        'end_time_seconds': chunk['end_time'],
                        'confidence': float(score),
                        'dialogue': chunk['text'],
                        'is_keyword_match': False
                    }
                    results.append(result)

        query_time = (time.time() - query_start) * 1000  # Convert to ms
        return results, query_time, is_keyword_search
    
    # Demo queries
    demo_queries = [
        "artificial intelligence",
        "machine learning",
        "technology",
        "computer science",
        "programming"
    ]
    
    print("\n" + "=" * 50)
    print("🔥 FAST QUERY DEMONSTRATION")
    print("=" * 50)
    
    total_time = 0
    successful_queries = 0
    
    for i, query in enumerate(demo_queries, 1):
        try:
            print(f"\n🔍 Query {i}: '{query}'")
            results, query_time_ms, is_keyword = fast_query(query, top_k=5 if len(query.split()) == 1 else 3)
            total_time += query_time_ms
            successful_queries += 1

            print(f"⚡ Response time: {query_time_ms:.1f}ms")

            if query_time_ms < 700:
                print("✅ MEETS <700ms REQUIREMENT!")
            else:
                print("❌ Exceeds 700ms requirement")

            if results:
                if is_keyword and len(results) > 1:
                    print(f"\n📋 Keyword '{query}' found in {len(results)} locations:")
                    for result in results:
                        print(f"   {result['rank']}. 🎬 {result['video_id']} | ⏰ {result['timestamp']} | 💬 \"{result['dialogue'][:60]}...\"")
                else:
                    print(f"\n📋 Top Result:")
                    result = results[0]
                    print(f"   🎬 Video: {result['video_id']}")
                    print(f"   ⏰ Timestamp: {result['timestamp']}")
                    print(f"   📍 Context: {result['start_time']} - {result['end_time']}")
                    print(f"   🎯 Confidence: {result['confidence']:.3f}")
                    print(f"   💬 Dialogue: \"{result['dialogue'][:100]}...\"")
                    print(f"   📊 Raw: {result['timestamp_seconds']:.1f}s")
            else:
                print("   ❌ No results found")

        except Exception as e:
            print(f"   ❌ Query failed: {e}")
    
    if successful_queries > 0:
        avg_time = total_time / successful_queries
        print("\n" + "=" * 50)
        print("📊 PERFORMANCE SUMMARY")
        print("=" * 50)
        print(f"🎯 Target: <700ms per query")
        print(f"⚡ Average response time: {avg_time:.1f}ms")
        print(f"🚀 Successful queries: {successful_queries}/{len(demo_queries)}")
        
        if avg_time < 700:
            print("\n🎉 SYSTEM MEETS PERFORMANCE REQUIREMENTS!")
        else:
            print("\n⚠️  System needs optimization")
    
    # Interactive mode
    print("\n" + "=" * 50)
    print("🎮 INTERACTIVE MODE")
    print("=" * 50)
    print("Enter your queries (or 'quit' to exit):")
    
    while True:
        try:
            user_query = input("\n🔍 Your query: ").strip()
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_query:
                continue

            # Determine if keyword search and adjust top_k
            is_single_word = len(user_query.strip().split()) == 1
            search_top_k = 5 if is_single_word else 3

            results, query_time_ms, is_keyword = fast_query(user_query, top_k=search_top_k)

            print(f"⚡ Response time: {query_time_ms:.1f}ms")

            if query_time_ms < 700:
                print("✅ MEETS <700ms REQUIREMENT!")
            else:
                print("❌ Exceeds 700ms requirement")

            if results:
                if is_keyword and len(results) > 1:
                    print(f"\n📋 Keyword '{user_query}' Results:")
                    for result in results:
                        print(f"   {result['rank']}. 🎬 {result['video_id']} | ⏰ {result['timestamp']} | 💬 \"{result['dialogue']}\"")
                        print(f"      🎯 Confidence: {result['confidence']:.3f}")
                        print()
                else:
                    print("\n📋 Results:")
                    for result in results:
                        print(f"   {result['rank']}. 🎬 {result['video_id']}")
                        print(f"      ⏰ Timestamp: {result['timestamp']}")
                        print(f"      💬 Dialogue: \"{result['dialogue']}\"")
                        print(f"      🎯 Confidence: {result['confidence']:.3f}")
                        print(f"      📍 Context: {result['start_time']} - {result['end_time']}")
                        print()
            else:
                print("❌ No results found")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Thanks for trying Quickscene Production System!")

if __name__ == "__main__":
    process_single_video_fast()
