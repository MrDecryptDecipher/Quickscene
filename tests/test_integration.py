"""
Integration tests for Quickscene system.

Tests end-to-end pipeline functionality and module interactions.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.transcriber import Transcriber
from app.chunker import Chunker
from app.embedder import Embedder
from app.indexer import Indexer
from app.query_handler import QueryHandler


class TestIntegration:
    """Integration test suite for Quickscene system."""
    
    @pytest.fixture
    def config(self):
        """Test configuration for integration tests."""
        return {
            'asr': {
                'mode': 'whisper',
                'whisper_model': 'base',
                'language': 'en',
                'device': 'cpu'
            },
            'embedding': {
                'model_name': 'all-MiniLM-L6-v2',
                'batch_size': 32,
                'device': 'cpu'
            },
            'chunking': {
                'duration_sec': 15,
                'overlap_sec': 0,
                'respect_word_boundaries': True,
                'min_chunk_length': 10
            },
            'index': {
                'type': 'IndexFlatIP',
                'metric': 'METRIC_INNER_PRODUCT'
            },
            'query': {
                'max_results': 5,
                'similarity_threshold': 0.3,
                'response_timeout_sec': 1.0
            },
            'paths': {
                'videos': './test_data/videos',
                'transcripts': './test_data/transcripts',
                'chunks': './test_data/chunks',
                'embeddings': './test_data/embeddings',
                'index': './test_data/index',
                'index_file': 'test_index.index',
                'metadata_file': 'test_metadata.json'
            },
            'performance': {
                'max_concurrent_transcriptions': 2
            }
        }
    
    @pytest.fixture
    def sample_transcript(self):
        """Sample transcript for integration testing."""
        return {
            'video_id': 'integration_test_video',
            'duration': 60.0,
            'language': 'en',
            'segments': [
                {
                    'start': 0.0,
                    'end': 15.0,
                    'text': 'Welcome to this video about artificial intelligence and machine learning concepts.'
                },
                {
                    'start': 15.0,
                    'end': 30.0,
                    'text': 'We will discuss deep learning algorithms and neural network architectures.'
                },
                {
                    'start': 30.0,
                    'end': 45.0,
                    'text': 'Computer vision and natural language processing are important AI applications.'
                },
                {
                    'start': 45.0,
                    'end': 60.0,
                    'text': 'Thank you for watching this introduction to artificial intelligence.'
                }
            ]
        }
    
    def test_transcriber_to_chunker_integration(self, config, sample_transcript):
        """Test data flow from transcriber to chunker."""
        # Initialize components
        chunker = Chunker(config)
        
        # Mock file operations
        with patch.object(chunker.file_manager, 'save_json', return_value=True):
            chunks = chunker.chunk_transcript(sample_transcript)
        
        # Verify chunks were created from transcript
        assert len(chunks) > 0
        assert all(chunk['video_id'] == sample_transcript['video_id'] for chunk in chunks)
        
        # Verify temporal consistency
        for chunk in chunks:
            assert chunk['start_time'] >= 0
            assert chunk['end_time'] <= sample_transcript['duration']
            assert chunk['start_time'] < chunk['end_time']
    
    def test_chunker_to_embedder_integration(self, config):
        """Test data flow from chunker to embedder."""
        # Sample chunks
        chunks = [
            {
                'video_id': 'test_video',
                'chunk_id': 'test_video_0000',
                'start_time': 0.0,
                'end_time': 15.0,
                'text': 'This is a test chunk for embedding generation.',
                'word_count': 9
            },
            {
                'video_id': 'test_video',
                'chunk_id': 'test_video_0001',
                'start_time': 15.0,
                'end_time': 30.0,
                'text': 'Another chunk with different content for testing.',
                'word_count': 8
            }
        ]
        
        # Initialize embedder
        with patch('app.embedder.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_embeddings = np.random.rand(2, 384).astype(np.float32)
            mock_model.encode.return_value = mock_embeddings
            mock_st.return_value = mock_model
            
            embedder = Embedder(config)
            
            with patch.object(embedder.file_manager, 'save_numpy', return_value=True):
                embeddings = embedder.embed_chunks(chunks)
        
        # Verify embeddings format
        assert embeddings.shape == (2, 384)
        assert embeddings.dtype == np.float32
    
    def test_embedder_to_indexer_integration(self, config):
        """Test data flow from embedder to indexer."""
        # Sample embeddings and chunks
        embeddings = [np.random.rand(2, 384).astype(np.float32)]
        chunks = [[
            {
                'video_id': 'test_video',
                'chunk_id': 'test_video_0000',
                'start_time': 0.0,
                'end_time': 15.0,
                'text': 'Test chunk for indexing'
            },
            {
                'video_id': 'test_video',
                'chunk_id': 'test_video_0001',
                'start_time': 15.0,
                'end_time': 30.0,
                'text': 'Another test chunk'
            }
        ]]
        
        # Initialize indexer
        with patch('app.indexer.faiss') as mock_faiss:
            mock_index = Mock()
            mock_index.d = 384
            mock_index.ntotal = 2
            mock_faiss.IndexFlatIP.return_value = mock_index
            
            indexer = Indexer(config)
            
            with patch.object(indexer, '_save_index'):
                indexer.build_index(embeddings, chunks)
        
        # Verify index was built
        assert indexer.index is not None
        assert len(indexer.metadata) == 2
        assert indexer.embedding_dim == 384
    
    def test_indexer_to_query_handler_integration(self, config):
        """Test data flow from indexer to query handler."""
        # Initialize query handler with mocked components
        with patch('app.query_handler.Embedder') as mock_embedder_class:
            with patch('app.query_handler.Indexer') as mock_indexer_class:
                # Mock embedder
                mock_embedder = Mock()
                mock_query_embedding = np.random.rand(1, 384).astype(np.float32)
                mock_embedder.embed_query.return_value = mock_query_embedding
                mock_embedder_class.return_value = mock_embedder
                
                # Mock indexer
                mock_indexer = Mock()
                mock_indexer.index = Mock()  # Non-None index
                mock_search_results = [
                    {
                        'video_id': 'test_video',
                        'chunk_id': 'test_video_0000',
                        'start_time': 0.0,
                        'end_time': 15.0,
                        'text': 'Test search result',
                        'confidence': 0.8,
                        'global_index': 0
                    }
                ]
                mock_indexer.search.return_value = mock_search_results
                mock_indexer_class.return_value = mock_indexer
                
                query_handler = QueryHandler(config)
                results = query_handler.process_query("test query")
        
        # Verify query processing
        assert len(results) > 0
        assert results[0]['video_id'] == 'test_video'
        assert 'confidence' in results[0]
    
    def test_full_video_processing_pipeline(self, config, sample_transcript):
        """Test complete video processing pipeline."""
        video_id = sample_transcript['video_id']
        
        # Step 1: Mock transcription (already have transcript)
        
        # Step 2: Chunking
        chunker = Chunker(config)
        with patch.object(chunker.file_manager, 'save_json', return_value=True):
            chunks = chunker.chunk_transcript(sample_transcript)
        
        # Step 3: Embedding
        with patch('app.embedder.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_embeddings = np.random.rand(len(chunks), 384).astype(np.float32)
            mock_model.encode.return_value = mock_embeddings
            mock_st.return_value = mock_model
            
            embedder = Embedder(config)
            
            with patch.object(embedder.file_manager, 'save_numpy', return_value=True):
                embeddings = embedder.embed_chunks(chunks)
        
        # Step 4: Indexing
        with patch('app.indexer.faiss') as mock_faiss:
            mock_index = Mock()
            mock_index.d = 384
            mock_index.ntotal = len(chunks)
            mock_faiss.IndexFlatIP.return_value = mock_index
            
            indexer = Indexer(config)
            
            with patch.object(indexer, '_save_index'):
                indexer.build_index([embeddings], [chunks])
        
        # Verify pipeline completion
        assert len(chunks) > 0
        assert embeddings.shape[0] == len(chunks)
        assert indexer.index is not None
        assert len(indexer.metadata) == len(chunks)
    
    def test_query_to_result_pipeline(self, config):
        """Test complete query processing pipeline."""
        # Initialize all components with mocks
        with patch('app.query_handler.Embedder') as mock_embedder_class:
            with patch('app.query_handler.Indexer') as mock_indexer_class:
                # Mock embedder
                mock_embedder = Mock()
                mock_query_embedding = np.random.rand(1, 384).astype(np.float32)
                mock_embedder.embed_query.return_value = mock_query_embedding
                mock_embedder_class.return_value = mock_embedder
                
                # Mock indexer with realistic search results
                mock_indexer = Mock()
                mock_indexer.index = Mock()
                mock_search_results = [
                    {
                        'video_id': 'video1',
                        'chunk_id': 'video1_0000',
                        'start_time': 0.0,
                        'end_time': 15.0,
                        'text': 'Artificial intelligence is transforming technology',
                        'confidence': 0.9,
                        'global_index': 0
                    },
                    {
                        'video_id': 'video1',
                        'chunk_id': 'video1_0001',
                        'start_time': 15.0,
                        'end_time': 30.0,
                        'text': 'Machine learning algorithms are powerful tools',
                        'confidence': 0.7,
                        'global_index': 1
                    }
                ]
                mock_indexer.search.return_value = mock_search_results
                mock_indexer_class.return_value = mock_indexer
                
                # Process query
                query_handler = QueryHandler(config)
                results = query_handler.process_query("artificial intelligence")
        
        # Verify end-to-end query processing
        assert len(results) > 0
        assert all('video_id' in result for result in results)
        assert all('start_time' in result for result in results)
        assert all('end_time' in result for result in results)
        assert all('confidence' in result for result in results)
        
        # Verify results are ranked
        if len(results) > 1:
            confidences = [result['confidence'] for result in results]
            # Should be sorted in descending order (or by ranking score)
            assert confidences == sorted(confidences, reverse=True) or \
                   all('ranking_score' in result for result in results)
    
    def test_batch_video_processing(self, config):
        """Test batch processing of multiple videos."""
        # Sample transcripts for multiple videos
        transcripts = [
            {
                'video_id': 'video1',
                'duration': 30.0,
                'segments': [
                    {'start': 0.0, 'end': 15.0, 'text': 'First video content about AI'},
                    {'start': 15.0, 'end': 30.0, 'text': 'More AI content in first video'}
                ]
            },
            {
                'video_id': 'video2',
                'duration': 30.0,
                'segments': [
                    {'start': 0.0, 'end': 15.0, 'text': 'Second video about machine learning'},
                    {'start': 15.0, 'end': 30.0, 'text': 'Deep learning in second video'}
                ]
            }
        ]
        
        all_chunks = []
        all_embeddings = []
        
        # Process each transcript
        for transcript in transcripts:
            # Chunking
            chunker = Chunker(config)
            with patch.object(chunker.file_manager, 'save_json', return_value=True):
                chunks = chunker.chunk_transcript(transcript)
            all_chunks.append(chunks)
            
            # Embedding
            with patch('app.embedder.SentenceTransformer') as mock_st:
                mock_model = Mock()
                mock_model.get_sentence_embedding_dimension.return_value = 384
                mock_embeddings = np.random.rand(len(chunks), 384).astype(np.float32)
                mock_model.encode.return_value = mock_embeddings
                mock_st.return_value = mock_model
                
                embedder = Embedder(config)
                
                with patch.object(embedder.file_manager, 'save_numpy', return_value=True):
                    embeddings = embedder.embed_chunks(chunks)
            all_embeddings.append(embeddings)
        
        # Build combined index
        with patch('app.indexer.faiss') as mock_faiss:
            mock_index = Mock()
            mock_index.d = 384
            total_chunks = sum(len(chunks) for chunks in all_chunks)
            mock_index.ntotal = total_chunks
            mock_faiss.IndexFlatIP.return_value = mock_index
            
            indexer = Indexer(config)
            
            with patch.object(indexer, '_save_index'):
                indexer.build_index(all_embeddings, all_chunks)
        
        # Verify batch processing
        assert len(all_chunks) == 2
        assert len(all_embeddings) == 2
        assert indexer.index is not None
        assert len(indexer.metadata) == total_chunks
    
    def test_incremental_index_updates(self, config):
        """Test adding new videos to existing index."""
        # Initial index with one video
        initial_chunks = [
            {
                'video_id': 'video1',
                'chunk_id': 'video1_0000',
                'start_time': 0.0,
                'end_time': 15.0,
                'text': 'Initial video content'
            }
        ]
        initial_embeddings = np.random.rand(1, 384).astype(np.float32)
        
        # Build initial index
        with patch('app.indexer.faiss') as mock_faiss:
            mock_index = Mock()
            mock_index.d = 384
            mock_index.ntotal = 1
            mock_faiss.IndexFlatIP.return_value = mock_index
            
            indexer = Indexer(config)
            
            with patch.object(indexer, '_save_index'):
                indexer.build_index([initial_embeddings], [initial_chunks])
        
        # Add new video
        new_chunks = [
            {
                'video_id': 'video2',
                'chunk_id': 'video2_0000',
                'start_time': 0.0,
                'end_time': 15.0,
                'text': 'New video content'
            }
        ]
        new_embeddings = np.random.rand(1, 384).astype(np.float32)
        
        with patch.object(indexer, '_save_index'):
            indexer.add_embeddings(new_embeddings, new_chunks)
        
        # Verify incremental update
        assert len(indexer.metadata) == 2
        mock_index.add.assert_called_with(new_embeddings)
    
    def test_error_handling_chain(self, config):
        """Test error handling throughout the pipeline."""
        # Test chunker error handling
        invalid_transcript = {'invalid': 'data'}
        
        chunker = Chunker(config)
        with pytest.raises(ValueError, match="Invalid transcript"):
            chunker.chunk_transcript(invalid_transcript)
        
        # Test embedder error handling
        invalid_chunks = [{'invalid': 'chunk'}]
        
        with patch('app.embedder.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model
            
            embedder = Embedder(config)
            
            with pytest.raises(ValueError, match="Invalid chunks"):
                embedder.embed_chunks(invalid_chunks)
    
    def test_partial_failure_recovery(self, config):
        """Test system resilience to partial failures."""
        # Test with some valid and some invalid data
        mixed_transcripts = [
            {
                'video_id': 'valid_video',
                'duration': 30.0,
                'segments': [
                    {'start': 0.0, 'end': 15.0, 'text': 'Valid content'}
                ]
            },
            {
                'video_id': 'invalid_video',
                # Missing required fields
            }
        ]
        
        chunker = Chunker(config)
        successful_chunks = []
        
        for transcript in mixed_transcripts:
            try:
                with patch.object(chunker.file_manager, 'save_json', return_value=True):
                    chunks = chunker.chunk_transcript(transcript)
                successful_chunks.append(chunks)
            except ValueError:
                # Expected for invalid transcript
                continue
        
        # Should have one successful result
        assert len(successful_chunks) == 1
        assert len(successful_chunks[0]) > 0
    
    @pytest.mark.integration
    def test_end_to_end_system_validation(self, config, sample_transcript):
        """Test complete end-to-end system validation."""
        # This test validates the entire pipeline works together
        
        # Step 1: Chunking
        chunker = Chunker(config)
        with patch.object(chunker.file_manager, 'save_json', return_value=True):
            chunks = chunker.chunk_transcript(sample_transcript)
        
        # Step 2: Embedding
        with patch('app.embedder.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_embeddings = np.random.rand(len(chunks), 384).astype(np.float32)
            mock_model.encode.return_value = mock_embeddings
            mock_st.return_value = mock_model
            
            embedder = Embedder(config)
            
            with patch.object(embedder.file_manager, 'save_numpy', return_value=True):
                embeddings = embedder.embed_chunks(chunks)
        
        # Step 3: Indexing
        with patch('app.indexer.faiss') as mock_faiss:
            mock_index = Mock()
            mock_index.d = 384
            mock_index.ntotal = len(chunks)
            mock_faiss.IndexFlatIP.return_value = mock_index
            
            indexer = Indexer(config)
            
            with patch.object(indexer, '_save_index'):
                indexer.build_index([embeddings], [chunks])
        
        # Step 4: Query processing
        with patch('app.query_handler.Embedder') as mock_embedder_class:
            with patch('app.query_handler.Indexer') as mock_indexer_class:
                # Use the same indexer instance
                mock_indexer_class.return_value = indexer
                
                # Mock embedder for query
                mock_embedder = Mock()
                mock_query_embedding = np.random.rand(1, 384).astype(np.float32)
                mock_embedder.embed_query.return_value = mock_query_embedding
                mock_embedder_class.return_value = mock_embedder
                
                # Mock search results based on our chunks
                mock_search_results = [
                    {
                        'video_id': chunks[0]['video_id'],
                        'chunk_id': chunks[0]['chunk_id'],
                        'start_time': chunks[0]['start_time'],
                        'end_time': chunks[0]['end_time'],
                        'text': chunks[0]['text'],
                        'confidence': 0.8,
                        'global_index': 0
                    }
                ]
                indexer.search = Mock(return_value=mock_search_results)
                
                query_handler = QueryHandler(config)
                results = query_handler.process_query("artificial intelligence")
        
        # Verify complete pipeline
        assert len(chunks) > 0
        assert embeddings.shape[0] == len(chunks)
        assert indexer.index is not None
        assert len(results) > 0
        assert results[0]['video_id'] == sample_transcript['video_id']
