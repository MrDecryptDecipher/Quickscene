"""
Unit tests for Query Handler module.

Tests query processing, result formatting, and latency requirements.
"""

import pytest
import time
import numpy as np
from unittest.mock import Mock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.query_handler import QueryHandler


class TestQueryHandler:
    """Test suite for Query Handler module."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {
            'query': {
                'max_results': 5,
                'similarity_threshold': 0.3,
                'response_timeout_sec': 1.0
            },
            'embedding': {
                'model_name': 'all-MiniLM-L6-v2',
                'batch_size': 32,
                'device': 'cpu'
            },
            'index': {
                'type': 'IndexFlatIP'
            },
            'paths': {
                'embeddings': './test_data/embeddings',
                'index': './test_data/index'
            }
        }
    
    @pytest.fixture
    def query_handler(self, config):
        """Create query handler instance for testing."""
        with patch('app.query_handler.Embedder') as mock_embedder_class:
            with patch('app.query_handler.Indexer') as mock_indexer_class:
                # Mock embedder
                mock_embedder = Mock()
                mock_embedder_class.return_value = mock_embedder
                
                # Mock indexer
                mock_indexer = Mock()
                mock_indexer.index = Mock()  # Non-None index
                mock_indexer_class.return_value = mock_indexer
                
                return QueryHandler(config)
    
    @pytest.fixture
    def sample_search_results(self):
        """Sample search results from indexer."""
        return [
            {
                'video_id': 'video1',
                'chunk_id': 'video1_0000',
                'start_time': 0.0,
                'end_time': 15.0,
                'text': 'This is the first chunk about artificial intelligence and machine learning.',
                'confidence': 0.85,
                'global_index': 0
            },
            {
                'video_id': 'video1',
                'chunk_id': 'video1_0001',
                'start_time': 15.0,
                'end_time': 30.0,
                'text': 'Second chunk discusses deep learning algorithms and neural networks.',
                'confidence': 0.72,
                'global_index': 1
            },
            {
                'video_id': 'video2',
                'chunk_id': 'video2_0000',
                'start_time': 0.0,
                'end_time': 15.0,
                'text': 'Different video talking about computer vision and image processing.',
                'confidence': 0.68,
                'global_index': 2
            }
        ]
    
    def test_query_handler_initialization(self, config):
        """Test query handler initialization."""
        with patch('app.query_handler.Embedder') as mock_embedder_class:
            with patch('app.query_handler.Indexer') as mock_indexer_class:
                mock_indexer = Mock()
                mock_indexer.index = Mock()
                mock_indexer_class.return_value = mock_indexer
                
                handler = QueryHandler(config)
                
                assert handler.max_results == 5
                assert handler.similarity_threshold == 0.3
                assert handler.response_timeout == 1.0
    
    def test_process_query_success(self, query_handler, sample_search_results):
        """Test successful query processing."""
        query = "artificial intelligence machine learning"
        
        # Mock embedder response
        mock_embedding = np.random.rand(1, 384).astype(np.float32)
        query_handler.embedder.embed_query.return_value = mock_embedding
        
        # Mock indexer response
        query_handler.indexer.search.return_value = sample_search_results
        
        results = query_handler.process_query(query)
        
        assert len(results) <= query_handler.max_results
        assert all('rank' in result for result in results)
        assert all('confidence' in result for result in results)
        assert all('video_id' in result for result in results)
    
    def test_process_query_invalid_input(self, query_handler):
        """Test query processing with invalid input."""
        with pytest.raises(ValueError, match="Invalid query"):
            query_handler.process_query("")
        
        with pytest.raises(ValueError, match="Invalid query"):
            query_handler.process_query("ab")  # Too short
    
    def test_process_query_no_index(self, query_handler):
        """Test query processing without available index."""
        query_handler.indexer.index = None
        
        with pytest.raises(RuntimeError, match="Search index not available"):
            query_handler.process_query("test query")
    
    def test_query_to_chunk_matching(self, query_handler, sample_search_results):
        """Test that query matches return relevant chunks."""
        query = "artificial intelligence"
        
        # Mock embedder and indexer
        mock_embedding = np.random.rand(1, 384).astype(np.float32)
        query_handler.embedder.embed_query.return_value = mock_embedding
        query_handler.indexer.search.return_value = sample_search_results
        
        results = query_handler.process_query(query)
        
        # Verify results contain expected information
        assert len(results) > 0
        for result in results:
            assert 'video_id' in result
            assert 'start_time' in result
            assert 'end_time' in result
            assert 'text' in result
            assert 'confidence' in result
    
    def test_best_chunk_in_top_results(self, query_handler, sample_search_results):
        """Test that best matching chunk appears in top results."""
        query = "artificial intelligence"
        
        # Mock responses
        mock_embedding = np.random.rand(1, 384).astype(np.float32)
        query_handler.embedder.embed_query.return_value = mock_embedding
        query_handler.indexer.search.return_value = sample_search_results
        
        results = query_handler.process_query(query, top_k=3)
        
        # Best result should have highest confidence
        assert len(results) <= 3
        if len(results) > 1:
            # Results should be sorted by ranking score (descending)
            for i in range(len(results) - 1):
                current_score = results[i].get('ranking_score', results[i]['confidence'])
                next_score = results[i + 1].get('ranking_score', results[i + 1]['confidence'])
                assert current_score >= next_score
    
    def test_latency_requirements(self, query_handler, sample_search_results):
        """Test that query processing meets latency requirements."""
        query = "test query for latency"
        
        # Mock fast responses
        mock_embedding = np.random.rand(1, 384).astype(np.float32)
        query_handler.embedder.embed_query.return_value = mock_embedding
        query_handler.indexer.search.return_value = sample_search_results
        
        start_time = time.time()
        results = query_handler.process_query(query)
        elapsed_time = time.time() - start_time
        
        # Should complete within timeout (with mocked operations, should be very fast)
        assert elapsed_time < query_handler.response_timeout
        assert len(results) > 0
    
    def test_result_formatting(self, query_handler, sample_search_results):
        """Test that results are properly formatted."""
        query = "test query"
        
        # Mock responses
        mock_embedding = np.random.rand(1, 384).astype(np.float32)
        query_handler.embedder.embed_query.return_value = mock_embedding
        query_handler.indexer.search.return_value = sample_search_results
        
        results = query_handler.process_query(query)
        
        # Verify result format
        for i, result in enumerate(results):
            assert result['rank'] == i + 1
            assert isinstance(result['start_time'], (int, float))
            assert isinstance(result['end_time'], (int, float))
            assert isinstance(result['confidence'], float)
            assert isinstance(result['text'], str)
            assert 'timestamp' in result
            assert 'duration' in result
    
    def test_similarity_threshold_filtering(self, query_handler):
        """Test that results below similarity threshold are filtered."""
        query = "test query"
        
        # Create results with varying confidence scores
        low_confidence_results = [
            {
                'video_id': 'video1',
                'chunk_id': 'video1_0000',
                'start_time': 0.0,
                'end_time': 15.0,
                'text': 'Low confidence result',
                'confidence': 0.1,  # Below threshold
                'global_index': 0
            },
            {
                'video_id': 'video2',
                'chunk_id': 'video2_0000',
                'start_time': 0.0,
                'end_time': 15.0,
                'text': 'High confidence result',
                'confidence': 0.8,  # Above threshold
                'global_index': 1
            }
        ]
        
        # Mock responses
        mock_embedding = np.random.rand(1, 384).astype(np.float32)
        query_handler.embedder.embed_query.return_value = mock_embedding
        query_handler.indexer.search.return_value = low_confidence_results
        
        results = query_handler.process_query(query)
        
        # Should only return high confidence results
        assert len(results) == 1
        assert results[0]['confidence'] >= query_handler.similarity_threshold
    
    def test_batch_query_processing(self, query_handler, sample_search_results):
        """Test batch processing of multiple queries."""
        queries = [
            "artificial intelligence",
            "machine learning",
            "deep learning"
        ]
        
        # Mock responses
        mock_embedding = np.random.rand(1, 384).astype(np.float32)
        query_handler.embedder.embed_query.return_value = mock_embedding
        query_handler.indexer.search.return_value = sample_search_results
        
        all_results = query_handler.batch_query(queries)
        
        assert len(all_results) == len(queries)
        for results in all_results:
            assert isinstance(results, list)
    
    def test_batch_query_partial_failure(self, query_handler, sample_search_results):
        """Test batch query processing with some failures."""
        queries = ["good query", "bad query"]
        
        # Mock embedder to fail on second query
        def mock_embed_query(query):
            if "bad" in query:
                raise Exception("Embedding failed")
            return np.random.rand(1, 384).astype(np.float32)
        
        query_handler.embedder.embed_query.side_effect = mock_embed_query
        query_handler.indexer.search.return_value = sample_search_results
        
        all_results = query_handler.batch_query(queries)
        
        assert len(all_results) == 2
        assert len(all_results[0]) > 0  # First query succeeded
        assert len(all_results[1]) == 0  # Second query failed
    
    def test_get_similar_chunks(self, query_handler, sample_search_results):
        """Test finding similar chunks to a specific video segment."""
        video_id = "video1"
        start_time = 0.0
        end_time = 15.0
        
        # Mock indexer metadata
        query_handler.indexer.metadata = [
            {
                'video_id': 'video1',
                'start_time': 0.0,
                'end_time': 15.0,
                'text': 'Target chunk text'
            }
        ]
        
        # Mock query processing
        mock_embedding = np.random.rand(1, 384).astype(np.float32)
        query_handler.embedder.embed_query.return_value = mock_embedding
        query_handler.indexer.search.return_value = sample_search_results
        
        similar_chunks = query_handler.get_similar_chunks(video_id, start_time, end_time)
        
        # Should return similar chunks (excluding the original)
        assert isinstance(similar_chunks, list)
    
    def test_get_query_suggestions(self, query_handler, sample_search_results):
        """Test getting query suggestions."""
        partial_query = "artificial"
        
        # Mock responses
        mock_embedding = np.random.rand(1, 384).astype(np.float32)
        query_handler.embedder.embed_query.return_value = mock_embedding
        query_handler.indexer.search.return_value = sample_search_results
        
        suggestions = query_handler.get_query_suggestions(partial_query)
        
        assert isinstance(suggestions, list)
        # Suggestions should be longer than partial query
        for suggestion in suggestions:
            assert len(suggestion) > len(partial_query)
    
    def test_get_system_status(self, query_handler):
        """Test getting system status information."""
        # Mock component status
        query_handler.indexer.get_index_stats.return_value = {
            'index_exists': True,
            'total_vectors': 100
        }
        query_handler.embedder.get_model_info.return_value = {
            'model_name': 'test-model',
            'embedding_dimension': 384
        }
        
        status = query_handler.get_system_status()
        
        assert 'query_handler' in status
        assert 'index' in status
        assert 'embedder' in status
        assert 'ready_for_queries' in status
    
    def test_warm_up(self, query_handler, sample_search_results):
        """Test system warm-up functionality."""
        # Mock responses
        mock_embedding = np.random.rand(1, 384).astype(np.float32)
        query_handler.embedder.embed_query.return_value = mock_embedding
        query_handler.indexer.search.return_value = sample_search_results
        
        # Should not raise exception
        query_handler.warm_up()
        
        # Verify warm-up query was processed
        query_handler.embedder.embed_query.assert_called()
    
    def test_warm_up_no_index(self, query_handler):
        """Test warm-up with no available index."""
        query_handler.indexer.index = None
        
        # Should handle gracefully without raising exception
        query_handler.warm_up()
    
    def test_clear_caches(self, query_handler):
        """Test clearing system caches."""
        # Should not raise exception
        query_handler.clear_caches()
        
        # Verify embedder cache was cleared
        query_handler.embedder.clear_cache.assert_called_once()
    
    def test_ranking_score_calculation(self, query_handler):
        """Test ranking score calculation for results."""
        query = "artificial intelligence"
        
        result = {
            'confidence': 0.8,
            'text': 'This is a long text about artificial intelligence and machine learning with many words',
            'start_time': 0.0,
            'end_time': 15.0
        }
        
        score = query_handler._calculate_ranking_score(result, query)
        
        # Score should be based on confidence and other factors
        assert isinstance(score, float)
        assert score > 0
    
    def test_edge_case_queries(self, query_handler, sample_search_results):
        """Test edge case query handling."""
        # Mock responses
        mock_embedding = np.random.rand(1, 384).astype(np.float32)
        query_handler.embedder.embed_query.return_value = mock_embedding
        query_handler.indexer.search.return_value = sample_search_results
        
        # Test very long query
        long_query = "artificial intelligence " * 50
        results = query_handler.process_query(long_query)
        assert isinstance(results, list)
        
        # Test query with special characters
        special_query = "AI & ML: deep-learning (2024)!"
        results = query_handler.process_query(special_query)
        assert isinstance(results, list)
    
    @pytest.mark.unit
    def test_query_preprocessing(self, query_handler):
        """Test query preprocessing and validation."""
        # Valid queries should pass
        valid_queries = [
            "artificial intelligence",
            "machine learning algorithms",
            "deep neural networks"
        ]
        
        for query in valid_queries:
            is_valid, errors = query_handler.validators.validate_query(query)
            assert is_valid, f"Query '{query}' should be valid: {errors}"
    
    @pytest.mark.unit
    def test_result_validation(self, query_handler, sample_search_results):
        """Test that generated results pass validation."""
        query = "test query"
        
        # Mock responses
        mock_embedding = np.random.rand(1, 384).astype(np.float32)
        query_handler.embedder.embed_query.return_value = mock_embedding
        query_handler.indexer.search.return_value = sample_search_results
        
        results = query_handler.process_query(query)
        
        # Validate results format
        is_valid, errors = query_handler.validators.validate_search_results(results)
        assert is_valid, f"Results should be valid: {errors}"
    
    @pytest.mark.performance
    def test_query_performance_benchmark(self, query_handler, sample_search_results):
        """Test query processing performance."""
        query = "performance test query"
        
        # Mock fast responses
        mock_embedding = np.random.rand(1, 384).astype(np.float32)
        query_handler.embedder.embed_query.return_value = mock_embedding
        query_handler.indexer.search.return_value = sample_search_results
        
        # Measure performance
        start_time = time.time()
        results = query_handler.process_query(query)
        elapsed_time = time.time() - start_time
        
        # Should be very fast with mocked operations
        assert elapsed_time < 0.1
        assert len(results) > 0
