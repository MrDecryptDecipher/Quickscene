"""
Unit tests for Embedder module.

Tests embedding shape, dtype validation, and batch processing.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.embedder import Embedder


class TestEmbedder:
    """Test suite for Embedder module."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {
            'embedding': {
                'model_name': 'all-MiniLM-L6-v2',
                'batch_size': 32,
                'max_seq_length': 512,
                'device': 'cpu'
            },
            'paths': {
                'embeddings': './test_data/embeddings'
            }
        }
    
    @pytest.fixture
    def embedder(self, config):
        """Create embedder instance for testing."""
        with patch('app.embedder.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model
            
            embedder = Embedder(config)
            embedder.embedding_dim = 384
            return embedder
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample chunks for testing."""
        return [
            {
                'video_id': 'test_video',
                'chunk_id': 'test_video_0000',
                'start_time': 0.0,
                'end_time': 15.0,
                'text': 'This is the first chunk of text for embedding generation.',
                'word_count': 11
            },
            {
                'video_id': 'test_video',
                'chunk_id': 'test_video_0001',
                'start_time': 15.0,
                'end_time': 30.0,
                'text': 'This is the second chunk with different content for testing.',
                'word_count': 11
            },
            {
                'video_id': 'test_video',
                'chunk_id': 'test_video_0002',
                'start_time': 30.0,
                'end_time': 45.0,
                'text': 'Third chunk contains more text to verify embedding functionality.',
                'word_count': 10
            }
        ]
    
    def test_embedder_initialization(self, config):
        """Test embedder initialization."""
        with patch('app.embedder.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model
            
            embedder = Embedder(config)
            
            assert embedder.model_name == 'all-MiniLM-L6-v2'
            assert embedder.batch_size == 32
            assert embedder.device == 'cpu'
            assert embedder.embedding_dim == 384
            mock_st.assert_called_once_with('all-MiniLM-L6-v2', device='cpu')
    
    def test_embed_chunks_success(self, embedder, sample_chunks):
        """Test successful chunk embedding generation."""
        # Mock embedding generation
        mock_embeddings = np.random.rand(3, 384).astype(np.float32)
        embedder.model.encode.return_value = mock_embeddings
        
        with patch.object(embedder.file_manager, 'save_numpy', return_value=True):
            result = embedder.embed_chunks(sample_chunks)
        
        # Verify result shape and type
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 384)
        assert result.dtype == np.float32
    
    def test_embed_chunks_empty_input(self, embedder):
        """Test embedding generation with empty input."""
        result = embedder.embed_chunks([])
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 384)
        assert result.dtype == np.float32
    
    def test_embed_chunks_existing_embeddings(self, embedder, sample_chunks):
        """Test loading existing embeddings instead of regenerating."""
        existing_embeddings = np.random.rand(3, 384).astype(np.float32)
        
        with patch.object(embedder.file_manager, 'load_numpy', return_value=existing_embeddings):
            with patch.object(embedder, 'get_embeddings_path') as mock_path:
                mock_path.return_value.exists.return_value = True
                
                result = embedder.embed_chunks(sample_chunks)
        
        np.testing.assert_array_equal(result, existing_embeddings)
    
    def test_embed_chunks_invalid_input(self, embedder):
        """Test embedding generation with invalid chunks."""
        invalid_chunks = [{'invalid': 'chunk'}]
        
        with pytest.raises(ValueError, match="Invalid chunks"):
            embedder.embed_chunks(invalid_chunks)
    
    def test_embedding_shape_validation(self, embedder, sample_chunks):
        """Test that embeddings have correct shape (N, 384)."""
        mock_embeddings = np.random.rand(3, 384).astype(np.float32)
        embedder.model.encode.return_value = mock_embeddings
        
        with patch.object(embedder.file_manager, 'save_numpy', return_value=True):
            result = embedder.embed_chunks(sample_chunks)
        
        assert result.ndim == 2
        assert result.shape[0] == len(sample_chunks)
        assert result.shape[1] == 384
    
    def test_embedding_dtype_validation(self, embedder, sample_chunks):
        """Test that embeddings are float32 type."""
        mock_embeddings = np.random.rand(3, 384).astype(np.float64)  # Wrong dtype
        embedder.model.encode.return_value = mock_embeddings
        
        with patch.object(embedder.file_manager, 'save_numpy', return_value=True):
            result = embedder.embed_chunks(sample_chunks)
        
        # Should be converted to float32
        assert result.dtype == np.float32
    
    def test_deterministic_output(self, embedder):
        """Test that same input produces same output."""
        test_text = "This is a test sentence for deterministic output verification."
        
        # Mock consistent output
        mock_embedding = np.array([[0.1, 0.2, 0.3] + [0.0] * 381]).astype(np.float32)
        embedder.model.encode.return_value = mock_embedding
        
        result1 = embedder.embed_query(test_text)
        result2 = embedder.embed_query(test_text)  # Should use cache
        
        np.testing.assert_array_equal(result1, result2)
    
    def test_batch_processing(self, embedder, sample_chunks):
        """Test batch processing of multiple chunks."""
        # Mock batch embedding
        mock_embeddings = np.random.rand(3, 384).astype(np.float32)
        embedder.model.encode.return_value = mock_embeddings
        
        with patch.object(embedder.file_manager, 'save_numpy', return_value=True):
            result = embedder.embed_chunks(sample_chunks)
        
        # Verify model was called with batch processing
        embedder.model.encode.assert_called()
        call_args = embedder.model.encode.call_args[0]
        assert len(call_args[0]) == 3  # Batch of 3 texts
    
    def test_empty_text_handling(self, embedder):
        """Test handling of empty or whitespace-only text."""
        empty_chunks = [
            {
                'video_id': 'test_video',
                'chunk_id': 'test_video_0000',
                'start_time': 0.0,
                'end_time': 15.0,
                'text': '',
                'word_count': 0
            }
        ]
        
        # Should handle empty text gracefully
        mock_embeddings = np.random.rand(1, 384).astype(np.float32)
        embedder.model.encode.return_value = mock_embeddings
        
        with patch.object(embedder.file_manager, 'save_numpy', return_value=True):
            result = embedder.embed_chunks(empty_chunks)
        
        assert result.shape == (1, 384)
    
    def test_special_characters_handling(self, embedder):
        """Test handling of text with special characters."""
        special_chunks = [
            {
                'video_id': 'test_video',
                'chunk_id': 'test_video_0000',
                'start_time': 0.0,
                'end_time': 15.0,
                'text': 'Text with émojis 🎉 and spëcial chàracters!',
                'word_count': 7
            }
        ]
        
        mock_embeddings = np.random.rand(1, 384).astype(np.float32)
        embedder.model.encode.return_value = mock_embeddings
        
        with patch.object(embedder.file_manager, 'save_numpy', return_value=True):
            result = embedder.embed_chunks(special_chunks)
        
        assert result.shape == (1, 384)
    
    def test_embed_query_success(self, embedder):
        """Test successful query embedding generation."""
        query = "test query for embedding"
        mock_embedding = np.random.rand(1, 384).astype(np.float32)
        embedder.model.encode.return_value = mock_embedding
        
        result = embedder.embed_query(query)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 384)
        assert result.dtype == np.float32
    
    def test_embed_query_invalid(self, embedder):
        """Test query embedding with invalid input."""
        with pytest.raises(ValueError, match="Invalid query"):
            embedder.embed_query("")
        
        with pytest.raises(ValueError, match="Invalid query"):
            embedder.embed_query(123)  # Non-string input
    
    def test_embed_query_caching(self, embedder):
        """Test that query embeddings are cached."""
        query = "test query for caching"
        mock_embedding = np.random.rand(1, 384).astype(np.float32)
        embedder.model.encode.return_value = mock_embedding
        
        # First call
        result1 = embedder.embed_query(query)
        
        # Second call should use cache
        result2 = embedder.embed_query(query)
        
        # Model should only be called once
        assert embedder.model.encode.call_count == 1
        np.testing.assert_array_equal(result1, result2)
    
    def test_embedding_speed_benchmark(self, embedder, sample_chunks):
        """Test embedding generation speed (performance test)."""
        import time
        
        mock_embeddings = np.random.rand(3, 384).astype(np.float32)
        embedder.model.encode.return_value = mock_embeddings
        
        start_time = time.time()
        
        with patch.object(embedder.file_manager, 'save_numpy', return_value=True):
            embedder.embed_chunks(sample_chunks)
        
        elapsed_time = time.time() - start_time
        
        # Should complete quickly (this is a mock test, real performance depends on hardware)
        assert elapsed_time < 1.0  # Should be very fast with mocked model
    
    def test_memory_efficiency(self, embedder):
        """Test memory usage with large batch."""
        # Create large batch of chunks
        large_chunks = []
        for i in range(100):
            large_chunks.append({
                'video_id': 'test_video',
                'chunk_id': f'test_video_{i:04d}',
                'start_time': i * 15.0,
                'end_time': (i + 1) * 15.0,
                'text': f'This is chunk number {i} with some test content.',
                'word_count': 10
            })
        
        mock_embeddings = np.random.rand(100, 384).astype(np.float32)
        embedder.model.encode.return_value = mock_embeddings
        
        with patch.object(embedder.file_manager, 'save_numpy', return_value=True):
            result = embedder.embed_chunks(large_chunks)
        
        assert result.shape == (100, 384)
    
    def test_batch_embed_chunks(self, embedder):
        """Test batch embedding of multiple chunk files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Mock chunk files
            chunk_data = [
                {
                    'video_id': 'video1',
                    'chunk_id': 'video1_0000',
                    'start_time': 0.0,
                    'end_time': 15.0,
                    'text': 'Test chunk',
                    'word_count': 2
                }
            ]
            
            mock_embeddings = np.random.rand(1, 384).astype(np.float32)
            
            with patch.object(embedder.file_manager, 'load_json', return_value=chunk_data):
                with patch.object(embedder, 'embed_chunks', return_value=mock_embeddings):
                    results = embedder.batch_embed_chunks(temp_path)
            
            assert len(results) == 0  # No actual files in temp directory
    
    def test_embeddings_exist(self, embedder):
        """Test checking if embeddings exist."""
        video_id = 'test_video'
        
        with patch.object(embedder, 'get_embeddings_path') as mock_path:
            mock_path.return_value.exists.return_value = True
            
            assert embedder.embeddings_exist(video_id) is True
            mock_path.assert_called_once_with(video_id)
    
    def test_load_embeddings_success(self, embedder):
        """Test loading existing embeddings."""
        video_id = 'test_video'
        embeddings_data = np.random.rand(5, 384).astype(np.float32)
        
        with patch.object(embedder.file_manager, 'load_numpy', return_value=embeddings_data):
            with patch.object(embedder, 'get_embeddings_path') as mock_path:
                mock_path.return_value.exists.return_value = True
                
                result = embedder.load_embeddings(video_id)
        
        np.testing.assert_array_equal(result, embeddings_data)
    
    def test_load_embeddings_not_found(self, embedder):
        """Test loading embeddings that don't exist."""
        video_id = 'nonexistent_video'
        
        with patch.object(embedder, 'get_embeddings_path') as mock_path:
            mock_path.return_value.exists.return_value = False
            
            result = embedder.load_embeddings(video_id)
        
        assert result is None
    
    def test_clear_cache(self, embedder):
        """Test clearing embedding cache."""
        # Add something to cache
        query = "test query"
        mock_embedding = np.random.rand(1, 384).astype(np.float32)
        embedder.model.encode.return_value = mock_embedding
        
        embedder.embed_query(query)
        assert len(embedder._embedding_cache) > 0
        
        embedder.clear_cache()
        assert len(embedder._embedding_cache) == 0
    
    def test_get_model_info(self, embedder):
        """Test getting model information."""
        info = embedder.get_model_info()
        
        assert 'model_name' in info
        assert 'embedding_dimension' in info
        assert 'device' in info
        assert info['model_name'] == 'all-MiniLM-L6-v2'
        assert info['embedding_dimension'] == 384
    
    @pytest.mark.unit
    def test_model_loading(self, config):
        """Test that SentenceTransformer model loads correctly."""
        with patch('app.embedder.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model
            
            embedder = Embedder(config)
            
            mock_st.assert_called_once_with('all-MiniLM-L6-v2', device='cpu')
            assert embedder.model == mock_model
    
    @pytest.mark.unit
    def test_embedding_format_validation(self, embedder, sample_chunks):
        """Test that generated embeddings have correct format."""
        mock_embeddings = np.random.rand(3, 384).astype(np.float32)
        embedder.model.encode.return_value = mock_embeddings
        
        with patch.object(embedder.file_manager, 'save_numpy', return_value=True):
            result = embedder.embed_chunks(sample_chunks)
        
        # Validate format
        assert result.ndim == 2
        assert result.shape[1] == 384
        assert result.dtype == np.float32
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
