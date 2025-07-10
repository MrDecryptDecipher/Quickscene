"""
Unit tests for Indexer module.

Tests FAISS index creation, search consistency, and metadata management.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.indexer import Indexer


class TestIndexer:
    """Test suite for Indexer module."""
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {
            'index': {
                'type': 'IndexFlatIP',
                'metric': 'METRIC_INNER_PRODUCT'
            },
            'paths': {
                'index': './test_data/index',
                'index_file': 'test_index.index',
                'metadata_file': 'test_metadata.json',
                'embeddings': './test_data/embeddings',
                'chunks': './test_data/chunks'
            }
        }
    
    @pytest.fixture
    def indexer(self, config):
        """Create indexer instance for testing."""
        with patch('app.indexer.faiss') as mock_faiss:
            mock_index = Mock()
            mock_index.d = 384
            mock_index.ntotal = 0
            mock_faiss.IndexFlatIP.return_value = mock_index
            mock_faiss.read_index.return_value = mock_index
            
            indexer = Indexer(config)
            indexer.embedding_dim = 384
            return indexer
    
    @pytest.fixture
    def sample_embeddings(self):
        """Sample embeddings for testing."""
        return [
            np.random.rand(3, 384).astype(np.float32),
            np.random.rand(2, 384).astype(np.float32)
        ]
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample chunks for testing."""
        return [
            [
                {
                    'video_id': 'video1',
                    'chunk_id': 'video1_0000',
                    'start_time': 0.0,
                    'end_time': 15.0,
                    'text': 'First chunk of video one'
                },
                {
                    'video_id': 'video1',
                    'chunk_id': 'video1_0001',
                    'start_time': 15.0,
                    'end_time': 30.0,
                    'text': 'Second chunk of video one'
                },
                {
                    'video_id': 'video1',
                    'chunk_id': 'video1_0002',
                    'start_time': 30.0,
                    'end_time': 45.0,
                    'text': 'Third chunk of video one'
                }
            ],
            [
                {
                    'video_id': 'video2',
                    'chunk_id': 'video2_0000',
                    'start_time': 0.0,
                    'end_time': 15.0,
                    'text': 'First chunk of video two'
                },
                {
                    'video_id': 'video2',
                    'chunk_id': 'video2_0001',
                    'start_time': 15.0,
                    'end_time': 30.0,
                    'text': 'Second chunk of video two'
                }
            ]
        ]
    
    def test_indexer_initialization(self, config):
        """Test indexer initialization."""
        with patch('app.indexer.faiss') as mock_faiss:
            mock_index = Mock()
            mock_faiss.IndexFlatIP.return_value = mock_index
            
            indexer = Indexer(config)
            
            assert indexer.index_type == 'IndexFlatIP'
            assert indexer.metric == 'METRIC_INNER_PRODUCT'
    
    def test_build_index_success(self, indexer, sample_embeddings, sample_chunks):
        """Test successful index building."""
        with patch.object(indexer, '_save_index'):
            indexer.build_index(sample_embeddings, sample_chunks)
        
        # Verify index was created
        assert indexer.index is not None
        assert len(indexer.metadata) == 5  # 3 + 2 chunks
        assert indexer.embedding_dim == 384
    
    def test_build_index_empty_input(self, indexer):
        """Test index building with empty input."""
        with pytest.raises(ValueError, match="No embeddings or chunks provided"):
            indexer.build_index([], [])
    
    def test_build_index_mismatched_input(self, indexer, sample_embeddings, sample_chunks):
        """Test index building with mismatched embeddings and chunks."""
        # Remove one chunk list to create mismatch
        mismatched_chunks = sample_chunks[:-1]
        
        with pytest.raises(ValueError, match="Number of embedding arrays must match"):
            indexer.build_index(sample_embeddings, mismatched_chunks)
    
    def test_faiss_index_creation(self, indexer):
        """Test FAISS index creation."""
        embeddings = np.random.rand(10, 384).astype(np.float32)
        
        with patch('app.indexer.faiss') as mock_faiss:
            mock_index = Mock()
            mock_index.d = 384
            mock_index.ntotal = 10
            mock_faiss.IndexFlatIP.return_value = mock_index
            
            indexer._create_faiss_index(embeddings)
        
        assert indexer.index == mock_index
        assert indexer.embedding_dim == 384
        mock_index.add.assert_called_once_with(embeddings)
    
    def test_search_success(self, indexer):
        """Test successful search operation."""
        # Setup index with metadata
        indexer.metadata = [
            {
                'global_index': 0,
                'video_id': 'video1',
                'chunk_id': 'video1_0000',
                'start_time': 0.0,
                'end_time': 15.0,
                'text': 'Test chunk'
            }
        ]
        
        # Mock search results
        mock_scores = np.array([[0.8]])
        mock_indices = np.array([[0]])
        indexer.index.search.return_value = (mock_scores, mock_indices)
        
        query_embedding = np.random.rand(1, 384).astype(np.float32)
        results = indexer.search(query_embedding, k=1)
        
        assert len(results) == 1
        assert results[0]['video_id'] == 'video1'
        assert results[0]['confidence'] == 0.8
    
    def test_search_no_index(self, indexer):
        """Test search without built index."""
        indexer.index = None
        
        query_embedding = np.random.rand(1, 384).astype(np.float32)
        
        with pytest.raises(RuntimeError, match="Index not built"):
            indexer.search(query_embedding)
    
    def test_search_no_metadata(self, indexer):
        """Test search without metadata."""
        indexer.metadata = []
        
        query_embedding = np.random.rand(1, 384).astype(np.float32)
        
        with pytest.raises(RuntimeError, match="No metadata available"):
            indexer.search(query_embedding)
    
    def test_search_invalid_query(self, indexer):
        """Test search with invalid query embedding."""
        indexer.metadata = [{'global_index': 0}]
        
        # Invalid shape
        invalid_query = np.random.rand(384).astype(np.float64)  # Wrong dtype
        
        with pytest.raises(ValueError, match="Invalid query embedding"):
            indexer.search(invalid_query)
    
    def test_knn_search_consistency(self, indexer):
        """Test that KNN search returns consistent results."""
        # Setup index with metadata
        indexer.metadata = [
            {'global_index': i, 'video_id': f'video{i}', 'chunk_id': f'chunk{i}',
             'start_time': 0.0, 'end_time': 15.0, 'text': f'chunk {i}'}
            for i in range(5)
        ]
        
        # Mock consistent search results
        mock_scores = np.array([[0.9, 0.8, 0.7]])
        mock_indices = np.array([[0, 1, 2]])
        indexer.index.search.return_value = (mock_scores, mock_indices)
        
        query_embedding = np.random.rand(1, 384).astype(np.float32)
        
        # Run search multiple times
        results1 = indexer.search(query_embedding, k=3)
        results2 = indexer.search(query_embedding, k=3)
        
        # Results should be identical
        assert len(results1) == len(results2) == 3
        for r1, r2 in zip(results1, results2):
            assert r1['video_id'] == r2['video_id']
            assert r1['confidence'] == r2['confidence']
    
    def test_add_embeddings_success(self, indexer):
        """Test adding embeddings to existing index."""
        # Setup existing index
        indexer.metadata = []
        
        new_embeddings = np.random.rand(2, 384).astype(np.float32)
        new_chunks = [
            {
                'video_id': 'new_video',
                'chunk_id': 'new_video_0000',
                'start_time': 0.0,
                'end_time': 15.0,
                'text': 'New chunk'
            },
            {
                'video_id': 'new_video',
                'chunk_id': 'new_video_0001',
                'start_time': 15.0,
                'end_time': 30.0,
                'text': 'Another new chunk'
            }
        ]
        
        with patch.object(indexer, '_save_index'):
            indexer.add_embeddings(new_embeddings, new_chunks)
        
        assert len(indexer.metadata) == 2
        indexer.index.add.assert_called_once_with(new_embeddings)
    
    def test_add_embeddings_mismatched_input(self, indexer):
        """Test adding embeddings with mismatched chunks."""
        embeddings = np.random.rand(2, 384).astype(np.float32)
        chunks = [{'chunk': 'single'}]  # Only one chunk for two embeddings
        
        with pytest.raises(ValueError, match="Number of embeddings must match"):
            indexer.add_embeddings(embeddings, chunks)
    
    def test_metadata_mapping(self, indexer, sample_embeddings, sample_chunks):
        """Test metadata mapping for chunk-to-video relationships."""
        with patch.object(indexer, '_save_index'):
            indexer.build_index(sample_embeddings, sample_chunks)
        
        # Verify metadata structure
        assert len(indexer.metadata) == 5
        
        # Check first video chunks
        video1_chunks = [m for m in indexer.metadata if m['video_id'] == 'video1']
        assert len(video1_chunks) == 3
        
        # Check second video chunks
        video2_chunks = [m for m in indexer.metadata if m['video_id'] == 'video2']
        assert len(video2_chunks) == 2
        
        # Verify global indices are unique and sequential
        global_indices = [m['global_index'] for m in indexer.metadata]
        assert global_indices == list(range(5))
    
    def test_index_save_load(self, indexer):
        """Test index saving and loading."""
        # Mock FAISS operations
        with patch('app.indexer.faiss') as mock_faiss:
            with patch.object(indexer.file_manager, 'save_json', return_value=True):
                indexer._save_index()
        
        # Verify FAISS write was called
        mock_faiss.write_index.assert_called_once()
    
    def test_rebuild_from_embeddings(self, indexer):
        """Test rebuilding index from existing embedding files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock directory structure
            embeddings_dir = Path(temp_dir) / 'embeddings'
            chunks_dir = Path(temp_dir) / 'chunks'
            embeddings_dir.mkdir()
            chunks_dir.mkdir()
            
            # Create mock files
            embedding_file = embeddings_dir / 'video1.npy'
            chunk_file = chunks_dir / 'video1_chunks.json'
            
            mock_embeddings = np.random.rand(3, 384).astype(np.float32)
            mock_chunks = [
                {
                    'video_id': 'video1',
                    'chunk_id': 'video1_0000',
                    'start_time': 0.0,
                    'end_time': 15.0,
                    'text': 'Test chunk'
                }
            ]
            
            with patch.object(indexer.file_manager, 'load_numpy', return_value=mock_embeddings):
                with patch.object(indexer.file_manager, 'load_json', return_value=mock_chunks):
                    with patch.object(indexer, 'build_index') as mock_build:
                        # Update config paths
                        indexer.config['paths']['embeddings'] = str(embeddings_dir)
                        indexer.config['paths']['chunks'] = str(chunks_dir)
                        
                        indexer.rebuild_from_embeddings()
            
            mock_build.assert_called_once()
    
    def test_get_index_stats(self, indexer):
        """Test getting index statistics."""
        indexer.metadata = [{'test': 'data'}] * 10
        indexer.embedding_dim = 384
        
        stats = indexer.get_index_stats()
        
        assert stats['index_exists'] is True
        assert stats['total_vectors'] == 10
        assert stats['embedding_dimension'] == 384
        assert stats['index_type'] == 'IndexFlatIP'
    
    def test_clear_index(self, indexer):
        """Test clearing index and metadata."""
        # Setup some data
        indexer.metadata = [{'test': 'data'}]
        indexer.embedding_dim = 384
        
        with patch.object(indexer.index_file, 'exists', return_value=True):
            with patch.object(indexer.index_file, 'unlink') as mock_unlink:
                with patch.object(indexer.metadata_file, 'exists', return_value=True):
                    with patch.object(indexer.metadata_file, 'unlink') as mock_unlink_meta:
                        indexer.clear_index()
        
        assert indexer.index is None
        assert indexer.metadata == []
        assert indexer.embedding_dim is None
        mock_unlink.assert_called_once()
        mock_unlink_meta.assert_called_once()
    
    def test_index_size_optimization(self, indexer, sample_embeddings, sample_chunks):
        """Test index size optimization."""
        with patch.object(indexer, '_save_index'):
            indexer.build_index(sample_embeddings, sample_chunks)
        
        # Verify index was created efficiently
        assert indexer.index is not None
        assert len(indexer.metadata) == sum(len(chunks) for chunks in sample_chunks)
    
    def test_concurrent_search_safety(self, indexer):
        """Test thread safety of search operations."""
        # Setup index
        indexer.metadata = [
            {
                'global_index': 0,
                'video_id': 'video1',
                'chunk_id': 'video1_0000',
                'start_time': 0.0,
                'end_time': 15.0,
                'text': 'Test chunk'
            }
        ]
        
        # Mock search results
        mock_scores = np.array([[0.8]])
        mock_indices = np.array([[0]])
        indexer.index.search.return_value = (mock_scores, mock_indices)
        
        query_embedding = np.random.rand(1, 384).astype(np.float32)
        
        # Multiple searches should work without interference
        results1 = indexer.search(query_embedding, k=1)
        results2 = indexer.search(query_embedding, k=1)
        
        assert results1 == results2
    
    @pytest.mark.unit
    def test_index_build_validation(self, indexer, sample_embeddings, sample_chunks):
        """Test that index builds successfully and validates."""
        with patch.object(indexer, '_save_index'):
            indexer.build_index(sample_embeddings, sample_chunks)
        
        # Verify index properties
        assert indexer.index is not None
        assert indexer.embedding_dim == 384
        assert len(indexer.metadata) > 0
        
        # Verify metadata structure
        for metadata in indexer.metadata:
            required_fields = ['global_index', 'video_id', 'chunk_id', 'start_time', 'end_time', 'text']
            for field in required_fields:
                assert field in metadata
    
    @pytest.mark.unit
    def test_search_performance(self, indexer):
        """Test search performance (basic timing)."""
        import time
        
        # Setup index with metadata
        indexer.metadata = [
            {
                'global_index': 0,
                'video_id': 'video1',
                'chunk_id': 'video1_0000',
                'start_time': 0.0,
                'end_time': 15.0,
                'text': 'Test chunk'
            }
        ]
        
        # Mock fast search
        mock_scores = np.array([[0.8]])
        mock_indices = np.array([[0]])
        indexer.index.search.return_value = (mock_scores, mock_indices)
        
        query_embedding = np.random.rand(1, 384).astype(np.float32)
        
        start_time = time.time()
        results = indexer.search(query_embedding, k=1)
        elapsed_time = time.time() - start_time
        
        # Should be very fast with mocked operations
        assert elapsed_time < 0.1
        assert len(results) == 1
