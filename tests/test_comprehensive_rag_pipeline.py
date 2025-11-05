"""
Unit and integration tests for RAG pipeline
Updated for current RAGPipeline implementation
"""

import pytest
import time
from src.rag_pipeline import RAGPipeline
from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
import os
import numpy as np

@pytest.fixture
def rag_pipeline():
    """Fixture for RAG pipeline"""
    return RAGPipeline()

@pytest.fixture
def sample_documents():
    """Fixture for sample test documents"""
    return ["tests/test_data/sample1.txt", "tests/test_data/sample2.txt"]

class TestDocumentProcessor:
    """Test document processing"""

    def test_load_txt(self):
        processor = DocumentProcessor()
        text = processor.load_txt("tests/test_data/sample.txt")
        assert isinstance(text, str)
        assert len(text) > 0

    def test_chunking(self):
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        text = "a" * 250
        chunks = processor.chunk_text(text)
        assert len(chunks) > 1
        assert all(len(chunk) <= 100 for chunk in chunks)

    def test_chunk_overlap(self):
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        text = "abcdefghij" * 30
        chunks = processor.chunk_text(text)
        if len(chunks) > 1:
            assert chunks[0][-20:] == chunks[1][:20]

class TestEmbeddings:
    """Test embedding generation"""

    def test_embedding_shape(self):
        embedder = EmbeddingGenerator()
        embedding = embedder.generate_embedding("test text")
        assert embedding.shape[0] == embedder.embedding_dim

    def test_batch_embeddings(self):
        embedder = EmbeddingGenerator()
        texts = ["text1", "text2", "text3"]
        embeddings = embedder.generate_embeddings_batch(texts)
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == embedder.embedding_dim

    def test_semantic_similarity(self):
        embedder = EmbeddingGenerator()
        emb1 = embedder.generate_embedding("dog")
        emb2 = embedder.generate_embedding("puppy")
        emb3 = embedder.generate_embedding("car")
        sim_dog_puppy = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        sim_dog_car = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))
        assert sim_dog_puppy > sim_dog_car

class TestRAGPipeline:
    """Test complete RAG pipeline"""

    def test_document_ingestion(self, rag_pipeline, sample_documents):
        result = rag_pipeline.ingest_document(sample_documents[0])
        assert result['status'] == 'success'
        assert result['chunks_created'] > 0

    def test_query_response(self, rag_pipeline):
        # Ingest a known document first
        rag_pipeline.ingest_document("tests/test_data/known_content.txt")
        result = rag_pipeline.query("What is the main topic?")
        assert 'answer' in result
        assert isinstance(result['answer'], str)
        assert len(result['answer']) > 0

    def test_source_citations(self, rag_pipeline):
        rag_pipeline.ingest_document("tests/test_data/known_content.txt")
        result = rag_pipeline.query("test question", include_sources=True)
        assert 'sources' in result
        assert isinstance(result['sources'], list)

    def test_confidence_scoring(self, rag_pipeline):
        rag_pipeline.ingest_document("tests/test_data/known_content.txt")
        result = rag_pipeline.query("test question")
        assert 'confidence' in result
        assert result['confidence'] in ['low', 'medium', 'high']

class TestPerformance:
    """Test performance benchmarks"""

    def test_query_response_time(self, rag_pipeline):
        rag_pipeline.ingest_document("tests/test_data/known_content.txt")
        start = time.time()
        result = rag_pipeline.query("test question")
        end = time.time()
        query_time = end - start
        assert query_time < 5.0

    def test_batch_ingestion(self, rag_pipeline, sample_documents):
        start = time.time()
        results = rag_pipeline.ingest_multiple_documents(sample_documents)
        end = time.time()
        assert all(r['status'] == 'success' for r in results)
        print(f"Batch ingestion time: {end - start:.2f}s")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
