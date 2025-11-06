import pytest
from src.rag_pipeline import RAGPipeline
from src.optimization import RAGOptimizer, QueryOptimizer
import numpy as np

def test_deduplication_and_compression():
    chunks = ["AI is great.", "AI is great.", "Machine learning is cool."]
    deduped = RAGOptimizer.deduplicate_chunks(chunks)
    assert len(deduped) < len(chunks)

    dummy_embeddings = np.random.rand(10, 256)
    compressed = RAGOptimizer.compress_embeddings(dummy_embeddings, target_dim=5)
    assert compressed.shape[1] == 5

def test_query_expansion():
    variations = QueryOptimizer.expand_query("explain why AI is useful")
    assert len(variations) >= 1
    assert any("describe" in v or "reason" in v for v in variations)

def test_query_optimized_pipeline():
    pipeline = RAGPipeline()     
    pipeline.ingest_document("tests/test_data/sample1.txt")
    result = pipeline.query_optimized("What is AI?")
    assert "answer" in result
    assert isinstance(result["answer"], str)
