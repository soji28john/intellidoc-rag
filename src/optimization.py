"""
Performance optimization techniques
"""

from functools import lru_cache
import hashlib
from typing import List
import numpy as np

class RAGOptimizer:
    """Optimization utilities for RAG pipeline"""
    

    @staticmethod
    @lru_cache(maxsize=1000)
    def cached_embedding(text: str, model_name: str =="default"):
        """
        Cache embeddings to avoid regenerating for same text
        Uses LRU cache to limit memory usage
        """
        from src.embeddings import EmbeddingGenerator
        embedder = EmbeddingGenerator(model_name=model_name)
        return embedder.generate_embedding(text)
    
    @staticmethod
    def batch_with_size_limit(items: List, max_batch_size: int = 32):
        """
        Split large batches into smaller chunks for processing
        Prevents memory issues with very large document sets
        """
        for i in range(0, len(items), max_batch_size):
            yield items[i:i + max_batch_size]
    
    @staticmethod
    def deduplicate_chunks(chunks: List[str], threshold: float = 0.95):
        """
        Remove highly similar duplicate chunks
        Uses embeddings to find near-duplicates
        """
        from src.embeddings import EmbeddingGenerator
        
        if len(chunks) < 2:
            return chunks
        
        embedder = EmbeddingGenerator()
        embeddings = embedder.generate_embeddings_batch(chunks)
        unique_indices = [0]  
        
        for i in range(1, len(embeddings)):
            is_unique = True
            for j in unique_indices:
                similarity = np.dot(embeddings[i], embeddings[j]) / \
                           (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                if similarity > threshold:
                    is_unique = False
                    break
            
            if is_unique:
                unique_indices.append(i)
        
        return [chunks[i] for i in unique_indices]
    
    @staticmethod
    def compress_embeddings(embeddings: np.ndarray, target_dim: int = 128):
        """
        Reduce embedding dimensionality using PCA
        Speeds up vector search with minimal accuracy loss
        """
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=target_dim)
        compressed = pca.fit_transform(embeddings)
        
        print(f"Variance retained: {pca.explained_variance_ratio_.sum():.2%}")
        return compressed

# Implement query optimization
class QueryOptimizer:
    """Optimize query processing"""
    
    @staticmethod
    @lru_cache(maxsize=1)
    def get_cross_encoder():
        """Load and cache CrossEncoder model"""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError("Please install sentence-transformers and torch to use CrossEncoder.")
        return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    @staticmethod
    def expand_query(query: str) -> List[str]:
        """
        Generate query variations to improve retrieval
        Techniques:
        - Synonym expansion
        - Question reformulation
        - Key phrase extraction
        """
        variations = [query]      
        synonyms = {
            'find': ['locate', 'discover', 'identify'],
            'explain': ['describe', 'clarify', 'elaborate'],
            'why': ['reason', 'cause', 'purpose']
        }
        
        words = query.lower().split()
        for word in words:
            if word in synonyms:
                for synonym in synonyms[word]:
                    new_query = query.lower().replace(word, synonym)
                    variations.append(new_query)
        
        return variations[:5]  
    
        
    @staticmethod
    def rerank_results(query: str, documents: List[str], top_k: int = 5):
        """
        Re-rank retrieved documents using cross-encoder
        More accurate than bi-encoder but slower
        """
        if not documents:
            return []
        model = QueryOptimizer.get_cross_encoder()     
        pairs = [[query, doc] for doc in documents]
        scores = model.predict(pairs)          
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [documents[i] for i in ranked_indices]
    
    if __name__ == "__main__":
        import numpy as np

    texts = ["AI is transforming the world.", "Artificial intelligence changes industries."]
    from src.optimization import RAGOptimizer

    optimizer = RAGOptimizer()
    deduped = optimizer.deduplicate_chunks(texts)
    print("Deduplicated chunks:", deduped)


    dummy_embeddings = np.random.rand(10, 512)
    target_dim = min(dummy_embeddings.shape)
    compressed = optimizer.compress_embeddings(dummy_embeddings, target_dim=target_dim)
    print("Compressed shape:", compressed.shape)
