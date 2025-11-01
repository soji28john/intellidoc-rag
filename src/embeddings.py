"""
Embedding Generation Module
Creates vector representations of text
"""

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class EmbeddingGenerator:
    """
    Generate embeddings using Sentence-BERT its an open-source model.
     """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embedding model
        
        Model chosen from Sentence-BERT options:
        - 'all-MiniLM-L6-v2': It's Fast, good quality, 384 dimensions
            """
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for single text"""
        return self.model.encode(text)
    
    def generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts (more efficient)"""
        return self.model.encode(texts, show_progress_bar=True)
