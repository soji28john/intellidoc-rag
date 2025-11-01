"""
Vector Store Module
Manages storage and retrieval of document embeddings
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict
import uuid

class VectorStore:
    """
    Interface to ChromaDB vector database
    """
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize ChromaDB client"""
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"description": "Document embeddings for RAG"}
        )
    
    def add_documents(self, 
                     documents: List[str], 
                     embeddings: List[List[float]], 
                     metadatas: List[Dict]):
        """
        Add documents to vector store
        
        Args:
            documents: List of text chunks
            embeddings: Corresponding embeddings
            metadatas: Metadata for each document
        """
        # Generate unique IDs
        ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    
    def search(self, 
               query_embedding: List[float], 
               n_results: int = 5) -> Dict:
        """
        Search for similar documents
        
        Args:
            query_embedding: Embedding of search query
            n_results: Number of results to return
        
        Returns:
            Dictionary with documents, distances, and metadata
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return results
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about stored documents"""
        return {
            "total_documents": self.collection.count(),
            "name": self.collection.name
        }