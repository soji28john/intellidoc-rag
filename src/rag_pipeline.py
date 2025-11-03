"""
RAG Pipeline Module
Orchestrates the complete RAG workflow
"""

from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.llm_interface import LLMInterface
from typing import Dict, List
import time

class RAGPipeline:
    """
    Complete RAG pipeline- workflow
    document ingestion(processing, embedding,storage)
    semantic search( retrieval of relevant chunks)
    Answer generation vial LLM.
    
    Attributes:
    doc_processor handle extraction and parsing of documents
    embedding_gen generates vector embeddings from the text
    vector_store manages storage and retrieval of embeddings
    llm interfaces with the language model for answer generation
    """
    # initialization of all components
    def __init__(self):
        
        self.doc_processor = DocumentProcessor()
        self.embedding_gen = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.llm = LLMInterface()
        
        #print("RAG Pipeline initialized")
    
    def ingest_document(self, file_path: str) -> Dict:
        """
        Ingest a document into the RAG system
        
        Args:
            file_path: Path to the input document file
        
        Returns:
            Statistics about ingestion process
            file:document path
            chunks_created: Number of text chunks created
            time_taken: Time taken for ingestion
            status: Ingestion status
        """
        start_time = time.time()
        
        # Step 1: Process document
        print(f"Processing document: {file_path}")
        docs = self.doc_processor.process_document(file_path)
        
        # Step 2: Generate embeddings
        print(f"Generating embeddings for {len(docs)} chunks...")
        texts = [doc['content'] for doc in docs]
        embeddings = self.embedding_gen.generate_embeddings_batch(texts)
        
        # Step 3: Store in vector database
        print("Storing in vector database")
        self.vector_store.add_documents(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=[doc['metadata'] for doc in docs]
        )
        
        elapsed_time = time.time() - start_time
        
        return {
            'file': file_path,
            'chunks_created': len(docs),
            'time_taken': f"{elapsed_time:.2f} seconds",
            'status': 'success'
        }
    
    def ingest_multiple_documents(self, file_paths: List[str]) -> List[Dict]:
        """Ingest multiple documents sequentially
        Args:
            file_paths: (List[str]:List of document file paths
        Returns:
            List[Dict]:List of ingestion statistics for each document
        """
        
        results = []
        for file_path in file_paths:
            result = self.ingest_document(file_path)
            results.append(result)
        return results
    # Querying the system
    def query(self, 
             question: str, 
             n_results: int = 5,
             include_sources: bool = True,
             return_chunks: bool = False) -> Dict:
        
        """
        Ask a query and retrieve an answer using the RAG system
        
        Args:
            question(str): User's question
            n_results(int): Number of relevant chunks to retrieve
            include_sources(bool): Whether to include source citations

        Returns:
            Dict with answer, sources, and metadata(query_time, retrieved_chunks)
        """
        start_time = time.time()
        
        # Step 1: Generate query embedding
        print(f"Query: {question}")
        query_embedding = self.embedding_gen.generate_embedding(question)
        
        # Step 2: Retrieve relevant documents
        print(f"Retrieving top {n_results} relevant documents...")
        search_results = self.vector_store.search(query_embedding, n_results)
        
        # Step 3: Prepare context
        context = []
        for i, doc in enumerate(search_results['documents'][0]):
            context.append({
                'content': doc,
                'metadata': search_results['metadatas'][0][i]
            })
        
        # Step 4: Generate answer
        if return_chunks:
            return {
        "chunks": context,
        "retrieved_texts": [c["content"] for c in context],
        "metadatas": [c["metadata"] for c in context]
    }
        print("Generating answer...")
        if include_sources:
            result = self.llm.generate_with_citations(question, context)
        else:
            answer = self.llm.generate_response(
                question, 
                [ctx['content'] for ctx in context]
            )
            result = {'answer': answer}
        
        # Add timing info
        result['query_time'] = f"{time.time() - start_time:.2f} seconds"
        result['retrieved_chunks'] = n_results
        
        return result
    
    def get_system_stats(self) -> Dict:
        """Get statistics about the RAG system"""
        return {
            'total_documents': self.vector_store.get_collection_stats()['total_documents'],
            'embedding_model': self.embedding_gen.model,
            'embedding_dimension': self.embedding_gen.embedding_dim,
            'llm_model': self.llm.model
        }