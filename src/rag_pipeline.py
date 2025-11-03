"""
RAG Pipeline Module with advanced features
conversational memory, re-ranking, Hybrid search
Orchestrates the complete RAG workflow
"""

from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.llm_interface import LLMInterface
from typing import Dict, List
import time

try: 
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None
from rank_bm25 import BM25Okapi

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
    def __init__(self, use_reranker: bool = False, reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        

        self.doc_processor = DocumentProcessor()
        self.embedding_gen = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.llm = LLMInterface()
        self.use_reranker = use_reranker
        self.reranker_model = reranker_model
        self.reranker = None
        self._bm25 = None
        self._bm25_corpus = []
        if self.use_reranker:
            if CrossEncoder is None:
                raise ImportError("crossEncoder not available. Install sentence-transformers and torch.")
            
            self.reranker = CrossEncoder(self.reranker_model)
        
            
        print("RAG Pipeline initialized (use_reranker=%s)" % self.use_reranker)
    
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
        
        self._bm25_corpus.extend([text.split() for text in texts])
        if self._bm25_corpus:
            self._bm25 = BM25Okapi(self._bm25_corpus)
        
        
        return {
            'file': file_path,
            'chunks_created': len(docs),
            'time_taken': f"{time.time() - start_time:.2f}s",
            'status': 'success'
        }
    
    def ingest_multiple_documents(self, file_paths: List[str]):
        """Ingest multiple documents sequentially
        Args:
            file_paths: (List[str]:List of document file paths
        Returns:
            List[Dict]:List of ingestion statistics for each document
        """
        return [self.ingest_document(path) for path in file_paths]

        
    
    def retrieve_semantic(self, query: str, top_k: int = 10):
        query_emb = self.embedding_gen.generate_embedding(query)
        results = self.vector_store.search(query_emb, n_results=top_k)
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        return [{"content": d, "metadata": m} for d, m in zip(docs, metas)]

    def retrieve_bm25(self, query: str, top_k: int = 10):
        if not self._bm25:
            return []
        tokenized = query.split()
        top = self._bm25.get_top_n(tokenized, self._bm25_corpus, n=top_k)
        # top are token lists; join back to string
        return [" ".join(t) for t in top]
    
    def rerank_results(self, query: str, docs: List[str]):
        """
        Rerank the provided docs list using cross-encoder (query, doc).
        Returns list of dicts: {'content': doc, 'score': float}
        """
        if not self.reranker:
            # fallback: return docs with default score
            return [{"content": d, "score": 0.0} for d in docs]

        # prepare pairs
        pairs = [(query, d) for d in docs]
        scores = self.reranker.predict(pairs, show_progress_bar=False)  # numpy array
        scored = [{"content": d, "score": float(s)} for d, s in zip(docs, scores)]
        # sort desc
        return sorted(scored, key=lambda x: x["score"], reverse=True)
        
    def hybrid_search(self, query: str, top_k_semantic: int = 10, top_k_bm25: int = 10, final_k: int = 5):
        # semantic
        sem = self.retrieve_semantic(query, top_k_semantic)
        sem_texts = [r["content"] for r in sem]

        # bm25
        bm25_texts = self.retrieve_bm25(query, top_k_bm25)

        # merge, preserve order from semantic then bm25 
        seen, merged = set(),[]
        
        for t in sem_texts + bm25_texts:
            if t not in seen:
                seen.add(t)
                merged.append(t)

      
        if self.use_reranker:
            reranked = self.rerank_results(query, merged)
            merged_sorted = [r["content"] for r in reranked][:final_k]
        else:
            merged_sorted = merged[:final_k]

        # return as context dicts
        return [{"content": c, "metadata": {}} for c in merged_sorted]
    
    # Querying the system
    def query(self, 
             question: str, 
             n_results: int = 5,
             include_sources: bool = True) -> Dict:
        
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
        print(f"Query: {question}")

        candidates = self.retrieve_semantic(question, top_k=n_results * 3)
        candidate_texts = [c["content"] for c in candidates]

        if self.use_reranker:
            reranked = self.rerank_results(question, candidate_texts)
            final_chunks = [{"content": r["content"], "metadata": {}} for r in reranked[:n_results]]
        else:
            final_chunks = candidates[:n_results]

        context_texts = [c["content"] for c in final_chunks]

        if include_sources:
            answer_obj = self.llm.generate_with_citations(question, final_chunks)
        else:
            answer_text = self.llm.generate_response(question, context_texts)
            answer_obj = {"answer": answer_text}

        answer_obj["query_time"] = f"{time.time() - start_time:.2f} seconds"
        answer_obj["retrieved_chunks"] = len(final_chunks)
        
        return answer_obj

    def get_system_stats(self):
        return {
        "total_documents": len(self.vector_store.embeddings) if hasattr(self.vector_store, "embeddings") else "N/A",
        "embedding_model": self.embedding_gen.model_name,
        "llm_model": self.llm.model
    }

 # Conversational Memory
class ConversationalRAG(RAGPipeline):
    def __init__(self, use_reranker: bool = False, reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        super().__init__(use_reranker=use_reranker, reranker_model=reranker_model)
        self.conversation_history: List[Dict] = []

    def query_with_history(self, question: str, n_results: int = 5, include_sources: bool = True):
        # prepare history snippet 
        history_snippet = "\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in self.conversation_history[-5:]])
        # incorporate history into the question 
        question_with_history = (history_snippet + "\n\n" + question) if history_snippet else question

        result = self.query(question_with_history, n_results=n_results, include_sources=include_sources)

        # store canonical question and answer 
        self.conversation_history.append({"question": question, "answer": result.get("answer", "")})
        return result