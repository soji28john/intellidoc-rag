"""
RAG Pipeline Module with advanced features
Conversational memory, re-ranking, Hybrid search
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
from src.optimization import QueryOptimizer, RAGOptimizer

class RAGPipeline:
    """Complete RAG pipeline with document ingestion, semantic search, and LLM answering."""

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
        self.ingested_files = []

        if self.use_reranker:
            if CrossEncoder is None:
                raise ImportError("CrossEncoder not available. Install sentence-transformers and torch.")
            self.reranker = CrossEncoder(self.reranker_model)

        print(f"RAG Pipeline initialized (use_reranker={self.use_reranker})")

    def reset(self):
        """Reset vector store and ingested files."""
        self.vector_store.reset()
        self.ingested_files = []
        self._bm25 = None
        self._bm25_corpus = []

    def ingest_document(self, file_path: str) -> Dict:
        """Ingest a single document into the pipeline."""
        start_time = time.time()
        docs = self.doc_processor.process_document(file_path)
        if not docs:
            return {"file": file_path, "chunks_created": 0, "time_taken": "0s", "status": "no_content"}

        texts = [doc['content'] for doc in docs]
        embeddings = self.embedding_gen.generate_embeddings_batch(texts)

        # Add to vector store
        self.vector_store.add_documents(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=[doc['metadata'] for doc in docs]
        )
        self.ingested_files.append(file_path)

        # Update BM25 corpus
        self._bm25_corpus.extend([t.split() for t in texts])
        if self._bm25_corpus:
            self._bm25 = BM25Okapi(self._bm25_corpus)

        return {
            'file': file_path,
            'chunks_created': len(docs),
            'time_taken': f"{time.time() - start_time:.2f}s",
            'status': 'success'
        }

    def ingest_multiple_documents(self, file_paths: List[str]):
        """Ingest multiple documents sequentially."""
        results = []
        for path in file_paths:
            result = self.ingest_document(path)
            results.append(result)
        return results

    def retrieve_semantic(self, query: str, top_k: int = 10):
        if not hasattr(self.vector_store, "embeddings") or len(self.vector_store.embeddings) == 0:
            return []

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
        return [" ".join(t) for t in top]

    def rerank_results(self, query: str, docs: List[str]):
        if not self.reranker:
            return [{"content": d, "score": 0.0} for d in docs]
        pairs = [(query, d) for d in docs]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        scored = [{"content": d, "score": float(s)} for d, s in zip(docs, scores)]
        return sorted(scored, key=lambda x: x["score"], reverse=True)

    def hybrid_search(self, query: str, top_k_semantic: int = 10, top_k_bm25: int = 10, final_k: int = 5):
        sem = self.retrieve_semantic(query, top_k_semantic)
        sem_texts = [r["content"] for r in sem]
        bm25_texts = self.retrieve_bm25(query, top_k_bm25)
        seen, merged = set(), []
        for t in sem_texts + bm25_texts:
            if t not in seen:
                seen.add(t)
                merged.append(t)
        if self.use_reranker:
            reranked = self.rerank_results(query, merged)
            merged_sorted = [r["content"] for r in reranked][:final_k]
        else:
            merged_sorted = merged[:final_k]
        return [{"content": c, "metadata": {}} for c in merged_sorted]

    def query(self, question: str, n_results: int = 5, include_sources: bool = True) -> Dict:
        start_time = time.time()
        candidates = self.retrieve_semantic(question, top_k=n_results * 3)
        if not candidates:
            return {
                "answer": "No documents found. Please ingest documents first.",
                "sources": [],
                "confidence": "low",
                "query_time": "0s",
                "retrieved_chunks": 0
            }

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
            "total_documents": len(self.vector_store.embeddings) if hasattr(self.vector_store, "embeddings") else 0,
            "embedding_model": getattr(self.embedding_gen, "model_name", "unknown"),
            "llm_model": getattr(self.llm, "model", "unknown")
        }
    

    def query_optimized(self, question: str, n_results: int = 5):
        """
        Optimized query execution with:
        - Query expansion (semantic reformulation)
        - Multi-variation retrieval
        - Deduplication of chunks
        - Cross-encoder re-ranking
        """

        print(f"[Optimized Query] Starting for: '{question}'")
        # Expand query into multiple reformulations and Retrieve results for each variation
        query_variations = QueryOptimizer.expand_query(question)
        print(f"Generated {len(query_variations)} query variations.")   
        all_results = []
        for var in query_variations:
            query_emb = self.embedding_gen.generate_embedding(var)
            retrieved = self.vector_store.search(query_emb, n_results=n_results)
            docs = retrieved.get("documents", [[]])[0]
            all_results.extend(docs)

        if not all_results:
            return {"answer": "No relevant documents found.", "query": question}

        print(f"Retrieved {len(all_results)} raw results across all variations.")
        unique_results = RAGOptimizer.deduplicate_chunks(all_results)
        print(f"Deduplicated to {len(unique_results)} unique results.")
        # Re-rank for best relevance and generate final LLM answer with citations
        reranked_docs = QueryOptimizer.rerank_results(question, unique_results, top_k=n_results)
        print(f"Re-ranked and selected top {len(reranked_docs)} results.")

        context = [{"content": doc, "metadata": {}} for doc in reranked_docs]
        answer_obj = self.llm.generate_with_citations(question, context)

        answer_obj.update({
            "optimized": True,
            "retrieved_chunks": len(reranked_docs),
        })

        return answer_obj


# Conversational RAG
class ConversationalRAG(RAGPipeline):
    def __init__(self, use_reranker: bool = False, reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        super().__init__(use_reranker=use_reranker, reranker_model=reranker_model)
        self.conversation_history: List[Dict] = []

    def query_with_history(self, question: str, n_results: int = 5, include_sources: bool = True):
        history_snippet = "\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in self.conversation_history[-5:]])
        question_with_history = (history_snippet + "\n\n" + question) if history_snippet else question
        result = self.query(question_with_history, n_results=n_results, include_sources=include_sources)
        self.conversation_history.append({"question": question, "answer": result.get("answer", "")})
        return result
