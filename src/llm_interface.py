"""
LLM Interface for Ollama - Llama-3 model
Handles communication with large language models
"""
import requests
from openai import OpenAI
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()

class LLMInterface:
    """
    Interface to interact with Ollama-based GPT models(Llama-3)
    This class sends user requests and retrieved context to the model and returns the AI responses(RAG)
    """
    
    def __init__(self, model: str = "llama3", temperature: float = 0.2):
        """
        Initialize LLM interface
        
        Args:
            model: Model name in Ollama(llama3)
            temperature: controls creativity (0=deterministic, 1=creative)
        """
        
        self.model = model
        self.temperature = temperature
        self.ollama_url = "http://localhost:11434/api/chat"
    
    def generate_response(self, 
                         query: str, 
                         context: List[str],
                         system_prompt: [str] = None
                         ) -> str:
        """
        Generate response using user's queryand retrieved context
        
        Args:
            query: User's question
            context: text chunks of retrieved relevant documents
            system_prompt: System instructions for the model
        
        Returns:
            Generated AI answer
        """
       
        # Default system prompt if none provided
        if system_prompt is None:
            system_prompt = """You are a helpful AI assistant that answers questions based on provided context.
            
            Rules:
            1. Answer only using the information from the provided context
            2. If the answer is not in the context, say "I don't have enough information to answer that."
            3. Cite the source when making claims
            4. Be concise but thorough
            5. If you're uncertain, express your uncertainty
            """
        
        # Format context into prompt
        context_text = "\n\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(context)])

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
            ]
        }
        try:
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()  # Raise an error for HTTP errors
            data = response.json()
            return data.get("message", {}).get("content", "")
        
        except Exception as e:
            return f"Error communicating with LLM: {str(e)}"
            

    
    def generate_with_citations(self, 
                               query: str, 
                               context: List[Dict]) -> Dict:
        """
        RAG Generate response with source citations
        
        Args:
            query: User's question
            context: List of dicts with 'content' and 'metadata'
        
        Returns:
            Dict with 'answer' and 'sources'
        """
        
        # Extract text and prepare citation info
        context_texts = []
        sources = []
        
        for i, chunk in enumerate(context):
            context_texts.append(chunk['content'])
            sources.append({
                'id': i + 1,
                'source': chunk['metadata'].get('source', 'Unknown'),
                'chunk_id': chunk['metadata'].get('chunk_id', 0)
            })
        
        # Enhanced system prompt for citations
        
        system_prompt = (
            "Answer ONLY using the provided context.\n"
            "Cite sources like [Source 1], [Source 2] when stating facts.\n"
            "If the context lacks the answer, say so."
            "Be precise and factual."
        )
        answer = self.generate_response(query, context_texts, system_prompt)
        
        return {
            'answer': answer,
            'sources': sources,
            'confidence': self._estimate_confidence(answer, context_texts)
        }
    
    def _estimate_confidence(self, answer: str, context: List[str]) -> str:
        """
        Simple heuristic to estimate confidence
        Args:
            answer: Generated answer
            context: Retrieved context
        """
        if "I don't have" in answer or "not enough information" in answer:
            return "low"
        elif "may" in answer or "possibly" in answer or "might" in answer:
            return "medium"
        else:
            return "high"