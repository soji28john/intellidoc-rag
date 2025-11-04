"""
FastAPI Backend for RAG System
RESTful API for programmatic access
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
from src.rag_pipeline import RAGPipeline

# Initialize FastAPI app
app = FastAPI(
    title="IntelliDoc API",
    description="RAG-powered document Q&A system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()

# Pydantic models for request/response
class Query(BaseModel):
    question: str
    n_results: Optional[int] = 5
    include_sources: Optional[bool] = True

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[dict]] = None
    confidence: Optional[str] = None
    query_time: str
    retrieved_chunks: int

class DocumentIngestionResponse(BaseModel):
    file: str
    chunks_created: int
    time_taken: str
    status: str

# Routes
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "IntelliDoc API is running",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.post("/ingest", response_model=DocumentIngestionResponse)
async def ingest_document(file: UploadFile = File(...)):
    """
    Ingest a document into the RAG system
    
    Args:
        file: Document file (PDF, TXT, DOCX)
    
    Returns:
        Ingestion statistics
    """
    # Validate file type
    allowed_types = ['application/pdf', 'text/plain', 
                     'application/vnd.openxmlformats-officedocument.wordprocessingml.document']
    
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    # Save uploaded file
    os.makedirs("data/raw", exist_ok=True)
    file_path = os.path.join("data/raw", file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Ingest document
    try:
        result = rag_pipeline.ingest_document(file_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_documents(query: Query):
    """
    Query the RAG system
    
    Args:
        query: Query object with question and parameters
    
    Returns:
        Answer with sources and metadata
    """
    try:
        result = rag_pipeline.query(
            query.question,
            n_results=query.n_results,
            include_sources=query.include_sources
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        stats = rag_pipeline.get_system_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/reset")
async def reset_system():
    """Reset the vector database """
    try:
        rag_pipeline.reset()
        # Implementation to clear vector store
        return {"message": "System reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)