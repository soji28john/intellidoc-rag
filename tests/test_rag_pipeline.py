from src.rag_pipeline import RAGPipeline

# Initialize pipeline
rag = RAGPipeline()

# Ingest documents
documents = [
    'data/raw/sample.pdf',
    'tests/test_data/sample.pdf',
    'tests/test_data/sample.docx',
    'tests/test_data/sample.txt',
    #'tests/test_data/sample.csv',
]

for doc in documents:
    result = rag.ingest_document(doc)
    print(result)

# Query the system
questions = [
    "What are the main findings?",
    "Who are the authors?",
    "What methodology was used?",
    "What are the limitations?"
]

for q in questions:
    result = rag.query(q)
    
    print(f"Q: {q}")
    print(f"A: {result['answer']}")
    print(f"Sources: {result.get('sources', [])}")
    print(f"Confidence: {result.get('confidence', 'N/A')}")