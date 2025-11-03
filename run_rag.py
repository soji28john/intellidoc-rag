from src.rag_pipeline import RAGPipeline

def main():
    rag = RAGPipeline(use_reranker=False)  # change to True if want reranking

    docs = ["data/raw/sample.pdf"]  # change your PDF path
    rag.ingest_multiple_documents(docs)

    question = "What is the objective of the project?"
    response = rag.query(question)

    print("\nAnswer:")
    print(response["answer"])

if __name__ == "__main__":
    main()

