from src.rag_pipeline import ConversationalRAG

def main():
    chat = ConversationalRAG()

    docs = ["data/raw/sample.pdf"]
    chat.ingest_multiple_documents(docs)

    print("\nUser: What is this document about?")
    print(chat.query_with_history("What is this document about?")["answer"])

    print("\nUser: Who is the target audience?")
    print(chat.query_with_history("Who is the target audience?")["answer"])

    print("\nUser: Summarize our conversation.")
    print(chat.query_with_history("Summarize our conversation.")["answer"])

if __name__ == "__main__":
    main()
