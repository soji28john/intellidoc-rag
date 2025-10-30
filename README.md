
# IntelliDoc: RAG-powered document intelligent system
# It's a Retrieval augmented Generation system that designed for the extraction of information from multiple docs(pdf,text..) and provide the accurate, context based answers to the user queries.
# This project encompasses the following features
Ingest multiple document types: PDFs, Word docs, and text files
Processed and stored information parsed from the documents in a vector database
Retrieve relevant context based on user queries
Generate accurate answers using an LLM. Here I'm using Llama 3 
Provide citations and confidence scores
Deploy as a web application for web interface i'm using Streamlit or FastAPI
### Chunking strategy
In the "Intellidoc" RAG system the loaded documents splits into smaller and overlapping text chunks. 
This ensures that The LLM can process large documents and also, preserve the context across the textchunks overlap, help to retrieve the relevant informations more accurately. 
For more understanding take a look at the code(chunking algorithm-chunk size , chunk overlap)
This allows the system to maintain the context across segments, that improves the quality of the retrieved queries answers.