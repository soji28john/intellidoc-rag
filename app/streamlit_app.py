"""
Streamlit Web Interface for RAG System
"""
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# add project root and src folder to sys.path
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from src.rag_pipeline import RAGPipeline
import streamlit as st

# Page config
st.set_page_config(
    page_title="IntelliDoc - RAG Q&A System",
    page_icon="@@", layout="wide"
)

# Initialize RAG pipeline 
@st.cache_resource
def load_rag_pipeline():
    """Load RAG pipeline once and cache it"""
    return RAGPipeline()

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stats-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .answer-box {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
    }
    .source-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main app
def main():
    # Header
    st.markdown("<h1 class='main-header'> IntelliDoc</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem;'>RAG-Powered Document Intelligence System</p>", unsafe_allow_html=True)
    
    # Initialize pipeline
    rag = load_rag_pipeline()
    
    # Sidebar for document upload and settings
    with st.sidebar:
        st.header(" Configuration")
        
        # Document upload section
        st.subheader(" Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    for uploaded_file in uploaded_files:
                        # Save uploaded file temporarily
                        file_path = os.path.join("data/raw", uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Ingest into RAG system
                        result = rag.ingest_document(file_path)
                        st.success(f"✓ Processed {uploaded_file.name}")
                        st.json(result)
        
        st.divider()
        
        # Settings
        st.subheader("Query Settings")
        n_results = st.slider("Number of chunks to retrieve", 1, 10, 5)
        include_sources = st.checkbox("Include source citations", value=True)
        
        st.divider()
        
        # System stats
        st.subheader(" System Statistics")
        stats = rag.get_system_stats()
        st.markdown(f"""
        <div class='stats-box'>
            <p><b>Total Documents:</b> {stats['total_documents']}</p>
            <p><b>Embedding Model:</b> {stats['embedding_model']}</p>
            <p><b>LLM Model:</b> {stats['llm_model']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    st.header(" Ask Your Questions")
    
    # Create tabs for different interfaces
    tab1, tab2, tab3 = st.tabs([" Q&A Interface", " Batch Processing", " Experiment"])
    
    with tab1:
        # Single question interface
        query = st.text_input(
            "Enter your question:",
            placeholder="What are the main findings in the documents?"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            ask_button = st.button("Ask",  use_container_width=True)
        
        if ask_button and query:
            with st.spinner(" Thinking"):
                result = rag.query(query, n_results=n_results, include_sources=include_sources)
                
                # Display answer
                st.markdown("Answer")
                st.markdown(f"""
                <div class='answer-box'>
                    {result['answer']}
                </div>
                """, unsafe_allow_html=True)
                
                # Display metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Query Time", result.get('query_time', 'N/A'))
                with col2:
                    st.metric("Chunks Retrieved", result.get('retrieved_chunks', 'N/A'))
                with col3:
                    confidence = result.get('confidence', 'N/A')
                    st.metric("Confidence", confidence.upper())
                
                # Display sources
                if include_sources and 'sources' in result:
                    st.markdown("###  Sources")
                    for source in result['sources']:
                        st.markdown(f"""
                        <div class='source-box'>
                            <b>Source {source['id']}:</b> {source['source']}<br>
                            <small>Chunk ID: {source['chunk_id']}</small>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Download answer
                st.download_button(
                    label=" Download Answer",
                    data=result['answer'],
                    file_name=f"answer_{query[:20]}.txt",
                    mime="text/plain"
                )
    
    with tab2:
        # Batch processing interface
        st.markdown(" Process Multiple Questions at Once")
        
        questions_text = st.text_area(
            "Enter questions (one per line):",
            height=200,
            placeholder="What is the main topic?\nWho are the authors?\nWhat are the conclusions?"
        )
        
        if st.button("Process All Questions"):
            questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
            
            if questions:
                results_data = []
                progress_bar = st.progress(0)
                
                for i, q in enumerate(questions):
                    result = rag.query(q, n_results=n_results)
                    results_data.append({
                        'Question': q,
                        'Answer': result['answer'],
                        'Confidence': result.get('confidence', 'N/A'),
                        'Time': result.get('query_time', 'N/A')
                    })
                    progress_bar.progress((i + 1) / len(questions))
                
                # Display results in expandable sections
                for i, result in enumerate(results_data):
                    with st.expander(f"Q{i+1}: {result['Question'][:50]}"):
                        st.markdown(f"**Answer:** {result['Answer']}")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Confidence", result['Confidence'])
                        with col2:
                            st.metric("Time", result['Time'])
                
                # Export results
                import pandas as pd
                df = pd.DataFrame(results_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    " Download Results (CSV)",
                    csv,
                    "batch_results.csv",
                    "text/csv"
                )
    
    with tab3:
        # Experimentation interface
        st.markdown("  Experiment with Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            exp_query = st.text_input("Test Query:", "What is the main topic?")
            exp_chunks = st.slider("Chunks to retrieve:", 1, 15, 5, key="exp_chunks")
        
        with col2:
            exp_temperature = st.slider("LLM Temperature:", 0.0, 1.0, 0.7, 0.1)
            exp_model = st.selectbox("LLM Model:", ["llama3", "llama2", "mistral"])
        
        if st.button("Run Experiment"):
            # Run with different parameters
            with st.spinner("Running experiment"):
                # Temporarily change LLM settings
                original_model = rag.llm.model
                original_temp = rag.llm.temperature
                
                rag.llm.model = exp_model
                rag.llm.temperature = exp_temperature
                
                result = rag.query(exp_query, n_results=exp_chunks)
                
                # Restore original settings
                rag.llm.model = original_model
                rag.llm.temperature = original_temp
                
                # Display results
                st.json(result)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with using LangChain, ChromaDB, and Streamlit</p>
        <p><a href='https://github.com/soji28john/intellidoc-rag'>View on GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()