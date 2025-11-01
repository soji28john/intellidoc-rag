import os
import sys
import pytest
#import PyPDF2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.document_processor import DocumentProcessor

# Setup test files paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "test_data")

PDF_FILE = os.path.join(DATA_DIR, "sample.pdf")
DOCX_FILE = os.path.join(DATA_DIR, "sample.docx")
TXT_FILE = os.path.join(DATA_DIR, "sample.txt")


@pytest.fixture
def processor():
    """Initialize DocumentProcessor with small chunk size for testing"""
    return DocumentProcessor(chunk_size=50, chunk_overlap=10)

# Test document loading 

def test_load_pdf(processor):
    docs = processor.process_document(PDF_FILE)
    assert isinstance(docs, list)
    assert all("content" in d and "metadata" in d for d in docs)
    assert docs[0]["metadata"]["source"] == PDF_FILE

def test_load_docx(processor):
    docs = processor.process_document(DOCX_FILE)
    assert isinstance(docs, list)
    assert all("content" in d and "metadata" in d for d in docs)
    assert docs[0]["metadata"]["source"] == DOCX_FILE

def test_load_txt(processor):
    docs = processor.process_document(TXT_FILE)
    assert isinstance(docs, list)
    assert all("content" in d and "metadata" in d for d in docs)
    assert docs[0]["metadata"]["source"] == TXT_FILE

# Test chunking 

def test_chunk_text_length(processor):
    text = "A" * 120  # 120 characters
    chunks = processor.chunk_text(text)
    # chunk_size=50, chunk_overlap=10 that meansfirst chunk 0-50, second 40-90, third 80-120
    assert len(chunks) == 3
    assert chunks[0] == "A" * 50
    assert chunks[-1] == "A" * 40

def test_empty_text(processor):
    chunks = processor.chunk_text("")
    assert chunks == []

# Test unsupported file 

def test_unsupported_file(processor):
    unsupported_file = os.path.join(DATA_DIR, "sample.csv")
    with pytest.raises(ValueError):
        processor.process_document(unsupported_file)
