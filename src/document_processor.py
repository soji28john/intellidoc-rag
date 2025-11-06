"""
Document Processor Module responsible for reading the documents
Handles loading and processing of various document types
"""

# Import the libraries
from typing import List, Dict
import PyPDF2
from docx import Document
import os

#class definition handles loading the documents and splitting them into chunks and add the metadata like chunk no, source ....
class DocumentProcessor:
    """
    Process various document formats into text chunks
    """
# constructor to initialize the chunk size and chunk overlap   
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        """
        Args:
            chunk_size: Size of each text chunk in characters
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        with open(file_path, 'rb') as file: #rb mode for reading binary
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    
    def load_docx(self, file_path: str) -> str:
        """Extract text from Word document"""
        doc = Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    
    def load_txt(self, file_path: str) -> str:
        """Load plain text file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        Overlap helps maintain context across chunks
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def process_document(self, file_path: str) -> Dict:
        """
        Process document: load, chunk, and prepare metadata
        """
        # Determine file type and load
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            text = self.load_pdf(file_path)
        elif file_extension == '.docx':
            text = self.load_docx(file_path)
        elif file_extension == '.txt':
            text = self.load_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Create chunks
        chunks = self.chunk_text(text)
        
        # Prepare metadata for each chunk
        documents = []
        for i, chunk in enumerate(chunks):
            documents.append({
                'content': chunk,
                'metadata': {
                    'source': file_path,
                    'chunk_id': i,
                    'total_chunks': len(chunks)
                }
            })
        
        return documents