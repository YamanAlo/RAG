from typing import List, Dict, Any
import pdfplumber
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.config import CHUNK_SIZE, CHUNK_OVERLAP

class PDFProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
        )

    def extract_text(self, file_path: str | Path) -> str:
        """Extract text from PDF file."""
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text

    def process_document(self, file_path: str | Path) -> List[Dict[str, Any]]:
        """Process PDF document and return chunks with metadata."""
        text = self.extract_text(file_path)
        chunks = self.text_splitter.split_text(text)
        
        # Add metadata to chunks
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunks.append({
                'text': chunk,
                'source': Path(file_path).name,
                'chunk_index': i
            })
        return processed_chunks 