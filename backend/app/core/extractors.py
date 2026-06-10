from typing import List, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class DocumentExtractor(ABC):
    @abstractmethod
    def extract(self, file_path: str) -> str:
        pass

    @abstractmethod
    def chunk(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        pass


class PDFExtractor(DocumentExtractor):
    def extract(self, file_path: str) -> str:
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return ""

    def chunk(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        if not text:
            return []
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            
            if start > 0:
                search_start = max(0, start - overlap)
                newline_pos = text.rfind('\n', search_start, start)
                if newline_pos != -1 and newline_pos >= search_start:
                    start = newline_pos + 1
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)
            
            start = end
        
        return chunks


class TextExtractor(DocumentExtractor):
    def extract(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return ""

    def chunk(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        if not text:
            return []
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)
            start = end
        
        return chunks


class WebExtractor(DocumentExtractor):
    def extract(self, url: str) -> str:
        try:
            import requests
            from bs4 import BeautifulSoup
            import html2text
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for script in soup(["script", "style"]):
                script.decompose()
            
            h = html2text.HTML2Text()
            h.ignore_links = False
            text = h.handle(str(soup))
            
            return text.strip()
        except Exception as e:
            logger.error(f"Web extraction failed: {e}")
            return ""

    def chunk(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        if not text:
            return []
        
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks


class DocxExtractor(DocumentExtractor):
    def extract(self, file_path: str) -> str:
        try:
            from docx import Document
            doc = Document(file_path)
            return "\n\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            return ""

    def chunk(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        if not text:
            return []
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)
            start = end
        
        return chunks


def get_extractor(source_type: str) -> DocumentExtractor:
    extractors = {
        "pdf": PDFExtractor(),
        "txt": TextExtractor(),
        "text": TextExtractor(),
        "web": WebExtractor(),
        "url": WebExtractor(),
        "docx": DocxExtractor(),
    }
    
    return extractors.get(source_type.lower(), TextExtractor())
