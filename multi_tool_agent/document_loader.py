# document_loader.py
from typing import List
import fitz  # PyMuPDF

def load_pdf(path: str) -> str:
    """Extracts and concatenates all text from a PDF file."""
    doc = fitz.open(path)
    pages = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        pages.append(text)
    return "\n\n".join(pages)

def chunk_text(
    text: str, 
    max_chars: int = 2000, 
    overlap_chars: int = 200
) -> List[str]:
    """
    Splits `text` into chunks of ~max_chars, overlapping by overlap_chars.
    """
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap_chars
    return chunks
