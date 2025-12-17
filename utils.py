import fitz
from typing import List
import openai
import tiktoken
import numpy as np
from sentence_transformers import SentenceTransformer

def extract_text_from_pdf(file) -> str:
    """Extract full text from uploaded PDF"""
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def split_text(text: str, max_tokens=500) -> List[str]:
    """Split text into chunks of max token length"""
    enc = tiktoken.get_encoding("cl100k_base")
    words = text.split()
    
    chunks, chunk = [], []
    tokens = 0

    for word in words:
        word_tokens = len(enc.encode(word))
        if tokens + word_tokens > max_tokens:
            chunks.append(" ".join(chunk))
            chunk = []
            tokens = 0
        chunk.append(word)
        tokens += word_tokens

    if chunk:
        chunks.append(" ".join(chunk))

    return chunks

def create_embeddings(chunks: List[str]) -> List[dict]:
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=chunks
    )
    embeddings = []
    for chunk, data in zip(chunks, response.data):
        embeddings.append({"text": chunk, "embedding": data.embedding})
    return embeddings


