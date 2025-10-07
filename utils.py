
# utils.py
import os
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from PyPDF2 import PdfReader
from tqdm import tqdm

# Load embedding model (global) — smallest good tradeoff for speed & quality
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


from io import BytesIO
from PyPDF2 import PdfReader

def load_pdf_bytes(file_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyPDF2."""
    stream = BytesIO(file_bytes)  # wrap bytes in BytesIO
    reader = PdfReader(stream)
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n\n".join(texts)



def simple_text_split(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks.

    - chunk_size: approx tokens chars per chunk
    - overlap: characters of overlap between chunks
    """
    text = text.replace("\r", "\n")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 2 <= chunk_size:
            current = (current + "\n\n" + p).strip()
        else:
            if current:
                chunks.append(current)
            # if paragraph itself bigger than chunk_size, split it
            if len(p) > chunk_size:
                for i in range(0, len(p), chunk_size - overlap):
                    chunks.append(p[i : i + chunk_size].strip())
                current = ""
            else:
                current = p
    if current:
        chunks.append(current)

    # Post-process to ensure overlap
    final = []
    for i, c in enumerate(chunks):
        if i == 0:
            final.append(c)
        else:
            # create overlap slice from previous chunk tail
            prev = final[-1]
            if overlap > 0:
                tail = prev[-overlap:]
                merged = (tail + " \n\n" + c).strip()
                final.append(merged)
            else:
                final.append(c)
    return final


def embed_texts(texts: List[str]) -> np.ndarray:
    """Return embeddings as a 2D numpy array (n x d)."""
    model = get_embedding_model()
    emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return emb.astype('float32')


def build_faiss_index(embeddings: np.ndarray) -> Tuple[faiss.IndexFlatIP, int]:
    """Build a FAISS inner-product (cosine-style) index and add normalized embeddings.

    Returns (index, dim).
    """
    # Normalize embeddings to unit length for cosine similarity using inner product
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, dim


def search_index(index: faiss.IndexFlatIP, query_emb: np.ndarray, top_k: int = 5):
    """Search FAISS index. query_emb should be (1, d) and already normalized."""
    faiss.normalize_L2(query_emb)
    scores, indices = index.search(query_emb, top_k)
    return scores[0], indices[0]


def retrieve_top_k(query: str, docs: List[str], index: faiss.IndexFlatIP, top_k: int = 5):
    model = get_embedding_model()
    q_emb = model.encode([query], convert_to_numpy=True).astype('float32')
    scores, ids = search_index(index, q_emb, top_k=top_k)
    results = []
    for s, i in zip(scores, ids):
        if i < 0 or i >= len(docs):
            continue
        results.append({
            'chunk': docs[int(i)],
            'score': float(s),
            'id': int(i)
        })
    return results
