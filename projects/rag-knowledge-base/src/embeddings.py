#!/usr/bin/env python3
"""
Embeddings + Vector Store Builder
Uses SentenceTransformers for embeddings, FAISS for vector storage.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import faiss

# Paths
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
VECTOR_DIR = DATA_DIR / "vector_store"
CORPUS_PATH = RAW_DIR / "arxiv_corpus.json"

# Model config
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 768  # Larger chunks = fewer total chunks
BATCH_ENCODE = 128  # Larger batch size for faster encoding


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = 128) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks


def build_embeddings():
    """Build embeddings and FAISS index from ArXiv corpus."""
    print("Loading corpus...")
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        papers = json.load(f)
    
    print(f"Loaded {len(papers)} papers")
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Prepare chunks with metadata
    print("Chunking abstracts...")
    chunks = []
    metadata = []
    
    for paper in papers:
        abstract = paper.get("abstract", "")
        if not abstract or len(abstract) < 50:
            continue
        
        text_chunks = chunk_text(abstract)
        for i, chunk in enumerate(text_chunks):
            chunks.append(chunk)
            metadata.append({
                "paper_id": paper.get("id", ""),
                "title": paper.get("title", ""),
                "authors": paper.get("authors", []),
                "published": paper.get("published", ""),
                "categories": paper.get("categories", []),
                "url": paper.get("url", ""),
                "chunk_index": i,
                "total_chunks": len(text_chunks),
            })
    
    print(f"Total chunks: {len(chunks)}")
    
    # Build embeddings
    print("Encoding chunks (this may take a few minutes)...")
    embeddings = model.encode(chunks, show_progress_bar=True, batch_size=BATCH_ENCODE, convert_to_numpy=True)
    embeddings = np.array(embeddings).astype("float32")
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Build FAISS index
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product = cosine on normalized vectors
    index.add(embeddings)
    
    # Save index
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(VECTOR_DIR / "faiss.index"))
    
    # Save metadata and chunks
    with open(VECTOR_DIR / "metadata.pkl", "wb") as f:
        pickle.dump({"chunks": chunks, "metadata": metadata, "model": EMBEDDING_MODEL}, f)
    
    # Save embeddings separately
    np.save(VECTOR_DIR / "embeddings.npy", embeddings)
    
    print(f"\nVector store built successfully!")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Dimension: {dimension}")
    print(f"  Index saved to: {VECTOR_DIR / 'faiss.index'}")
    print(f"  Metadata saved to: {VECTOR_DIR / 'metadata.pkl'}")
    
    return index, chunks, metadata


def load_vector_store() -> Tuple[faiss.Index, List[str], List[Dict]]:
    """Load pre-built vector store."""
    index = faiss.read_index(str(VECTOR_DIR / "faiss.index"))
    with open(VECTOR_DIR / "metadata.pkl", "rb") as f:
        data = pickle.load(f)
    return index, data["chunks"], data["metadata"]


if __name__ == "__main__":
    build_embeddings()
