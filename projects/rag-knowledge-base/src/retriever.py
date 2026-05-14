#!/usr/bin/env python3
"""
Retriever Module
Implements query transformation (expansion, decomposition, HyDE) + FAISS retrieval.
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import faiss

# Config
VECTOR_DIR = Path("data/vector_store")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# LLM API for query transformation (uses any available API key)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
KIMI_API_KEY = os.getenv("KIMI_API_KEY")


def get_llm_client():
    """Get LLM client from available API keys."""
    if OPENAI_API_KEY:
        from openai import OpenAI
        return OpenAI(api_key=OPENAI_API_KEY)
    elif KIMI_API_KEY:
        from openai import OpenAI
        return OpenAI(api_key=KIMI_API_KEY, base_url="https://api.moonshot.cn/v1")
    return None


def expand_query(query: str, num_variants: int = 3) -> List[str]:
    """Generate semantically equivalent query variants using LLM."""
    client = get_llm_client()
    if not client:
        # Fallback: simple rule-based expansion
        return _rule_based_expansion(query)
    
    prompt = f"""Generate {num_variants} alternative ways to ask this question.
Each should use different vocabulary but seek the same information.

Original: {query}

Alternatives (one per line, numbered):"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini" if OPENAI_API_KEY else "kimi-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        
        variants = [query]
        for line in response.choices[0].message.content.split("\n"):
            line = line.strip()
            if line and "." in line[:3]:
                variant = line.split(".", 1)[1].strip()
                if variant:
                    variants.append(variant)
        return variants[:num_variants + 1]
    except Exception as e:
        print(f"Query expansion failed: {e}")
        return [query]


def decompose_query(query: str) -> List[str]:
    """Break complex queries into sub-queries."""
    client = get_llm_client()
    if not client:
        return [query]
    
    prompt = f"""Break this complex question into 2-3 simpler sub-questions.
Each sub-question should be answerable independently.

Question: {query}

Sub-questions:"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini" if OPENAI_API_KEY else "kimi-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        
        sub_queries = []
        for line in response.choices[0].message.content.split("\n"):
            line = line.strip()
            if line and line.endswith("?"):
                sub_queries.append(line)
        return sub_queries if sub_queries else [query]
    except Exception as e:
        print(f"Query decomposition failed: {e}")
        return [query]


def _rule_based_expansion(query: str) -> List[str]:
    """Fallback expansion when no LLM is available."""
    # Simple synonym expansion
    synonyms = {
        "machine learning": ["ML", "machine learning algorithms"],
        "neural network": ["deep learning", "ANN"],
        "NLP": ["natural language processing", "text processing"],
        "computer vision": ["image recognition", "visual AI"],
        "training": ["learning", "optimization"],
    }
    
    variants = [query]
    for key, alts in synonyms.items():
        if key.lower() in query.lower():
            for alt in alts:
                variants.append(query.lower().replace(key.lower(), alt))
    return variants[:4]


def generate_hypothetical_document(query: str) -> str:
    """Generate hypothetical document for HyDE."""
    client = get_llm_client()
    if not client:
        return query  # Fallback: use query itself
    
    prompt = f"""Write a short passage (2-3 sentences) that would answer this question.
The passage should sound like it's from a scientific paper or technical documentation.

Question: {query}

Passage:"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini" if OPENAI_API_KEY else "kimi-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"HyDE generation failed: {e}")
        return query


class Retriever:
    """Document retriever with query transformation and HyDE."""
    
    def __init__(self, vector_dir: Path = VECTOR_DIR):
        self.vector_dir = vector_dir
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.chunks = []
        self.metadata = []
        self._load()
    
    def _load(self):
        """Load vector store."""
        print("Loading vector store...")
        self.index = faiss.read_index(str(self.vector_dir / "faiss.index"))
        with open(self.vector_dir / "metadata.pkl", "rb") as f:
            data = pickle.load(f)
            self.chunks = data["chunks"]
            self.metadata = data["metadata"]
        print(f"Loaded {len(self.chunks)} chunks")
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query string."""
        embedding = self.model.encode([query])
        embedding = np.array(embedding).astype("float32")
        faiss.normalize_L2(embedding)
        return embedding
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        use_expansion: bool = True,
        use_hyde: bool = True,
        hyde_alpha: float = 0.6
    ) -> List[Tuple[str, Dict, float]]:
        """
        Retrieve documents using query transformation + HyDE.
        
        Returns: List of (chunk_text, metadata, score) tuples, sorted by score.
        """
        import time
        start = time.time()
        
        # Query transformation
        if use_expansion:
            queries = expand_query(query)
            print(f"Expanded to {len(queries)} queries")
        else:
            queries = [query]
        
        # Generate HyDE document
        if use_hyde:
            hypo_doc = generate_hypothetical_document(query)
            print(f"HyDE document generated")
        else:
            hypo_doc = None
        
        # Embed all query variants
        query_embeddings = []
        for q in queries:
            emb = self.embed_query(q)
            query_embeddings.append(emb[0])
        
        # Combine embeddings
        if hypo_doc and use_hyde:
            hyde_emb = self.embed_query(hypo_doc)[0]
            # Weighted combination: HyDE gets higher weight
            combined = np.mean(query_embeddings + [hyde_emb * (hyde_alpha * 2)], axis=0)
        else:
            combined = np.mean(query_embeddings, axis=0)
        
        # Normalize combined embedding
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        
        # Search FAISS
        combined = combined.reshape(1, -1).astype("float32")
        distances, indices = self.index.search(combined, top_k * 2)  # Over-fetch for re-ranking
        
        # Build results
        results = []
        seen_ids = set()
        for idx, score in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            meta = self.metadata[idx]
            paper_id = meta.get("paper_id", "")
            if paper_id in seen_ids:
                continue
            seen_ids.add(paper_id)
            results.append((self.chunks[idx], meta, float(score)))
        
        elapsed = (time.time() - start) * 1000
        print(f"Retrieval completed in {elapsed:.0f}ms")
        
        return results[:top_k]


if __name__ == "__main__":
    import pickle
    retriever = Retriever()
    query = "What are the latest techniques in transformer architecture?"
    results = retriever.retrieve(query, top_k=5)
    
    print(f"\nQuery: {query}")
    for i, (chunk, meta, score) in enumerate(results, 1):
        print(f"\n[{i}] Score: {score:.3f}")
        print(f"  Title: {meta['title'][:80]}...")
        print(f"  Chunk: {chunk[:150]}...")
