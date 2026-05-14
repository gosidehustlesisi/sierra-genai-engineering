#!/usr/bin/env python3
"""
Re-ranker Module
Cross-encoder re-ranking for precision improvement over bi-encoder retrieval.
"""

import os
import numpy as np
from typing import List, Tuple, Dict
from sentence_transformers import CrossEncoder

# Config
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# LLM fallback for re-ranking
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


class ReRanker:
    """Cross-encoder re-ranker for precise relevance scoring."""
    
    def __init__(self, model_name: str = CROSS_ENCODER_MODEL):
        print(f"Loading cross-encoder: {model_name}...")
        self.model = CrossEncoder(model_name)
        print("Cross-encoder loaded")
    
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, Dict, float]],
        top_k: int = 5
    ) -> List[Tuple[str, Dict, float]]:
        """
        Re-rank candidates using cross-encoder.
        
        Args:
            query: User query
            candidates: List of (chunk, metadata, score) from first-stage retrieval
            top_k: Number of results to return
        
        Returns:
            Re-ranked list of (chunk, metadata, score) tuples
        """
        if not candidates:
            return []
        
        # Prepare query-document pairs
        pairs = [(query, chunk) for chunk, _, _ in candidates]
        
        # Score with cross-encoder
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Combine with original scores (weighted average)
        results = []
        for i, (chunk, meta, orig_score) in enumerate(candidates):
            # Cross-encoder score (0 to ~10) + original cosine score
            combined = 0.7 * float(scores[i]) + 0.3 * orig_score
            results.append((chunk, meta, combined))
        
        # Sort by combined score descending
        results.sort(key=lambda x: x[2], reverse=True)
        
        return results[:top_k]
    
    def rerank_with_llm(
        self,
        query: str,
        candidates: List[Tuple[str, Dict, float]],
        top_k: int = 5
    ) -> List[Tuple[str, Dict, float]]:
        """Fallback LLM-as-judge re-ranking when cross-encoder unavailable."""
        client = get_llm_client()
        if not client:
            # No LLM available, return candidates as-is
            return candidates[:top_k]
        
        scored = []
        for chunk, meta, _ in candidates:
            prompt = f"""Rate how relevant this document is to answering the query.

Query: {query}
Document: {chunk[:500]}

Rate from 0 (completely irrelevant) to 10 (directly answers the query).
Respond with ONLY a number."""
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini" if OPENAI_API_KEY else "kimi-latest",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=5
                )
                score = float(response.choices[0].message.content.strip()) / 10.0
            except (ValueError, Exception):
                score = 0.0
            
            scored.append((chunk, meta, score))
        
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]


if __name__ == "__main__":
    # Test with sample candidates
    candidates = [
        ("Transformers use self-attention mechanisms to process sequences.", {}, 0.8),
        ("RNNs process sequences sequentially using hidden states.", {}, 0.6),
        ("CNNs use convolutional filters for image processing.", {}, 0.4),
    ]
    
    reranker = ReRanker()
    results = reranker.rerank("How do transformers work?", candidates, top_k=2)
    
    print("Re-ranked results:")
    for chunk, meta, score in results:
        print(f"  [{score:.3f}] {chunk[:80]}...")
