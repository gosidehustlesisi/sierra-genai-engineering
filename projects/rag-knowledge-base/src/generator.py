#!/usr/bin/env python3
"""
Generator Module
Answer synthesis from retrieved documents with citations.
"""

import os
import re
from typing import List, Tuple, Dict

# LLM API
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


def format_context(chunks: List[Tuple[str, Dict, float]]) -> str:
    """Format retrieved chunks with citations for the prompt."""
    context_parts = []
    for i, (chunk, meta, score) in enumerate(chunks, 1):
        title = meta.get("title", "Untitled")
        authors = ", ".join(meta.get("authors", [])[:2])
        published = meta.get("published", "")[:4]  # Year only
        url = meta.get("url", "")
        
        citation = f"[{i}] {title}"
        if authors:
            citation += f" — {authors}"
        if published:
            citation += f" ({published})"
        
        context_parts.append(f"{citation}\n{chunk}\nSource: {url}")
    
    return "\n\n".join(context_parts)


def synthesize_answer(
    query: str,
    chunks: List[Tuple[str, Dict, float]],
    temperature: float = 0.3,
    max_tokens: int = 500
) -> str:
    """
    Synthesize answer from retrieved chunks with citations.
    
    Args:
        query: User question
        chunks: List of (chunk, metadata, score) from retriever/reranker
        temperature: Generation temperature
        max_tokens: Max output tokens
    
    Returns:
        Generated answer with citations
    """
    client = get_llm_client()
    if not client:
        # Fallback: return top chunk summary
        if not chunks:
            return "No relevant documents found."
        top = chunks[0]
        return f"Based on the most relevant source: {top[0][:300]}..."
    
    context = format_context(chunks)
    
    prompt = f"""Answer the question using ONLY the provided context.
Cite sources with [number] format after each claim.
If the context doesn't contain enough information, say so.
Be concise but thorough.

Context:
{context}

Question: {query}

Answer:"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini" if OPENAI_API_KEY else "kimi-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Answer synthesis failed: {e}")
        return "Error generating answer."


def iterative_synthesis(
    query: str,
    chunks: List[Tuple[str, Dict, float]],
    iterations: int = 2
) -> str:
    """Multi-pass answer refinement."""
    # Pass 1: Generate draft
    draft = synthesize_answer(query, chunks, temperature=0.5)
    
    if iterations <= 1:
        return draft
    
    # Pass 2: Refine
    client = get_llm_client()
    if not client:
        return draft
    
    context = format_context(chunks)
    
    refine_prompt = f"""Improve this answer for accuracy, completeness, and clarity.
Ensure all claims are supported by the context below.
Add citations [number] where missing.

Context:
{context}

Draft Answer: {draft}

Improved Answer:"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini" if OPENAI_API_KEY else "kimi-latest",
            messages=[{"role": "user", "content": refine_prompt}],
            temperature=0.2,
            max_tokens=600
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Refinement failed: {e}")
        return draft


def extract_citations(answer: str) -> List[int]:
    """Extract citation numbers from an answer."""
    citations = re.findall(r'\[(\d+)\]', answer)
    return [int(c) for c in citations]


if __name__ == "__main__":
    # Test
    test_chunks = [
        ("Self-attention allows transformers to weigh the importance of different tokens.", 
         {"title": "Attention Is All You Need", "authors": ["Vaswani et al."], "published": "2017"}, 0.9),
        ("Multi-head attention uses multiple attention heads in parallel.",
         {"title": "BERT: Pre-training", "authors": ["Devlin et al."], "published": "2019"}, 0.8),
    ]
    
    answer = synthesize_answer("How do transformers work?", test_chunks)
    print("Generated answer:")
    print(answer)
    print(f"\nCitations: {extract_citations(answer)}")
