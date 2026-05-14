#!/usr/bin/env python3
"""
RAG Pipeline
End-to-end orchestration: query → transform → retrieve → rerank → generate.
"""

import time
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path

from src.retriever import Retriever
from src.reranker import ReRanker
from src.generator import synthesize_answer, iterative_synthesis


@dataclass
class RAGResult:
    """Result container for RAG pipeline."""
    query: str
    answer: str
    sources: List[Dict]
    retrieval_time_ms: float
    total_time_ms: float
    strategy_used: str
    num_retrieved: int
    num_reranked: int


class RAGPipeline:
    """Production-grade RAG pipeline for scientific literature search."""
    
    def __init__(
        self,
        vector_dir: Path = Path("data/vector_store"),
        use_reranker: bool = True,
        use_llm_transform: bool = True,
        use_hyde: bool = True,
        retrieval_k: int = 20,
        final_k: int = 5
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            vector_dir: Path to FAISS index and metadata
            use_reranker: Enable cross-encoder re-ranking
            use_llm_transform: Enable LLM query transformation
            use_hyde: Enable HyDE
            retrieval_k: Number of candidates from first-stage retrieval
            final_k: Number of final sources to synthesize from
        """
        print("Initializing RAG Pipeline...")
        self.retriever = Retriever(vector_dir)
        
        self.use_reranker = use_reranker
        self.reranker = None
        if use_reranker:
            try:
                self.reranker = ReRanker()
            except Exception as e:
                print(f"Cross-encoder not available: {e}")
                self.use_reranker = False
        
        self.use_llm_transform = use_llm_transform
        self.use_hyde = use_hyde
        self.retrieval_k = retrieval_k
        self.final_k = final_k
        
        print(f"  Reranker: {self.use_reranker}")
        print(f"  LLM Transform: {use_llm_transform}")
        print(f"  HyDE: {use_hyde}")
        print("Pipeline ready")
    
    def query(self, question: str) -> RAGResult:
        """
        Execute full RAG pipeline.
        
        Args:
            question: User question
        
        Returns:
            RAGResult with answer, sources, and timing
        """
        total_start = time.time()
        
        # Stage 1: Retrieve
        print(f"\n{'='*50}")
        print(f"Query: {question}")
        print(f"{'='*50}")
        
        retrieval_start = time.time()
        candidates = self.retriever.retrieve(
            question,
            top_k=self.retrieval_k,
            use_expansion=self.use_llm_transform,
            use_hyde=self.use_hyde
        )
        retrieval_time = (time.time() - retrieval_start) * 1000
        
        print(f"Retrieved {len(candidates)} candidates")
        
        # Stage 2: Re-rank
        if self.use_reranker and self.reranker:
            print("Re-ranking with cross-encoder...")
            top_chunks = self.reranker.rerank(question, candidates, top_k=self.final_k)
        else:
            # Sort by original score
            candidates.sort(key=lambda x: x[2], reverse=True)
            top_chunks = candidates[:self.final_k]
        
        print(f"Selected top {len(top_chunks)} chunks")
        
        # Stage 3: Generate
        print("Synthesizing answer...")
        answer = synthesize_answer(question, top_chunks)
        
        total_time = (time.time() - total_start) * 1000
        
        # Build sources
        sources = []
        for chunk, meta, score in top_chunks:
            sources.append({
                "content": chunk[:300] + "...",
                "title": meta.get("title", ""),
                "authors": meta.get("authors", []),
                "published": meta.get("published", ""),
                "url": meta.get("url", ""),
                "relevance_score": round(float(score), 3),
                "categories": meta.get("categories", [])
            })
        
        strategy = []
        if self.use_llm_transform:
            strategy.append("query_transform")
        if self.use_hyde:
            strategy.append("hyde")
        if self.use_reranker:
            strategy.append("rerank")
        strategy.append("synthesize")
        
        return RAGResult(
            query=question,
            answer=answer,
            sources=sources,
            retrieval_time_ms=retrieval_time,
            total_time_ms=total_time,
            strategy_used=" + ".join(strategy),
            num_retrieved=len(candidates),
            num_reranked=len(top_chunks)
        )
    
    def query_iterative(self, question: str) -> RAGResult:
        """Execute pipeline with iterative synthesis for higher quality."""
        result = self.query(question)
        
        # Re-fetch top chunks for iterative synthesis
        candidates = self.retriever.retrieve(question, top_k=self.retrieval_k)
        if self.use_reranker and self.reranker:
            top_chunks = self.reranker.rerank(question, candidates, top_k=self.final_k)
        else:
            candidates.sort(key=lambda x: x[2], reverse=True)
            top_chunks = candidates[:self.final_k]
        
        # Iterative refinement
        print("Applying iterative refinement...")
        result.answer = iterative_synthesis(question, top_chunks, iterations=2)
        result.strategy_used += " + iterative_refine"
        
        return result


def demo():
    """Run a demo query through the pipeline."""
    pipeline = RAGPipeline()
    
    test_queries = [
        "What are the latest advances in transformer architectures?",
        "How does reinforcement learning improve large language models?",
        "What techniques improve few-shot learning in computer vision?",
    ]
    
    for q in test_queries:
        result = pipeline.query(q)
        
        print(f"\n{'='*60}")
        print(f"ANSWER:")
        print(f"{'='*60}")
        print(result.answer)
        print(f"\nSources: {len(result.sources)}")
        for i, src in enumerate(result.sources, 1):
            print(f"  [{i}] {src['title'][:60]}... (score: {src['relevance_score']})")
        print(f"\nTiming: {result.retrieval_time_ms:.0f}ms retrieval | {result.total_time_ms:.0f}ms total")
        print(f"Strategy: {result.strategy_used}")


if __name__ == "__main__":
    demo()
