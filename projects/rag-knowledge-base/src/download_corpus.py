#!/usr/bin/env python3
"""
ArXiv Corpus Downloader
Fetches 2,000+ real abstracts via export.arxiv.org API.
Queries: machine learning, NLP, data science, computer vision
"""

import requests
import xml.etree.ElementTree as ET
import json
import time
import os
from pathlib import Path

# ArXiv API endpoint
ARXIV_API = "http://export.arxiv.org/api/query"

# Search queries to fetch diverse abstracts
QUERIES = [
    "machine learning",
    "natural language processing",
    "deep learning",
    "computer vision",
    "data science",
    "reinforcement learning",
    "neural networks",
    "transformer",
    "generative AI",
    "large language models",
]

# Config
MAX_PER_QUERY = 300
BATCH_SIZE = 100  # ArXiv recommends max 100 per request
SLEEP_SECONDS = 3  # Rate limit: 1 request every 3 seconds
OUTPUT_DIR = Path("data/raw")


def fetch_arxiv(query: str, start: int, max_results: int) -> list[dict]:
    """Fetch a batch of ArXiv results."""
    params = {
        "search_query": f"all:{query}",
        "start": start,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    
    response = requests.get(ARXIV_API, params=params, timeout=30)
    response.raise_for_status()
    
    # Parse Atom XML
    root = ET.fromstring(response.content)
    
    # Atom namespace
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    
    entries = []
    for entry in root.findall("atom:entry", ns):
        # Skip the query info entry (has no id with arxiv.org)
        id_elem = entry.find("atom:id", ns)
        if id_elem is None or "arxiv.org" not in id_elem.text:
            continue
        
        title = entry.find("atom:title", ns)
        summary = entry.find("atom:summary", ns)
        published = entry.find("atom:published", ns)
        
        # Authors
        authors = []
        for author in entry.findall("atom:author", ns):
            name = author.find("atom:name", ns)
            if name is not None:
                authors.append(name.text)
        
        # Categories
        categories = []
        for cat in entry.findall("atom:category", ns):
            term = cat.get("term")
            if term:
                categories.append(term)
        
        entries.append({
            "id": id_elem.text.strip() if id_elem is not None else "",
            "title": title.text.strip().replace("\n", " ") if title is not None else "",
            "abstract": summary.text.strip().replace("\n", " ") if summary is not None else "",
            "authors": authors,
            "published": published.text.strip() if published is not None else "",
            "categories": categories,
            "url": id_elem.text.strip().replace("abs", "abs") if id_elem is not None else "",
            "query_source": query,
        })
    
    return entries


def download_all():
    """Download abstracts across all queries."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_papers = []
    seen_ids = set()
    
    for query in QUERIES:
        print(f"\n[Query] '{query}'")
        query_papers = []
        
        for start in range(0, MAX_PER_QUERY, BATCH_SIZE):
            batch_size = min(BATCH_SIZE, MAX_PER_QUERY - start)
            print(f"  Fetching start={start}, max={batch_size}...", end=" ")
            
            try:
                papers = fetch_arxiv(query, start, batch_size)
                print(f"got {len(papers)} papers")
                
                for paper in papers:
                    if paper["id"] not in seen_ids:
                        seen_ids.add(paper["id"])
                        query_papers.append(paper)
                
                time.sleep(SLEEP_SECONDS)
                
            except Exception as e:
                print(f"ERROR: {e}")
                time.sleep(SLEEP_SECONDS * 2)
                continue
        
        print(f"  [Query total] {len(query_papers)} unique papers")
        all_papers.extend(query_papers)
    
    print(f"\n{'='*50}")
    print(f"Total unique papers: {len(all_papers)}")
    
    # Save full corpus
    corpus_path = OUTPUT_DIR / "arxiv_corpus.json"
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(all_papers, f, indent=2, ensure_ascii=False)
    print(f"Saved to: {corpus_path}")
    
    # Save sample (50 papers)
    sample = all_papers[:50]
    sample_path = Path("data/sample") / "arxiv_sample.json"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    print(f"Sample (50 papers) saved to: {sample_path}")
    
    # Save metadata summary
    categories = {}
    for p in all_papers:
        for cat in p.get("categories", [])[:1]:  # Primary category
            categories[cat] = categories.get(cat, 0) + 1
    
    top_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10]
    
    summary = {
        "total_papers": len(all_papers),
        "queries_used": QUERIES,
        "top_categories": top_cats,
        "date_downloaded": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "export.arxiv.org/api/query",
    }
    
    summary_path = OUTPUT_DIR / "corpus_metadata.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Metadata saved to: {summary_path}")


if __name__ == "__main__":
    download_all()
