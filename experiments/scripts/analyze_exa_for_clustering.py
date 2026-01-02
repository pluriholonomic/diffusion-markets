#!/usr/bin/env python3
"""
Analyze Exa headline data for semantic clustering / structural grouping.

This script explores whether headline embeddings can provide structural
groupings for stat arb, instead of unreliable price correlations.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

def load_exa_data() -> pd.DataFrame:
    """Load Exa-enriched Polymarket data."""
    for path in [
        Path("/root/diffusion-markets/data/polymarket/turtel_exa_enriched.parquet"),
        Path("/root/polymarket_data/derived/turtel_exa_enriched.parquet"),
        Path("data/polymarket/turtel_exa_enriched.parquet"),
    ]:
        if path.exists():
            return pd.read_parquet(path)
    raise FileNotFoundError("No Exa data found")


def analyze_headline_structure(df: pd.DataFrame) -> None:
    """Analyze the structure of headline data."""
    print("=" * 70)
    print("EXA HEADLINE DATA ANALYSIS")
    print("=" * 70)
    
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Headline coverage
    n_with = (df["n_headlines"] > 0).sum()
    print(f"\nHeadline coverage: {n_with}/{len(df)} ({100*n_with/len(df):.1f}%)")
    print(f"Mean headlines/market: {df['n_headlines'].mean():.1f}")
    
    # Sample markets
    print("\n" + "=" * 70)
    print("SAMPLE MARKETS WITH HEADLINES")
    print("=" * 70)
    
    df_with = df[df["n_headlines"] > 0].head(5)
    for _, row in df_with.iterrows():
        print(f"\n--- {row['slug'][:50]} ---")
        print(f"Question: {row['question'][:80]}...")
        print(f"Category: {row['category']}")
        
        # Try to parse headlines
        try:
            if pd.notna(row.get("headlines_text")):
                text = row["headlines_text"]
                if isinstance(text, str):
                    lines = text.split("\n")[:3]
                    print(f"Headlines ({row['n_headlines']}):")
                    for line in lines:
                        if line.strip():
                            print(f"  - {line[:70]}")
        except Exception as e:
            print(f"  (Error parsing headlines: {e})")


def find_semantic_groups_by_question(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Find semantic groups by question text similarity.
    Uses simple keyword matching as a fast baseline.
    """
    print("\n" + "=" * 70)
    print("SEMANTIC GROUPING BY QUESTION TEXT")
    print("=" * 70)
    
    # Extract key terms from questions
    groups = defaultdict(list)
    
    for idx, row in df.iterrows():
        question = str(row.get("question", "")).lower()
        slug = row.get("slug", str(idx))
        
        # Group by key political figures
        if "trump" in question:
            groups["trump"].append(slug)
        if "biden" in question:
            groups["biden"].append(slug)
        if "harris" in question:
            groups["harris"].append(slug)
        
        # Group by crypto
        if "bitcoin" in question or "btc" in question:
            groups["bitcoin"].append(slug)
        if "ethereum" in question or "eth" in question:
            groups["ethereum"].append(slug)
        
        # Group by events
        if "election" in question:
            groups["election"].append(slug)
        if "popular vote" in question:
            groups["popular_vote"].append(slug)
        if "swing state" in question:
            groups["swing_state"].append(slug)
    
    # Show significant groups
    print(f"\nFound {len(groups)} keyword-based groups")
    for name, members in sorted(groups.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"  {name}: {len(members)} markets")
        for m in members[:3]:
            print(f"    - {m[:50]}")
    
    return dict(groups)


def find_headline_based_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Find groups based on shared headlines.
    Markets that share the same news stories are likely related.
    """
    print("\n" + "=" * 70)
    print("GROUPING BY SHARED HEADLINES")
    print("=" * 70)
    
    # Extract headline fingerprints
    market_headlines = {}
    
    for idx, row in df[df["n_headlines"] > 0].iterrows():
        slug = row.get("slug", str(idx))
        
        try:
            headlines = row.get("headlines_json")
            if isinstance(headlines, str):
                headlines = json.loads(headlines)
            
            if isinstance(headlines, list):
                # Extract unique headline identifiers (URLs or titles)
                fingerprints = set()
                for h in headlines:
                    if isinstance(h, dict):
                        url = h.get("url", "")
                        title = h.get("title", "")
                        fingerprints.add(url or title)
                    elif isinstance(h, str):
                        fingerprints.add(h)
                
                if fingerprints:
                    market_headlines[slug] = fingerprints
        except Exception:
            pass
    
    print(f"Markets with parseable headlines: {len(market_headlines)}")
    
    # Find markets with overlapping headlines
    overlap_pairs = []
    slugs = list(market_headlines.keys())
    
    for i, slug1 in enumerate(slugs[:500]):  # Limit for speed
        for slug2 in slugs[i+1:500]:
            h1 = market_headlines[slug1]
            h2 = market_headlines[slug2]
            overlap = len(h1 & h2)
            if overlap > 0:
                jaccard = overlap / len(h1 | h2)
                if jaccard > 0.2:  # Significant overlap
                    overlap_pairs.append((jaccard, slug1, slug2))
    
    overlap_pairs.sort(reverse=True)
    
    print(f"\nPairs with significant headline overlap (Jaccard > 0.2): {len(overlap_pairs)}")
    for jaccard, s1, s2 in overlap_pairs[:10]:
        print(f"  Jaccard={jaccard:.2f}: {s1[:40]} <-> {s2[:40]}")
    
    return {}


def compute_embedding_similarity(df: pd.DataFrame) -> None:
    """
    Compute semantic similarity using sentence embeddings.
    This is the CORRECT approach for structural grouping.
    """
    print("\n" + "=" * 70)
    print("SEMANTIC EMBEDDING SIMILARITY")
    print("=" * 70)
    
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("sentence-transformers not installed. Install with:")
        print("  pip install sentence-transformers")
        return
    
    # Get questions with good data
    df_good = df[df["question"].notna()].head(200)  # Limit for speed
    questions = df_good["question"].tolist()
    slugs = df_good["slug"].tolist()
    
    print(f"Embedding {len(questions)} questions...")
    
    # Load model and compute embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(questions, show_progress_bar=True)
    
    # Compute similarity matrix
    sim_matrix = cosine_similarity(embeddings)
    
    # Find high-similarity pairs (NOT on diagonal)
    pairs = []
    n = len(questions)
    for i in range(n):
        for j in range(i+1, n):
            if sim_matrix[i, j] > 0.7:  # High similarity threshold
                pairs.append((sim_matrix[i, j], i, j))
    
    pairs.sort(reverse=True)
    
    print(f"\nHigh-similarity pairs (>0.7): {len(pairs)}")
    for sim, i, j in pairs[:15]:
        print(f"\nSim={sim:.3f}:")
        print(f"  1: {questions[i][:70]}")
        print(f"  2: {questions[j][:70]}")
    
    # Cluster
    print("\n" + "=" * 70)
    print("AGGLOMERATIVE CLUSTERING")
    print("=" * 70)
    
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.3,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings)
    
    # Group by cluster
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[label].append((slugs[i], questions[i]))
    
    # Show non-trivial clusters
    non_trivial = {k: v for k, v in clusters.items() if len(v) >= 2}
    print(f"Total clusters: {len(clusters)}")
    print(f"Non-trivial clusters (>=2): {len(non_trivial)}")
    
    for cid, members in sorted(non_trivial.items(), key=lambda x: -len(x[1]))[:5]:
        print(f"\n[Cluster {cid}] ({len(members)} markets):")
        for slug, q in members[:5]:
            print(f"  - {q[:60]}")


def main():
    """Main analysis."""
    df = load_exa_data()
    
    # Basic structure analysis
    analyze_headline_structure(df)
    
    # Keyword-based grouping (fast)
    find_semantic_groups_by_question(df)
    
    # Headline overlap grouping
    find_headline_based_groups(df)
    
    # Embedding-based clustering (if available)
    compute_embedding_similarity(df)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
For stat arb on prediction markets, we should use:

1. QUESTION EMBEDDING SIMILARITY
   - Markets about the same event (e.g., "Trump wins" variants)
   - Trade when their prices diverge from logical consistency

2. SHARED HEADLINE GROUPS  
   - Markets driven by the same news
   - Mean-revert within groups when one market reacts faster

3. CATEGORY + ENTITY MATCHING
   - Same entity (Trump, Bitcoin, etc.) across different questions
   - Exploit delayed information propagation

The key insight: Structural grouping by MEANING, not by price correlation.
Price correlations are near-zero because events are mostly independent.
But markets about the SAME event should have constrained prices.
""")


if __name__ == "__main__":
    main()
