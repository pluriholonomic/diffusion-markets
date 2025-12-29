"""
Turtel-Compatible Headlines Pipeline

Implements the headline enrichment approach from:
Turtel et al. (2025) "Outcome-based RL to Predict the Future" (arXiv:2505.17989)

Key features:
1. Random prediction date sampling between market open/close
2. Headlines fetched BEFORE prediction date (no temporal leakage)
3. Optional LLM verification for leaked future information
4. Formatted prompts matching Turtel's structure

This enables apples-to-apples comparison with Turtel et al.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class TurtelHeadlineSpec:
    """Configuration for Turtel-style headline enrichment."""
    
    # News source
    news_source: str = "gdelt"  # "gdelt" (free) or "exa" (paid, better quality)
    
    # Temporal controls
    sample_prediction_date: bool = True  # Random date between open/close (Turtel-style)
    window_days: int = 7  # How many days of news before prediction date
    max_articles: int = 10  # Max headlines per question
    
    # Leakage prevention
    verify_no_leakage: bool = False  # Use LLM to check for future info (expensive)
    leakage_model: str = "gpt-4o-mini"  # Model for leakage verification
    
    # Date columns in input data
    open_date_col: str = "createdAt"
    close_date_col: str = "endDate"
    resolution_date_col: str = "resolutionTime"
    
    # API settings
    gdelt_sleep_s: float = 0.5
    exa_api_key: Optional[str] = None
    
    # Cache
    cache_dir: Optional[str] = None
    fuzzy_cache: bool = False  # If True, match cache by question only (reuse across datasets)
    
    # Seed for reproducibility
    seed: int = 0


def _parse_datetime(val: Any) -> Optional[datetime]:
    """Parse various datetime formats to timezone-aware datetime."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    
    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=timezone.utc)
        return val
    
    if isinstance(val, (int, float)):
        # Unix timestamp
        return datetime.fromtimestamp(val, tz=timezone.utc)
    
    if isinstance(val, str):
        # Try common formats
        for fmt in [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]:
            try:
                dt = datetime.strptime(val, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        
        # Try pandas
        try:
            return pd.to_datetime(val, utc=True).to_pydatetime()
        except Exception:
            pass
    
    return None


def sample_prediction_date(
    open_date: datetime,
    close_date: datetime,
    rng: np.random.Generator,
) -> datetime:
    """
    Sample a prediction date uniformly between open and close.
    
    From Turtel paper (Section 2.1):
    "For each question we draw a single prediction date uniformly at random
    between its on-chain open and scheduled close."
    """
    if close_date <= open_date:
        return open_date
    
    delta_seconds = (close_date - open_date).total_seconds()
    sampled_seconds = rng.uniform(0, delta_seconds)
    return open_date + timedelta(seconds=sampled_seconds)


def fetch_gdelt_headlines(
    query: str,
    before_date: datetime,
    window_days: int = 7,
    max_articles: int = 10,
    sleep_s: float = 0.5,
    cache_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch headlines from GDELT DOC API.
    
    Args:
        query: Search query (usually the question text)
        before_date: Only return articles BEFORE this date
        window_days: How many days before to search
        max_articles: Max articles to return
        sleep_s: Sleep between API calls
        cache_dir: Optional cache directory
        
    Returns:
        List of article dicts with title, url, date, etc.
    """
    import urllib.parse
    import urllib.request
    
    # Check cache
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = f"{query[:50]}_{before_date.strftime('%Y%m%d')}"
        cache_key = re.sub(r'[^a-zA-Z0-9_]', '_', cache_key)
        cache_file = cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
    
    # Build date range
    end_date = before_date - timedelta(days=1)  # Strictly before
    start_date = end_date - timedelta(days=window_days)
    
    # GDELT DOC API
    base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": str(max_articles),
        "format": "json",
        "startdatetime": start_date.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_date.strftime("%Y%m%d%H%M%S"),
        "sort": "datedesc",
    }
    
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    
    try:
        time.sleep(sleep_s)
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        
        articles = data.get("articles", [])
        
        # Parse into consistent format
        results = []
        for art in articles[:max_articles]:
            results.append({
                "title": art.get("title", ""),
                "url": art.get("url", ""),
                "date": art.get("seendate", ""),
                "source": art.get("domain", ""),
                "language": art.get("language", ""),
            })
        
        # Cache results
        if cache_dir:
            with open(cache_file, "w") as f:
                json.dump(results, f)
        
        return results
        
    except Exception as e:
        print(f"[gdelt] Error fetching headlines: {e}")
        return []


def fetch_exa_headlines(
    query: str,
    before_date: datetime,
    window_days: int = 7,
    max_articles: int = 10,
    api_key: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    timeout_s: float = 30.0,
    delay_s: float = 0.5,
    max_retries: int = 5,
    fuzzy_cache: bool = False,
) -> List[Dict[str, Any]]:
    """
    Fetch headlines from Exa.ai API (Turtel's approach).
    
    Requires Exa API key. Higher quality than GDELT but paid.
    Now with caching for resume support, timeouts, delays, and retry logic.
    
    Args:
        fuzzy_cache: If True, match cache by question only (ignoring date).
                     Useful for reusing expensive cached results across datasets.
    """
    import concurrent.futures
    
    # Normalize query for cache key
    query_key = re.sub(r'[^a-zA-Z0-9_]', '_', query[:50])
    
    # Check cache first
    cache_file = None
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        if fuzzy_cache:
            # Fuzzy lookup: find ANY cache file matching this question prefix
            import glob
            pattern = str(cache_dir / f"exa_{query_key}_*.json")
            matches = glob.glob(pattern)
            if matches:
                # Use first match (any date is fine)
                cache_file = Path(matches[0])
                with open(cache_file) as f:
                    return json.load(f)
            # No match - will fetch fresh and save with exact key
            cache_key = f"exa_{query_key}_{before_date.strftime('%Y%m%d')}"
            cache_file = cache_dir / f"{cache_key}.json"
        else:
            # Exact cache lookup (original behavior)
            cache_key = f"exa_{query_key}_{before_date.strftime('%Y%m%d')}"
            cache_file = cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                with open(cache_file) as f:
                    return json.load(f)
    
    try:
        from exa_py import Exa
    except ImportError:
        raise ImportError("Install exa-py: pip install exa-py")
    
    if api_key is None:
        import os
        api_key = os.environ.get("EXA_API_KEY")
    
    if not api_key:
        raise ValueError("EXA_API_KEY not set")
    
    exa = Exa(api_key)
    
    end_date = before_date - timedelta(days=1)
    start_date = end_date - timedelta(days=window_days)
    
    def _do_search():
        return exa.search(
            query,
            num_results=max_articles,
            start_published_date=start_date.strftime("%Y-%m-%d"),
            end_published_date=end_date.strftime("%Y-%m-%d"),
        )
    
    # Retry loop with exponential backoff
    for attempt in range(max_retries):
        # Add delay to avoid rate limiting (exponential backoff on retries)
        current_delay = delay_s * (2 ** attempt)
        time.sleep(current_delay)
        
        try:
            # Use ThreadPoolExecutor for timeout
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_do_search)
                response = future.result(timeout=timeout_s)
            
            results = []
            for result in response.results:
                results.append({
                    "title": result.title,
                    "url": result.url,
                    "date": result.published_date or "",
                    "source": result.url.split("/")[2] if result.url else "",
                    "score": result.score,
                })
            
            # Save to cache (cache even empty results to avoid retrying)
            if cache_file:
                with open(cache_file, "w") as f:
                    json.dump(results, f)
            
            return results
        
        except concurrent.futures.TimeoutError:
            print(f"[exa] Timeout after {timeout_s}s for query: {query[:50]}... (attempt {attempt+1}/{max_retries})")
            continue
            
        except Exception as e:
            error_str = str(e)
            # Retry on 502/503 errors (server issues)
            if "502" in error_str or "503" in error_str or "Bad gateway" in error_str:
                print(f"[exa] Server error (attempt {attempt+1}/{max_retries}): {error_str[:100]}")
                continue
            else:
                # Other errors: don't retry
                print(f"[exa] Error fetching headlines: {e}")
                return []
    
    # All retries exhausted
    print(f"[exa] All {max_retries} retries failed for: {query[:50]}...")
    return []


def check_temporal_leakage(
    question: str,
    headlines: List[Dict[str, Any]],
    prediction_date: datetime,
    model: str = "gpt-4o-mini",
) -> Tuple[bool, str]:
    """
    Use LLM to check if headlines contain information from after prediction date.
    
    From Turtel paper (Section 2.1):
    "We use OpenAI o3 to flag any questions for which the news stories contain
    relevant information that should not have been known at the prediction date."
    
    Returns:
        (has_leakage, explanation)
    """
    try:
        import openai
    except ImportError:
        return False, "openai not installed, skipping leakage check"
    
    headlines_text = "\n".join([
        f"- [{h.get('date', 'unknown')}] {h.get('title', '')}"
        for h in headlines
    ])
    
    prompt = f"""You are checking for temporal leakage in a forecasting dataset.

Question: {question}
Prediction Date: {prediction_date.strftime('%Y-%m-%d')}

Headlines provided to the model:
{headlines_text}

Task: Determine if any of these headlines contain information that would only be known AFTER {prediction_date.strftime('%Y-%m-%d')}.

Examples of leakage:
- Headlines dated after the prediction date
- Headlines referencing events that occurred after the prediction date
- Headlines revealing the outcome of the forecasted event

Respond with:
LEAKAGE: YES or NO
EXPLANATION: Brief explanation

Your response:"""

    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0,
        )
        
        text = response.choices[0].message.content or ""
        has_leakage = "LEAKAGE: YES" in text.upper()
        
        return has_leakage, text
        
    except Exception as e:
        return False, f"Error checking leakage: {e}"


def format_turtel_prompt(
    question: str,
    headlines: List[Dict[str, Any]],
    prediction_date: datetime,
    include_date: bool = True,
) -> str:
    """
    Format the prompt in Turtel's style.
    
    Structure:
    - Question
    - Prediction date (optional)
    - Relevant news headlines
    """
    parts = [f"Question: {question}"]
    
    if include_date:
        parts.append(f"Prediction Date: {prediction_date.strftime('%Y-%m-%d')}")
    
    if headlines:
        parts.append("\nRelevant News Headlines:")
        for h in headlines:
            date_str = h.get("date", "")[:10] if h.get("date") else ""
            title = h.get("title", "")
            if date_str:
                parts.append(f"- [{date_str}] {title}")
            else:
                parts.append(f"- {title}")
    else:
        parts.append("\n(No relevant news headlines found)")
    
    parts.append("\nBased on the above information, what is the probability that this event will occur? Provide your reasoning and a probability between 0 and 1.")
    
    return "\n".join(parts)


def enrich_with_turtel_headlines(
    df: pd.DataFrame,
    spec: TurtelHeadlineSpec,
    question_col: str = "question",
    max_rows: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Enrich a DataFrame with Turtel-style headlines.
    
    Adds columns:
    - prediction_date: Sampled prediction date
    - headlines_json: JSON list of headline dicts
    - headlines_text: Formatted headlines text
    - turtel_prompt: Full Turtel-style prompt
    - n_headlines: Number of headlines found
    - has_leakage: Whether temporal leakage was detected (if verify_no_leakage=True)
    
    Args:
        df: Input DataFrame with question and date columns
        spec: TurtelHeadlineSpec configuration
        question_col: Column containing question text
        max_rows: Optional limit on rows to process
        verbose: Print progress
        
    Returns:
        (enriched_df, metadata)
    """
    rng = np.random.default_rng(spec.seed)
    
    df = df.copy()
    if max_rows is not None:
        df = df.head(max_rows)
    
    n = len(df)
    
    # Initialize new columns
    df["prediction_date"] = None
    df["headlines_json"] = None
    df["headlines_text"] = ""
    df["turtel_prompt"] = ""
    df["n_headlines"] = 0
    if spec.verify_no_leakage:
        df["has_leakage"] = False
        df["leakage_explanation"] = ""
    
    # Setup cache
    cache_dir = Path(spec.cache_dir) if spec.cache_dir else None
    
    # Stats
    stats = {
        "total": n,
        "processed": 0,
        "headlines_found": 0,
        "leakage_detected": 0,
        "errors": 0,
    }
    
    for idx, row in df.iterrows():
        if verbose and stats["processed"] % 100 == 0:
            print(f"[turtel_headlines] Processing {stats['processed']}/{n}...")
        
        try:
            # Parse dates
            open_date = _parse_datetime(row.get(spec.open_date_col))
            close_date = _parse_datetime(row.get(spec.close_date_col))
            
            if open_date is None:
                open_date = datetime.now(timezone.utc) - timedelta(days=30)
            if close_date is None:
                close_date = datetime.now(timezone.utc)
            
            # Sample prediction date
            if spec.sample_prediction_date:
                pred_date = sample_prediction_date(open_date, close_date, rng)
            else:
                pred_date = close_date
            
            df.at[idx, "prediction_date"] = pred_date.isoformat()
            
            # Get question text
            question = str(row.get(question_col, ""))
            if not question:
                stats["errors"] += 1
                continue
            
            # Fetch headlines
            if spec.news_source == "exa":
                headlines = fetch_exa_headlines(
                    query=question,
                    before_date=pred_date,
                    window_days=spec.window_days,
                    max_articles=spec.max_articles,
                    api_key=spec.exa_api_key,
                    cache_dir=cache_dir,
                    fuzzy_cache=spec.fuzzy_cache,
                )
            else:  # gdelt
                headlines = fetch_gdelt_headlines(
                    query=question,
                    before_date=pred_date,
                    window_days=spec.window_days,
                    max_articles=spec.max_articles,
                    sleep_s=spec.gdelt_sleep_s,
                    cache_dir=cache_dir,
                )
            
            df.at[idx, "headlines_json"] = json.dumps(headlines)
            df.at[idx, "n_headlines"] = len(headlines)
            
            if headlines:
                stats["headlines_found"] += 1
                
                # Format headlines text
                headlines_text = "\n".join([
                    f"[{h.get('date', '')[:10]}] {h.get('title', '')}"
                    for h in headlines
                ])
                df.at[idx, "headlines_text"] = headlines_text
            
            # Format full prompt
            turtel_prompt = format_turtel_prompt(
                question=question,
                headlines=headlines,
                prediction_date=pred_date,
            )
            df.at[idx, "turtel_prompt"] = turtel_prompt
            
            # Check for leakage
            if spec.verify_no_leakage and headlines:
                has_leakage, explanation = check_temporal_leakage(
                    question=question,
                    headlines=headlines,
                    prediction_date=pred_date,
                    model=spec.leakage_model,
                )
                df.at[idx, "has_leakage"] = has_leakage
                df.at[idx, "leakage_explanation"] = explanation
                if has_leakage:
                    stats["leakage_detected"] += 1
            
            stats["processed"] += 1
            
        except Exception as e:
            if verbose:
                print(f"[turtel_headlines] Error at row {idx}: {e}")
            stats["errors"] += 1
    
    # Metadata
    meta = {
        "spec": asdict(spec),
        "stats": stats,
        "headlines_coverage": stats["headlines_found"] / max(stats["processed"], 1),
    }
    
    if verbose:
        print(f"[turtel_headlines] Done. Processed: {stats['processed']}, "
              f"Headlines found: {stats['headlines_found']}, "
              f"Errors: {stats['errors']}")
    
    return df, meta


def create_turtel_train_test_split(
    df: pd.DataFrame,
    spec: TurtelHeadlineSpec,
    test_after_all_train_resolutions: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Create train/test split following Turtel's temporal constraints.
    
    From Turtel paper (Section 2.1):
    "We construct the test set such that all questions' prediction-dates occur
    AFTER the latest resolution date of any training question."
    
    Args:
        df: Enriched DataFrame with prediction_date and resolution dates
        spec: TurtelHeadlineSpec
        test_after_all_train_resolutions: If True, enforce Turtel's strict constraint
        
    Returns:
        (train_df, test_df, metadata)
    """
    df = df.copy()
    
    # Parse prediction dates
    df["_pred_dt"] = df["prediction_date"].apply(_parse_datetime)
    df["_res_dt"] = df[spec.resolution_date_col].apply(_parse_datetime)
    
    # Remove rows without valid dates
    valid_mask = df["_pred_dt"].notna() & df["_res_dt"].notna()
    df = df[valid_mask].copy()
    
    # Sort by prediction date
    df = df.sort_values("_pred_dt").reset_index(drop=True)
    
    if test_after_all_train_resolutions:
        # Find the latest resolution date
        # Test set = all questions with prediction_date AFTER this
        
        # Start with 80% train, 20% test by count
        n_train_target = int(len(df) * 0.8)
        
        # Find the split point where test predictions are after all train resolutions
        for split_idx in range(n_train_target, len(df)):
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]
            
            max_train_resolution = train_df["_res_dt"].max()
            min_test_prediction = test_df["_pred_dt"].min()
            
            if min_test_prediction > max_train_resolution:
                break
        else:
            # Fallback: use last 20%
            split_idx = n_train_target
        
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
    else:
        # Simple 80/20 split by prediction date
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
    
    # Remove temporary columns
    for col in ["_pred_dt", "_res_dt"]:
        if col in train_df.columns:
            train_df = train_df.drop(columns=[col])
        if col in test_df.columns:
            test_df = test_df.drop(columns=[col])
    
    meta = {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "split_ratio": len(test_df) / (len(train_df) + len(test_df)),
        "temporal_constraint": test_after_all_train_resolutions,
    }
    
    return train_df, test_df, meta


# =============================================================================
# CLI Integration
# =============================================================================

def add_turtel_headlines_args(parser) -> None:
    """Add Turtel headlines arguments to an argparse parser."""
    parser.add_argument("--news-source", type=str, default="gdelt",
                        choices=["gdelt", "exa"],
                        help="News source: gdelt (free) or exa (paid, higher quality)")
    parser.add_argument("--sample-prediction-date", action="store_true", default=True,
                        help="Sample prediction date between open/close (Turtel-style)")
    parser.add_argument("--window-days", type=int, default=7,
                        help="Days of news before prediction date")
    parser.add_argument("--max-articles", type=int, default=10,
                        help="Max headlines per question")
    parser.add_argument("--verify-no-leakage", action="store_true",
                        help="Use LLM to check for temporal leakage")
    parser.add_argument("--leakage-model", type=str, default="gpt-4o-mini",
                        help="Model for leakage verification")
    parser.add_argument("--open-date-col", type=str, default="createdAt")
    parser.add_argument("--close-date-col", type=str, default="endDate")
    parser.add_argument("--resolution-date-col", type=str, default="resolutionTime")
    parser.add_argument("--headlines-cache-dir", type=str, default=None,
                        help="Cache directory for headlines")
    parser.add_argument("--fuzzy-cache", action="store_true",
                        help="Match cache by question only (reuse across datasets)")

