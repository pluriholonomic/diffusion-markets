#!/usr/bin/env python3
"""
Fast Polymarket Goldsky Data Processing

Optimized for 150M+ rows using vectorized operations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

def main():
    print("="*70)
    print("FAST POLYMARKET GOLDSKY PROCESSING")
    print(f"Started: {datetime.now()}")
    print("="*70)
    
    # Load markets
    print("\n1. Loading markets...")
    markets = pd.read_csv('data/polymarket_goldsky/markets.csv')
    print(f"   Markets: {len(markets):,}")
    
    # Create token -> market_id mapping (vectorized)
    print("\n2. Creating token mappings...")
    token1_map = markets.set_index('token1')['id'].astype(str).to_dict()
    token2_map = markets.set_index('token2')['id'].astype(str).to_dict()
    
    # Combine
    token_to_market = {str(k): str(v) for k, v in token1_map.items()}
    token_to_market.update({str(k): str(v) for k, v in token2_map.items()})
    print(f"   Token mappings: {len(token_to_market):,}")
    
    # Process trades (already processed by poly_data)
    print("\n3. Loading processed trades...")
    trades_path = Path('data/polymarket_goldsky/processed/trades.csv')
    
    # The trades.csv already has market_id, price, timestamp
    # Just need to aggregate
    
    chunk_size = 10_000_000
    all_aggs = []
    
    for i, chunk in enumerate(pd.read_csv(trades_path, chunksize=chunk_size)):
        print(f"   Processing chunk {i+1}...")
        
        # Aggregate by market_id
        agg = chunk.groupby('market_id').agg({
            'price': ['first', 'last', 'mean', 'count'],
            'usd_amount': 'sum',
            'timestamp': ['min', 'max']
        })
        agg.columns = ['first_price', 'last_price', 'avg_price', 'n_trades', 
                      'total_volume', 'first_time', 'last_time']
        agg = agg.reset_index()
        all_aggs.append(agg)
        
        print(f"      Chunk markets: {len(agg):,}")
    
    print("\n4. Combining aggregations...")
    combined = pd.concat(all_aggs, ignore_index=True)
    
    # Re-aggregate across chunks
    final_agg = combined.groupby('market_id').agg({
        'first_price': 'first',  # From first chunk
        'last_price': 'last',    # From last chunk
        'avg_price': 'mean',
        'n_trades': 'sum',
        'total_volume': 'sum',
        'first_time': 'min',
        'last_time': 'max'
    }).reset_index()
    
    print(f"   Total markets with trades: {len(final_agg):,}")
    
    # The market_id in trades is a float like 240380.0
    # Need to see if this maps to markets somehow
    print("\n5. Checking market_id format...")
    print(f"   Trade market_ids: {final_agg['market_id'].head().tolist()}")
    print(f"   Markets ids: {markets['id'].head().tolist()}")
    
    # Convert both to string without .0
    final_agg['market_id'] = final_agg['market_id'].apply(lambda x: str(int(float(x))) if pd.notna(x) else None)
    markets['id_str'] = markets['id'].astype(str)
    
    # Check overlap
    trade_ids = set(final_agg['market_id'].dropna())
    market_ids = set(markets['id_str'])
    overlap = trade_ids & market_ids
    print(f"   Overlap: {len(overlap):,}")
    
    if len(overlap) == 0:
        print("\n   WARNING: No direct ID overlap. Checking if trade market_ids are valid...")
        print(f"   Sample trade IDs: {list(trade_ids)[:5]}")
        print(f"   Sample market IDs: {list(market_ids)[:5]}")
        
        # The trade market_ids might be line numbers or internal IDs
        # Let's just use the trade data as-is for backtesting
        print("\n   Using trade data directly (market_id is internal)...")
    
    # Save aggregated prices
    output_dir = Path('data/polymarket_goldsky_processed')
    output_dir.mkdir(exist_ok=True)
    
    final_agg.to_parquet(output_dir / 'trade_prices.parquet')
    print(f"\n   Saved: {output_dir / 'trade_prices.parquet'}")
    
    # For markets, merge what we can
    print("\n6. Creating backtest dataset...")
    
    # If IDs overlap, merge
    if len(overlap) > 0:
        merged = markets.merge(
            final_agg,
            left_on='id_str',
            right_on='market_id',
            how='left'
        )
    else:
        # Just use markets as-is, trades separately
        merged = markets.copy()
        merged['first_price'] = np.nan
    
    merged.to_parquet(output_dir / 'markets_with_prices.parquet')
    print(f"   Saved: {output_dir / 'markets_with_prices.parquet'}")
    
    # Summary
    with_prices = merged[merged['first_price'].notna()] if 'first_price' in merged.columns else pd.DataFrame()
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"""
Markets: {len(markets):,}
Markets with prices: {len(with_prices):,}
Trade price records: {len(final_agg):,}

Files:
  data/polymarket_goldsky_processed/trade_prices.parquet
  data/polymarket_goldsky_processed/markets_with_prices.parquet
  
Finished: {datetime.now()}
""")

if __name__ == "__main__":
    main()
