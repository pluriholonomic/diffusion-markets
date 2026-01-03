#!/usr/bin/env python3
"""
Multi-Platform Trading Dashboard

Separate views for:
- Polymarket positions and metrics
- Kalshi positions and metrics
- Combined portfolio view
"""

import os
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from flask import Flask, jsonify, render_template_string, request
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PlatformData:
    """Data for a single platform."""
    name: str
    open_positions: List[Dict] = field(default_factory=list)
    closed_trades: List[Dict] = field(default_factory=list)
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    exposure: float = 0.0
    win_rate: float = 0.0
    strategies: Dict[str, Any] = field(default_factory=dict)
    last_updated: str = ""


@dataclass
class DashboardData:
    """Combined dashboard data."""
    polymarket: PlatformData = field(default_factory=lambda: PlatformData("polymarket"))
    kalshi: PlatformData = field(default_factory=lambda: PlatformData("kalshi"))
    combined: Dict[str, Any] = field(default_factory=dict)
    mode: str = "hybrid"
    timestamp: str = ""


DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Multi-Platform Trading Dashboard</title>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="30">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0a; 
            color: #e0e0e0;
            padding: 20px;
        }
        
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid #333;
        }
        
        .header h1 { color: #fff; font-size: 24px; }
        .mode-badge { 
            background: #2563eb; 
            padding: 5px 15px; 
            border-radius: 20px;
            font-size: 14px;
            text-transform: uppercase;
        }
        
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .tab {
            padding: 12px 24px;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 16px;
        }
        
        .tab:hover { background: #252525; }
        .tab.active { 
            background: #2563eb; 
            border-color: #2563eb;
            color: white;
        }
        
        .tab.polymarket.active { background: #7c3aed; border-color: #7c3aed; }
        .tab.kalshi.active { background: #059669; border-color: #059669; }
        
        .platform-content { display: none; }
        .platform-content.active { display: block; }
        
        .grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 15px;
            margin-bottom: 25px;
        }
        
        .card {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 10px;
            padding: 20px;
        }
        
        .card-label { color: #888; font-size: 12px; text-transform: uppercase; margin-bottom: 8px; }
        .card-value { font-size: 28px; font-weight: 600; }
        .card-value.positive { color: #22c55e; }
        .card-value.negative { color: #ef4444; }
        .card-value.neutral { color: #3b82f6; }
        
        .section { margin-bottom: 30px; }
        .section-title { 
            font-size: 18px; 
            font-weight: 600; 
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #333;
        }
        
        table { width: 100%; border-collapse: collapse; }
        th, td { 
            padding: 12px 10px; 
            text-align: left; 
            border-bottom: 1px solid #222;
            font-size: 13px;
        }
        th { color: #888; text-transform: uppercase; font-size: 11px; }
        tr:hover { background: #151515; }
        
        .badge {
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 500;
            text-transform: uppercase;
        }
        .badge.yes { background: #22c55e20; color: #22c55e; }
        .badge.no { background: #ef444420; color: #ef4444; }
        .badge.polymarket { background: #7c3aed20; color: #7c3aed; }
        .badge.kalshi { background: #05966920; color: #059669; }
        
        .combined-summary {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .platform-summary {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 10px;
            padding: 20px;
        }
        
        .platform-summary.polymarket { border-left: 4px solid #7c3aed; }
        .platform-summary.kalshi { border-left: 4px solid #059669; }
        
        .platform-summary h3 { margin-bottom: 15px; }
        
        .summary-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #252525;
        }
        
        .summary-row:last-child { border: none; }
        
        .auto-refresh { color: #666; font-size: 12px; }
        
        .empty-state {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .live-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #22c55e;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1><span class="live-indicator"></span>Multi-Platform Trading Dashboard</h1>
        <div>
            <span class="mode-badge">{{ mode }}</span>
            <span class="auto-refresh">Auto-refresh: 30s | {{ timestamp }}</span>
        </div>
    </div>
    
    <div class="tabs">
        <div class="tab active" onclick="showTab('combined')">📊 Combined</div>
        <div class="tab polymarket" onclick="showTab('polymarket')">🟣 Polymarket</div>
        <div class="tab kalshi" onclick="showTab('kalshi')">🟢 Kalshi</div>
    </div>
    
    <!-- Combined View -->
    <div id="combined" class="platform-content active">
        <div class="grid">
            <div class="card">
                <div class="card-label">Total Positions</div>
                <div class="card-value neutral">{{ combined.total_positions }}</div>
            </div>
            <div class="card">
                <div class="card-label">Total Exposure</div>
                <div class="card-value neutral">${{ '{:,.0f}'.format(combined.total_exposure) }}</div>
            </div>
            <div class="card">
                <div class="card-label">Total Unrealized PnL</div>
                <div class="card-value {{ 'positive' if combined.total_upnl >= 0 else 'negative' }}">${{ '{:,.2f}'.format(combined.total_upnl) }}</div>
            </div>
            <div class="card">
                <div class="card-label">Total Realized PnL</div>
                <div class="card-value {{ 'positive' if combined.total_rpnl >= 0 else 'negative' }}">${{ '{:,.2f}'.format(combined.total_rpnl) }}</div>
            </div>
            <div class="card">
                <div class="card-label">Combined Win Rate</div>
                <div class="card-value neutral">{{ '{:.1f}%'.format(combined.win_rate * 100) }}</div>
            </div>
            <div class="card">
                <div class="card-label">Total Strategies</div>
                <div class="card-value neutral">{{ combined.total_strategies }}</div>
            </div>
        </div>
        
        <div class="combined-summary">
            <div class="platform-summary polymarket">
                <h3>🟣 Polymarket</h3>
                <div class="summary-row">
                    <span>Open Positions</span>
                    <strong>{{ polymarket.open_positions|length }}</strong>
                </div>
                <div class="summary-row">
                    <span>Exposure</span>
                    <strong>${{ '{:,.0f}'.format(polymarket.exposure) }}</strong>
                </div>
                <div class="summary-row">
                    <span>Unrealized PnL</span>
                    <strong class="{{ 'positive' if polymarket.unrealized_pnl >= 0 else 'negative' }}">${{ '{:,.2f}'.format(polymarket.unrealized_pnl) }}</strong>
                </div>
                <div class="summary-row">
                    <span>Win Rate</span>
                    <strong>{{ '{:.1f}%'.format(polymarket.win_rate * 100) }}</strong>
                </div>
            </div>
            
            <div class="platform-summary kalshi">
                <h3>🟢 Kalshi</h3>
                <div class="summary-row">
                    <span>Open Positions</span>
                    <strong>{{ kalshi.open_positions|length }}</strong>
                </div>
                <div class="summary-row">
                    <span>Exposure</span>
                    <strong>${{ '{:,.0f}'.format(kalshi.exposure) }}</strong>
                </div>
                <div class="summary-row">
                    <span>Unrealized PnL</span>
                    <strong class="{{ 'positive' if kalshi.unrealized_pnl >= 0 else 'negative' }}">${{ '{:,.2f}'.format(kalshi.unrealized_pnl) }}</strong>
                </div>
                <div class="summary-row">
                    <span>Win Rate</span>
                    <strong>{{ '{:.1f}%'.format(kalshi.win_rate * 100) }}</strong>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">All Open Positions</h2>
            {% if all_positions %}
            <table>
                <thead>
                    <tr>
                        <th>Platform</th>
                        <th>Opened</th>
                        <th>Strategy</th>
                        <th>Market</th>
                        <th>Side</th>
                        <th>Entry $</th>
                        <th>Current $</th>
                        <th>Size</th>
                        <th>Unrealized PnL</th>
                    </tr>
                </thead>
                <tbody>
                    {% for pos in all_positions[:30] %}
                    <tr>
                        <td><span class="badge {{ pos.platform }}">{{ pos.platform }}</span></td>
                        <td>{{ pos.opened_at[:16] if pos.opened_at else 'N/A' }}</td>
                        <td>{{ pos.strategy }}</td>
                        <td style="max-width: 250px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" 
                            title="{{ pos.market_question or pos.market_id }}">
                            {{ (pos.market_question[:40] + '...') if pos.market_question and pos.market_question|length > 40 else (pos.market_question or pos.market_id[:20] + '...') }}
                        </td>
                        <td><span class="badge {{ pos.side }}">{{ pos.side }}</span></td>
                        <td>{{ '{:.3f}'.format(pos.entry_price) }}</td>
                        <td>{{ '{:.3f}'.format(pos.current_price if pos.side == 'yes' else (1 - pos.current_price)) }}</td>
                        <td>${{ '{:,.0f}'.format(pos.size) }}</td>
                        <td class="{{ 'positive' if pos.unrealized_pnl >= 0 else 'negative' }}">${{ '{:,.2f}'.format(pos.unrealized_pnl) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% else %}
            <div class="empty-state">No open positions</div>
            {% endif %}
        </div>
    </div>
    
    <!-- Polymarket View -->
    <div id="polymarket" class="platform-content">
        <div class="grid">
            <div class="card">
                <div class="card-label">Open Positions</div>
                <div class="card-value neutral">{{ polymarket.open_positions|length }}</div>
            </div>
            <div class="card">
                <div class="card-label">Exposure</div>
                <div class="card-value neutral">${{ '{:,.0f}'.format(polymarket.exposure) }}</div>
            </div>
            <div class="card">
                <div class="card-label">Unrealized PnL</div>
                <div class="card-value {{ 'positive' if polymarket.unrealized_pnl >= 0 else 'negative' }}">${{ '{:,.2f}'.format(polymarket.unrealized_pnl) }}</div>
            </div>
            <div class="card">
                <div class="card-label">Realized PnL</div>
                <div class="card-value {{ 'positive' if polymarket.realized_pnl >= 0 else 'negative' }}">${{ '{:,.2f}'.format(polymarket.realized_pnl) }}</div>
            </div>
            <div class="card">
                <div class="card-label">Win Rate</div>
                <div class="card-value neutral">{{ '{:.1f}%'.format(polymarket.win_rate * 100) }}</div>
            </div>
            <div class="card">
                <div class="card-label">Strategies</div>
                <div class="card-value neutral">{{ polymarket.strategies|length }}</div>
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">🟣 Polymarket Positions</h2>
            {% if polymarket.open_positions %}
            <table>
                <thead>
                    <tr>
                        <th>Opened</th>
                        <th>Strategy</th>
                        <th>Market</th>
                        <th>Side</th>
                        <th>Entry $</th>
                        <th>Current $</th>
                        <th>Size</th>
                        <th>Unrealized PnL</th>
                    </tr>
                </thead>
                <tbody>
                    {% for pos in polymarket.open_positions %}
                    <tr>
                        <td>{{ pos.opened_at[:16] if pos.opened_at else 'N/A' }}</td>
                        <td>{{ pos.strategy }}</td>
                        <td style="max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" 
                            title="{{ pos.market_question or pos.market_id }}">
                            {{ (pos.market_question[:50] + '...') if pos.market_question and pos.market_question|length > 50 else (pos.market_question or pos.market_id[:20] + '...') }}
                        </td>
                        <td><span class="badge {{ pos.side }}">{{ pos.side }}</span></td>
                        <td>{{ '{:.3f}'.format(pos.entry_price) }}</td>
                        <td>{{ '{:.3f}'.format(pos.current_price if pos.side == 'yes' else (1 - pos.current_price)) }}</td>
                        <td>${{ '{:,.0f}'.format(pos.size) }}</td>
                        <td class="{{ 'positive' if pos.unrealized_pnl >= 0 else 'negative' }}">${{ '{:,.2f}'.format(pos.unrealized_pnl) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% else %}
            <div class="empty-state">No Polymarket positions</div>
            {% endif %}
        </div>
    </div>
    
    <!-- Kalshi View -->
    <div id="kalshi" class="platform-content">
        <div class="grid">
            <div class="card">
                <div class="card-label">Open Positions</div>
                <div class="card-value neutral">{{ kalshi.open_positions|length }}</div>
            </div>
            <div class="card">
                <div class="card-label">Exposure</div>
                <div class="card-value neutral">${{ '{:,.0f}'.format(kalshi.exposure) }}</div>
            </div>
            <div class="card">
                <div class="card-label">Unrealized PnL</div>
                <div class="card-value {{ 'positive' if kalshi.unrealized_pnl >= 0 else 'negative' }}">${{ '{:,.2f}'.format(kalshi.unrealized_pnl) }}</div>
            </div>
            <div class="card">
                <div class="card-label">Realized PnL</div>
                <div class="card-value {{ 'positive' if kalshi.realized_pnl >= 0 else 'negative' }}">${{ '{:,.2f}'.format(kalshi.realized_pnl) }}</div>
            </div>
            <div class="card">
                <div class="card-label">Win Rate</div>
                <div class="card-value neutral">{{ '{:.1f}%'.format(kalshi.win_rate * 100) }}</div>
            </div>
            <div class="card">
                <div class="card-label">Strategies</div>
                <div class="card-value neutral">{{ kalshi.strategies|length }}</div>
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">🟢 Kalshi Positions</h2>
            {% if kalshi.open_positions %}
            <table>
                <thead>
                    <tr>
                        <th>Opened</th>
                        <th>Strategy</th>
                        <th>Market</th>
                        <th>Side</th>
                        <th>Entry $</th>
                        <th>Current $</th>
                        <th>Size</th>
                        <th>Unrealized PnL</th>
                    </tr>
                </thead>
                <tbody>
                    {% for pos in kalshi.open_positions %}
                    <tr>
                        <td>{{ pos.opened_at[:16] if pos.opened_at else 'N/A' }}</td>
                        <td>{{ pos.strategy }}</td>
                        <td style="max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" 
                            title="{{ pos.market_question or pos.market_id }}">
                            {{ (pos.market_question[:50] + '...') if pos.market_question and pos.market_question|length > 50 else (pos.market_question or pos.market_id[:20] + '...') }}
                        </td>
                        <td><span class="badge {{ pos.side }}">{{ pos.side }}</span></td>
                        <td>{{ '{:.3f}'.format(pos.entry_price) }}</td>
                        <td>{{ '{:.3f}'.format(pos.current_price if pos.side == 'yes' else (1 - pos.current_price)) }}</td>
                        <td>${{ '{:,.0f}'.format(pos.size) }}</td>
                        <td class="{{ 'positive' if pos.unrealized_pnl >= 0 else 'negative' }}">${{ '{:,.2f}'.format(pos.unrealized_pnl) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% else %}
            <div class="empty-state">No Kalshi positions</div>
            {% endif %}
        </div>
    </div>
    
    <script>
        function showTab(tabName) {
            // Update tab buttons
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            event.target.classList.add('active');
            
            // Show correct content
            document.querySelectorAll('.platform-content').forEach(content => {
                content.classList.remove('active');
            });
            document.getElementById(tabName).classList.add('active');
        }
    </script>
</body>
</html>
'''


class MultiPlatformDashboard:
    """Dashboard server for multi-platform trading."""
    
    def __init__(self, log_dir: str = "logs/paper_trading", port: int = 5002):
        self.log_dir = Path(log_dir)
        self.port = port
        self.app = Flask(__name__)
        self._data = DashboardData()
        self._setup_routes()
    
    def _setup_routes(self):
        """Set up Flask routes."""
        
        @self.app.route('/')
        def index():
            self._refresh_data()
            
            # Combine positions
            all_positions = []
            for pos in self._data.polymarket.open_positions:
                pos_copy = dict(pos)
                pos_copy['platform'] = 'polymarket'
                all_positions.append(pos_copy)
            for pos in self._data.kalshi.open_positions:
                pos_copy = dict(pos)
                pos_copy['platform'] = 'kalshi'
                all_positions.append(pos_copy)
            
            # Sort by opened time
            all_positions.sort(key=lambda p: p.get('opened_at', ''), reverse=True)
            
            return render_template_string(
                DASHBOARD_TEMPLATE,
                mode=self._data.mode,
                timestamp=self._data.timestamp,
                polymarket=self._data.polymarket,
                kalshi=self._data.kalshi,
                combined=self._data.combined,
                all_positions=all_positions,
            )
        
        @self.app.route('/api/status')
        def api_status():
            self._refresh_data()
            return jsonify({
                'polymarket': asdict(self._data.polymarket),
                'kalshi': asdict(self._data.kalshi),
                'combined': self._data.combined,
                'mode': self._data.mode,
                'timestamp': self._data.timestamp,
            })
        
        @self.app.route('/api/polymarket')
        def api_polymarket():
            self._refresh_data()
            return jsonify(asdict(self._data.polymarket))
        
        @self.app.route('/api/kalshi')
        def api_kalshi():
            self._refresh_data()
            return jsonify(asdict(self._data.kalshi))
    
    def _refresh_data(self):
        """Refresh data from log files."""
        self._data.timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # Load position state
        state_file = self.log_dir / "position_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                
                # Separate by platform
                pm_positions = []
                kalshi_positions = []
                
                for pos in state.get('open_positions', []):
                    platform = pos.get('platform', 'polymarket').lower()
                    if platform == 'kalshi':
                        kalshi_positions.append(pos)
                    else:
                        pm_positions.append(pos)
                
                self._data.polymarket.open_positions = pm_positions
                self._data.kalshi.open_positions = kalshi_positions
                
                # Calculate metrics for each platform
                for platform_data, positions in [
                    (self._data.polymarket, pm_positions),
                    (self._data.kalshi, kalshi_positions)
                ]:
                    platform_data.exposure = sum(p.get('size', 0) for p in positions)
                    platform_data.unrealized_pnl = sum(p.get('unrealized_pnl', 0) for p in positions)
                    
                    # Get strategies
                    strategies = {}
                    for pos in positions:
                        strat = pos.get('strategy', 'unknown')
                        if strat not in strategies:
                            strategies[strat] = {'count': 0, 'exposure': 0, 'pnl': 0}
                        strategies[strat]['count'] += 1
                        strategies[strat]['exposure'] += pos.get('size', 0)
                        strategies[strat]['pnl'] += pos.get('unrealized_pnl', 0)
                    platform_data.strategies = strategies
                
                platform_data.last_updated = self._data.timestamp
                
            except Exception as e:
                logger.warning(f"Failed to load position state: {e}")
        
        # Load closed trades
        closed_file = self.log_dir / "closed_trades.jsonl"
        if closed_file.exists():
            try:
                pm_closed = []
                kalshi_closed = []
                
                with open(closed_file) as f:
                    for line in f:
                        if line.strip():
                            trade = json.loads(line)
                            platform = trade.get('platform', 'polymarket').lower()
                            if platform == 'kalshi':
                                kalshi_closed.append(trade)
                            else:
                                pm_closed.append(trade)
                
                self._data.polymarket.closed_trades = pm_closed[-100:]
                self._data.kalshi.closed_trades = kalshi_closed[-100:]
                
                # Calculate realized PnL
                for platform_data, trades in [
                    (self._data.polymarket, pm_closed),
                    (self._data.kalshi, kalshi_closed)
                ]:
                    platform_data.realized_pnl = sum(t.get('pnl', 0) for t in trades)
                    platform_data.wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
                    platform_data.losses = sum(1 for t in trades if t.get('pnl', 0) <= 0)
                    total = platform_data.wins + platform_data.losses
                    platform_data.win_rate = platform_data.wins / max(1, total)
                    platform_data.total_trades = len(trades)
                
            except Exception as e:
                logger.warning(f"Failed to load closed trades: {e}")
        
        # Calculate combined metrics
        self._data.combined = {
            'total_positions': len(self._data.polymarket.open_positions) + len(self._data.kalshi.open_positions),
            'total_exposure': self._data.polymarket.exposure + self._data.kalshi.exposure,
            'total_upnl': self._data.polymarket.unrealized_pnl + self._data.kalshi.unrealized_pnl,
            'total_rpnl': self._data.polymarket.realized_pnl + self._data.kalshi.realized_pnl,
            'total_strategies': len(self._data.polymarket.strategies) + len(self._data.kalshi.strategies),
            'win_rate': (self._data.polymarket.wins + self._data.kalshi.wins) / max(1, self._data.polymarket.total_trades + self._data.kalshi.total_trades),
        }
        
        # Detect mode
        self._data.mode = "hybrid"
    
    def run(self, debug: bool = False):
        """Start the dashboard server."""
        logger.info(f"Starting Multi-Platform Dashboard on port {self.port}")
        self.app.run(host='0.0.0.0', port=self.port, debug=debug, threaded=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Platform Trading Dashboard")
    parser.add_argument("--log-dir", type=str, default="logs/paper_trading")
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    dashboard = MultiPlatformDashboard(
        log_dir=args.log_dir,
        port=args.port,
    )
    dashboard.run(debug=args.debug)


if __name__ == "__main__":
    main()
