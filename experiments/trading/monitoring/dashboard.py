#!/usr/bin/env python3
"""
Trading Dashboard with Risk Metrics

A web-based dashboard for monitoring paper trading performance.
Includes Sharpe ratio, Expected Shortfall, VaR, and fill rate metrics.
"""

import json
import os
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import threading
import time
import numpy as np

try:
    from flask import Flask, render_template_string, jsonify, request
except ImportError:
    Flask = None
    print("Flask not installed. Run: pip install flask")


def compute_risk_metrics(pnl_series: List[float], initial_bankroll: float = 10000) -> Dict[str, float]:
    """
    Compute risk metrics from PnL series.
    
    Returns:
        - sharpe: Sharpe ratio (annualized, assuming 252 trading days)
        - sortino: Sortino ratio (downside deviation only)
        - var_95: Value at Risk at 95% confidence
        - var_99: Value at Risk at 99% confidence
        - cvar_95: Conditional VaR (Expected Shortfall) at 95%
        - cvar_99: CVaR at 99%
        - max_drawdown: Maximum drawdown percentage
        - win_rate: Percentage of winning trades
    """
    if not pnl_series or len(pnl_series) < 2:
        return {
            'sharpe': 0.0,
            'sortino': 0.0,
            'var_95': 0.0,
            'var_99': 0.0,
            'cvar_95': 0.0,
            'cvar_99': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'total_return': 0.0,
        }
    
    pnl = np.array(pnl_series)
    
    # Returns (as fraction of initial bankroll)
    returns = pnl / initial_bankroll
    
    # Mean and std
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1) if len(returns) > 1 else 0.0001
    
    # Sharpe ratio (annualized, assuming ~252 trading days, ~20 trades/day)
    # Scale factor depends on trade frequency
    trades_per_year = 252 * 20  # Approximate
    sharpe = (mean_return / std_return) * np.sqrt(trades_per_year) if std_return > 0 else 0
    
    # Sortino ratio (downside deviation only)
    negative_returns = returns[returns < 0]
    if len(negative_returns) > 0:
        downside_std = np.std(negative_returns, ddof=1)
        sortino = (mean_return / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0
    else:
        sortino = sharpe * 2  # No downside, very good
    
    # VaR (Value at Risk) - percentile of losses
    var_95 = -np.percentile(pnl, 5)  # 95% confidence
    var_99 = -np.percentile(pnl, 1)  # 99% confidence
    
    # CVaR / Expected Shortfall - average loss beyond VaR
    losses_beyond_95 = pnl[pnl <= -var_95] if var_95 > 0 else pnl[pnl < np.percentile(pnl, 5)]
    losses_beyond_99 = pnl[pnl <= -var_99] if var_99 > 0 else pnl[pnl < np.percentile(pnl, 1)]
    
    cvar_95 = -np.mean(losses_beyond_95) if len(losses_beyond_95) > 0 else var_95
    cvar_99 = -np.mean(losses_beyond_99) if len(losses_beyond_99) > 0 else var_99
    
    # Maximum drawdown
    cumulative = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cumulative + initial_bankroll)
    drawdowns = (running_max - (cumulative + initial_bankroll)) / running_max
    max_drawdown = np.max(drawdowns) * 100 if len(drawdowns) > 0 else 0
    
    # Win rate
    wins = np.sum(pnl > 0)
    total = len(pnl)
    win_rate = (wins / total * 100) if total > 0 else 0
    
    # Total return
    total_return = (np.sum(pnl) / initial_bankroll) * 100
    
    return {
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'var_95': float(var_95),
        'var_99': float(var_99),
        'cvar_95': float(cvar_95),
        'cvar_99': float(cvar_99),
        'max_drawdown': float(max_drawdown),
        'win_rate': float(win_rate),
        'total_return': float(total_return),
    }


# HTML template for the dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Trading Dashboard</title>
    <meta http-equiv="refresh" content="30">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0f;
            color: #e0e0e0;
            padding: 20px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #2a2a3a;
        }
        .header h1 { color: #fff; font-size: 28px; }
        .header-right { display: flex; align-items: center; gap: 15px; }
        .mode-badge {
            padding: 6px 12px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: 600;
            background: #1a3a4a;
            color: #60a5fa;
        }
        .status-badge {
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 14px;
        }
        .status-active { background: #1a4a1a; color: #4ade80; }
        .status-halted { background: #4a1a1a; color: #ef4444; }
        
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 25px; }
        .grid-4 { grid-template-columns: repeat(4, 1fr); }
        
        .card {
            background: #12121a;
            border: 1px solid #2a2a3a;
            border-radius: 12px;
            padding: 18px;
        }
        .card-title { color: #888; font-size: 11px; text-transform: uppercase; margin-bottom: 6px; letter-spacing: 0.5px; }
        .card-value { font-size: 28px; font-weight: 700; }
        .card-value.positive { color: #4ade80; }
        .card-value.negative { color: #ef4444; }
        .card-value.neutral { color: #fbbf24; }
        .card-subtitle { color: #666; font-size: 11px; margin-top: 4px; }
        
        .section-title { font-size: 16px; margin-bottom: 12px; color: #fff; font-weight: 600; }
        
        .risk-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 25px; }
        .risk-card { 
            background: linear-gradient(135deg, #1a1a2a 0%, #12121a 100%);
            border: 1px solid #3a3a5a;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 25px;
            font-size: 13px;
        }
        th, td {
            text-align: left;
            padding: 10px 14px;
            border-bottom: 1px solid #2a2a3a;
        }
        th { color: #888; font-weight: 500; font-size: 11px; text-transform: uppercase; }
        tr:hover { background: #1a1a2a; }
        
        .strategy-status { display: flex; align-items: center; gap: 8px; }
        .status-dot { width: 8px; height: 8px; border-radius: 50%; }
        .status-dot.active { background: #4ade80; }
        .status-dot.inactive { background: #666; }
        
        .alerts {
            background: #1a1a2a;
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 25px;
        }
        .alert { padding: 10px; margin: 5px 0; border-radius: 6px; font-size: 13px; }
        .alert.warning { background: #4a3a1a; color: #fbbf24; }
        .alert.error { background: #4a1a1a; color: #ef4444; }
        .alert.info { background: #1a3a4a; color: #60a5fa; }
        
        .flow-container {
            background: #12121a;
            border: 1px solid #2a2a3a;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 25px;
        }
        .flow-items { display: flex; justify-content: space-around; align-items: center; text-align: center; }
        .flow-arrow { font-size: 20px; color: #4a4a5a; }
        
        .timestamp { color: #666; font-size: 11px; text-align: right; margin-top: 15px; }
        
        .pnl-display { font-size: 14px; margin-top: 5px; }
        .win { color: #4ade80; }
        .loss { color: #ef4444; }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Trading Dashboard</h1>
        <div class="header-right">
            <span class="mode-badge">{{ mode | upper }} MODE</span>
            <span class="status-badge {{ 'status-halted' if status.trading_halted else 'status-active' }}">
                {{ 'HALTED' if status.trading_halted else 'ACTIVE' }}
            </span>
        </div>
    </div>
    
    <!-- Main Metrics -->
    <div class="grid" style="grid-template-columns: repeat(6, 1fr);">
        <div class="card">
            <div class="card-title">Bankroll</div>
            <div class="card-value">${{ '{:,.0f}'.format(status.bankroll) }}</div>
            <div class="card-subtitle">Initial: ${{ '{:,.0f}'.format(status.initial_bankroll) }}</div>
        </div>
        <div class="card">
            <div class="card-title">Realized P&L</div>
            <div class="card-value {{ 'positive' if status.realized_pnl >= 0 else 'negative' }}">
                {{ '+' if status.realized_pnl >= 0 else '' }}${{ '{:,.0f}'.format(status.realized_pnl) }}
            </div>
            <div class="card-subtitle">Closed/Resolved</div>
        </div>
        <div class="card">
            <div class="card-title">Unrealized P&L</div>
            <div class="card-value {{ 'positive' if status.unrealized_pnl >= 0 else 'negative' }}" style="font-size: 24px;">
                {{ '+' if status.unrealized_pnl >= 0 else '' }}${{ '{:,.0f}'.format(status.unrealized_pnl) }}
            </div>
            <div class="card-subtitle">{{ status.open_positions }} open positions</div>
        </div>
        <div class="card">
            <div class="card-title">Total P&L</div>
            <div class="card-value {{ 'positive' if status.total_pnl >= 0 else 'negative' }}">
                {{ '+' if status.total_pnl >= 0 else '' }}${{ '{:,.0f}'.format(status.total_pnl) }}
            </div>
            <div class="card-subtitle">Return: {{ '{:.1f}'.format(risk.total_return) }}%</div>
        </div>
        <div class="card">
            <div class="card-title">Win Rate</div>
            <div class="card-value {{ 'positive' if risk.win_rate >= 55 else 'negative' if risk.win_rate < 45 else 'neutral' }}">
                {{ '{:.1f}'.format(risk.win_rate) }}%
            </div>
            <div class="card-subtitle">{{ status.wins }}W / {{ status.losses }}L</div>
        </div>
        <div class="card">
            <div class="card-title">Trades</div>
            <div class="card-value">{{ status.total_trades }}</div>
            <div class="card-subtitle">Fill rate: {{ '{:.1f}'.format(status.fill_rate) }}%</div>
        </div>
    </div>
    
    <!-- Risk Metrics -->
    <h2 class="section-title">📈 Risk Metrics</h2>
    <div class="risk-grid">
        <div class="card risk-card">
            <div class="card-title">Sharpe Ratio</div>
            <div class="card-value {{ 'positive' if risk.sharpe >= 1 else 'negative' if risk.sharpe < 0 else 'neutral' }}">
                {{ '{:.2f}'.format(risk.sharpe) }}
            </div>
            <div class="card-subtitle">Annualized</div>
        </div>
        <div class="card risk-card">
            <div class="card-title">Sortino Ratio</div>
            <div class="card-value {{ 'positive' if risk.sortino >= 1.5 else 'negative' if risk.sortino < 0 else 'neutral' }}">
                {{ '{:.2f}'.format(risk.sortino) }}
            </div>
            <div class="card-subtitle">Downside only</div>
        </div>
        <div class="card risk-card">
            <div class="card-title">VaR (95%)</div>
            <div class="card-value negative">${{ '{:,.0f}'.format(risk.var_95) }}</div>
            <div class="card-subtitle">Per trade</div>
        </div>
        <div class="card risk-card">
            <div class="card-title">CVaR / ES (95%)</div>
            <div class="card-value negative">${{ '{:,.0f}'.format(risk.cvar_95) }}</div>
            <div class="card-subtitle">Expected Shortfall</div>
        </div>
    </div>
    
    <div class="grid grid-4">
        <div class="card">
            <div class="card-title">Max Drawdown</div>
            <div class="card-value {{ 'negative' if risk.max_drawdown > 10 else 'neutral' }}">
                {{ '{:.1f}'.format(risk.max_drawdown) }}%
            </div>
        </div>
        <div class="card">
            <div class="card-title">VaR (99%)</div>
            <div class="card-value negative">${{ '{:,.0f}'.format(risk.var_99) }}</div>
        </div>
        <div class="card">
            <div class="card-title">CVaR (99%)</div>
            <div class="card-value negative">${{ '{:,.0f}'.format(risk.cvar_99) }}</div>
        </div>
        <div class="card">
            <div class="card-title">Expected Fill Rate</div>
            <div class="card-value {{ 'positive' if status.fill_rate >= 80 else 'neutral' }}">
                {{ '{:.0f}'.format(status.fill_rate) }}%
            </div>
        </div>
    </div>
    
    {% if status.trading_halted %}
    <div class="alerts">
        <div class="alert error">⚠️ Trading halted: {{ status.halt_reason }}</div>
    </div>
    {% endif %}
    
    <!-- Trade Flow -->
    <h2 class="section-title">🔄 Trade Flow</h2>
    <div class="flow-container">
        <div class="flow-items">
            <div>
                <div class="card-title">Markets Scanned</div>
                <div class="card-value">{{ flow.markets_scanned }}</div>
            </div>
            <div class="flow-arrow">→</div>
            <div>
                <div class="card-title">Signals Generated</div>
                <div class="card-value">{{ flow.signals_generated }}</div>
            </div>
            <div class="flow-arrow">→</div>
            <div>
                <div class="card-title">Orders Attempted</div>
                <div class="card-value">{{ flow.trades_attempted }}</div>
            </div>
            <div class="flow-arrow">→</div>
            <div>
                <div class="card-title">Orders Filled</div>
                <div class="card-value positive">{{ flow.trades_filled }}</div>
            </div>
        </div>
    </div>
    
    <!-- Position Management Settings -->
    <h2 class="section-title">⚙️ Position Management Thresholds</h2>
    <div class="grid grid-4">
        <div class="card">
            <div class="card-title">Take Profit</div>
            <div class="card-value positive">+{{ '{:.1f}'.format(status.profit_take_pct) }}%</div>
            <div class="card-subtitle">Exit winners at this gain</div>
        </div>
        <div class="card">
            <div class="card-title">Stop Loss</div>
            <div class="card-value negative">-{{ '{:.1f}'.format(status.stop_loss_pct) }}%</div>
            <div class="card-subtitle">Exit losers at this loss</div>
        </div>
        <div class="card">
            <div class="card-title">Online Learning</div>
            <div class="card-value {{ 'positive' if status.online_learning else 'neutral' }}">
                {{ 'Enabled' if status.online_learning else 'Disabled' }}
            </div>
            <div class="card-subtitle">Adaptive thresholds</div>
        </div>
        <div class="card">
            <div class="card-title">Open Positions</div>
            <div class="card-value neutral">{{ status.open_positions }}</div>
            <div class="card-subtitle">Profit takes: {{ status.profit_takes }} | Stop losses: {{ status.stop_losses }}</div>
        </div>
    </div>
    
    <!-- Strategies Table -->
    <h2 class="section-title">📋 Strategies</h2>
    <table>
        <thead>
            <tr>
                <th>Strategy</th>
                <th>Platform</th>
                <th>Status</th>
                <th>Signals</th>
                <th>Trades</th>
                <th>Win Rate</th>
                <th>PnL</th>
            </tr>
        </thead>
        <tbody>
            {% for name, strat in strategies.items() %}
            <tr>
                <td>
                    <div class="strategy-status">
                        <span class="status-dot {{ 'active' if strat.signals_count > 0 else 'inactive' }}"></span>
                        {{ name }}
                    </div>
                </td>
                <td>{{ strat.platform }}</td>
                <td>{{ strat.calibration_status }}</td>
                <td>{{ strat.signals_count }}</td>
                <td>{{ strat.trades_count }}</td>
                <td>{{ '{:.0f}'.format(strat.win_rate) }}%</td>
                <td class="{{ 'win' if strat.pnl >= 0 else 'loss' }}">
                    {{ '+' if strat.pnl >= 0 else '' }}${{ '{:.0f}'.format(strat.pnl) }}
                </td>
            </tr>
            {% endfor %}
            {% if not strategies %}
            <tr><td colspan="7" style="text-align: center; color: #666;">No strategies loaded</td></tr>
            {% endif %}
        </tbody>
    </table>
    
    <!-- Open Positions (Unrealized PnL) -->
    <h2 class="section-title">📊 Open Positions (Mark-to-Market)</h2>
    <table>
        <thead>
            <tr>
                <th>Opened</th>
                <th>Strategy</th>
                <th>Platform</th>
                <th style="max-width: 300px;">Market Question</th>
                <th>Side</th>
                <th title="Price paid for this side (YES or NO)">Entry $</th>
                <th title="Current price of this side">Current $</th>
                <th>Size</th>
                <th>Unrealized PnL</th>
            </tr>
        </thead>
        <tbody>
            {% for pos in open_positions %}
            <tr>
                <td>{{ pos.entry_time[:16] if pos.entry_time else '' }}</td>
                <td>{{ pos.strategy }}</td>
                <td>{{ pos.platform }}</td>
                <td style="max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" 
                    title="{{ pos.market_question or pos.market_id }}">
                    {{ (pos.market_question[:50] + '...') if pos.market_question and pos.market_question|length > 50 else (pos.market_question or pos.market_id[:20] + '...') }}
                </td>
                <td>{{ pos.side | upper }}</td>
                <td>{{ '{:.3f}'.format(pos.entry_price) }}</td>
                {# Show current price as SIDE price (same convention as entry) #}
                {# For YES: current_price is already YES price #}
                {# For NO: convert YES price to NO price (1 - current_price) #}
                <td>{{ '{:.3f}'.format(pos.current_price if pos.side == 'yes' else (1 - pos.current_price)) }}</td>
                <td>${{ '{:.0f}'.format(pos.size) }}</td>
                <td class="{{ 'win' if pos.unrealized_pnl >= 0 else 'loss' }}">
                    {{ '+' if pos.unrealized_pnl >= 0 else '' }}${{ '{:.0f}'.format(pos.unrealized_pnl) }}
                </td>
            </tr>
            {% endfor %}
            {% if not open_positions %}
            <tr><td colspan="9" style="text-align: center; color: #666;">No open positions</td></tr>
            {% endif %}
        </tbody>
    </table>
    
    <!-- Recent Trades (Realized PnL) -->
    <h2 class="section-title">📝 Closed Trades (Realized PnL)</h2>
    <table>
        <thead>
            <tr>
                <th>Time</th>
                <th>Strategy</th>
                <th>Platform</th>
                <th>Market</th>
                <th>Side</th>
                <th>Size</th>
                <th>Price</th>
                <th>Result</th>
                <th>Realized PnL</th>
            </tr>
        </thead>
        <tbody>
            {% for trade in closed_trades %}
            <tr>
                <td>{{ trade.time }}</td>
                <td>{{ trade.strategy }}</td>
                <td>{{ trade.platform }}</td>
                <td title="{{ trade.market_id }}">{{ trade.market_id[:12] }}...</td>
                <td>{{ trade.side | upper }}</td>
                <td>${{ '{:.0f}'.format(trade.size) }}</td>
                <td>{{ '{:.2f}'.format(trade.entry_price) }} → {{ '{:.2f}'.format(trade.exit_price) }}</td>
                <td class="{{ trade.result }}">{{ trade.result | upper }}</td>
                <td class="{{ 'win' if trade.pnl >= 0 else 'loss' }}">
                    {{ '+' if trade.pnl >= 0 else '' }}${{ '{:.0f}'.format(trade.pnl) }}
                </td>
            </tr>
            {% endfor %}
            {% if not closed_trades %}
            <tr><td colspan="9" style="text-align: center; color: #666;">No closed trades yet</td></tr>
            {% endif %}
        </tbody>
    </table>
    
    <!-- Risk Limits Panel -->
    <details style="margin-top: 30px;">
        <summary style="cursor: pointer; font-size: 18px; font-weight: bold; color: #fff; padding: 15px; background: #1a1a2a; border-radius: 8px; margin-bottom: 15px;">
            ⚠️ Risk Limits & Configuration (click to expand)
        </summary>
        <div style="background: #12121a; border: 1px solid #2a2a3a; border-radius: 12px; padding: 20px; margin-top: 10px;">
            <div class="grid" style="grid-template-columns: repeat(3, 1fr); gap: 20px;">
                <!-- Position Management -->
                <div style="background: #1a1a2a; border-radius: 8px; padding: 15px;">
                    <h3 style="color: #60a5fa; margin-bottom: 15px; font-size: 14px;">📊 Position Management</h3>
                    <table style="width: 100%; font-size: 13px;">
                        <tr><td style="color: #888;">Take Profit</td><td style="color: #4ade80; text-align: right;">+{{ '{:.1f}'.format(limits.profit_take_pct) }}%</td></tr>
                        <tr><td style="color: #888;">Stop Loss</td><td style="color: #ef4444; text-align: right;">-{{ '{:.1f}'.format(limits.stop_loss_pct) }}%</td></tr>
                        <tr><td style="color: #888;">Max Positions</td><td style="text-align: right;">{{ limits.max_positions }}</td></tr>
                        <tr><td style="color: #888;">Max Per Market</td><td style="text-align: right;">{{ limits.max_per_market }}</td></tr>
                        <tr><td style="color: #888;">Max Position Size</td><td style="text-align: right;">${{ '{:,.0f}'.format(limits.max_position_size) }}</td></tr>
                        <tr><td style="color: #888;">Max Hold Time</td><td style="text-align: right;">{{ limits.max_hold_hours }}h</td></tr>
                        <tr><td style="color: #888;">Online Learning</td><td style="text-align: right;">{{ 'Enabled' if limits.online_learning else 'Disabled' }}</td></tr>
                    </table>
                </div>
                
                <!-- Global Risk Limits -->
                <div style="background: #1a1a2a; border-radius: 8px; padding: 15px;">
                    <h3 style="color: #fbbf24; margin-bottom: 15px; font-size: 14px;">🛡️ Global Risk Limits</h3>
                    <table style="width: 100%; font-size: 13px;">
                        <tr><td style="color: #888;">Max Position %</td><td style="text-align: right;">{{ '{:.0f}'.format(limits.max_position_pct * 100) }}%</td></tr>
                        <tr><td style="color: #888;">Daily Loss Stop</td><td style="color: #ef4444; text-align: right;">{{ '{:.0f}'.format(limits.max_daily_loss_pct * 100) }}%</td></tr>
                        <tr><td style="color: #888;">Max Drawdown</td><td style="color: #ef4444; text-align: right;">{{ '{:.0f}'.format(limits.max_drawdown_pct * 100) }}%</td></tr>
                        <tr><td style="color: #888;">Kelly Fraction</td><td style="text-align: right;">{{ limits.kelly_fraction }}</td></tr>
                        <tr><td style="color: #888;">Min Edge</td><td style="text-align: right;">{{ '{:.0f}'.format(limits.min_edge * 100) }}%</td></tr>
                        <tr><td style="color: #888;">Min Liquidity</td><td style="text-align: right;">${{ '{:,.0f}'.format(limits.min_liquidity) }}</td></tr>
                        <tr><td style="color: #888;">Max Concentration</td><td style="text-align: right;">{{ '{:.0f}'.format(limits.max_concentration * 100) }}%</td></tr>
                    </table>
                </div>
                
                <!-- Execution Settings -->
                <div style="background: #1a1a2a; border-radius: 8px; padding: 15px;">
                    <h3 style="color: #a78bfa; margin-bottom: 15px; font-size: 14px;">⚡ Execution Settings</h3>
                    <table style="width: 100%; font-size: 13px;">
                        <tr><td style="color: #888;">Initial Bankroll</td><td style="text-align: right;">${{ '{:,.0f}'.format(limits.initial_bankroll) }}</td></tr>
                        <tr><td style="color: #888;">Max Orders/Run</td><td style="text-align: right;">{{ limits.max_orders_per_run }}</td></tr>
                        <tr><td style="color: #888;">Min Confidence</td><td style="text-align: right;">{{ limits.min_confidence }}</td></tr>
                        <tr><td style="color: #888;">Mode</td><td style="text-align: right;">{{ limits.mode | upper }}</td></tr>
                    </table>
                    <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #2a2a3a;">
                        <h4 style="color: #888; font-size: 12px; margin-bottom: 8px;">Definitions</h4>
                        <p style="font-size: 11px; color: #666; line-height: 1.5;">
                            <b>Fill Rate</b> = positions / order attempts<br>
                            <b>Win Rate</b> = wins / (wins + losses)<br>
                            <b>Total Trades</b> = open + closed positions
                        </p>
                    </div>
                </div>
            </div>
        </div>
    </details>
    
    <div class="timestamp">Last updated: {{ timestamp }}</div>
</body>
</html>
"""


@dataclass
class DashboardData:
    """Container for dashboard data."""
    status: Dict[str, Any]
    risk: Dict[str, float]
    strategies: Dict[str, Dict]
    recent_signals: List[Dict]
    recent_trades: List[Dict]
    open_positions: List[Dict]   # Open positions with unrealized PnL
    closed_trades: List[Dict]    # Closed trades with realized PnL
    flow: Dict[str, int]
    mode: str
    timestamp: str
    pnl_series: List[float] = field(default_factory=list)


class TradingDashboard:
    """Flask-based trading dashboard with risk metrics."""
    
    def __init__(
        self,
        log_dir: str = "logs/paper_trading",
        port: int = 8080,
        mode: str = "hybrid",
        initial_bankroll: float = 10000,
    ):
        self.log_dir = Path(log_dir)
        self.port = port
        self.mode = mode
        self.initial_bankroll = initial_bankroll
        
        if Flask is None:
            raise ImportError("Flask not installed")
        
        self.app = Flask(__name__)
        self._setup_routes()
        
        # Cached data
        self._data = self._create_empty_data()
    
    def _create_empty_data(self) -> DashboardData:
        """Create empty dashboard data."""
        return DashboardData(
            status={
                'trading_halted': False,
                'halt_reason': '',
                'bankroll': self.initial_bankroll,
                'initial_bankroll': self.initial_bankroll,
                'peak_bankroll': self.initial_bankroll,
                'current_drawdown': 0.0,
                'realized_pnl': 0,      # Closed/resolved positions
                'unrealized_pnl': 0,    # Open positions (mark-to-market)
                'total_pnl': 0,         # Realized + Unrealized
                'open_positions': 0,    # Number of open positions
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'fill_rate': 0.0,
            },
            risk={
                'sharpe': 0.0,
                'sortino': 0.0,
                'var_95': 0.0,
                'var_99': 0.0,
                'cvar_95': 0.0,
                'cvar_99': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'total_return': 0.0,
            },
            strategies={},
            recent_signals=[],
            recent_trades=[],
            open_positions=[],
            closed_trades=[],
            flow={'markets_scanned': 0, 'signals_generated': 0, 'trades_attempted': 0, 'trades_filled': 0},
            mode=self.mode,
            timestamp=datetime.utcnow().isoformat(),
            pnl_series=[],
        )
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            self._refresh_data()
            
            # Build limits object for display
            limits_data = {
                # Position Management
                'profit_take_pct': self._data.status.get('profit_take_pct', 20.0),
                'stop_loss_pct': self._data.status.get('stop_loss_pct', 20.0),
                'max_positions': 50,
                'max_per_market': 1,
                'max_position_size': 1000,
                'max_hold_hours': 72,
                'online_learning': self._data.status.get('online_learning', True),
                # Global Risk Limits
                'max_position_pct': 0.10,
                'max_daily_loss_pct': 0.20,
                'max_drawdown_pct': 0.30,
                'kelly_fraction': 0.25,
                'min_edge': 0.05,
                'min_liquidity': 1000,
                'max_concentration': 0.30,
                # Execution Settings
                'initial_bankroll': self._data.status.get('initial_bankroll', 10000),
                'max_orders_per_run': 20,
                'min_confidence': 0.3,
                'mode': self._data.mode,
            }
            
            return render_template_string(
                DASHBOARD_HTML,
                status=type('Status', (), self._data.status)(),
                risk=type('Risk', (), self._data.risk)(),
                strategies=self._data.strategies,
                recent_signals=self._data.recent_signals,
                recent_trades=self._data.recent_trades,
                open_positions=self._data.open_positions,
                closed_trades=self._data.closed_trades,
                flow=type('Flow', (), self._data.flow)(),
                limits=type('Limits', (), limits_data)(),
                mode=self._data.mode,
                timestamp=self._data.timestamp,
            )
        
        @self.app.route('/api/status')
        def api_status():
            self._refresh_data()
            return jsonify({
                'status': self._data.status,
                'risk': self._data.risk,
                'strategies': self._data.strategies,
                'flow': self._data.flow,
                'mode': self._data.mode,
                'timestamp': self._data.timestamp,
                'open_positions': self._data.open_positions,
                'closed_trades': self._data.closed_trades,
            })
        
        @self.app.route('/api/trades')
        def api_trades():
            self._refresh_data()
            return jsonify({
                'trades': self._data.recent_trades,
                'pnl_series': self._data.pnl_series[-100:],  # Last 100
                'risk': self._data.risk,
                'timestamp': self._data.timestamp,
            })
        
        @self.app.route('/api/risk')
        def api_risk():
            self._refresh_data()
            return jsonify({
                'risk': self._data.risk,
                'pnl_series': self._data.pnl_series,
                'timestamp': self._data.timestamp,
            })
    
    def _refresh_data(self):
        """Refresh data from log files."""
        try:
            # Read signals
            signals = []
            signal_by_strategy = {}
            signal_files = sorted(self.log_dir.glob("signals_*.jsonl"), reverse=True)
            
            for sf in signal_files[:5]:  # Read last 5 signal files
                try:
                    with open(sf) as f:
                        for line in f:
                            try:
                                s = json.loads(line)
                                strategy = s.get('strategy', 'unknown')
                                
                                signals.append({
                                    'time': s.get('timestamp', '')[:19],
                                    'strategy': strategy,
                                    'market_id': s.get('market_id', ''),
                                    'side': s.get('side', ''),
                                    'edge': s.get('edge', 0),
                                    'confidence': s.get('confidence', 0),
                                    'platform': s.get('platform', ''),
                                })
                                
                                if strategy not in signal_by_strategy:
                                    signal_by_strategy[strategy] = {'count': 0, 'platform': s.get('platform', '')}
                                signal_by_strategy[strategy]['count'] += 1
                            except:
                                pass
                except:
                    pass
            
            # Read trades
            trades = []
            trade_by_strategy = {}
            pnl_series = []
            realized_pnl = 0      # Closed/resolved positions
            unrealized_pnl = 0    # Open positions (mark-to-market)
            open_positions = 0    # Count of open positions
            wins = 0
            losses = 0
            
            trade_files = sorted(self.log_dir.glob("trades_*.jsonl"), reverse=True)
            for tf in trade_files[:5]:  # Read last 5 trade files
                if 'archive' in str(tf):
                    continue
                try:
                    with open(tf) as f:
                        for line in f:
                            try:
                                t = json.loads(line)
                                order = t.get('order', {})
                                signal = t.get('signal', {})
                                order_meta = order.get('metadata', {})
                                
                                pnl = order_meta.get('pnl', 0)
                                result = order_meta.get('result', 'pending')
                                strategy = signal.get('strategy', order_meta.get('strategy', 'unknown'))
                                
                                trade = {
                                    'time': t.get('timestamp', '')[:19],
                                    'strategy': strategy,
                                    'platform': signal.get('platform', order.get('platform', '')),
                                    'market_id': signal.get('market_id', order.get('market_id', '')),
                                    'side': signal.get('side', order.get('side', '')),
                                    'size': order.get('size', 0),
                                    'price': order.get('price', 0),
                                    'status': order.get('status', 'unknown'),
                                    'result': result,
                                    'pnl': pnl,
                                    'edge': signal.get('edge', 0),
                                }
                                trades.append(trade)
                                
                                # Separate realized vs unrealized PnL
                                if result == 'pending':
                                    # Open position - compute mark-to-market PnL
                                    # For now, use 0 as we don't have current prices
                                    # In production, this would fetch current market prices
                                    unrealized_pnl += pnl  # Will be 0 for pending
                                    open_positions += 1
                                else:
                                    # Closed position - realized PnL
                                    realized_pnl += pnl
                                    pnl_series.append(pnl)
                                
                                if result == 'win':
                                    wins += 1
                                elif result == 'loss':
                                    losses += 1
                                
                                # Track by strategy
                                if strategy not in trade_by_strategy:
                                    trade_by_strategy[strategy] = {
                                        'count': 0, 'wins': 0, 'losses': 0, 'pnl': 0,
                                        'platform': trade['platform']
                                    }
                                trade_by_strategy[strategy]['count'] += 1
                                trade_by_strategy[strategy]['pnl'] += pnl
                                if result == 'win':
                                    trade_by_strategy[strategy]['wins'] += 1
                                elif result == 'loss':
                                    trade_by_strategy[strategy]['losses'] += 1
                            except:
                                pass
                except:
                    pass
            
            # Try to read position state for unrealized PnL and thresholds
            profit_take_pct = 60.0  # Default
            stop_loss_pct = 38.5    # Default
            online_learning = True
            profit_takes = 0
            stop_losses = 0
            
            position_file = self.log_dir / "positions.json"
            if position_file.exists():
                try:
                    with open(position_file) as f:
                        positions_data = json.load(f)
                        unrealized_pnl = positions_data.get('unrealized_pnl', unrealized_pnl)
                        open_positions = positions_data.get('open_count', open_positions)
                        # Get threshold settings
                        profit_take_pct = positions_data.get('profit_take_pct', profit_take_pct)
                        stop_loss_pct = positions_data.get('stop_loss_pct', stop_loss_pct)
                        online_learning = positions_data.get('online_learning_enabled', online_learning)
                        profit_takes = positions_data.get('profit_takes', 0)
                        stop_losses = positions_data.get('stop_losses', 0)
                except:
                    pass
            
            total_pnl = realized_pnl + unrealized_pnl
            
            # Compute risk metrics
            risk_metrics = compute_risk_metrics(pnl_series, self.initial_bankroll)
            
            # Build strategies dict
            strategies = {}
            all_strategy_names = set(signal_by_strategy.keys()) | set(trade_by_strategy.keys())
            
            for name in all_strategy_names:
                sig_info = signal_by_strategy.get(name, {'count': 0, 'platform': ''})
                trade_info = trade_by_strategy.get(name, {'count': 0, 'wins': 0, 'losses': 0, 'pnl': 0, 'platform': ''})
                
                total_trades = trade_info['wins'] + trade_info['losses']
                win_rate = (trade_info['wins'] / total_trades * 100) if total_trades > 0 else 0
                
                strategies[name] = {
                    'platform': sig_info.get('platform') or trade_info.get('platform', ''),
                    'calibration_status': 'active' if sig_info['count'] > 0 else 'inactive',
                    'samples': 0,  # Would need to load from strategy
                    'signals_count': sig_info['count'],
                    'trades_count': trade_info['count'],
                    'wins': trade_info['wins'],
                    'losses': trade_info['losses'],
                    'win_rate': win_rate,
                    'pnl': trade_info['pnl'],
                }
            
            # Compute fill rate: positions opened / unique order attempts
            # Count unique markets attempted (not duplicate attempts for same market)
            unique_markets_attempted = len(set(t.get('market_id', '') for t in trades if t.get('market_id')))
            actual_positions = open_positions + wins + losses
            fill_rate = (actual_positions / unique_markets_attempted * 100) if unique_markets_attempted > 0 else 0
            
            # Update data
            self._data.status = {
                'trading_halted': False,
                'halt_reason': '',
                'bankroll': self.initial_bankroll + realized_pnl,  # Only count realized
                'initial_bankroll': self.initial_bankroll,
                'peak_bankroll': max(self.initial_bankroll, self.initial_bankroll + realized_pnl),
                'current_drawdown': risk_metrics['max_drawdown'],
                'realized_pnl': realized_pnl,
                'unrealized_pnl': unrealized_pnl,
                'total_pnl': total_pnl,
                'open_positions': open_positions,
                'total_trades': open_positions + wins + losses,  # Actual positions: open + closed
                'wins': wins,
                'losses': losses,
                'fill_rate': fill_rate,
                # Position management thresholds
                'profit_take_pct': profit_take_pct,
                'stop_loss_pct': stop_loss_pct,
                'online_learning': online_learning,
                'profit_takes': profit_takes,
                'stop_losses': stop_losses,
            }
            self._data.risk = risk_metrics
            self._data.strategies = strategies
            self._data.recent_signals = signals[-20:][::-1]
            self._data.recent_trades = trades[-30:][::-1]
            
            # Get open positions from positions.json
            open_positions_list = []
            position_file = self.log_dir / "positions.json"
            if position_file.exists():
                try:
                    with open(position_file) as f:
                        positions_data = json.load(f)
                        if 'positions' in positions_data:
                            open_positions_list = positions_data['positions']
                except Exception as e:
                    print(f"Error reading positions.json: {e}")
            
            # Get closed trades from closed_trades.jsonl
            closed_trades_list = []
            closed_file = self.log_dir / "closed_trades.jsonl"
            if closed_file.exists():
                try:
                    with open(closed_file) as f:
                        for line in f:
                            if line.strip():
                                try:
                                    trade = json.loads(line)
                                    closed_trades_list.append({
                                        'time': trade.get('exit_time', trade.get('timestamp', ''))[:16],
                                        'strategy': trade.get('strategy', 'unknown'),
                                        'platform': trade.get('platform', 'unknown'),
                                        'market_id': trade.get('market_id', ''),
                                        'market_question': trade.get('market_question', ''),
                                        'side': trade.get('side', 'yes'),
                                        'entry_price': trade.get('entry_price', 0),
                                        'exit_price': trade.get('exit_price', 0),
                                        'size': trade.get('size', 0),
                                        'pnl': trade.get('pnl', 0),
                                        'return_pct': trade.get('return_pct', 0),
                                        'exit_reason': trade.get('exit_reason', 'unknown'),
                                        'hold_time_hours': trade.get('hold_time_hours', 0),
                                        'status': 'closed',
                                        'result': trade.get('exit_reason', 'closed'),
                                    })
                                    # Update wins/losses based on closed trades
                                    if trade.get('pnl', 0) > 0:
                                        wins += 1
                                    else:
                                        losses += 1
                                    realized_pnl += trade.get('pnl', 0)
                                    pnl_series.append(trade.get('pnl', 0))
                                except:
                                    pass
                except Exception as e:
                    print(f"Error reading closed_trades.jsonl: {e}")
            
            self._data.open_positions = open_positions_list[-20:][::-1]
            self._data.closed_trades = closed_trades_list[-30:][::-1]
            
            # Count actual unique positions (not duplicate order attempts)
            # trades_filled = open positions + closed positions (unique trades)
            unique_positions = len(open_positions_list) + len(closed_trades_list)
            
            # Count unique markets attempted (first attempt per market)
            unique_markets = len(set(t.get('market_id', '') for t in trades if t.get('market_id')))
            
            self._data.pnl_series = pnl_series
            self._data.flow = {
                'markets_scanned': len(signals) * 5,  # Estimate
                'signals_generated': len(signals),
                'trades_attempted': unique_markets,
                'trades_filled': unique_positions,
            }
            self._data.timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            
        except Exception as e:
            print(f"Error refreshing data: {e}")
            import traceback
            traceback.print_exc()
    
    def update_status(self, status: Dict):
        """Update status from engine."""
        self._data.status.update(status)
    
    def update_strategies(self, strategies: Dict):
        """Update strategy info."""
        self._data.strategies = strategies
    
    def run(self, debug: bool = False):
        """Run the dashboard server."""
        print(f"\n{'='*60}")
        print(f"Starting Trading Dashboard on http://localhost:{self.port}")
        print(f"Mode: {self.mode.upper()}")
        print(f"{'='*60}\n")
        self.app.run(host='0.0.0.0', port=self.port, debug=debug)
    
    def run_background(self):
        """Run dashboard in background thread."""
        thread = threading.Thread(target=lambda: self.app.run(
            host='0.0.0.0', 
            port=self.port, 
            debug=False,
            use_reloader=False,
        ))
        thread.daemon = True
        thread.start()
        return thread


def run_dashboard(log_dir: str = "logs/paper_trading", port: int = 8080, 
                  mode: str = "hybrid", initial_bankroll: float = 10000):
    """Run standalone dashboard."""
    dashboard = TradingDashboard(
        log_dir=log_dir, 
        port=port, 
        mode=mode,
        initial_bankroll=initial_bankroll,
    )
    dashboard.run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs/paper_trading")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--mode", default="hybrid", choices=["simulated", "hybrid", "live"])
    parser.add_argument("--bankroll", type=float, default=10000)
    args = parser.parse_args()
    
    run_dashboard(args.log_dir, args.port, args.mode, args.bankroll)
