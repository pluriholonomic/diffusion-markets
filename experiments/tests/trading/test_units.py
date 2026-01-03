"""
Unit tests for the canonical trading unit system.

Tests:
- TradeUnits basic operations
- PositionUnits add/reduce with proper PnL accounting
- Edge cases (extreme prices, zero stake)
- Consistency between different calculation methods
"""

import pytest
from datetime import datetime
import math
import sys
from pathlib import Path

# Ensure experiments directory is on path for trading module import
_exp_dir = Path(__file__).resolve().parent.parent.parent
if str(_exp_dir) not in sys.path:
    sys.path.insert(0, str(_exp_dir))

from trading.core.units import TradeUnits, PositionUnits, compute_kelly_stake


class TestTradeUnits:
    """Tests for TradeUnits class."""
    
    def test_basic_yes_trade(self):
        """Basic YES trade calculations."""
        trade = TradeUnits(stake_usd=100, entry_side_price=0.40, side='yes')
        
        # Contracts = stake / price
        assert trade.contracts == 250.0
        
        # MTM at same price = stake
        assert trade.mtm_value(0.40) == 100.0
        assert trade.unrealized_pnl(0.40) == 0.0
        
        # MTM at higher price
        assert trade.mtm_value(0.60) == 150.0  # 250 * 0.60
        assert trade.unrealized_pnl(0.60) == 50.0  # 150 - 100
        
        # Return percentage
        assert trade.unrealized_return_pct(0.60) == 50.0  # 50%
    
    def test_basic_no_trade(self):
        """Basic NO trade calculations."""
        # NO at 0.30 means we pay $0.30 per NO contract
        trade = TradeUnits(stake_usd=100, entry_side_price=0.30, side='no')
        
        # Contracts = 100 / 0.30 = 333.33
        assert abs(trade.contracts - 333.333) < 0.01
        
        # If NO price rises to 0.40
        assert abs(trade.mtm_value(0.40) - 133.33) < 0.1
        assert abs(trade.unrealized_pnl(0.40) - 33.33) < 0.1
    
    def test_resolution_yes_wins(self):
        """Test PnL when YES wins."""
        trade = TradeUnits(stake_usd=100, entry_side_price=0.40, side='yes')
        
        # YES wins: payout = 250 * 1.0 = $250
        # PnL = $250 - $100 = $150
        assert trade.resolution_pnl(1) == 150.0
        
        # YES loses: payout = 0
        # PnL = $0 - $100 = -$100
        assert trade.resolution_pnl(0) == -100.0
    
    def test_resolution_no_wins(self):
        """Test PnL when NO wins."""
        trade = TradeUnits(stake_usd=100, entry_side_price=0.30, side='no')
        
        # NO wins (outcome=0): payout = 333.33 * 1.0 = $333.33
        # PnL = $333.33 - $100 = $233.33
        pnl = trade.resolution_pnl(0)
        assert abs(pnl - 233.33) < 0.1
        
        # NO loses (outcome=1): payout = 0
        # PnL = -$100
        assert trade.resolution_pnl(1) == -100.0
    
    def test_exit_with_costs(self):
        """Test exit with transaction costs."""
        trade = TradeUnits(stake_usd=100, entry_side_price=0.40, side='yes')
        
        # Exit at 0.60 with 2% cost
        # Exit value = 250 * 0.60 = $150
        # Cost = $150 * 0.02 = $3
        # PnL = $150 - $100 - $3 = $47
        pnl = trade.exit_pnl(0.60, exit_cost_rate=0.02)
        assert pnl == 47.0
    
    def test_from_market_order_with_slippage(self):
        """Test creation with slippage."""
        # YES order with 2% slippage
        trade = TradeUnits.from_market_order(
            stake_usd=100,
            market_yes_price=0.50,
            side='yes',
            slippage_pct=2.0,
        )
        
        # Should pay 0.50 * 1.02 = 0.51
        assert trade.entry_side_price == 0.51
        
        # NO order with 2% slippage
        trade_no = TradeUnits.from_market_order(
            stake_usd=100,
            market_yes_price=0.50,  # NO price = 0.50
            side='no',
            slippage_pct=2.0,
        )
        
        # NO price = 1 - 0.50 = 0.50, with slippage = 0.51
        assert trade_no.entry_side_price == 0.51
    
    def test_extreme_prices(self):
        """Test with extreme prices near 0 and 1."""
        # Very low price
        trade_low = TradeUnits(stake_usd=100, entry_side_price=0.01, side='yes')
        assert trade_low.contracts == 10000.0
        assert trade_low.resolution_pnl(1) == 9900.0  # 10000 - 100
        
        # Very high price
        trade_high = TradeUnits(stake_usd=100, entry_side_price=0.99, side='yes')
        assert abs(trade_high.contracts - 101.01) < 0.1
        # If YES wins: 101.01 - 100 = $1.01 profit
        assert abs(trade_high.resolution_pnl(1) - 1.01) < 0.1
    
    def test_validation(self):
        """Test input validation."""
        # Negative stake
        with pytest.raises(ValueError):
            TradeUnits(stake_usd=-100, entry_side_price=0.40, side='yes')
        
        # Price out of range
        with pytest.raises(ValueError):
            TradeUnits(stake_usd=100, entry_side_price=0, side='yes')
        with pytest.raises(ValueError):
            TradeUnits(stake_usd=100, entry_side_price=1.5, side='yes')
        
        # Invalid side
        with pytest.raises(ValueError):
            TradeUnits(stake_usd=100, entry_side_price=0.40, side='maybe')
    
    def test_zero_stake(self):
        """Test edge case with zero stake."""
        trade = TradeUnits(stake_usd=0, entry_side_price=0.40, side='yes')
        assert trade.contracts == 0.0
        assert trade.mtm_value(0.60) == 0.0
        assert trade.unrealized_return_pct(0.60) == 0.0


class TestPositionUnits:
    """Tests for PositionUnits class."""
    
    def test_add_contracts(self):
        """Test adding contracts to a position."""
        pos = PositionUnits(market_id="test", side='yes')
        
        # Add 100 contracts at 0.40
        cost = pos.add_contracts(100, 0.40)
        assert cost == 40.0  # 100 * 0.40
        assert pos.total_contracts == 100.0
        assert pos.avg_entry_price == 0.40
        assert pos.total_stake_usd == 40.0
        
        # Add 100 more at 0.60
        cost2 = pos.add_contracts(100, 0.60)
        assert cost2 == 60.0
        assert pos.total_contracts == 200.0
        # Weighted average: (100*0.40 + 100*0.60) / 200 = 0.50
        assert pos.avg_entry_price == 0.50
        assert pos.total_stake_usd == 100.0
    
    def test_reduce_contracts_pnl(self):
        """Test reducing contracts realizes correct PnL."""
        pos = PositionUnits(market_id="test", side='yes')
        
        # Add 100 contracts at 0.40 (cost $40)
        pos.add_contracts(100, 0.40)
        
        # Close 50 contracts at 0.60
        # PnL = 50 * (0.60 - 0.40) = $10
        pnl = pos.reduce_contracts(50, 0.60)
        assert pnl == pytest.approx(10.0)
        assert pos.realized_pnl == pytest.approx(10.0)
        assert pos.total_contracts == 50.0
        # Stake reduced proportionally: 40 * 0.5 = 20
        assert pos.total_stake_usd == pytest.approx(20.0)
    
    def test_reduce_all_contracts(self):
        """Test closing entire position."""
        pos = PositionUnits(market_id="test", side='yes')
        pos.add_contracts(100, 0.40)
        
        # Close all at 0.70
        # PnL = 100 * (0.70 - 0.40) = $30
        pnl = pos.reduce_contracts(100, 0.70)
        assert pnl == pytest.approx(30.0)
        assert pos.total_contracts == 0.0
        assert not pos.is_open
    
    def test_reduce_at_loss(self):
        """Test reducing at a loss."""
        pos = PositionUnits(market_id="test", side='yes')
        pos.add_contracts(100, 0.60)
        
        # Close at 0.40 (loss)
        # PnL = 100 * (0.40 - 0.60) = -$20
        pnl = pos.reduce_contracts(100, 0.40)
        assert pnl == pytest.approx(-20.0)
        assert pos.realized_pnl == pytest.approx(-20.0)
    
    def test_resolution_yes_side(self):
        """Test resolution for YES position."""
        pos = PositionUnits(market_id="test", side='yes')
        pos.add_contracts(250, 0.40)  # Cost $100
        
        # YES wins
        pnl = pos.close_at_resolution(1)
        # Payout = 250 * 1.0 = $250
        # PnL = $250 - $100 = $150
        assert pnl == 150.0
        assert not pos.is_open
    
    def test_resolution_no_side(self):
        """Test resolution for NO position."""
        pos = PositionUnits(market_id="test", side='no')
        pos.add_contracts(333.33, 0.30)  # Cost $100
        
        # NO wins (outcome=0)
        pnl = pos.close_at_resolution(0)
        # Payout = 333.33 * 1.0 = $333.33
        # PnL = $333.33 - $100 = $233.33
        assert abs(pnl - 233.33) < 0.1
    
    def test_unrealized_pnl(self):
        """Test unrealized PnL calculation."""
        pos = PositionUnits(market_id="test", side='yes')
        pos.add_contracts(100, 0.40)  # Cost $40
        
        # At current price 0.60
        # Value = 100 * 0.60 = $60
        # Unrealized = $60 - $40 = $20
        assert pos.unrealized_pnl(0.60) == 20.0
        
        # Total PnL = realized + unrealized
        assert pos.total_pnl(0.60) == 20.0
    
    def test_total_pnl_with_partial_close(self):
        """Test total PnL after partial close."""
        pos = PositionUnits(market_id="test", side='yes')
        pos.add_contracts(100, 0.40)  # Cost $40
        
        # Close 50 at 0.60: realized $10
        pos.reduce_contracts(50, 0.60)
        
        # Remaining 50 contracts at current price 0.70
        # Value = 50 * 0.70 = $35
        # Remaining stake = $20
        # Unrealized = $35 - $20 = $15
        assert pos.unrealized_pnl(0.70) == 15.0
        
        # Total = $10 realized + $15 unrealized = $25
        assert pos.total_pnl(0.70) == 25.0
    
    def test_serialization(self):
        """Test to_dict and from_dict."""
        pos = PositionUnits(market_id="test", side='yes', strategy='calibration')
        pos.add_contracts(100, 0.40)
        pos.reduce_contracts(50, 0.60)
        
        data = pos.to_dict()
        restored = PositionUnits.from_dict(data)
        
        assert restored.market_id == pos.market_id
        assert restored.side == pos.side
        assert restored.total_contracts == pos.total_contracts
        assert restored.realized_pnl == pos.realized_pnl
    
    def test_validation(self):
        """Test input validation for position operations."""
        pos = PositionUnits(market_id="test", side='yes')
        pos.add_contracts(100, 0.40)
        
        # Can't add negative contracts
        with pytest.raises(ValueError):
            pos.add_contracts(-50, 0.50)
        
        # Can't close more than we have
        with pytest.raises(ValueError):
            pos.reduce_contracts(200, 0.50)


class TestKellyStake:
    """Tests for Kelly stake calculation."""
    
    def test_positive_edge_yes(self):
        """Test Kelly stake with positive YES edge."""
        # True prob 0.60, market price 0.40 -> edge on YES
        stake = compute_kelly_stake(
            probability=0.60,
            market_price=0.40,
            bankroll=10000,
            kelly_fraction=0.25,
        )
        
        # Kelly = (0.60 - 0.40) / (1 - 0.40) = 0.333
        # Quarter Kelly = 0.333 * 0.25 = 0.083
        # Stake = 10000 * 0.083 = $833
        # But capped at 10% = $1000
        assert stake > 0
        assert stake <= 1000  # Max 10% of bankroll
    
    def test_positive_edge_no(self):
        """Test Kelly stake with positive NO edge."""
        # True prob 0.30, market price 0.60 -> edge on NO
        stake = compute_kelly_stake(
            probability=0.30,
            market_price=0.60,
            bankroll=10000,
            kelly_fraction=0.25,
        )
        
        assert stake > 0
        assert stake <= 1000
    
    def test_no_edge(self):
        """Test Kelly stake with no edge."""
        # True prob equals market price
        stake = compute_kelly_stake(
            probability=0.50,
            market_price=0.50,
            bankroll=10000,
        )
        
        assert stake == 0.0
    
    def test_max_stake_cap(self):
        """Test that stake is capped at max_stake_pct."""
        # Huge edge
        stake = compute_kelly_stake(
            probability=0.99,
            market_price=0.01,
            bankroll=10000,
            kelly_fraction=1.0,  # Full Kelly
            max_stake_pct=0.05,  # But capped at 5%
        )
        
        assert stake == 500.0  # 10000 * 0.05


class TestBacktestPnLConsistency:
    """
    Tests that ensure backtest and live PnL calculations are consistent.
    These are integration-style tests for the unit system.
    """
    
    def test_trade_to_position_consistency(self):
        """TradeUnits and PositionUnits should give same PnL."""
        # Create equivalent trade and position
        trade = TradeUnits(stake_usd=100, entry_side_price=0.40, side='yes')
        
        pos = PositionUnits(market_id="test", side='yes')
        pos.add_contracts(trade.contracts, trade.entry_side_price)
        
        # At same price, unrealized PnL should match
        current_price = 0.60
        assert abs(trade.unrealized_pnl(current_price) - pos.unrealized_pnl(current_price)) < 0.01
        
        # At resolution, PnL should match
        resolution_pnl_trade = trade.resolution_pnl(1)
        resolution_pnl_pos = pos.close_at_resolution(1)
        assert abs(resolution_pnl_trade - resolution_pnl_pos) < 0.01
    
    def test_symmetric_yes_no(self):
        """YES and NO bets should be symmetric around the price."""
        bankroll = 100
        yes_price = 0.40
        no_price = 0.60
        
        # Buy YES at 0.40
        yes_trade = TradeUnits(stake_usd=50, entry_side_price=yes_price, side='yes')
        # Buy NO at 0.60
        no_trade = TradeUnits(stake_usd=50, entry_side_price=no_price, side='no')
        
        # If YES wins: YES profit, NO loss
        yes_pnl_if_yes = yes_trade.resolution_pnl(1)  # +$75
        no_pnl_if_yes = no_trade.resolution_pnl(1)    # -$50
        
        # If NO wins: NO profit, YES loss
        yes_pnl_if_no = yes_trade.resolution_pnl(0)   # -$50
        no_pnl_if_no = no_trade.resolution_pnl(0)     # +$33.33
        
        # Combined PnL should reflect the edge
        assert yes_pnl_if_yes > 0
        assert no_pnl_if_yes < 0
        assert yes_pnl_if_no < 0
        assert no_pnl_if_no > 0
    
    def test_mtm_at_entry_is_stake(self):
        """Mark-to-market at entry price equals stake."""
        trade = TradeUnits(stake_usd=100, entry_side_price=0.40, side='yes')
        assert trade.mtm_value(0.40) == 100.0
        assert trade.unrealized_pnl(0.40) == 0.0
    
    def test_multiple_adds_then_resolution(self):
        """Test multiple adds followed by resolution."""
        pos = PositionUnits(market_id="test", side='yes')
        
        # Add positions at different prices
        pos.add_contracts(100, 0.40)  # $40
        pos.add_contracts(100, 0.50)  # $50
        pos.add_contracts(100, 0.60)  # $60
        
        # Total: 300 contracts, $150 stake
        assert pos.total_contracts == 300.0
        assert pos.total_stake_usd == 150.0
        assert pos.avg_entry_price == 0.50  # (40+50+60)/300 * price
        
        # YES wins: payout = 300 * 1.0 = $300
        # PnL = $300 - $150 = $150
        pnl = pos.close_at_resolution(1)
        assert pnl == 150.0
