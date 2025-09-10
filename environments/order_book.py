"""
Order Book Implementation for Trading Environment

This module implements a realistic order book that maintains buy and sell orders,
handles order matching, and provides market data for trading agents.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class OrderType(Enum):
    """Types of orders that can be placed."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderSide(Enum):
    """Order sides (buy or sell)."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Represents a trading order."""
    order_id: str
    agent_id: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    timestamp: float = 0.0
    filled_quantity: float = 0.0
    
    @property
    def remaining_quantity(self) -> float:
        """Get remaining quantity to be filled."""
        return self.quantity - self.filled_quantity
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.remaining_quantity <= 1e-8


@dataclass
class Trade:
    """Represents a completed trade."""
    trade_id: str
    buy_order_id: str
    sell_order_id: str
    quantity: float
    price: float
    timestamp: float
    buy_agent_id: str
    sell_agent_id: str


class OrderBook:
    """
    Realistic order book implementation with proper order matching.
    
    This order book maintains separate buy and sell order lists, handles
    order matching according to price-time priority, and provides market
    data for trading agents.
    """
    
    def __init__(self, tick_size: float = 0.01, max_depth: int = 10):
        """
        Initialize the order book.
        
        Args:
            tick_size: Minimum price increment
            max_depth: Maximum number of price levels to maintain
        """
        self.tick_size = tick_size
        self.max_depth = max_depth
        
        # Order storage
        self.orders: Dict[str, Order] = {}
        self.buy_orders: List[str] = []  # Sorted by price (desc), then time
        self.sell_orders: List[str] = []  # Sorted by price (asc), then time
        
        # Market data
        self.last_trade_price: Optional[float] = None
        self.last_trade_quantity: float = 0.0
        self.volume_traded: float = 0.0
        self.trades: List[Trade] = []
        
        # Statistics
        self.total_orders: int = 0
        self.total_trades: int = 0
        
    def add_order(self, order: Order) -> List[Trade]:
        """
        Add an order to the book and return any resulting trades.
        
        Args:
            order: The order to add
            
        Returns:
            List of trades that occurred from this order
        """
        self.orders[order.order_id] = order
        self.total_orders += 1
        
        trades = []
        
        if order.order_type == OrderType.MARKET:
            trades = self._match_market_order(order)
        elif order.order_type == OrderType.LIMIT:
            trades = self._match_limit_order(order)
        
        # Update market data
        if trades:
            self.last_trade_price = trades[-1].price
            self.last_trade_quantity = trades[-1].quantity
            self.volume_traded += sum(trade.quantity for trade in trades)
            self.trades.extend(trades)
            self.total_trades += len(trades)
        
        return trades
    
    def _match_market_order(self, order: Order) -> List[Trade]:
        """Match a market order against the book."""
        trades = []
        remaining_quantity = order.remaining_quantity
        
        if order.side == OrderSide.BUY:
            # Match against sell orders (ascending price)
            for sell_order_id in self.sell_orders[:]:
                if remaining_quantity <= 0:
                    break
                    
                sell_order = self.orders[sell_order_id]
                if sell_order.remaining_quantity <= 0:
                    continue
                
                # Market buy matches at sell order's limit price
                trade_quantity = min(remaining_quantity, sell_order.remaining_quantity)
                trade_price = sell_order.price
                
                trade = self._create_trade(order, sell_order, trade_quantity, trade_price)
                trades.append(trade)
                
                # Update order quantities
                order.filled_quantity += trade_quantity
                sell_order.filled_quantity += trade_quantity
                
                remaining_quantity -= trade_quantity
                
                # Remove filled sell orders
                if sell_order.is_filled:
                    self.sell_orders.remove(sell_order_id)
                    del self.orders[sell_order_id]
        
        else:  # SELL
            # Match against buy orders (descending price)
            for buy_order_id in self.buy_orders[:]:
                if remaining_quantity <= 0:
                    break
                    
                buy_order = self.orders[buy_order_id]
                if buy_order.remaining_quantity <= 0:
                    continue
                
                # Market sell matches at buy order's limit price
                trade_quantity = min(remaining_quantity, buy_order.remaining_quantity)
                trade_price = buy_order.price
                
                trade = self._create_trade(buy_order, order, trade_quantity, trade_price)
                trades.append(trade)
                
                # Update order quantities
                order.filled_quantity += trade_quantity
                buy_order.filled_quantity += trade_quantity
                
                remaining_quantity -= trade_quantity
                
                # Remove filled buy orders
                if buy_order.is_filled:
                    self.buy_orders.remove(buy_order_id)
                    del self.orders[buy_order_id]
        
        return trades
    
    def _match_limit_order(self, order: Order) -> List[Trade]:
        """Match a limit order against the book."""
        trades = []
        remaining_quantity = order.remaining_quantity
        
        if order.side == OrderSide.BUY:
            # Match against sell orders at or below our limit price
            for sell_order_id in self.sell_orders[:]:
                if remaining_quantity <= 0:
                    break
                    
                sell_order = self.orders[sell_order_id]
                if sell_order.price > order.price or sell_order.remaining_quantity <= 0:
                    continue
                
                trade_quantity = min(remaining_quantity, sell_order.remaining_quantity)
                trade_price = sell_order.price  # Match at better price
                
                trade = self._create_trade(order, sell_order, trade_quantity, trade_price)
                trades.append(trade)
                
                # Update order quantities
                order.filled_quantity += trade_quantity
                sell_order.filled_quantity += trade_quantity
                
                remaining_quantity -= trade_quantity
                
                # Remove filled sell orders
                if sell_order.is_filled:
                    self.sell_orders.remove(sell_order_id)
                    del self.orders[sell_order_id]
        
        else:  # SELL
            # Match against buy orders at or above our limit price
            for buy_order_id in self.buy_orders[:]:
                if remaining_quantity <= 0:
                    break
                    
                buy_order = self.orders[buy_order_id]
                if buy_order.price < order.price or buy_order.remaining_quantity <= 0:
                    continue
                
                trade_quantity = min(remaining_quantity, buy_order.remaining_quantity)
                trade_price = buy_order.price  # Match at better price
                
                trade = self._create_trade(buy_order, order, trade_quantity, trade_price)
                trades.append(trade)
                
                # Update order quantities
                order.filled_quantity += trade_quantity
                buy_order.filled_quantity += trade_quantity
                
                remaining_quantity -= trade_quantity
                
                # Remove filled buy orders
                if buy_order.is_filled:
                    self.buy_orders.remove(buy_order_id)
                    del self.orders[buy_order_id]
        
        # If order still has remaining quantity, add to book
        if remaining_quantity > 0:
            self._add_to_book(order)
        
        return trades
    
    def _add_to_book(self, order: Order):
        """Add a limit order to the appropriate side of the book."""
        if order.side == OrderSide.BUY:
            # Insert in descending price order, then by time
            inserted = False
            for i, existing_order_id in enumerate(self.buy_orders):
                existing_order = self.orders[existing_order_id]
                if order.price > existing_order.price:
                    self.buy_orders.insert(i, order.order_id)
                    inserted = True
                    break
            if not inserted:
                self.buy_orders.append(order.order_id)
        
        else:  # SELL
            # Insert in ascending price order, then by time
            inserted = False
            for i, existing_order_id in enumerate(self.sell_orders):
                existing_order = self.orders[existing_order_id]
                if order.price < existing_order.price:
                    self.sell_orders.insert(i, order.order_id)
                    inserted = True
                    break
            if not inserted:
                self.sell_orders.append(order.order_id)
    
    def _create_trade(self, buy_order: Order, sell_order: Order, 
                     quantity: float, price: float) -> Trade:
        """Create a trade record."""
        trade_id = f"trade_{self.total_trades + 1}_{buy_order.order_id}_{sell_order.order_id}"
        return Trade(
            trade_id=trade_id,
            buy_order_id=buy_order.order_id,
            sell_order_id=sell_order.order_id,
            quantity=quantity,
            price=price,
            timestamp=max(buy_order.timestamp, sell_order.timestamp),
            buy_agent_id=buy_order.agent_id,
            sell_agent_id=sell_order.agent_id
        )
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order from the book.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            True if order was cancelled, False if not found
        """
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        # Remove from appropriate side
        if order.side == OrderSide.BUY and order_id in self.buy_orders:
            self.buy_orders.remove(order_id)
        elif order.side == OrderSide.SELL and order_id in self.sell_orders:
            self.sell_orders.remove(order_id)
        
        del self.orders[order_id]
        return True
    
    def get_best_bid(self) -> Optional[float]:
        """Get the best (highest) bid price."""
        if not self.buy_orders:
            return None
        return self.orders[self.buy_orders[0]].price
    
    def get_best_ask(self) -> Optional[float]:
        """Get the best (lowest) ask price."""
        if not self.sell_orders:
            return None
        return self.orders[self.sell_orders[0]].price
    
    def get_spread(self) -> Optional[float]:
        """Get the current bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid is None or best_ask is None:
            return None
        return best_ask - best_bid
    
    def get_mid_price(self) -> Optional[float]:
        """Get the mid price (average of best bid and ask)."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid is None or best_ask is None:
            return None
        return (best_bid + best_ask) / 2
    
    def get_market_depth(self, side: OrderSide, levels: int = 5) -> List[Tuple[float, float]]:
        """
        Get market depth for a given side.
        
        Args:
            side: BUY or SELL
            levels: Number of price levels to return
            
        Returns:
            List of (price, quantity) tuples
        """
        if side == OrderSide.BUY:
            orders = self.buy_orders[:levels]
        else:
            orders = self.sell_orders[:levels]
        
        depth = []
        for order_id in orders:
            order = self.orders[order_id]
            depth.append((order.price, order.remaining_quantity))
        
        return depth
    
    def get_market_data(self) -> Dict:
        """Get comprehensive market data for agents."""
        return {
            "best_bid": self.get_best_bid(),
            "best_ask": self.get_best_ask(),
            "spread": self.get_spread(),
            "mid_price": self.get_mid_price(),
            "last_trade_price": self.last_trade_price,
            "last_trade_quantity": self.last_trade_quantity,
            "volume_traded": self.volume_traded,
            "buy_depth": self.get_market_depth(OrderSide.BUY),
            "sell_depth": self.get_market_depth(OrderSide.SELL),
            "total_orders": self.total_orders,
            "total_trades": self.total_trades,
        }

