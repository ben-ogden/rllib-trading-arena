"""
Market Simulator for Trading Environment

This module implements realistic market dynamics including price movements,
volatility, liquidity changes, and external market events that affect
the trading environment.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import random


class MarketEvent(Enum):
    """Types of market events that can occur."""
    NORMAL = "normal"
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    NEWS_EVENT = "news_event"
    FLASH_CRASH = "flash_crash"
    FLASH_RALLY = "flash_rally"


@dataclass
class MarketState:
    """Current state of the market."""
    price: float
    volatility: float
    liquidity: float
    volume: float
    event: MarketEvent
    event_duration: int
    event_intensity: float


class MarketSimulator:
    """
    Realistic market simulator that generates price movements and market events.
    
    This simulator creates realistic market dynamics including:
    - Mean-reverting price movements
    - Volatility clustering
    - Liquidity variations
    - Market events (news, crashes, rallies)
    - Volume patterns
    """
    
    def __init__(self, 
                 initial_price: float = 100.0,
                 volatility: float = 0.02,
                 liquidity_factor: float = 0.1,
                 mean_reversion: float = 0.1,
                 event_probability: float = 0.05):
        """
        Initialize the market simulator.
        
        Args:
            initial_price: Starting price of the asset
            volatility: Base volatility level
            liquidity_factor: Base liquidity level
            mean_reversion: Mean reversion strength
            event_probability: Probability of market events per step
        """
        self.initial_price = initial_price
        self.base_volatility = volatility
        self.base_liquidity = liquidity_factor
        self.mean_reversion = mean_reversion
        self.event_probability = event_probability
        
        # Current state
        self.current_price = initial_price
        self.current_volatility = volatility
        self.current_liquidity = liquidity_factor
        self.current_volume = 0.0
        
        # Market event state
        self.current_event = MarketEvent.NORMAL
        self.event_duration = 0
        self.event_intensity = 0.0
        
        # Price history for trend calculation
        self.price_history: List[float] = [initial_price]
        self.max_history = 100
        
        # Market parameters
        self.price_range = (initial_price * 0.5, initial_price * 2.0)
        self.volatility_range = (volatility * 0.1, volatility * 5.0)
        self.liquidity_range = (liquidity_factor * 0.1, liquidity_factor * 3.0)
        
        # Event parameters
        self.event_durations = {
            MarketEvent.VOLATILITY_SPIKE: (5, 20),
            MarketEvent.LIQUIDITY_CRISIS: (10, 30),
            MarketEvent.NEWS_EVENT: (3, 10),
            MarketEvent.FLASH_CRASH: (1, 3),
            MarketEvent.FLASH_RALLY: (1, 3),
        }
        
        self.event_intensities = {
            MarketEvent.VOLATILITY_SPIKE: (2.0, 5.0),
            MarketEvent.LIQUIDITY_CRISIS: (0.1, 0.3),
            MarketEvent.NEWS_EVENT: (1.5, 3.0),
            MarketEvent.FLASH_CRASH: (5.0, 10.0),
            MarketEvent.FLASH_RALLY: (5.0, 10.0),
        }
    
    def step(self) -> MarketState:
        """
        Advance the market simulation by one step.
        
        Returns:
            Current market state
        """
        # Handle current market event
        if self.current_event != MarketEvent.NORMAL:
            self.event_duration -= 1
            if self.event_duration <= 0:
                self.current_event = MarketEvent.NORMAL
                self.event_intensity = 0.0
        
        # Check for new market events
        if self.current_event == MarketEvent.NORMAL and random.random() < self.event_probability:
            self._trigger_market_event()
        
        # Generate price movement
        price_change = self._generate_price_movement()
        self.current_price = max(self.price_range[0], 
                               min(self.price_range[1], 
                                   self.current_price + price_change))
        
        # Update volatility (volatility clustering)
        self._update_volatility()
        
        # Update liquidity
        self._update_liquidity()
        
        # Update volume
        self._update_volume()
        
        # Update price history
        self.price_history.append(self.current_price)
        if len(self.price_history) > self.max_history:
            self.price_history.pop(0)
        
        return MarketState(
            price=self.current_price,
            volatility=self.current_volatility,
            liquidity=self.current_liquidity,
            volume=self.current_volume,
            event=self.current_event,
            event_duration=self.event_duration,
            event_intensity=self.event_intensity
        )
    
    def _generate_price_movement(self) -> float:
        """Generate realistic price movement based on current market conditions."""
        # Base random walk
        random_component = np.random.normal(0, self.current_volatility)
        
        # Mean reversion component
        price_deviation = (self.current_price - self.initial_price) / self.initial_price
        mean_reversion_component = -self.mean_reversion * price_deviation
        
        # Event-based component
        event_component = 0.0
        if self.current_event != MarketEvent.NORMAL:
            event_component = self._get_event_price_impact()
        
        # Trend component (momentum)
        trend_component = self._get_trend_component()
        
        # Combine all components
        total_change = (random_component + mean_reversion_component + 
                       event_component + trend_component)
        
        return self.current_price * total_change
    
    def _get_event_price_impact(self) -> float:
        """Get price impact from current market event."""
        if self.current_event == MarketEvent.VOLATILITY_SPIKE:
            # Increased volatility but no directional bias
            return np.random.normal(0, self.current_volatility * self.event_intensity)
        
        elif self.current_event == MarketEvent.LIQUIDITY_CRISIS:
            # Reduced liquidity leads to larger price swings
            return np.random.normal(0, self.current_volatility * 2.0)
        
        elif self.current_event == MarketEvent.NEWS_EVENT:
            # News events have directional bias
            direction = np.random.choice([-1, 1])
            return direction * self.current_volatility * self.event_intensity
        
        elif self.current_event == MarketEvent.FLASH_CRASH:
            # Sudden downward movement
            return -self.current_volatility * self.event_intensity
        
        elif self.current_event == MarketEvent.FLASH_RALLY:
            # Sudden upward movement
            return self.current_volatility * self.event_intensity
        
        return 0.0
    
    def _get_trend_component(self) -> float:
        """Calculate trend component based on recent price history."""
        if len(self.price_history) < 10:
            return 0.0
        
        # Calculate short-term momentum
        recent_prices = self.price_history[-10:]
        price_changes = np.diff(recent_prices) / recent_prices[:-1]
        momentum = np.mean(price_changes)
        
        # Apply momentum with some persistence
        return momentum * 0.1
    
    def _update_volatility(self):
        """Update volatility with clustering effects."""
        # Volatility clustering: high volatility tends to persist
        volatility_persistence = 0.95
        volatility_shock = np.random.normal(0, 0.01)
        
        # Event-based volatility changes
        if self.current_event == MarketEvent.VOLATILITY_SPIKE:
            self.current_volatility = min(self.volatility_range[1], 
                                        self.current_volatility * self.event_intensity)
        elif self.current_event == MarketEvent.LIQUIDITY_CRISIS:
            self.current_volatility = min(self.volatility_range[1], 
                                        self.current_volatility * 1.5)
        else:
            # Normal volatility evolution
            self.current_volatility = (volatility_persistence * self.current_volatility + 
                                     volatility_shock)
            self.current_volatility = max(self.volatility_range[0], 
                                        min(self.volatility_range[1], 
                                            self.current_volatility))
    
    def _update_liquidity(self):
        """Update liquidity based on market conditions."""
        # Liquidity is inversely related to volatility
        volatility_factor = 1.0 / (1.0 + self.current_volatility * 10)
        
        # Event-based liquidity changes
        if self.current_event == MarketEvent.LIQUIDITY_CRISIS:
            self.current_liquidity = self.base_liquidity * self.event_intensity
        elif self.current_event == MarketEvent.VOLATILITY_SPIKE:
            self.current_liquidity = self.base_liquidity * 0.5
        else:
            # Normal liquidity evolution
            liquidity_noise = np.random.normal(0, 0.05)
            self.current_liquidity = (self.base_liquidity * volatility_factor + 
                                    liquidity_noise)
        
        # Clamp to valid range
        self.current_liquidity = max(self.liquidity_range[0], 
                                   min(self.liquidity_range[1], 
                                       self.current_liquidity))
    
    def _update_volume(self):
        """Update trading volume based on market conditions."""
        # Base volume
        base_volume = 1000.0
        
        # Volume increases with volatility
        volatility_factor = 1.0 + self.current_volatility * 20
        
        # Volume increases during events
        event_factor = 1.0
        if self.current_event != MarketEvent.NORMAL:
            event_factor = 2.0 + self.event_intensity
        
        # Add some randomness
        volume_noise = np.random.normal(1.0, 0.2)
        
        self.current_volume = base_volume * volatility_factor * event_factor * volume_noise
        self.current_volume = max(100.0, self.current_volume)  # Minimum volume
    
    def _trigger_market_event(self):
        """Trigger a random market event."""
        # Weight events by their likelihood
        event_weights = {
            MarketEvent.VOLATILITY_SPIKE: 0.3,
            MarketEvent.LIQUIDITY_CRISIS: 0.1,
            MarketEvent.NEWS_EVENT: 0.4,
            MarketEvent.FLASH_CRASH: 0.1,
            MarketEvent.FLASH_RALLY: 0.1,
        }
        
        # Select event based on weights
        events = list(event_weights.keys())
        weights = list(event_weights.values())
        self.current_event = np.random.choice(events, p=weights)
        
        # Set event duration and intensity
        duration_range = self.event_durations[self.current_event]
        intensity_range = self.event_intensities[self.current_event]
        
        self.event_duration = random.randint(duration_range[0], duration_range[1])
        self.event_intensity = random.uniform(intensity_range[0], intensity_range[1])
    
    def get_market_conditions(self) -> Dict:
        """Get current market conditions for agents."""
        return {
            "price": self.current_price,
            "volatility": self.current_volatility,
            "liquidity": self.current_liquidity,
            "volume": self.current_volume,
            "event": self.current_event.value,
            "event_duration": self.event_duration,
            "event_intensity": self.event_intensity,
            "price_change": (self.current_price - self.initial_price) / self.initial_price,
            "recent_volatility": self._calculate_recent_volatility(),
        }
    
    def _calculate_recent_volatility(self, window: int = 20) -> float:
        """Calculate recent volatility from price history."""
        if len(self.price_history) < 2:
            return 0.0
        
        recent_prices = self.price_history[-window:]
        if len(recent_prices) < 2:
            return 0.0
        
        returns = np.diff(recent_prices) / recent_prices[:-1]
        return np.std(returns) if len(returns) > 0 else 0.0
    
    def reset(self):
        """Reset the market simulator to initial state."""
        self.current_price = self.initial_price
        self.current_volatility = self.base_volatility
        self.current_liquidity = self.base_liquidity
        self.current_volume = 0.0
        self.current_event = MarketEvent.NORMAL
        self.event_duration = 0
        self.event_intensity = 0.0
        self.price_history = [self.initial_price]

