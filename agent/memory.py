"""
Memory System for Finance Agent
==============================

This module implements both short-term and long-term memory systems:
- ShortTermMemory: Stores current session state and context
- LongTermMemory: Stores learned patterns, successful strategies, and lessons
"""

import json
import time
import pickle
import os
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


@dataclass
class MemoryEntry:
    """Base class for memory entries"""
    timestamp: float
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        return cls(**data)


@dataclass
class StrategyEntry(MemoryEntry):
    """Entry for successful trading strategies"""
    symbol: str
    strategy_type: str
    performance_metrics: Dict[str, float]
    conditions: Dict[str, Any]
    success_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LessonEntry(MemoryEntry):
    """Entry for learned lessons and patterns"""
    lesson_type: str
    description: str
    context: Dict[str, Any]
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ShortTermMemory:
    """
    Short-term memory for current session state
    Stores temporary data, current context, and session variables
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.memory: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self.session_start = time.time()
    
    def store(self, key: str, value: Any, metadata: Dict[str, Any] = None) -> None:
        """
        Store a value in short-term memory
        
        Args:
            key: Memory key
            value: Value to store
            metadata: Optional metadata
        """
        self.memory[key] = {
            'value': value,
            'metadata': metadata or {},
            'timestamp': time.time()
        }
        self.access_times[key] = time.time()
        
        # Clean up old entries if memory is full
        if len(self.memory) > self.max_size:
            self._cleanup_old_entries()
    
    def retrieve(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value from short-term memory
        
        Args:
            key: Memory key
            default: Default value if key not found
        
        Returns:
            Stored value or default
        """
        if key in self.memory:
            self.access_times[key] = time.time()
            return self.memory[key]['value']
        return default
    
    def retrieve_with_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve value with metadata
        
        Args:
            key: Memory key
        
        Returns:
            Dictionary with value and metadata, or None
        """
        if key in self.memory:
            self.access_times[key] = time.time()
            return self.memory[key]
        return None
    
    def exists(self, key: str) -> bool:
        """Check if key exists in memory"""
        return key in self.memory
    
    def remove(self, key: str) -> bool:
        """
        Remove a key from memory
        
        Args:
            key: Memory key
        
        Returns:
            True if key was removed, False if not found
        """
        if key in self.memory:
            del self.memory[key]
            del self.access_times[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all memory"""
        self.memory.clear()
        self.access_times.clear()
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information"""
        return {
            'session_duration': time.time() - self.session_start,
            'memory_size': len(self.memory),
            'keys': list(self.memory.keys()),
            'oldest_entry': min(self.access_times.values()) if self.access_times else None,
            'newest_entry': max(self.access_times.values()) if self.access_times else None
        }
    
    def _cleanup_old_entries(self) -> None:
        """Remove oldest entries when memory is full"""
        if len(self.memory) <= self.max_size:
            return
        
        # Sort by access time and remove oldest
        sorted_keys = sorted(self.access_times.keys(), key=lambda k: self.access_times[k])
        keys_to_remove = sorted_keys[:len(self.memory) - self.max_size + 10]  # Remove 10 extra
        
        for key in keys_to_remove:
            self.remove(key)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary"""
        return {
            'memory': self.memory,
            'access_times': self.access_times,
            'session_start': self.session_start,
            'max_size': self.max_size
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load memory from dictionary"""
        self.memory = data.get('memory', {})
        self.access_times = data.get('access_times', {})
        self.session_start = data.get('session_start', time.time())
        self.max_size = data.get('max_size', 1000)


class LongTermMemory:
    """
    Long-term memory for persistent learning
    Stores successful strategies, lessons learned, and patterns
    """
    
    def __init__(self, storage_path: str = "memory_storage"):
        self.storage_path = storage_path
        self.strategies: List[StrategyEntry] = []
        self.lessons: List[LessonEntry] = []
        self.patterns: Dict[str, Any] = {}
        self.performance_history: List[Dict[str, Any]] = []
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_path, exist_ok=True)
        
        # Load existing data
        self._load_from_storage()
    
    def store_strategy(self, symbol: str, strategy_type: str, 
                      performance_metrics: Dict[str, float], 
                      conditions: Dict[str, Any], 
                      success_rate: float,
                      metadata: Dict[str, Any] = None) -> None:
        """
        Store a successful trading strategy
        
        Args:
            symbol: Stock symbol
            strategy_type: Type of strategy
            performance_metrics: Performance metrics (MAE, Sharpe, etc.)
            conditions: Market conditions when strategy worked
            success_rate: Success rate of the strategy
            metadata: Additional metadata
        """
        entry = StrategyEntry(
            timestamp=time.time(),
            content={
                'symbol': symbol,
                'strategy_type': strategy_type,
                'performance_metrics': performance_metrics,
                'conditions': conditions,
                'success_rate': success_rate
            },
            metadata=metadata or {},
            symbol=symbol,
            strategy_type=strategy_type,
            performance_metrics=performance_metrics,
            conditions=conditions,
            success_rate=success_rate
        )
        
        self.strategies.append(entry)
        self._save_to_storage()
    
    def store_lesson(self, lesson_type: str, description: str, 
                    context: Dict[str, Any], confidence: float,
                    metadata: Dict[str, Any] = None) -> None:
        """
        Store a learned lesson
        
        Args:
            lesson_type: Type of lesson (e.g., 'market_pattern', 'model_performance')
            description: Description of the lesson
            context: Context when lesson was learned
            confidence: Confidence in the lesson (0-1)
            metadata: Additional metadata
        """
        entry = LessonEntry(
            timestamp=time.time(),
            content={
                'lesson_type': lesson_type,
                'description': description,
                'context': context,
                'confidence': confidence
            },
            metadata=metadata or {},
            lesson_type=lesson_type,
            description=description,
            context=context,
            confidence=confidence
        )
        
        self.lessons.append(entry)
        self._save_to_storage()
    
    def store_pattern(self, pattern_name: str, pattern_data: Any, 
                     confidence: float = 1.0) -> None:
        """
        Store a learned pattern
        
        Args:
            pattern_name: Name of the pattern
            pattern_data: Pattern data
            confidence: Confidence in the pattern
        """
        self.patterns[pattern_name] = {
            'data': pattern_data,
            'confidence': confidence,
            'timestamp': time.time()
        }
        self._save_to_storage()
    
    def record_performance(self, symbol: str, model_type: str, 
                          metrics: Dict[str, float], 
                          context: Dict[str, Any]) -> None:
        """
        Record model performance for learning
        
        Args:
            symbol: Stock symbol
            model_type: Type of model used
            metrics: Performance metrics
            context: Context of the prediction
        """
        performance_entry = {
            'timestamp': time.time(),
            'symbol': symbol,
            'model_type': model_type,
            'metrics': metrics,
            'context': context
        }
        
        self.performance_history.append(performance_entry)
        
        # Keep only last 1000 entries
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        
        self._save_to_storage()
    
    def get_similar_strategies(self, symbol: str, conditions: Dict[str, Any], 
                             limit: int = 5) -> List[StrategyEntry]:
        """
        Find similar strategies for a given symbol and conditions
        
        Args:
            symbol: Stock symbol
            conditions: Current market conditions
            limit: Maximum number of strategies to return
        
        Returns:
            List of similar strategies
        """
        # Simple similarity based on symbol and basic conditions
        similar = []
        for strategy in self.strategies:
            if strategy.symbol == symbol:
                # Add simple condition matching here
                # In a more sophisticated implementation, you'd use proper similarity metrics
                similar.append(strategy)
        
        # Sort by success rate and return top results
        similar.sort(key=lambda x: x.success_rate, reverse=True)
        return similar[:limit]
    
    def get_relevant_lessons(self, context: Dict[str, Any], 
                           lesson_type: str = None, limit: int = 5) -> List[LessonEntry]:
        """
        Get relevant lessons for current context
        
        Args:
            context: Current context
            lesson_type: Filter by lesson type
            limit: Maximum number of lessons to return
        
        Returns:
            List of relevant lessons
        """
        relevant = []
        for lesson in self.lessons:
            if lesson_type is None or lesson.lesson_type == lesson_type:
                # Simple relevance check - in practice, you'd use more sophisticated matching
                relevant.append(lesson)
        
        # Sort by confidence and recency
        relevant.sort(key=lambda x: (x.confidence, x.timestamp), reverse=True)
        return relevant[:limit]
    
    def get_pattern(self, pattern_name: str) -> Optional[Dict[str, Any]]:
        """Get a stored pattern"""
        return self.patterns.get(pattern_name)
    
    def get_performance_summary(self, symbol: str = None, 
                              model_type: str = None, 
                              days_back: int = 30) -> Dict[str, Any]:
        """
        Get performance summary
        
        Args:
            symbol: Filter by symbol
            model_type: Filter by model type
            days_back: Number of days to look back
        
        Returns:
            Performance summary
        """
        cutoff_time = time.time() - (days_back * 24 * 60 * 60)
        
        filtered_performance = [
            p for p in self.performance_history
            if p['timestamp'] >= cutoff_time
            and (symbol is None or p['symbol'] == symbol)
            and (model_type is None or p['model_type'] == model_type)
        ]
        
        if not filtered_performance:
            return {'count': 0, 'avg_metrics': {}}
        
        # Calculate average metrics
        all_metrics = [p['metrics'] for p in filtered_performance]
        avg_metrics = {}
        for metric_name in all_metrics[0].keys():
            values = [m[metric_name] for m in all_metrics if metric_name in m]
            if values:
                avg_metrics[metric_name] = np.mean(values)
        
        return {
            'count': len(filtered_performance),
            'avg_metrics': avg_metrics,
            'symbol': symbol,
            'model_type': model_type,
            'days_back': days_back
        }
    
    def _save_to_storage(self) -> None:
        """Save memory to persistent storage"""
        try:
            data = {
                'strategies': [s.to_dict() for s in self.strategies],
                'lessons': [l.to_dict() for l in self.lessons],
                'patterns': self.patterns,
                'performance_history': self.performance_history
            }
            
            with open(os.path.join(self.storage_path, 'long_term_memory.json'), 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Warning: Failed to save long-term memory: {str(e)}")
    
    def _load_from_storage(self) -> None:
        """Load memory from persistent storage"""
        try:
            file_path = os.path.join(self.storage_path, 'long_term_memory.json')
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Load strategies
                self.strategies = [
                    StrategyEntry.from_dict(s) for s in data.get('strategies', [])
                ]
                
                # Load lessons
                self.lessons = [
                    LessonEntry.from_dict(l) for l in data.get('lessons', [])
                ]
                
                # Load patterns and performance history
                self.patterns = data.get('patterns', {})
                self.performance_history = data.get('performance_history', [])
                
        except Exception as e:
            print(f"Warning: Failed to load long-term memory: {str(e)}")
            # Initialize empty if loading fails
            self.strategies = []
            self.lessons = []
            self.patterns = {}
            self.performance_history = []
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            'strategies_count': len(self.strategies),
            'lessons_count': len(self.lessons),
            'patterns_count': len(self.patterns),
            'performance_entries': len(self.performance_history),
            'storage_path': self.storage_path,
            'oldest_strategy': min([s.timestamp for s in self.strategies]) if self.strategies else None,
            'newest_strategy': max([s.timestamp for s in self.strategies]) if self.strategies else None
        }


# Example usage and testing
if __name__ == "__main__":
    print("Testing Memory System...")
    
    # Test ShortTermMemory
    stm = ShortTermMemory()
    stm.store("current_symbol", "PTT.BK", {"source": "user_input"})
    stm.store("model_performance", {"mae": 0.05, "r2": 0.7}, {"timestamp": time.time()})
    
    print(f"✅ ShortTermMemory: Stored and retrieved: {stm.retrieve('current_symbol')}")
    print(f"✅ ShortTermMemory: Session info: {stm.get_session_info()}")
    
    # Test LongTermMemory
    ltm = LongTermMemory("test_memory")
    ltm.store_strategy(
        symbol="PTT.BK",
        strategy_type="momentum",
        performance_metrics={"mae": 0.05, "sharpe": 1.2},
        conditions={"rsi": 30, "volume": "high"},
        success_rate=0.75
    )
    
    ltm.store_lesson(
        lesson_type="market_pattern",
        description="RSI below 30 with high volume often leads to reversal",
        context={"market": "bearish", "sector": "energy"},
        confidence=0.8
    )
    
    print(f"✅ LongTermMemory: Stored strategy and lesson")
    print(f"✅ LongTermMemory: Stats: {ltm.get_memory_stats()}")
    
    # Clean up test files
    import shutil
    if os.path.exists("test_memory"):
        shutil.rmtree("test_memory")
    
    print("Memory system testing completed!")
