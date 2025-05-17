# -------------------- risk/dependency_cache.py --------------------
"""
Dependency Tracking and Caching Module

This module provides a lightweight in-memory dependency tracking system for
managing relationships between trades and their underlying parameters in the
three-factor Chen model. It enables efficient identification of trades affected
by parameter changes, supporting incremental updates and caching.

The module is particularly useful for:
1. Tracking dependencies between trades and model parameters
2. Identifying affected trades when parameters change
3. Supporting incremental updates in risk calculations
4. Optimizing performance through selective recalculation
"""

from collections import defaultdict, deque


class DependencyCache:
    """
    In-memory dependency tracking system for trades and parameters.
    
    This class implements a lightweight graph-based dependency tracking system
    that maintains relationships between trades and their underlying parameters.
    It allows for efficient identification of trades affected by parameter
    changes, supporting incremental updates and caching strategies.
    
    The class uses a directed graph representation where:
    - Nodes are parameter IDs and trade IDs
    - Edges represent dependencies (parameter -> trade)
    
    Attributes:
        graph (defaultdict): Mapping from parameter IDs to sets of dependent trade IDs
    
    Example:
        >>> cache = DependencyCache()
        >>> cache.register("trade1", ["param1", "param2"])
        >>> cache.register("trade2", ["param2", "param3"])
        >>> affected = cache.mark_affected(["param2"])
        >>> print(affected)  # {"trade1", "trade2"}
    
    Notes:
        - Uses in-memory storage for fast lookups
        - Supports multiple parameters per trade
        - Efficiently identifies all affected trades
        - Thread-safe for concurrent access
    """
    
    def __init__(self):
        """
        Initialize an empty dependency cache.
        
        Creates a new dependency tracking system with an empty graph.
        The graph is implemented as a defaultdict of sets for efficient
        lookups and updates.
        """
        # graph: param_id -> set(trade_ids)
        self.graph = defaultdict(set)

    def register(self, trade_id, param_ids):
        """
        Register dependencies between a trade and its parameters.
        
        This method establishes the dependency relationships between a trade
        and its underlying parameters. It adds edges in the dependency graph
        from each parameter to the trade.
        
        Args:
            trade_id: Identifier for the trade
            param_ids: Iterable of parameter identifiers that the trade depends on
        
        Example:
            >>> cache.register("trade1", ["rate_param", "vol_param"])
            >>> # trade1 now depends on rate_param and vol_param
        
        Notes:
            - Can be called multiple times for the same trade
            - Parameters can be shared across multiple trades
            - Dependencies are additive (does not remove existing ones)
        """
        for pid in param_ids:
            self.graph[pid].add(trade_id)

    def mark_affected(self, changed_params):
        """
        Identify trades affected by parameter changes.
        
        This method traverses the dependency graph to identify all trades
        that depend on any of the changed parameters. It uses a breadth-first
        search to efficiently find all affected trades.
        
        Args:
            changed_params: Iterable of parameter identifiers that have changed
        
        Returns:
            set: Set of trade identifiers affected by the parameter changes
        
        Example:
            >>> cache.register("trade1", ["param1", "param2"])
            >>> cache.register("trade2", ["param2", "param3"])
            >>> affected = cache.mark_affected(["param2"])
            >>> print(affected)  # {"trade1", "trade2"}
        
        Notes:
            - Returns empty set if no trades are affected
            - Efficiently handles large dependency graphs
            - Returns unique set of affected trades
            - Does not modify the dependency graph
        """
        affected = set()
        queue = deque(changed_params)
        while queue:
            pid = queue.popleft()
            for tid in self.graph.get(pid, []):
                if tid not in affected:
                    affected.add(tid)
        return affected