# -------------------- risk/dependency_cache.py --------------------
"""Lightweight dependency cache using in-memory graph."""
from collections import defaultdict, deque

class DependencyCache:
    def __init__(self):
        # graph: param_id -> set(trade_ids)
        self.graph = defaultdict(set)

    def register(self, trade_id, param_ids):
        """Map a trade to the parameters it depends on."""
        for pid in param_ids:
            self.graph[pid].add(trade_id)

    def mark_affected(self, changed_params):
        """Return set of trade_ids affected by changed_params."""
        affected = set()
        queue = deque(changed_params)
        while queue:
            pid = queue.popleft()
            for tid in self.graph.get(pid, []):
                if tid not in affected:
                    affected.add(tid)
        return affected