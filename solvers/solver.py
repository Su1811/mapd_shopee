from __future__ import annotations

import collections
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from env import (
    DeliveryEnv,
    Order,
)


def default_result(method: str, cfg: dict, orders: List[Order], message: str = "Chưa cài đặt") -> dict:
    total_orders = len(orders)
    return {
        "method": method,
        "config_name": cfg.get("name", ""),
        "total_orders": total_orders,
        "delivered": 0,
        "on_time": 0,
        "late": 0,
        "missed": total_orders,
        "delivery_rate": 0.0,
        "on_time_rate": 0.0,
        "total_reward": 0.0,
        "total_movecost": 0.0,
        "net_reward": 0.0,
        "elapsed_sec": 0.0,
        "shipper_rewards": [],
        "status": message,
    }


class Solver(ABC):
    """Abstract solver base class for delivery strategies."""

    def __init__(self, env_or_cfg, grid: Optional[List[List[int]]] = None, orders: Optional[List[Order]] = None):
        if isinstance(env_or_cfg, DeliveryEnv):
            self.env = env_or_cfg
            self.cfg = env_or_cfg.cfg
            self.grid = env_or_cfg.grid
            self.orders = env_or_cfg.clone_orders()
        else:
            self.env = None
            self.cfg = env_or_cfg
            self.grid = grid if grid is not None else self.cfg["grid"]
            self.orders = orders if orders is not None else []

    @abstractmethod
    def run(self) -> dict:
        raise NotImplementedError


def bfs_path(grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]) -> List[str]:
    """Tìm một đường đi hợp lệ từ start đến goal bằng BFS."""
    if start == goal:
        return []

    n = len(grid)
    q = collections.deque([(start[0], start[1], [])])
    visited = {start}

    for_action = [("U", -1, 0), ("D", 1, 0), ("L", 0, -1), ("R", 0, 1)]

    while q:
        r, c, path = q.popleft()
        for action, dr, dc in for_action:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n and 0 <= nc < n):
                continue
            if grid[nr][nc] == 1 or (nr, nc) in visited:
                continue
            new_path = path + [action]
            if (nr, nc) == goal:
                return new_path
            visited.add((nr, nc))
            q.append((nr, nc, new_path))

    return []
