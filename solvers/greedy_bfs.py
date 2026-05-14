from __future__ import annotations

import time
from collections import deque
from typing import Dict, List, Optional, Tuple

from env import DeliveryEnv, Order, Shipper
from solvers.solver import Solver, default_result


Move = str
Pos = Tuple[int, int]
DIRS: List[Tuple[Move, int, int]] = [
    ("U", -1, 0),
    ("D", 1, 0),
    ("L", 0, -1),
    ("R", 0, 1),
]


class GreedyBFS(Solver):
    """
    Greedy BFS baseline cơ bản.
    """

    def __init__(self, env_or_cfg, grid: Optional[List[List[int]]] = None, orders: Optional[List[Order]] = None):
        super().__init__(env_or_cfg, grid, orders)

    def _inside_free(self, r: int, c: int) -> bool:
        n = len(self.grid)
        return 0 <= r < n and 0 <= c < n and self.grid[r][c] == 0

    def _bfs_next_move(self, start: Pos, goal: Pos) -> Move:
        if start == goal:
            return "S"

        q = deque([start])
        parent: Dict[Pos, Tuple[Optional[Pos], Move]] = {start: (None, "S")}

        while q:
            r, c = q.popleft()
            if (r, c) == goal:
                break
            for mv, dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                nxt = (nr, nc)
                if self._inside_free(nr, nc) and nxt not in parent:
                    parent[nxt] = ((r, c), mv)
                    q.append(nxt)

        if goal not in parent:
            return "S"

        cur = goal
        while parent[cur][0] != start:
            prev = parent[cur][0]
            if prev is None:
                return "S"
            cur = prev
        return parent[cur][1]

    def _bfs_distance(self, start: Pos, goal: Pos) -> int:
        if start == goal:
            return 0
        q = deque([(start, 0)])
        seen = {start}
        while q:
            (r, c), d = q.popleft()
            for _, dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                nxt = (nr, nc)
                if not self._inside_free(nr, nc) or nxt in seen:
                    continue
                if nxt == goal:
                    return d + 1
                seen.add(nxt)
                q.append((nxt, d + 1))
        return 10**9

    def _next_pos_after_move(self, pos: Pos, move: Move) -> Pos:
        for mv, dr, dc in DIRS:
            if mv == move:
                nr, nc = pos[0] + dr, pos[1] + dc
                return (nr, nc) if self._inside_free(nr, nc) else pos
        return pos

    def _can_pickup_from_obs(self, shipper: Shipper, order: Order, orders: Dict[int, Order]) -> bool:
        if order.picked or order.delivered:
            return False
        current_weight = sum(orders[oid].w for oid in shipper.bag if oid in orders)
        return len(shipper.bag) < shipper.K_max and current_weight + order.w <= shipper.W_max

    def _choose_delivery_order(self, shipper: Shipper, orders: Dict[int, Order]) -> Optional[Order]:
        best_order = None
        best_dist = 10**9
        for oid in shipper.bag:
            order = orders.get(oid)
            if order is None or order.delivered:
                continue
            dist = self._bfs_distance((shipper.r, shipper.c), (order.ex, order.ey))
            if dist < best_dist:
                best_dist = dist
                best_order = order
        return best_order

    def _choose_pickup_order(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        reserved_orders: set[int],
    ) -> Optional[Order]:
        best_order = None
        best_key = None
        for order in orders.values():
            if order.id in reserved_orders:
                continue
            if not self._can_pickup_from_obs(shipper, order, orders):
                continue
            dist = self._bfs_distance((shipper.r, shipper.c), (order.sx, order.sy))
            if dist >= 10**9:
                continue
            # Cơ bản: ưu tiên khoảng cách gần, sau đó đơn ưu tiên cao, deadline sớm.
            key = (dist, -order.p, order.et, order.id)
            if best_key is None or key < best_key:
                best_key = key
                best_order = order
        return best_order

    def _decide_actions(self, obs: dict) -> Dict[int, Tuple[Move, object]]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        actions: Dict[int, Tuple[Move, object]] = {}
        reserved_pickups: set[int] = set()

        for shipper in shippers:
            pos = (shipper.r, shipper.c)

            delivery_order = self._choose_delivery_order(shipper, orders)
            if delivery_order is not None:
                goal = (delivery_order.ex, delivery_order.ey)
                move = self._bfs_next_move(pos, goal)
                next_pos = self._next_pos_after_move(pos, move)
                op: object = ("deliver", delivery_order.id) if next_pos == goal else 0
                actions[shipper.id] = (move, op)
                continue

            pickup_order = self._choose_pickup_order(shipper, orders, reserved_pickups)
            if pickup_order is not None:
                reserved_pickups.add(pickup_order.id)
                goal = (pickup_order.sx, pickup_order.sy)
                move = self._bfs_next_move(pos, goal)
                next_pos = self._next_pos_after_move(pos, move)
                op = "pickup" if next_pos == goal else 0
                actions[shipper.id] = (move, op)
                continue

            actions[shipper.id] = ("S", 0)

        return actions

    def run(self) -> dict:
        if self.env is None:
            return default_result("GreedyBFS", self.cfg, self.orders)

        start = time.time()
        obs = self.env.reset()
        done = obs.get("done", False)
        while not done:
            actions = self._decide_actions(obs)
            obs, _, done, _ = self.env.step(actions)
        return self.env.result("GreedyBFS", elapsed_sec=time.time() - start)
