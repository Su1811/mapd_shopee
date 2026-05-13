from __future__ import annotations

import heapq
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

from env import Order, Shipper, delivery_reward, manhattan

Move = str
Pos = Tuple[int, int]
DIR_LIST: List[Tuple[Move, int, int]] = [
    ("S", 0, 0),
    ("U", -1, 0),
    ("D", 1, 0),
    ("L", 0, -1),
    ("R", 0, 1),
]


class OnlineGraphPolicySolver:
    """
    Solver mẫu cho môi trường online/RL.

    Điểm quan trọng: solver không nhận env.orders hidden từ đầu. Mỗi vòng lặp:
        obs -> chọn action -> env.step(action) -> obs mới.
    Tập orders trong obs chỉ gồm đơn đã xuất hiện, chưa hoàn tất.
    """

    def __init__(self, env, policy_name: str = "greedy"):
        self.env = env
        self.policy_name = policy_name
        self.N = env.N
        self.grid = env.grid
        self.T = env.T
        self._bfs_cache: Dict[Pos, Dict[Pos, int]] = {}
        self._next_move_cache: Dict[Tuple[Pos, Pos], Move] = {}
        self.recent_sources: deque[Tuple[int, Pos]] = deque(maxlen=120)
        self.source_heat: Dict[Pos, float] = defaultdict(float)
        self.target_by_shipper: Dict[int, Tuple[str, int]] = {}

    # --------------------------- graph shortest path ---------------------------
    def _inside_free(self, r: int, c: int) -> bool:
        return 0 <= r < self.N and 0 <= c < self.N and self.grid[r][c] == 0

    def _dist_map_from(self, src: Pos) -> Dict[Pos, int]:
        if src in self._bfs_cache:
            return self._bfs_cache[src]
        dist = {src: 0}
        q = deque([src])
        while q:
            r, c = q.popleft()
            for _, dr, dc in DIR_LIST[1:]:
                nr, nc = r + dr, c + dc
                if self._inside_free(nr, nc) and (nr, nc) not in dist:
                    dist[(nr, nc)] = dist[(r, c)] + 1
                    q.append((nr, nc))
        self._bfs_cache[src] = dist
        return dist

    def _distance(self, a: Pos, b: Pos) -> int:
        if a == b:
            return 0
        return self._dist_map_from(a).get(b, 10**8)

    def _next_move(self, start: Pos, goal: Pos) -> Move:
        if start == goal:
            return "S"
        key = (start, goal)
        if key in self._next_move_cache:
            return self._next_move_cache[key]
        best_move = "S"
        best_d = self._distance(start, goal)
        for mv, dr, dc in DIR_LIST[1:]:
            nr, nc = start[0] + dr, start[1] + dc
            if not self._inside_free(nr, nc):
                continue
            d = self._distance((nr, nc), goal)
            if d < best_d:
                best_d = d
                best_move = mv
        self._next_move_cache[key] = best_move
        return best_move

    # ----------------------------- online features -----------------------------
    def _update_online_history(self, obs: dict) -> None:
        t = obs["t"]
        orders: Dict[int, Order] = obs["orders"]
        for oid in obs.get("new_order_ids", []):
            o = orders.get(oid)
            if o is None:
                continue
            src = (o.sx, o.sy)
            self.recent_sources.append((t, src))
            self.source_heat[src] += 1.0

    def _local_density(self, src: Pos, t: int) -> float:
        density = 0.0
        for ts, p in self.recent_sources:
            age = max(0, t - ts)
            if age > 100:
                continue
            d = manhattan(src[0], src[1], p[0], p[1])
            if d <= 3:
                density += (1.0 / (1.0 + d)) * (1.0 / (1.0 + 0.03 * age))
        return density

    def _can_pickup(self, s: Shipper, o: Order, orders: Dict[int, Order]) -> bool:
        carried = sum(orders[oid].w for oid in s.bag if oid in orders)
        return len(s.bag) < s.K_max and carried + o.w <= s.W_max

    def _best_bag_order(self, s: Shipper, orders: Dict[int, Order], t: int) -> Optional[int]:
        best_oid = None
        best_score = -10**18
        pos = (s.r, s.c)
        for oid in s.bag:
            o = orders.get(oid)
            if o is None or o.delivered:
                continue
            d = self._distance(pos, (o.ex, o.ey))
            if d >= 10**8:
                continue
            eta = t + d
            slack = o.et - eta
            rew = delivery_reward(o, min(eta, self.T - 1), self.T)
            score = 55.0 * o.p + 1.5 * rew - 1.2 * d - max(0, -slack)
            if score > best_score:
                best_score = score
                best_oid = oid
        return best_oid

    def _pickup_score(self, s: Shipper, o: Order, orders: Dict[int, Order], t: int) -> float:
        pos = (s.r, s.c)
        d_pick = self._distance(pos, (o.sx, o.sy))
        d_del = self._distance((o.sx, o.sy), (o.ex, o.ey))
        if d_pick >= 10**8 or d_del >= 10**8:
            return -10**18
        eta = t + d_pick + d_del
        rew = delivery_reward(o, min(eta, self.T - 1), self.T)
        slack = o.et - eta
        feasible = 25.0 if slack >= 0 else -min(50.0, -slack)
        route_len = d_pick + d_del
        density = self._local_density((o.sx, o.sy), t)

        if self.policy_name == "greedy":
            return 12.0 * o.p + rew + feasible - 1.25 * route_len - 0.20 * d_pick
        if self.policy_name == "vrp":
            # Online dynamic VRP insertion: maximize reward density per route length.
            return (rew + 15.0 * o.p + feasible) / (1.0 + route_len) - 0.06 * d_pick
        if self.policy_name == "aco":
            # ACO-style: pheromone is estimated only from already observed order sources.
            pheromone = 1.0 + density
            heuristic = (rew + 10.0 * o.p + feasible) / (1.0 + route_len)
            return pheromone * heuristic - 0.04 * d_pick
        if self.policy_name == "cbs":
            # MAPD/CBS-style: avoid assigning multiple shippers to nearby pickup targets.
            conflict_penalty = 0.0
            for sid, (_, other_oid) in self.target_by_shipper.items():
                if sid == s.id:
                    continue
                oo = orders.get(other_oid)
                if oo and manhattan(o.sx, o.sy, oo.sx, oo.sy) <= 1:
                    conflict_penalty += 4.0
            return 12.0 * o.p + rew + feasible - route_len - conflict_penalty
        return rew - route_len

    def _waiting_orders(self, orders: Dict[int, Order]) -> List[Order]:
        return [o for o in orders.values() if (not o.picked) and (not o.delivered)]

    def _reserved_waiting(self) -> set[int]:
        return {oid for phase, oid in self.target_by_shipper.values() if phase == "pickup"}

    def _assign_targets(self, obs: dict) -> None:
        t = obs["t"]
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]

        # Remove completed or invalid targets.
        for sid in list(self.target_by_shipper):
            phase, oid = self.target_by_shipper[sid]
            o = orders.get(oid)
            if o is None or o.delivered:
                self.target_by_shipper.pop(sid, None)
            elif phase == "pickup" and o.picked:
                self.target_by_shipper.pop(sid, None)

        # Bagged orders have priority over new pickups.
        for s in shippers:
            if s.bag:
                oid = self._best_bag_order(s, orders, t)
                if oid is not None:
                    self.target_by_shipper[s.id] = ("deliver", oid)

        reserved = self._reserved_waiting()
        waiting = self._waiting_orders(orders)

        # Assign each idle shipper to one visible order only. No future order is used.
        heap: List[Tuple[float, int, int]] = []
        for s in shippers:
            if s.id in self.target_by_shipper and self.target_by_shipper[s.id][0] == "deliver":
                continue
            for o in waiting:
                if o.id in reserved:
                    continue
                if not self._can_pickup(s, o, orders):
                    continue
                score = self._pickup_score(s, o, orders, t)
                heapq.heappush(heap, (-score, s.id, o.id))

        assigned_s = {sid for sid, (phase, _) in self.target_by_shipper.items() if phase == "pickup"}
        assigned_o = set(reserved)
        shipper_by_id = {s.id: s for s in shippers}
        while heap:
            neg_score, sid, oid = heapq.heappop(heap)
            if sid in assigned_s or oid in assigned_o:
                continue
            s = shipper_by_id[sid]
            o = orders.get(oid)
            if o is None or o.picked or o.delivered or not self._can_pickup(s, o, orders):
                continue
            if -neg_score <= -10**17:
                continue
            self.target_by_shipper[sid] = ("pickup", oid)
            assigned_s.add(sid)
            assigned_o.add(oid)

    def _action_for_shipper(self, s: Shipper, orders: Dict[int, Order]) -> Tuple[str, object]:
        target = self.target_by_shipper.get(s.id)
        if target is None:
            return "S", 0
        phase, oid = target
        o = orders.get(oid)
        if o is None or o.delivered:
            self.target_by_shipper.pop(s.id, None)
            return "S", 0

        if phase == "deliver":
            goal = (o.ex, o.ey)
            mv = self._next_move((s.r, s.c), goal)
            op = ("deliver", oid) if (s.r, s.c) == goal else 0
            return mv, op

        goal = (o.sx, o.sy)
        mv = self._next_move((s.r, s.c), goal)
        op = "pickup" if (s.r, s.c) == goal else 0
        return mv, op

    def act(self, obs: dict) -> Dict[int, Tuple[str, object]]:
        self._update_online_history(obs)
        self._assign_targets(obs)
        orders: Dict[int, Order] = obs["orders"]
        actions: Dict[int, Tuple[str, object]] = {}
        for s in obs["shippers"]:
            actions[s.id] = self._action_for_shipper(s, orders)
        return actions

    def run(self) -> dict:
        start = time.time()
        obs = self.env.observe()
        done = obs.get("done", False)
        while not done:
            actions = self.act(obs)
            obs, _, done, _ = self.env.step(actions)
        return self.env.result(self.__class__.__name__, time.time() - start)
