from __future__ import annotations

import random
import time
from typing import Dict, List, Optional

from env import DIRS, DeliveryEnv, Order, Shipper, delivery_reward, manhattan, move_cost, SEED
from solvers.solver import Solver, bfs_path


class GreedyBFS(Solver):
    """
    Baseline Greedy BFS đơn giản
    """

    def __init__(self, env_or_cfg, grid: Optional[List[List[int]]] = None, orders: Optional[List[Order]] = None):
        super().__init__(env_or_cfg, grid, orders)
        order_list = self.orders
        self.orders: Dict[int, Order] = {o.id: o for o in order_list}
        self.N = self.cfg["N"]
        self.T = self.cfg["T"]
        self.rng = random.Random(SEED + 10)

    def _init_shippers(self) -> List[Shipper]:
        free_cells = [
            (r, c)
            for r in range(self.N)
            for c in range(self.N)
            if self.grid[r][c] == 0
        ]
        positions = self.rng.sample(free_cells, min(self.cfg["C"], len(free_cells)))
        shippers: List[Shipper] = []
        for i, (r, c) in enumerate(positions):
            shippers.append(
                Shipper(
                    id=i,
                    r=r,
                    c=c,
                    W_max=self.cfg["W_max"][i],
                    K_max=self.cfg["K_max"][i],
                )
            )
        return shippers

    def _choose_order(self, sh: Shipper, pending: List[Order], t: int) -> Optional[Order]:
        candidates = [o for o in pending if not o.picked and sh.can_pickup(o, self.orders)]
        if not candidates:
            return None

        # Bản baseline chỉ chọn đơn gần nhất, nếu hòa thì ưu tiên deadline sớm hơn.
        return min(
            candidates,
            key=lambda o: (
                manhattan(sh.r, sh.c, o.sx, o.sy),
                o.et,
                -o.p,
                o.id,
            ),
        )

    def _choose_delivery_from_bag(self, sh: Shipper) -> Optional[int]:
        if not sh.bag:
            return None
        return min(sh.bag, key=lambda oid: self.orders[oid].et)

    def _assign_orders(self, shippers: List[Shipper], pending: List[Order], t: int) -> None:
        reserved: set[int] = set()
        for sh in shippers:
            if sh.target_oid >= 0:
                continue
            if sh.bag:
                oid = self._choose_delivery_from_bag(sh)
                if oid is not None:
                    sh.target_oid = oid
                    sh.phase = "deliver"
                    sh.path = []
                continue

            order = self._choose_order(sh, [o for o in pending if o.id not in reserved], t)
            if order is not None:
                sh.target_oid = order.id
                sh.phase = "pickup"
                sh.path = []
                reserved.add(order.id)

    def _move_one_step(self, sh: Shipper, occupied_next: Dict[tuple[int, int], int]) -> None:
        if sh.target_oid < 0:
            return

        order = self.orders[sh.target_oid]
        if sh.phase == "pickup":
            target = (order.sx, order.sy)
        else:
            target = (order.ex, order.ey)

        if not sh.path:
            sh.path = bfs_path(self.grid, (sh.r, sh.c), target)

        move = sh.path.pop(0) if sh.path else "S"
        dr, dc = DIRS[move]
        nr, nc = sh.r + dr, sh.c + dc

        if move != "S" and (nr, nc) not in occupied_next:
            occupied_next[(nr, nc)] = sh.id
            sh.r, sh.c = nr, nc
        else:
            occupied_next[(sh.r, sh.c)] = sh.id
            sh.path = []

    def run(self) -> dict:
        orders = self.orders
        shippers = self._init_shippers()

        orders_by_t: Dict[int, List[Order]] = {}
        for order in orders.values():
            orders_by_t.setdefault(order.appear_t, []).append(order)

        pending: List[Order] = []
        in_transit: List[Order] = []
        delivered: List[Order] = []
        total_reward = 0.0
        total_movecost = 0.0

        t0 = time.time()
        for t in range(self.T):
            pending.extend(orders_by_t.get(t, []))
            self._assign_orders(shippers, pending, t)

            occupied_next: Dict[tuple[int, int], int] = {}
            for sh in shippers:
                old_pos = (sh.r, sh.c)
                old_weight = sh.w_carried(orders)
                self._move_one_step(sh, occupied_next)
                if (sh.r, sh.c) != old_pos:
                    cost = move_cost(old_weight, sh.W_max)
                    sh.total_reward += cost
                    sh.steps_moved += 1
                    total_movecost += cost

            for sh in shippers:
                here = [o for o in pending if o.sx == sh.r and o.sy == sh.c and not o.picked]
                here.sort(key=lambda o: (-o.p, o.et, o.id))
                for order in here:
                    if sh.can_pickup(order, orders):
                        order.picked = True
                        order.carrier = sh.id
                        sh.bag.append(order.id)
                        pending.remove(order)
                        in_transit.append(order)
                        if sh.target_oid == order.id:
                            sh.target_oid = order.id
                            sh.phase = "deliver"
                            sh.path = []

            for sh in shippers:
                deliver_now = [
                    o for o in in_transit
                    if o.carrier == sh.id and o.ex == sh.r and o.ey == sh.c and not o.delivered
                ]
                for order in deliver_now:
                    order.delivered = True
                    order.deliver_t = t
                    reward = delivery_reward(order, t, self.T)
                    sh.total_reward += reward
                    total_reward += reward
                    sh.bag.remove(order.id)
                    delivered.append(order)
                    if sh.target_oid == order.id:
                        sh.target_oid = -1
                        sh.phase = "idle"
                        sh.path = []

        elapsed = time.time() - t0
        on_time = sum(1 for o in delivered if o.deliver_t <= o.et)
        late = len(delivered) - on_time
        missed = len(orders) - len(delivered)

        return {
            "method": "Greedy BFS Simple Baseline",
            "config_name": self.cfg.get("name", ""),
            "total_orders": len(orders),
            "delivered": len(delivered),
            "on_time": on_time,
            "late": late,
            "missed": missed,
            "delivery_rate": round(len(delivered) / max(len(orders), 1) * 100, 2),
            "on_time_rate": round(on_time / max(len(orders), 1) * 100, 2),
            "total_reward": round(total_reward, 4),
            "total_movecost": round(total_movecost, 4),
            "net_reward": round(total_reward + total_movecost, 4),
            "elapsed_sec": round(elapsed, 2),
            "shipper_rewards": [round(sh.total_reward, 4) for sh in shippers],
        }
