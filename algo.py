"""
algo.py — File thuật toán sinh viên cần hoàn thiện
==================================================

File này chỉ chứa phần thuật toán.

To-dos: Sinh viên sửa file này để cài đặt các chiến lược điều phối khác nhau.

Chạy thử độc lập:
    python algo.py --config test_config.txt --out results/
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from env import (
    DIRS,
    SEED,
    DeliveryEnv,
    Order,
    Shipper,
    delivery_reward,
    load_config,
    manhattan,
    move_cost,
)


# ============================================================
# Helper kết quả mặc định cho các thuật toán chưa cài đặt
# ============================================================

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


# ============================================================
# Helper tìm đường cơ bản
# ============================================================

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


# ============================================================
# Baseline bắt buộc: Greedy BFS bản đơn giản
# ============================================================

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

    def _move_one_step(self, sh: Shipper, occupied_next: Dict[Tuple[int, int], int]) -> None:
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

            occupied_next: Dict[Tuple[int, int], int] = {}
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


# ============================================================
# Khung thuật toán 2: VRP + OR-Tools
# ============================================================

class VRPOrToolsSolver(Solver):
    """Sinh viên cài đặt thuật toán VRP + OR-Tools tại đây."""

    def __init__(self, env_or_cfg, grid: Optional[List[List[int]]] = None, orders: Optional[List[Order]] = None):
        super().__init__(env_or_cfg, grid, orders)

    def run(self) -> dict:
        # TODO: chuyển nghiệm VRP thành mô phỏng từng bước và trả về dict kết quả.
        return default_result("VRP + OR-Tools", self.cfg, self.orders)


# ============================================================
# Khung thuật toán 3: Ant Colony Optimization
# ============================================================

class ACOSolver(Solver):
    """Sinh viên cài đặt thuật toán Ant Colony Optimization tại đây."""

    def __init__(self, env_or_cfg, grid: Optional[List[List[int]]] = None, orders: Optional[List[Order]] = None):
        super().__init__(env_or_cfg, grid, orders)
        self.num_ants = 20
        self.num_iterations = 50
        self.evaporation_rate = 0.2


    def run(self) -> dict:
        # TODO: chạy ACO, lấy nghiệm tốt nhất, mô phỏng và trả về dict kết quả.
        return default_result("Ant Colony Optimization", self.cfg, self.orders)


# ============================================================
# Khung thuật toán 4: MAPD-CBS
# ============================================================

class MAPDCBSSolver(Solver):
    """Sinh viên cài đặt MAPD với Conflict-Based Search tại đây."""

    def __init__(self, env_or_cfg, grid: Optional[List[List[int]]] = None, orders: Optional[List[Order]] = None):
        super().__init__(env_or_cfg, grid, orders)

    def run(self) -> dict:
        # TODO: sinh task, chạy CBS, mô phỏng và trả về dict kết quả.
        return default_result("MAPD-CBS", self.cfg, self.orders)


# ============================================================
# Chạy thử độc lập
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MAPD student algorithm file")
    parser.add_argument("--config", required=True, help="Đường dẫn file config")
    parser.add_argument("--out", default="results", help="Thư mục lưu kết quả")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    configs = load_config(args.config)

    solver_classes = [GreedyBFS, VRPOrToolsSolver, ACOSolver, MAPDCBSSolver]
    results = []
    for cfg in configs:
        env = DeliveryEnv(cfg, rng)
        cfg_results = []
        for solver_cls in solver_classes:
            solver = solver_cls(env)
            result = solver.run()
            cfg_results.append(result)
            results.append(result)
            print(
                f"[{cfg['name']}] {result['method']}: "
                f"net_reward={result['net_reward']:.2f}, "
                f"delivered={result['delivered']}/{result['total_orders']}"
            )

        out_path = os.path.join(args.out, f"result_{cfg['name']}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"config_name": cfg.get("name", ""), "results": cfg_results}, f, ensure_ascii=False, indent=2)

    methods = sorted({r["method"] for r in results})
    summary = {
        "total_score_by_method": {
            m: sum(r["net_reward"] for r in results if r["method"] == m)
            for m in methods
        },
        "results": results,
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Tổng net reward theo phương pháp:")
    for method, score in summary["total_score_by_method"].items():
        print(f"- {method}: {score:.2f}")


if __name__ == "__main__":
    main()
