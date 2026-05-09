from __future__ import annotations
from typing import List, Optional

from env import DeliveryEnv, Order
from solver import Solver, default_result


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
