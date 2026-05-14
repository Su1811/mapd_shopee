from __future__ import annotations
from typing import List, Optional

from env import DeliveryEnv, Order
from solvers.solver import Solver, default_result


class VRPOrToolsSolver(Solver):
    """Sinh viên cài đặt VRP + OR-Tools tại đây."""

    def __init__(self, env_or_cfg, grid: Optional[List[List[int]]] = None, orders: Optional[List[Order]] = None):
        super().__init__(env_or_cfg, grid, orders)

    def run(self) -> dict:
        # TODO: mô hình hóa các đơn đã quan sát thành bài toán VRP động và trả về dict kết quả.
        return default_result("VRP-OrTools", self.cfg, self.orders)
