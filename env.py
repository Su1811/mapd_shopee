"""
env.py — MAPD Extended Environment / Test Setup
================================================
File này chứa toàn bộ phần setup cố định của bài toán:
- dataclass Order, Shipper
- hằng số reward/cost
- load_config cho định dạng [CONFIG]...[MAP]...[END]
- sinh đơn hàng theo Poisson không đồng nhất, surge window, hotspot

Sinh viên KHÔNG sửa file này khi nộp thuật toán, chỉ nộp file algo.py
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

SEED = 42
TIME_UNIT_PER_HOUR = 10
TIME_UNIT_PER_DAY = 240

ALPHA = {1: 1.0, 2: 2.0, 3: 3.0}
BETA = {1: 0.1, 2: 0.3, 3: 0.5}
GAMMA = 1.0

HOTSPOT_RADIUS = 3
HOTSPOT_PROB = 0.7

DIRS = {
    "S": (0, 0),
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}


@dataclass
class Order:
    """Đơn hàng g_i = <sx, sy, ex, ey, et, w, p>. Tọa độ dùng 0-index."""

    id: int
    sx: int
    sy: int
    ex: int
    ey: int
    et: int
    w: float
    p: int
    appear_t: int

    picked: bool = False
    delivered: bool = False
    carrier: int = -1
    deliver_t: int = -1


@dataclass
class Shipper:
    """Trạng thái shipper trong mô phỏng."""

    id: int
    r: int
    c: int
    W_max: float
    K_max: int

    bag: List[int] = field(default_factory=list)
    path: List[str] = field(default_factory=list)
    target_oid: int = -1
    phase: str = "idle"  # idle | pickup | deliver
    total_reward: float = 0.0
    steps_moved: int = 0

    def w_carried(self, orders: Dict[int, Order]) -> float:
        return sum(orders[oid].w for oid in self.bag if oid in orders)

    def can_pickup(self, order: Order, orders: Dict[int, Order]) -> bool:
        return len(self.bag) < self.K_max and self.w_carried(orders) + order.w <= self.W_max


def manhattan(r1: int, c1: int, r2: int, c2: int) -> int:
    return abs(r1 - r2) + abs(c1 - c2)


def r_base(w: float) -> float:
    r0 = 10.0
    if w <= 0.2:
        return r0 * 0.4
    if w <= 3.0:
        return r0 * 1.0
    if w <= 10.0:
        return r0 * 1.5
    if w <= 30.0:
        return r0 * 2.0
    return r0 * 3.0


def delivery_reward(order: Order, t_delivery: int, T: int) -> float:
    rb = r_base(order.w)
    if t_delivery <= order.et:
        bonus = max(0.0, (order.et - t_delivery) / max(order.et, 1))
        return ALPHA[order.p] * rb * (1.0 + bonus)
    factor = max(0.0, 1.0 - (t_delivery - order.et) / max(T, 1))
    return BETA[order.p] * rb * factor


def move_cost(w_carried: float, w_max: float) -> float:
    return -0.01 * (1.0 + GAMMA * w_carried / max(w_max, 1.0))


def parse_grid(lines: List[str], idx: int, N: int) -> Tuple[List[List[int]], int]:
    grid: List[List[int]] = []
    for row_i in range(N):
        if idx + row_i >= len(lines):
            raise ValueError(f"Thiếu dòng bản đồ thứ {row_i + 1}/{N}.")
        row = list(map(int, lines[idx + row_i].split()))
        if len(row) != N:
            raise ValueError(f"Dòng bản đồ {idx + row_i + 1} phải có đúng {N} cột.")
        grid.append(row)
    return grid, idx + N


def _parse_int_pairs(value: str, key: str) -> List[Tuple[int, int]]:
    nums = list(map(int, value.split())) if value.strip() else []
    if len(nums) % 2 != 0:
        raise ValueError(f"{key} phải là danh sách cặp số nguyên.")
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def _normalize_shipper_list(cfg: dict, key: str, typ):
    if key not in cfg:
        raise ValueError(f"Thiếu {key} trong config {cfg.get('name', '')}.")
    values = cfg[key]
    if len(values) == cfg["C"]:
        return values
    if len(values) == 1:
        return [typ(values[0])] * cfg["C"]
    raise ValueError(f"{key} phải có đúng C={cfg['C']} phần tử hoặc 1 phần tử để broadcast.")


def load_config(filepath: str) -> List[dict]:
    """
    Đọc file config theo cấu trúc:
        [CONFIG]
        name = C1
        N = 5
        C = 2
        G = 15
        T = 240
        K_max = 3 3
        W_max = 20.0 20.0
        lambda0 = 0.08                  # tùy chọn
        surge_amplitude = 3.0           # tùy chọn
        surge_windows = 60 120 200 280  # tùy chọn, cặp start end
        hotspots = 2 2 8 7              # tùy chọn, cặp row col
        [MAP]
        ... N dòng ...
        [END]
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.rstrip() for line in f]

    configs: List[dict] = []
    i = 0
    while i < len(lines):
        token = _strip_comment(lines[i])
        if token != "[CONFIG]":
            i += 1
            continue

        cfg: dict = {}
        i += 1
        while i < len(lines) and _strip_comment(lines[i]) != "[MAP]":
            line = _strip_comment(lines[i])
            if line and "=" in line:
                key, val = [x.strip() for x in line.split("=", 1)]
                if key == "K_max":
                    cfg[key] = list(map(int, val.split()))
                elif key == "W_max":
                    cfg[key] = list(map(float, val.split()))
                elif key in {"N", "C", "G", "T"}:
                    cfg[key] = int(val)
                elif key == "surge_windows":
                    cfg[key] = _parse_int_pairs(val, key)
                elif key == "hotspots":
                    cfg[key] = _parse_int_pairs(val, key)
                elif key in {"surge_amplitude", "lambda0"}:
                    cfg[key] = float(val)
                else:
                    cfg[key] = val
            i += 1

        for key in ["name", "N", "C", "G", "T", "K_max", "W_max"]:
            if key not in cfg:
                raise ValueError(f"Thiếu {key} trong một [CONFIG].")
        cfg["K_max"] = _normalize_shipper_list(cfg, "K_max", int)
        cfg["W_max"] = _normalize_shipper_list(cfg, "W_max", float)

        if i >= len(lines) or _strip_comment(lines[i]) != "[MAP]":
            raise ValueError(f"Config {cfg.get('name')} thiếu [MAP].")
        i += 1
        cfg["grid"], i = parse_grid(lines, i, cfg["N"])

        if i < len(lines) and _strip_comment(lines[i]) == "[END]":
            i += 1
        configs.append(cfg)

    return configs


def _poisson_draw(lam: float, rng: random.Random) -> int:
    """Sinh số nguyên theo Poisson(lam) bằng Knuth, có chia chunk chống underflow."""
    if lam <= 0:
        return 0
    max_chunk = 20.0
    if lam > max_chunk:
        total = 0
        remain = lam
        while remain > 0:
            chunk = min(remain, max_chunk)
            total += _poisson_draw(chunk, rng)
            remain -= chunk
        return total

    limit = math.exp(-lam)
    k = 0
    prod = 1.0
    while True:
        k += 1
        prod *= rng.random()
        if prod <= limit:
            return k - 1


def _stable_seed(name: str, base_seed: int = SEED) -> int:
    digest = hashlib.md5(f"{base_seed}:{name}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _free_cells(grid: List[List[int]]) -> List[Tuple[int, int]]:
    return [(i, j) for i, row in enumerate(grid) for j, val in enumerate(row) if val == 0]


def _resolve_generation_params(cfg: dict) -> dict:
    """
    Chuẩn hóa tham số sinh đơn.

    Config Phase 1 không công bố, env tự sinh tham số ẩn
    """
    params = dict(cfg)
    T, G, N = cfg["T"], cfg["G"], cfg["N"]
    cells = _free_cells(cfg["grid"])

    params.setdefault("lambda0", G / max(T, 1))

    has_public_surge = bool(params.get("surge_windows")) and bool(params.get("hotspots"))
    if has_public_surge:
        params.setdefault("surge_amplitude", 3.0)
        return params

    # Default surge for public Phase 1 configs.
    rng = random.Random(_stable_seed(str(cfg.get("name", "unknown"))))
    if not cells:
        params["surge_windows"] = []
        params["hotspots"] = []
        params.setdefault("surge_amplitude", 0.0)
        return params

    # C1 đơn giản: đúng 1 window + 1 hotspot. Config lớn hơn có thể có 1-2 window.
    # Trong Phase 2 số window có thể tăng lên nhiều
    name = str(cfg.get("name", ""))
    if name == "C1":
        n_windows, n_hotspots, amp = 1, 1, 2.0
    else:
        n_windows = 1 if N <= 10 else 2
        n_hotspots = min(max(1, cfg["C"] // 2), 3)
        amp = 2.5 if N <= 12 else 3.0

    windows: List[Tuple[int, int]] = []
    low = max(1, int(0.15 * T))
    high = max(low + 1, int(0.75 * T))
    duration = max(20, min(T // 5, TIME_UNIT_PER_DAY // 2))
    for _ in range(n_windows):
        start = rng.randint(low, max(low, high - duration))
        windows.append((start, min(T - 1, start + duration)))
    windows.sort()

    hotspots = rng.sample(cells, min(n_hotspots, len(cells)))
    params["surge_windows"] = windows
    params["hotspots"] = hotspots
    params.setdefault("surge_amplitude", amp)
    return params


def gen_orders_fixed(cfg: dict, rng: random.Random) -> List[Order]:
    return _gen_orders(cfg, rng)


def _gen_orders(cfg: dict, rng: random.Random) -> List[Order]:
    cfg = _resolve_generation_params(cfg)
    N, T, G, grid = cfg["N"], cfg["T"], cfg["G"], cfg["grid"]
    lam = float(cfg.get("lambda0", G / max(T, 1)))
    surge_amp = float(cfg.get("surge_amplitude", 3.0))
    surge_windows: List[Tuple[int, int]] = cfg.get("surge_windows", [])
    hotspots: List[Tuple[int, int]] = cfg.get("hotspots", [])

    cells = _free_cells(grid)
    if not cells:
        return []

    orders: List[Order] = []
    oid = 0
    for t in range(T):
        if oid >= G:
            break

        in_surge = any(ts <= t <= te for ts, te in surge_windows)
        effective_lam = lam * (1.0 + surge_amp) if in_surge else lam
        n_new = _poisson_draw(effective_lam, rng)

        for _ in range(n_new):
            if oid >= G:
                break

            src = None
            if in_surge and hotspots and rng.random() < HOTSPOT_PROB:
                center = rng.choice(hotspots)
                nearby = [
                    cell for cell in cells
                    if manhattan(cell[0], cell[1], center[0], center[1]) <= HOTSPOT_RADIUS
                ]
                src = rng.choice(nearby) if nearby else rng.choice(cells)
            if src is None:
                src = rng.choice(cells)

            candidates = [cell for cell in cells if cell != src]
            dst = rng.choice(candidates if candidates else cells)
            priority = rng.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
            weight = rng.choices([0.1, 1.0, 5.0, 15.0, 40.0], weights=[0.2, 0.4, 0.25, 0.1, 0.05])[0]

            hours_slack = rng.randint(1, 6) * (4 - priority)
            deadline = min(t + hours_slack * TIME_UNIT_PER_HOUR, T - 1)
            orders.append(Order(
                id=oid,
                sx=src[0], sy=src[1],
                ex=dst[0], ey=dst[1],
                et=deadline,
                w=weight,
                p=priority,
                appear_t=t,
            ))
            oid += 1

    return orders


class DeliveryEnv:

    def __init__(self, cfg: dict, rng: random.Random | None = None):
        self.cfg = cfg
        self.grid = cfg["grid"]
        self.N = cfg["N"]
        self.C = cfg["C"]
        self.G = cfg["G"]
        self.T = cfg["T"]
        self.rng = rng if rng is not None else random.Random(SEED)
        self.orders = gen_orders_fixed(cfg, self.rng)

    def clone_orders(self) -> List[Order]:
        """Tạo bản copy trạng thái sạch để nhiều thuật toán chạy trên cùng config."""
        return [Order(o.id, o.sx, o.sy, o.ex, o.ey, o.et, o.w, o.p, o.appear_t) for o in self.orders]
