"""
env.py — Online MAPD Graph/RL Environment
=========================================

Phiên bản này sửa theo đúng setup bài toán online/RL trên đồ thị:
- Chỉ số lượng đơn G được cố định từ đầu.
- Thuộc tính cụ thể của đơn hàng KHÔNG được sinh trước toàn bộ.
- Ở mỗi thời điểm t, môi trường mới sinh và reveal các đơn xuất hiện tại t.
- Solver chỉ nhận observation hiện tại: bản đồ, shipper states, và các đơn đã xuất hiện nhưng chưa hoàn tất.
- Shipper chịu trách nhiệm mô phỏng thao tác vật lý: kiểm tra vật cản, tính ô kế tiếp, di chuyển, nhặt/giao hàng.

Sinh viên chỉ cần cài solver trong thư mục solvers. Người ra đề/grader dùng file này.
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
    """
    Shipper mô phỏng một người giao hàng trên đồ thị lưới.

    Lớp này gom các thao tác vật lý của shipper:
    - kiểm tra tọa độ có nằm trong map không;
    - kiểm tra ô có phải vật cản không;
    - tính tọa độ mới nếu đi U/D/L/R/S;
    - kiểm tra capacity trước khi nhặt;
    - nhặt và giao đơn nếu đang đứng đúng ô.

    Tọa độ nội bộ dùng 0-index để nhất quán với grid Python.
    """

    id: int
    r: int
    c: int
    W_max: float
    K_max: int

    bag: List[int] = field(default_factory=list)
    total_reward: float = 0.0
    steps_moved: int = 0

    @property
    def position(self) -> Tuple[int, int]:
        return (self.r, self.c)

    def set_position(self, pos: Tuple[int, int]) -> None:
        self.r, self.c = pos

    def is_inside_map(self, pos: Tuple[int, int], grid: List[List[int]]) -> bool:
        r, c = pos
        return 0 <= r < len(grid) and 0 <= c < len(grid[0])

    def is_obstacle(self, pos: Tuple[int, int], grid: List[List[int]]) -> bool:
        if not self.is_inside_map(pos, grid):
            return True
        r, c = pos
        return grid[r][c] == 1

    def is_free_cell(self, pos: Tuple[int, int], grid: List[List[int]]) -> bool:
        return self.is_inside_map(pos, grid) and not self.is_obstacle(pos, grid)

    def next_position(self, move: str) -> Tuple[int, int]:
        dr, dc = DIRS.get(move, (0, 0))
        return (self.r + dr, self.c + dc)

    def valid_next_position(self, move: str, grid: List[List[int]]) -> Tuple[int, int]:
        """Trả về ô đích nếu move hợp lệ, ngược lại đứng yên tại vị trí hiện tại."""
        nxt = self.next_position(move)
        return nxt if self.is_free_cell(nxt, grid) else self.position

    def move_to(self, pos: Tuple[int, int], orders: Dict[int, Order]) -> float:
        """Cập nhật vị trí và trả về move cost nếu thật sự di chuyển."""
        if pos == self.position:
            return 0.0
        cost = move_cost(self.w_carried(orders), self.W_max)
        self.set_position(pos)
        self.total_reward += cost
        self.steps_moved += 1
        return cost

    def w_carried(self, orders: Dict[int, Order]) -> float:
        return sum(orders[oid].w for oid in self.bag if oid in orders)

    def can_pickup(self, order: Order, orders: Dict[int, Order]) -> bool:
        if order.picked or order.delivered:
            return False
        if (order.sx, order.sy) != self.position:
            return False
        return len(self.bag) < self.K_max and self.w_carried(orders) + order.w <= self.W_max

    def pickup_best(self, orders: Dict[int, Order]) -> Optional[int]:
        """Nhặt 1 đơn tốt nhất tại ô hiện tại: ưu tiên p cao, deadline sớm, id nhỏ."""
        candidates = [o for o in orders.values() if self.can_pickup(o, orders)]
        if not candidates:
            return None
        order = min(candidates, key=lambda o: (-o.p, o.et, o.id))
        order.picked = True
        order.carrier = self.id
        self.bag.append(order.id)
        return order.id

    def can_deliver(self, order: Order) -> bool:
        return (order.id in self.bag) and (not order.delivered) and (order.ex, order.ey) == self.position

    def deliver(self, order: Order, t: int, T: int) -> float:
        """Giao đơn nếu hợp lệ và trả về reward; nếu không hợp lệ trả về 0."""
        if not self.can_deliver(order):
            return 0.0
        rew = delivery_reward(order, t, T)
        order.delivered = True
        order.deliver_t = t
        order.carrier = self.id
        self.bag.remove(order.id)
        self.total_reward += rew
        return rew



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
    Đọc file config theo cấu trúc [CONFIG] ... [MAP] ... [END].
    G là tổng số đơn của episode. Chỉ G được cố định từ đầu;
    từng đơn cụ thể sẽ được sinh online trong DeliveryEnv tại thời điểm appear_t.
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


def _stable_seed(name: str, base_seed: int = SEED) -> int:
    digest = hashlib.md5(f"{base_seed}:{name}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _free_cells(grid: List[List[int]]) -> List[Tuple[int, int]]:
    return [(i, j) for i, row in enumerate(grid) for j, val in enumerate(row) if val == 0]


def _resolve_generation_params(cfg: dict, base_seed: int = SEED) -> dict:
    """
    Chuẩn hóa tham số sinh đơn. Phase 1 có thể không công bố surge/hotspot;
    môi trường tự sinh tham số ẩn nhưng solver không nhìn thấy các tham số này.
    """
    params = dict(cfg)
    T, N = cfg["T"], cfg["N"]
    cells = _free_cells(cfg["grid"])

    params.setdefault("lambda0", cfg["G"] / max(T, 1))

    has_public_surge = bool(params.get("surge_windows")) and bool(params.get("hotspots"))
    if has_public_surge:
        params.setdefault("surge_amplitude", 3.0)
        return params

    rng = random.Random(_stable_seed(str(cfg.get("name", "unknown")), base_seed))
    if not cells:
        params["surge_windows"] = []
        params["hotspots"] = []
        params.setdefault("surge_amplitude", 0.0)
        return params

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


def _binomial_draw(n: int, p: float, rng: random.Random) -> int:
    if n <= 0 or p <= 0:
        return 0
    if p >= 1:
        return n
    return sum(1 for _ in range(n) if rng.random() < p)


class DeliveryEnv:
    """
    Stateful online environment.

    API chính:
        env = DeliveryEnv(cfg, seed=42)
        obs = env.reset()
        while not obs["done"]:
            actions = solver_policy(obs)
            obs, reward, done, info = env.step(actions)

    actions: dict[int, tuple[str, Any]]
        move ∈ {"S", "U", "D", "L", "R"}
        cargo_op:
            0 / None / "none"            : không làm gì
            1 / "pickup"                : nhặt 1 đơn tốt nhất tại ô hiện tại
            ("deliver", oid) / "2 oid"  : giao đơn oid đang mang nếu ở đích
    """

    def __init__(self, cfg: dict, seed: int = SEED, rng: Optional[random.Random] = None):
        # Copy cfg để một solver không thể vô tình làm bẩn config dùng cho solver khác.
        # Không sinh sẵn danh sách đơn; chỉ chuẩn bị tham số generator nội bộ.
        cfg = copy.deepcopy(cfg)
        self.raw_cfg = copy.deepcopy(cfg)
        self.cfg = _resolve_generation_params(cfg, seed)
        self.public_cfg = copy.deepcopy(cfg)
        self.grid = copy.deepcopy(cfg["grid"])
        self.N = cfg["N"]
        self.C = cfg["C"]
        self.G = cfg["G"]
        self.T = cfg["T"]
        self.seed = seed
        self._external_rng = rng
        self._base_rng_seed = _stable_seed(str(cfg.get("name", "unknown")), seed)
        self.rng = rng if rng is not None else random.Random(self._base_rng_seed)
        self.free_cells = _free_cells(self.grid)
        self.reset()

    def reset(self) -> dict:
        if self._external_rng is None:
            self.rng = random.Random(self._base_rng_seed)
        self.t = 0
        self.next_order_id = 0
        self.generated_count = 0
        self.orders: Dict[int, Order] = {}
        self.new_orders_last_step: List[int] = []
        self.shippers = self._init_shippers()
        self.total_reward = 0.0
        self.total_movecost = 0.0
        self.delivered = 0
        self.on_time = 0
        self.late = 0
        self._reveal_orders_at_current_time()
        return self.observe()

    def _init_shippers(self) -> List[Shipper]:
        starts = self._spread_start_positions(self.C)
        return [
            Shipper(i, starts[i][0], starts[i][1], float(self.cfg["W_max"][i]), int(self.cfg["K_max"][i]))
            for i in range(self.C)
        ]

    def _spread_start_positions(self, c: int) -> List[Tuple[int, int]]:
        if not self.free_cells:
            raise ValueError("Bản đồ không có ô trống.")
        selected: List[Tuple[int, int]] = []
        candidates = list(self.free_cells)
        preferred = [(0, 0), (0, self.N - 1), (self.N - 1, 0), (self.N - 1, self.N - 1), (self.N // 2, self.N // 2)]
        for p in preferred:
            nearest = min(candidates, key=lambda x: manhattan(x[0], x[1], p[0], p[1]))
            if nearest not in selected:
                selected.append(nearest)
            if len(selected) == c:
                return selected
        while len(selected) < c:
            best = max(candidates, key=lambda x: min(manhattan(x[0], x[1], y[0], y[1]) for y in selected))
            if best not in selected:
                selected.append(best)
            else:
                selected.append(next(cell for cell in candidates if cell not in selected))
        return selected[:c]

    def _in_surge(self, t: int) -> bool:
        return any(ts <= t <= te for ts, te in self.cfg.get("surge_windows", []))

    def _intensity(self, t: int) -> float:
        lam = float(self.cfg.get("lambda0", self.G / max(self.T, 1)))
        amp = float(self.cfg.get("surge_amplitude", 3.0))
        return lam * (1.0 + amp) if self._in_surge(t) else lam

    def _draw_new_order_count(self) -> int:
        remaining_orders = self.G - self.generated_count
        if remaining_orders <= 0:
            return 0
        remaining_steps = self.T - self.t
        if remaining_steps <= 1:
            return remaining_orders

        current_weight = self._intensity(self.t)
        future_weight = sum(self._intensity(tt) for tt in range(self.t, self.T))
        p = current_weight / max(future_weight, 1e-12)
        return _binomial_draw(remaining_orders, p, self.rng)

    def _sample_order(self) -> Order:
        if not self.free_cells:
            raise ValueError("Bản đồ không có ô trống.")

        in_surge = self._in_surge(self.t)
        hotspots: List[Tuple[int, int]] = self.cfg.get("hotspots", [])
        src = None
        if in_surge and hotspots and self.rng.random() < HOTSPOT_PROB:
            center = self.rng.choice(hotspots)
            nearby = [
                cell for cell in self.free_cells
                if manhattan(cell[0], cell[1], center[0], center[1]) <= HOTSPOT_RADIUS
            ]
            src = self.rng.choice(nearby if nearby else self.free_cells)
        if src is None:
            src = self.rng.choice(self.free_cells)

        candidates = [cell for cell in self.free_cells if cell != src]
        dst = self.rng.choice(candidates if candidates else self.free_cells)
        priority = self.rng.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
        weight = self.rng.choices([0.1, 1.0, 5.0, 15.0, 40.0], weights=[0.2, 0.4, 0.25, 0.1, 0.05])[0]

        hours_slack = self.rng.randint(1, 6) * (4 - priority)
        deadline = min(self.t + hours_slack * TIME_UNIT_PER_HOUR, self.T - 1)
        oid = self.next_order_id
        self.next_order_id += 1
        return Order(oid, src[0], src[1], dst[0], dst[1], deadline, weight, priority, self.t)

    def _reveal_orders_at_current_time(self) -> None:
        self.new_orders_last_step = []
        n_new = self._draw_new_order_count()
        for _ in range(n_new):
            order = self._sample_order()
            self.orders[order.id] = order
            self.generated_count += 1
            self.new_orders_last_step.append(order.id)

    def observe(self) -> dict:
        visible_orders = {
            oid: Order(o.id, o.sx, o.sy, o.ex, o.ey, o.et, o.w, o.p, o.appear_t, o.picked, o.delivered, o.carrier, o.deliver_t)
            for oid, o in self.orders.items()
            if not o.delivered
        }
        shippers = [
            Shipper(s.id, s.r, s.c, s.W_max, s.K_max, list(s.bag), s.total_reward, s.steps_moved)
            for s in self.shippers
        ]
        return {
            "t": self.t,
            "N": self.N,
            "C": self.C,
            "G": self.G,
            "T": self.T,
            "grid": self.grid,
            "orders": visible_orders,
            "new_order_ids": list(self.new_orders_last_step),
            "shippers": shippers,
                        "done": self.t >= self.T,
        }

    def _parse_action(self, action: Any) -> Tuple[str, Any]:
        if action is None:
            return "S", 0
        if isinstance(action, str):
            return action if action in DIRS else "S", 0
        if isinstance(action, (tuple, list)) and len(action) >= 1:
            move = action[0] if action[0] in DIRS else "S"
            op = action[1] if len(action) >= 2 else 0
            return move, op
        return "S", 0

    def _inside_free(self, r: int, c: int) -> bool:
        # Tiện ích môi trường; logic kiểm tra vật cản chính nằm trong Shipper.is_free_cell.
        return 0 <= r < self.N and 0 <= c < self.N and self.grid[r][c] == 0

    def _apply_moves(self, moves: Dict[int, str]) -> float:
        """
        Pha di chuyển: mỗi Shipper tự tính ô muốn đi bằng valid_next_position().
        Environment chỉ giải quyết xung đột nhiều shipper muốn vào cùng một ô.
        Shipper id nhỏ hơn được ưu tiên giữ/chiếm ô.
        """
        move_reward = 0.0
        old_positions = {a.id: a.position for a in self.shippers}
        occupied = set(old_positions.values())
        desired: Dict[int, Tuple[int, int]] = {}

        for shipper in self.shippers:
            mv = moves.get(shipper.id, "S")
            desired[shipper.id] = shipper.valid_next_position(mv, self.grid)

        for shipper in sorted(self.shippers, key=lambda x: x.id):
            old = old_positions[shipper.id]
            occupied.discard(old)
            target = desired[shipper.id]

            # Nếu ô đích đã bị shipper ưu tiên hơn chiếm/giữ, shipper này đứng yên.
            if target in occupied:
                target = old

            occupied.add(target)
            cost = shipper.move_to(target, self.orders)
            move_reward += cost

        return move_reward

    def _is_pickup_op(self, op: Any) -> bool:
        return op == 1 or op == "pickup"

    def _deliver_oid(self, op: Any) -> Optional[int]:
        if isinstance(op, (tuple, list)) and len(op) == 2 and op[0] == "deliver":
            return int(op[1])
        if isinstance(op, str) and op.startswith("2 "):
            return int(op.split()[1])
        return None

    def _pickup_for_shipper(self, s: Shipper) -> Optional[int]:
        return s.pickup_best(self.orders)

    def _deliver_for_shipper(self, s: Shipper, oid: int) -> float:
        order = self.orders.get(oid)
        if order is None:
            return 0.0
        was_on_time = self.t <= order.et
        rew = s.deliver(order, self.t, self.T)
        if rew <= 0.0:
            return 0.0
        self.delivered += 1
        if was_on_time:
            self.on_time += 1
        else:
            self.late += 1
        return rew

    def _normalize_actions(self, actions: Any) -> Dict[int, Any]:
        """
        Chấp nhận cả hai format:
        - dict: {shipper_id: (move, cargo_op)}
        - list/tuple: [(move, cargo_op), ...] theo thứ tự shipper id, giống code mẫu zip cũ.
        Shipper thiếu action sẽ mặc định đứng yên và không thao tác hàng.
        """
        if actions is None:
            return {}
        if isinstance(actions, dict):
            return actions
        if isinstance(actions, (list, tuple)):
            return {i: actions[i] for i in range(min(len(actions), len(self.shippers)))}
        return {}

    def step(self, actions: Any) -> Tuple[dict, float, bool, dict]:
        if self.t >= self.T:
            return self.observe(), 0.0, True, self.info()

        actions = self._normalize_actions(actions)
        parsed = {sid: self._parse_action(action) for sid, action in actions.items()}
        moves = {sid: mv for sid, (mv, _) in parsed.items()}
        reward = self._apply_moves(moves)
        self.total_movecost += reward

        # Cargo operation after movement.
        for s in sorted(self.shippers, key=lambda x: x.id):
            _, op = parsed.get(s.id, ("S", 0))
            if self._is_pickup_op(op):
                self._pickup_for_shipper(s)
            else:
                oid = self._deliver_oid(op)
                if oid is not None:
                    reward += self._deliver_for_shipper(s, oid)

        self.total_reward += reward - self.total_movecost if False else reward
        self.t += 1
        done = self.t >= self.T
        if not done:
            self._reveal_orders_at_current_time()
        return self.observe(), reward, done, self.info()

    def info(self) -> dict:
        missed = self.G - self.delivered
        return {
            "generated": self.generated_count,
            "total_orders": self.G,
            "delivered": self.delivered,
            "on_time": self.on_time,
            "late": self.late,
            "missed": missed,
            "total_reward": self.total_reward,
            "total_movecost": self.total_movecost,
            "net_reward": self.total_reward,
        }

    def render(self) -> None:
        """In trạng thái map: # là vật cản, . là ô trống, A<i> là shipper."""
        canvas: List[List[str]] = [["#" if v == 1 else "." for v in row] for row in self.grid]
        for oid, order in self.orders.items():
            if not order.delivered and not order.picked:
                canvas[order.sx][order.sy] = "P"
            elif order.picked and not order.delivered:
                canvas[order.ex][order.ey] = "D"
        for shipper in self.shippers:
            canvas[shipper.r][shipper.c] = f"A{shipper.id}"
        for row in canvas:
            print("	".join(row))

    def result(self, method: str, elapsed_sec: float = 0.0) -> dict:
        delivered = self.delivered
        missed = self.G - delivered
        return {
            "method": method,
            "config_name": self.raw_cfg.get("name", "unknown"),
            "total_orders": self.G,
            "orders_generated": self.generated_count,
            "delivered": delivered,
            "on_time": self.on_time,
            "late": self.late,
            "missed": missed,
            "delivery_rate": 100.0 * delivered / max(self.G, 1),
            "on_time_rate": 100.0 * self.on_time / max(delivered, 1),
            "total_reward": round(self.total_reward - self.total_movecost, 4),
            "total_movecost": round(self.total_movecost, 4),
            "net_reward": round(self.total_reward, 4),
            "elapsed_sec": round(elapsed_sec, 4),
            "shipper_rewards": [round(s.total_reward, 4) for s in self.shippers],
            "status": "OK",
        }
