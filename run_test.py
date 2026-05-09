"""
run_test.py — Test Runner (KHÔNG SỬA FILE NÀY)
================================================
Script cố định dùng để chấm điểm tự động.

Kiến trúc:
- env.py chịu trách nhiệm load config và sinh đơn hàng/surge/hotspot.
- algo.py chứa các thuật toán của sinh viên.
- Runner chạy tất cả solver được khai báo trong SOLVER_CLASS_NAMES.

Cách dùng:
    python run_test.py --config test_config.txt --out results/
    python run_test.py --config test_config_phase2.txt --out results_phase2/
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from typing import Any

from env import DeliveryEnv, SEED, load_config

MAX_TOTAL_SECONDS = 3600

SOLVER_CLASS_NAMES = [
    "GreedyBFS",
    "VRPOrToolsSolver",
    "ACOSolver",
    "MAPDCBSSolver",
]


def load_algo():
    spec = importlib.util.spec_from_file_location("algo", "algo.py")
    if spec is None or spec.loader is None:
        sys.exit("[ERROR] Không tìm thấy algo.py trong thư mục hiện tại.")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def score_result(result: dict) -> float:
    return float(result.get("net_reward", 0.0))


def _error_result(method: str, cfg: dict, total_orders: int, error: str) -> dict:
    return {
        "method": method,
        "config_name": cfg.get("name", "unknown"),
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
        "status": "ERROR",
        "error": error,
    }


def _run_solver(solver_cls: Any, env: DeliveryEnv) -> dict:
    try:
        solver = solver_cls(env)
    except TypeError:
        solver = solver_cls(env.cfg, env.grid, env.clone_orders())
    return solver.run()


def _collect_solver_classes(algo) -> list[tuple[str, Any]]:
    solvers = []
    for class_name in SOLVER_CLASS_NAMES:
        solver_cls = getattr(algo, class_name, None)
        if solver_cls is not None:
            solvers.append((class_name, solver_cls))
    return solvers


def main():
    parser = argparse.ArgumentParser(description="run_test.py — Test runner chấm điểm")
    parser.add_argument("--config", required=True, help="Đường dẫn file test_config.txt")
    parser.add_argument("--out", default="results", help="Thư mục lưu kết quả")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    print("Đang load algo.py ...")
    algo = load_algo()
    solver_classes = _collect_solver_classes(algo)
    if not solver_classes:
        sys.exit("[ERROR] Không tìm thấy solver nào trong algo.py.")
    print("Load thành công.")
    print("Solver sẽ chạy:", ", ".join(name for name, _ in solver_classes), "\n")

    print(f"Đọc config bằng env.py: {args.config}")
    configs = load_config(args.config)
    print(f"Tìm thấy {len(configs)} config.\n")

    all_results = []
    results_by_config = []
    total_start = time.time()

    for cfg in configs:
        name = cfg.get("name", "unknown")
        remaining = MAX_TOTAL_SECONDS - (time.time() - total_start)
        if remaining <= 0:
            print(f"[TIMEOUT] Đã vượt quá {MAX_TOTAL_SECONDS // 60} phút. Dừng lại.")
            break

        print(f"[{name}] N={cfg['N']} C={cfg['C']} G={cfg['G']} T={cfg['T']}  (còn {remaining / 60:.1f} phút)")
        env = DeliveryEnv(cfg, rng)
        print(f"  Đơn hàng sinh ra bởi env.py: {len(env.orders)}")

        cfg_results = []
        for solver_name, solver_cls in solver_classes:
            solver_start = time.time()
            try:
                result = _run_solver(solver_cls, env)
            except Exception as e:
                result = _error_result(solver_name, cfg, len(env.orders), str(e))

            wall = time.time() - solver_start
            result["wall_sec"] = round(wall, 2)
            result.setdefault("config_name", name)
            result.setdefault("method", solver_name)
            result.setdefault("total_orders", len(env.orders))
            result.setdefault("delivered", 0)
            result.setdefault("on_time", 0)
            result.setdefault("late", 0)
            result.setdefault("missed", result["total_orders"] - result["delivered"])
            result.setdefault("delivery_rate", 0.0)
            result.setdefault("on_time_rate", 0.0)
            result.setdefault("net_reward", 0.0)
            result.setdefault("total_reward", 0.0)
            result.setdefault("total_movecost", 0.0)
            result.setdefault("shipper_rewards", [])

            print(f"  [{result['method']}] Net reward: {result['net_reward']:.2f}")
            print(
                f"    Giao/Tổng: {result['delivered']}/{result['total_orders']}  "
                f"đúng hạn={result['on_time']}  trễ={result['late']}  bỏ lỡ={result['missed']}  "
                f"t={wall:.2f}s"
            )

            cfg_results.append(result)
            all_results.append(result)

        print("")
        config_payload = {
            "config_name": name,
            "orders_generated": len(env.orders),
            "results": cfg_results,
        }
        results_by_config.append(config_payload)
        with open(os.path.join(args.out, f"result_{name}.json"), "w", encoding="utf-8") as f:
            json.dump(config_payload, f, ensure_ascii=False, indent=2)

    total_elapsed = time.time() - total_start
    methods = sorted({r.get("method", "unknown") for r in all_results})
    total_score_by_method = {
        method: round(sum(score_result(r) for r in all_results if r.get("method") == method), 4)
        for method in methods
    }

    print("=" * 95)
    print(f"{'Config':<10} {'Method':<28} {'Net Reward':>12} {'%Giao':>8} {'%Đúng hạn':>10} {'t(s)':>7}")
    print("-" * 95)
    for r in all_results:
        print(
            f"{r['config_name']:<10} {r['method']:<28} {r['net_reward']:>12.2f} "
            f"{r['delivery_rate']:>7.1f}% {r['on_time_rate']:>9.1f}% {r.get('wall_sec', 0):>7.1f}"
        )
    print("=" * 95)
    print("TỔNG ĐIỂM THEO PHƯƠNG PHÁP:")
    for method, score in total_score_by_method.items():
        print(f"- {method}: {score:.2f}")
    print(f"Tổng thời gian chạy: {total_elapsed:.1f}s / {MAX_TOTAL_SECONDS}s")

    summary = {
        "config_file": args.config,
        "seed": args.seed,
        "total_elapsed": round(total_elapsed, 2),
        "total_score_by_method": total_score_by_method,
        "results_by_config": results_by_config,
        "all_results": all_results,
    }
    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    all_results_path = os.path.join(args.out, "all_results.json")
    with open(all_results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nĐã lưu tổng kết vào {summary_path}")
    print(f"Đã lưu toàn bộ kết quả vào {all_results_path}")


if __name__ == "__main__":
    main()
