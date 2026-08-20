#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch


SHAPES = (
    (18432, 2048, 2048),
    (18432, 8192, 2048),
    (18432, 2048, 8192),
)


def benchmark(m: int, n: int, k: int, repeats: int) -> dict[str, object]:
    left = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    right = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)

    def mm(a, b):
        return torch.mm(a, b)

    started = time.perf_counter()
    compiled = torch.compile(mm, fullgraph=True, dynamic=False)
    output = compiled(left, right)
    torch.cuda.synchronize()
    compile_seconds = time.perf_counter() - started
    for _ in range(3):
        output = compiled(left, right)
    torch.cuda.synchronize()

    timings = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = compiled(left, right)
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
    if not torch.isfinite(output).all().item():
        raise RuntimeError(f"non-finite GEMM output for M={m}, N={n}, K={k}")
    return {
        "m": m,
        "n": n,
        "k": k,
        "dtype": "bfloat16",
        "compile_seconds": compile_seconds,
        "median_ms": statistics.median(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
        "samples_ms": timings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the dominant Anima BF16 GEMMs for Origami selection experiments.")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    results = [benchmark(*shape, args.repeats) for shape in SHAPES]
    print(json.dumps({
        "device": properties.name,
        "gcn_arch": getattr(properties, "gcnArchName", ""),
        "reported_compute_units": properties.multi_processor_count,
        "origami_modeled_physical_cus": properties.multi_processor_count * 2,
        "torch": torch.__version__,
        "rocm": torch.version.hip,
        "results": results,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
