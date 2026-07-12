#!/usr/bin/env python3
"""Run isolated Origami product-shape oracle experiments on Windows ROCm.

The workload command is intentionally supplied by the caller.  This keeps the
harness independent of ComfyUI server lifecycle and lets the same command run
with Origami top-k, DEFAULT without Origami, and an EXHAUSTIVE oracle.

Examples:
  python scripts/origami_product_oracle.py run --dry-run -- python workload.py
  python scripts/origami_product_oracle.py run --results-dir user/origami-oracle -- cmd /c run_workload.bat
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ORIGAMI_ENVIRONMENT = (
    "TORCHINDUCTOR_ORIGAMI",
    "TORCHINDUCTOR_ORIGAMI_TOPK",
    "TORCHINDUCTOR_MAX_AUTOTUNE",
    "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM",
    "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE",
    "TORCHINDUCTOR_USE_EXPERIMENTAL_BENCHMARKER",
    "TORCHINDUCTOR_CACHE_DIR",
)
GEMM_KERNEL = re.compile(
    r"triton_[^(=]+?\s*=\s*async_compile\.triton\([^,]+,\s*'''(?P<kernel>.*?)'''",
    re.DOTALL,
)
DIMENSION = re.compile(r"^\s*(?P<name>[MNK])\s*=\s*(?P<value>\d+)\s*$", re.MULTILINE)
DTYPE = re.compile(r"'arg_A': '\*(?P<dtype>[^']+)'", re.IGNORECASE)
ATEN = re.compile(r"Original ATen:\s*\[(?P<aten>[^]]+)\]")


@dataclass(frozen=True)
class Variant:
    name: str
    origami: bool
    search_space: str
    topk: int | None = None

    def environment(self, cache_dir: Path) -> dict[str, str]:
        values = {
            "TORCHINDUCTOR_ORIGAMI": "1" if self.origami else "0",
            "TORCHINDUCTOR_MAX_AUTOTUNE": "1",
            "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM": "1",
            "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE": self.search_space,
            "TORCHINDUCTOR_USE_EXPERIMENTAL_BENCHMARKER": "0",
            "TORCHINDUCTOR_CACHE_DIR": str(cache_dir),
        }
        if self.topk is not None:
            values["TORCHINDUCTOR_ORIGAMI_TOPK"] = str(self.topk)
        return values


@dataclass(frozen=True)
class GemmShape:
    m: int
    n: int
    k: int
    dtype: str
    aten: str
    source: str


def variants(topks: Iterable[int]) -> list[Variant]:
    result = [
        Variant("default-off", origami=False, search_space="DEFAULT"),
        Variant("exhaustive-off", origami=False, search_space="EXHAUSTIVE"),
    ]
    result.extend(Variant(f"origami-top{topk}", origami=True, search_space="DEFAULT", topk=topk) for topk in topks)
    return result


def parse_topks(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--topks must be comma-separated positive integers") from exc
    if not parsed or any(topk <= 0 for topk in parsed) or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("--topks must be unique positive integers")
    return parsed


def find_gemm_shapes(cache_dir: Path) -> list[GemmShape]:
    shapes: set[GemmShape] = set()
    if not cache_dir.exists():
        return []

    for source_path in cache_dir.rglob("*.py"):
        try:
            source = source_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for kernel_match in GEMM_KERNEL.finditer(source):
            kernel = kernel_match.group("kernel")
            dimensions = {match.group("name"): int(match.group("value")) for match in DIMENSION.finditer(kernel)}
            if set(dimensions) != {"M", "N", "K"}:
                continue
            prefix = source[max(0, kernel_match.start() - 4096):kernel_match.start()]
            aten_matches = list(ATEN.finditer(prefix))
            aten = aten_matches[-1].group("aten") if aten_matches else "unknown"
            if not any(name in aten for name in ("mm", "matmul", "addmm")):
                continue
            dtype_match = DTYPE.search(kernel)
            shapes.add(
                GemmShape(
                    m=dimensions["M"],
                    n=dimensions["N"],
                    k=dimensions["K"],
                    dtype=dtype_match.group("dtype") if dtype_match else "unknown",
                    aten=aten,
                    source=str(source_path.relative_to(cache_dir)),
                )
            )
    return sorted(shapes, key=lambda item: (item.m, item.n, item.k, item.dtype, item.aten, item.source))


def command_after_separator(command: list[str]) -> list[str]:
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("supply the actual Anima workload command after '--'")
    return command


def make_plan(args: argparse.Namespace, command: list[str]) -> list[dict[str, object]]:
    cache_root = args.cache_root.resolve()
    return [
        {
            "variant": asdict(variant),
            "cache_dir": str(cache_root / variant.name),
            "environment": variant.environment(cache_root / variant.name),
            "command": command,
        }
        for variant in variants(args.topks)
    ]


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def execute_plan(args: argparse.Namespace, plan: list[dict[str, object]]) -> int:
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    results_dir = args.results_dir.resolve()
    if results_dir.exists() and any(results_dir.iterdir()) and not args.allow_existing_results:
        raise ValueError(f"results directory is not empty: {results_dir}; choose a new path or pass --allow-existing-results")
    results_dir.mkdir(parents=True, exist_ok=True)
    write_json(results_dir / "plan.json", plan)

    failures = 0
    results: list[dict[str, object]] = []
    for item in plan:
        name = str(item["variant"]["name"])
        cache_dir = Path(str(item["cache_dir"]))
        if cache_dir.exists() and any(cache_dir.iterdir()) and not args.allow_existing_results:
            raise ValueError(f"cache directory is not empty: {cache_dir}; choose a new --cache-root or pass --allow-existing-results")
        environment = os.environ.copy()
        environment.pop("TORCHINDUCTOR_ORIGAMI_TOPK", None)
        environment.update(item["environment"])
        started = time.monotonic()
        completed = subprocess.run(item["command"], env=environment, check=False, capture_output=True, text=True)
        elapsed_seconds = time.monotonic() - started
        shapes = find_gemm_shapes(cache_dir)
        result = {
            "variant": item["variant"],
            "command": item["command"],
            "environment": item["environment"],
            "returncode": completed.returncode,
            "elapsed_seconds": elapsed_seconds,
            "gemm_shapes": [asdict(shape) for shape in shapes],
            "unique_gemm_shape_count": len({(shape.m, shape.n, shape.k, shape.dtype, shape.aten) for shape in shapes}),
        }
        write_json(results_dir / f"{name}.json", result)
        (results_dir / f"{name}.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (results_dir / f"{name}.stderr.log").write_text(completed.stderr, encoding="utf-8")
        results.append(result)
        print(f"{name}: returncode={completed.returncode}, shapes={result['unique_gemm_shape_count']}, elapsed={elapsed_seconds:.2f}s")
        if completed.returncode:
            failures += 1
            if not args.keep_going:
                break
    write_json(results_dir / "summary.json", {"results": results})
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    run = subparsers.add_parser("run", help="run isolated workload variants")
    run.add_argument("--topks", type=parse_topks, default=(4, 6, 8, 12), help="Origami top-k values (default: 4,6,8,12)")
    run.add_argument("--cache-root", type=Path, default=Path("user/origami-product-oracle-cache"), help="parent directory for per-variant caches")
    run.add_argument("--results-dir", type=Path, default=Path("user/origami-product-oracle-results"), help="directory for JSON manifests")
    run.add_argument("--dry-run", action="store_true", help="print the isolated process/cache plan without writing or running")
    run.add_argument("--allow-existing-results", action="store_true", help="allow writing into an existing results directory")
    run.add_argument("--keep-going", action="store_true", help="continue with later variants after a workload failure")
    run.add_argument("workload", nargs=argparse.REMAINDER, help="actual workload command, supplied after '--'")

    scan = subparsers.add_parser("scan", help="scan an existing Inductor cache")
    scan.add_argument("--scan-cache", type=Path, required=True, help="Inductor cache directory to scan")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.action == "scan":
        print(json.dumps([asdict(shape) for shape in find_gemm_shapes(args.scan_cache)], indent=2, sort_keys=True))
        return 0

    try:
        command = command_after_separator(args.workload)
        return execute_plan(args, make_plan(args, command))
    except ValueError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
