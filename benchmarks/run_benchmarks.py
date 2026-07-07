#!/usr/bin/env python3
"""Run the socialverse benchmark suite.

    python benchmarks/run_benchmarks.py            # all cases (downloads public data)
    python benchmarks/run_benchmarks.py --offline  # only built-in toy-data cases
    python benchmarks/run_benchmarks.py --only qca  # a single case by id substring
"""
import argparse
import importlib
import pkgutil
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

from benchmarks import _harness  # noqa: E402
from benchmarks import cases as cases_pkg  # noqa: E402


def load_cases():
    found = []
    for mod in pkgutil.iter_modules(cases_pkg.__path__):
        if not mod.name.startswith("c"):
            continue
        m = importlib.import_module(f"benchmarks.cases.{mod.name}")
        if hasattr(m, "CASE"):
            found.append(m.CASE)
    return sorted(found, key=lambda c: c.id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="only toy-data cases (no download)")
    ap.add_argument("--only", default=None, help="run cases whose id contains this substring")
    args = ap.parse_args()

    try:
        import socialverse as sv  # noqa: F401
    except ImportError:
        print("socialverse not importable — `pip install socialverse` (or add the repo to PYTHONPATH)")
        sys.exit(2)

    cases = load_cases()
    if args.offline:
        cases = [c for c in cases if c.offline]
    if args.only:
        cases = [c for c in cases if args.only in c.id]
    if not cases:
        print("no matching cases")
        sys.exit(1)

    results = []
    for c in cases:
        print(f"running {c.id} ...", flush=True)
        results.append((c, _harness.run_case(c)))
    ok = _harness.report(results)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
