"""Benchmark harness: run each case, compare to its expected value, print a report.

A **case** is a self-contained demo *and* regression test: it takes a natural-
language prompt (what a user would ask an omicos agent), runs the equivalent
socialverse chain, and asserts the result reproduces a published or known-truth
value. Cases live in ``benchmarks/cases/c*.py``; each exposes a module-level
``CASE = Case(...)``.

Run all:            python benchmarks/run_benchmarks.py
Only offline (toy): python benchmarks/run_benchmarks.py --offline
One case:           python benchmarks/run_benchmarks.py --only qca
"""
from __future__ import annotations

import io
import time
import traceback
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def fetch(url: str, filename: str) -> Path:
    """Download ``url`` to ``benchmarks/data/filename`` (cached; skipped if present).

    A browser User-Agent is sent because some hosts (Harvard Dataverse) reject the
    default urllib agent. Public replication data only.
    """
    path = DATA_DIR / filename
    if path.exists() and path.stat().st_size > 0:
        return path
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (socialverse benchmark)"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        path.write_bytes(resp.read())
    return path


def approx(got, target, tol) -> bool:
    try:
        return abs(float(got) - float(target)) <= float(tol)
    except (TypeError, ValueError):
        return False


@dataclass
class Case:
    id: str                      # short slug, e.g. "did_fect_hh2015"
    capability: str              # what it demonstrates
    agent: str                   # omicos agent it routes to
    skill: str                   # skill(s) exercised
    prompt: str                  # the natural-language prompt a user would type
    data: str                    # data description / source
    run: Callable[[], dict]      # () -> metrics dict
    check: Callable[[dict], list]  # (metrics) -> list[(desc, passed)]
    offline: bool = True         # True = uses built-in toy data (no download)
    tags: list = field(default_factory=list)


def run_case(case: Case) -> dict:
    t0 = time.time()
    try:
        metrics = case.run()
        checks = case.check(metrics)
        passed = all(ok for _d, ok in checks)
        return {"ok": passed, "metrics": metrics, "checks": checks,
                "secs": time.time() - t0, "error": None}
    except Exception as e:  # a broken case fails loudly but does not abort the suite
        return {"ok": False, "metrics": {}, "checks": [], "secs": time.time() - t0,
                "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}


def _fmt(v):
    if isinstance(v, float):
        return f"{v:,.4g}"
    return str(v)


def report(results: list[tuple[Case, dict]]) -> bool:
    print("\n" + "=" * 78)
    print("  socialverse benchmark — reproduce published / known-truth values")
    print("=" * 78)
    n_pass = 0
    for case, res in results:
        mark = "PASS" if res["ok"] else "FAIL"
        if res["ok"]:
            n_pass += 1
        print(f"\n[{mark}] {case.id}   ({case.capability})   ·  {res['secs']:.0f}s")
        print(f"       agent={case.agent}  skill={case.skill}")
        if res["error"]:
            print(f"       ERROR: {res['error']}")
            continue
        for desc, ok in res["checks"]:
            print(f"       {'✓' if ok else '✗'} {desc}")
    print("\n" + "-" * 78)
    print(f"  {n_pass}/{len(results)} cases passed")
    print("-" * 78)
    return n_pass == len(results)
