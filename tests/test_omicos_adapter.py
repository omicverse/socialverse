"""Tests for the OmicOS-facing query surface (sv.utils) — the adapter that lets
OmicOS's registry_lookup call socialverse exactly like it calls ov.utils."""
from __future__ import annotations

import warnings

warnings.simplefilter("ignore")

import socialverse as sv  # noqa: E402


def test_registry_lookup_signature_matches_omicverse():
    # ov.utils.registry_lookup(query, max_results=15) -> str
    out = sv.utils.registry_lookup("did", max_results=3)
    assert isinstance(out, str)
    # graceful when called positionally too (kernel falls back to 1-arg on TypeError)
    assert isinstance(sv.utils.registry_lookup("did"), str)


def test_registry_lookup_format_and_contract():
    out = sv.utils.registry_lookup("双重差分", max_results=2)
    assert out.startswith("Found ")
    assert "sv.tl.did(state, **kwargs)" in out
    assert "Must run first: parallel_trends" in out
    assert "Requires:" in out and "identification['parallel_trends']" in out
    assert "Produces:" in out and "models['did']" in out
    assert "Example:" in out


def test_registry_lookup_empty_query_is_helpful():
    out = sv.utils.registry_lookup("zzz-nonexistent-xyz")
    assert "No socialverse functions match" in out
    assert "registry_summary" in out


def test_registry_summary_lists_slots_and_categories():
    s = sv.utils.registry_summary()
    for slot in sv.SLOTS:
        assert slot in s
    assert "[causal]" in s and "did" in s
    assert "Typical chains" in s
    # points the agent at the query API, not code-guessing
    assert "resolve_plan" in s
