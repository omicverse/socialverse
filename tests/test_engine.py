"""Core engine tests — registry, StudyState, resolve_plan — independent of the
analysis modules (uses a fresh FunctionRegistry with inline functions)."""
from __future__ import annotations

import pytest

from socialverse import FunctionRegistry, RegistryError, StudyState
from socialverse._slots import VALID_SLOTS


@pytest.fixture
def reg():
    r = FunctionRegistry()

    @r.register(name="declare_design", produces={"design": ["panel_id", "time", "treatment"]},
                auto_fix="none", category="prep", aliases=["声明设计"])
    def declare_design(state, **k):
        for key in ("panel_id", "time", "treatment"):
            state.write("design", key, k.get(key, key))
        return state

    @r.register(name="parallel_trends",
                requires={"design": ["panel_id", "time", "treatment"], "estimand": ["target"]},
                produces={"diagnostics": ["pretrend"], "identification": ["parallel_trends"]},
                auto_fix="escalate", category="causal")
    def parallel_trends(state, **k):
        state.write("diagnostics", "pretrend", {"p": 0.4})
        state.write("identification", "parallel_trends", "pass")
        return state

    @r.register(name="did",
                requires={"design": ["panel_id", "time", "treatment"],
                          "identification": ["parallel_trends"]},
                produces={"models": ["did", "twfe"]},
                prerequisites={"functions": ["parallel_trends"]},
                auto_fix="escalate", category="causal", aliases=["双重差分", "DID"])
    def did(state, **k):
        state.write("models", "did", {"att": -0.13})
        return state

    return r


def test_find_fuzzy_and_alias(reg):
    assert reg.find("双重差分")[0]["name"] == "did"
    assert reg.find("DID")[0]["name"] == "did"
    assert reg.find("parallel")[0]["name"] == "parallel_trends"


def test_get_prerequisites_shape(reg):
    p = reg.get_prerequisites("did")
    assert p["required_functions"] == ["parallel_trends"]
    assert p["requires"]["identification"] == ["parallel_trends"]
    assert p["produces"]["models"] == ["did", "twfe"]
    assert p["auto_fix"] == "escalate"
    # the resolver tells you who satisfies each requirement
    assert "parallel_trends" in p["satisfied_by"]["identification.parallel_trends"]


def test_enforcement_blocks_unmet(reg):
    st = StudyState()
    did = reg.get_function("did")
    with pytest.raises(RegistryError) as e:
        did(st)
    assert "identification.parallel_trends" in str(e.value)
    assert "parallel_trends" in str(e.value)  # names the producer


def test_resolve_plan_orders_chain(reg):
    plan = reg.resolve_plan("did")
    names = [p.split(".")[-1] for p in plan["plan"]]
    assert names.index("declare_design") < names.index("parallel_trends") < names.index("did")
    # estimand has no producer -> user input
    assert any(n["slot"] == "estimand" for n in plan["needs_input"])


def test_full_chain_runs_and_records(reg):
    st = StudyState()
    st.write("estimand", "target", "ATT")
    reg.get_function("declare_design")(st)
    reg.get_function("parallel_trends")(st)
    reg.get_function("did")(st)
    assert st.models["did"]["att"] == -0.13
    assert [p["function"].split(".")[-1] for p in st.provenance] == \
        ["declare_design", "parallel_trends", "did"]


def test_slot_validation_rejects_bad_slot():
    r = FunctionRegistry()
    with pytest.raises(ValueError):
        @r.register(name="bad", produces={"not_a_slot": ["x"]})
        def bad(state, **k):
            return state


def test_studystate_missing_and_satisfies():
    st = StudyState()
    req = {"design": ["weights"]}
    assert st.missing(req) == [("design", "weights")]
    st.write("design", "weights", "w")
    assert st.satisfies(req)


def test_manifest_serializable(reg):
    import json
    blob = json.dumps(reg.manifest(), ensure_ascii=False)
    assert "functions" in json.loads(blob)
    assert set(reg.manifest()["slots"]) == set(VALID_SLOTS)
