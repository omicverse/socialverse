"""End-to-end chain tests — exercise the registered analysis pipelines on toy
data and assert both the registry wiring and the numerical results."""
from __future__ import annotations

import warnings

import pytest

warnings.simplefilter("ignore")

import socialverse as sv  # noqa: E402
from socialverse import datasets as ds  # noqa: E402


# --------------------------------------------------------------- registry wiring
def test_registry_populated():
    assert len(sv.registry) >= 30
    # every registered contract uses only valid slots (guaranteed, but assert)
    for fn in sv.registry.manifest()["functions"]:
        for slot in list(fn["requires"]) + list(fn["produces"]):
            assert slot in sv.VALID_SLOTS


def test_resolve_plan_causal_orders_and_flags_inputs():
    plan = sv.registry.resolve_plan("did")
    names = [p.split(".")[-1] for p in plan["plan"]]
    assert names.index("declare_design") < names.index("parallel_trends") < names.index("did")
    inputs = {(n["slot"], n["key"]) for n in plan["needs_input"]}
    assert ("estimand", "target") in inputs
    assert ("variables", "outcome") in inputs


# --------------------------------------------------------------------- causal
def test_causal_chain():
    st = sv.StudyState()
    st.write("estimand", "target", "ATT")
    st.write("variables", "outcome", "y")
    df = ds.load_did_panel(att=-0.8)
    sv.pp.ingest(st, data=df)
    sv.pp.declare_design(st, panel_id="firm_id", time="year",
                         treatment="treat_post", first_treated="first_treated")
    sv.tl.parallel_trends(st)
    sv.tl.did(st)
    sv.tl.event_study(st)

    # parallel trends hold in the toy data -> "pass"
    assert st.identification["parallel_trends"] == "pass"
    assert st.diagnostics["pretrend"]["p_value"] > 0.05
    # ATT recovers the true effect (-0.8) within tolerance
    assert st.models["did"]["outcome"] == "y"
    assert -1.1 < st.models["did"]["att"] < -0.5
    # event study: base period is ~0, post-periods near the true effect
    es = st.models["event_study"]["coefs"]
    assert abs(es["-1"][0]) < 1e-6
    assert -1.1 < es["0"][0] < -0.5


def test_did_blocks_without_parallel_trends():
    st = sv.StudyState()
    st.write("estimand", "target", "ATT")
    st.write("variables", "outcome", "y")
    df = ds.load_did_panel()
    sv.pp.ingest(st, data=df)
    sv.pp.declare_design(st, panel_id="firm_id", time="year",
                         treatment="treat_post", first_treated="first_treated")
    with pytest.raises(sv.RegistryError):
        sv.tl.did(st)   # identification.parallel_trends not yet produced


# --------------------------------------------------------------------- survey
def test_survey_chain():
    st = sv.StudyState()
    st.write("estimand", "target", "prevalence")
    df = ds.load_survey()
    sv.pp.ingest(st, data=df)
    sv.pp.declare_design(st, weights="weight", strata="strata", psu="psu", unit="row")
    st.write("variables", "outcome", "outcome")
    items = df[[c for c in df.columns if c.startswith("item")]]
    sv.tl.design_survey(st, items=items)
    sv.tl.survey_estimate(st, exposure="exposure")

    # 6 items loading on one latent factor -> high Cronbach alpha
    assert st.diagnostics["reliability"]["alpha"] > 0.7
    assert "weighted_reg" in st.models
    assert "coef" in st.models["weighted_reg"]


# ------------------------------------------------------------------ qualitative
def test_qualitative_chain():
    st = sv.StudyState()
    st.write("sources", "corpora", ds.load_corpus())
    sv.pp.build_corpus(st)
    assert st.corpus["units"]
    sv.pp.redact_pii(st)
    # PII gone from the redacted documents
    joined = " ".join(str(v) for v in st.corpus["documents"].values())
    assert "jane.doe@example.com" not in joined
    assert st.governance["pii_status"]
    sv.tl.code_themes(st, lexicon={"burnout": ["burnout", "burned out", "crushing"],
                                   "support": ["support", "belonging", "colleagues"]})
    assert st.codes["themes"]
    sv.tl.trace_quotes(st)
    assert "quote_index" in st.evidence
    sv.tl.reflexive_memo(st)
    # provenance ledger recorded every step
    steps = [p["function"].split(".")[-1] for p in st.provenance]
    assert steps[:2] == ["build_corpus", "redact_pii"]


# ------------------------------------------------------------------ governance
def test_governance_gates():
    st = sv.StudyState()
    df = ds.load_survey()
    st.write("sources", "datasets", df)
    st.write("design", "unit", "row")
    sv.gov.ethics_check(st, data=df, quasi_identifiers=["strata", "psu"])
    sv.gov.data_use_check(st, license="CC-BY-4.0")
    sv.gov.ai_use_disclosure(
        st, ai_log=[{"stage": "drafting", "tool": "LLM", "accepted": True, "verified": False}],
        policy="ICMJE")
    assert st.governance["ethics"]["verdict"] in {"PASS", "FIX", "NO-GO"}
    assert "cc" in str(st.governance["data_use"]["bucket"]).lower()
    assert st.governance["ai_disclosure"]


# ------------------------------------------------------------------ literature
def test_literature_chain():
    st = sv.StudyState()
    sv.lit.search_free(st, records=ds.load_bib())
    assert st.sources["bib"]
    sv.lit.verify_citations(st)
    vb = st.evidence["verified_bib"]
    # the suspicious (no-DOI) record is flagged
    statuses = {r["id"]: r["status"] for r in vb["records"]}
    assert statuses["sus1"] in {"suspicious", "not_found"}
    st.write("sources", "datasets", "manuscript")
    sv.lit.manuscript_review(st, manuscript="We prove causality. (Braun & Clarke, 2006)")
    assert st.diagnostics["coverage"]["verdict"] in {"READY", "MINOR", "MAJOR", "BLOCKER"}


# --------------------------------------------------------------- text/philology
def test_text_chain():
    st = sv.StudyState()
    docs = {"A": "the quick brown fox", "B": "the quick brown foxe", "C": "the quik brown fox"}
    st.write("sources", "corpora", docs)
    sv.pp.build_corpus(st)
    st.write("corpus", "documents", docs)
    sv.tl.philology_collate(st)
    sv.tl.tei_encode(st)
    assert "stemma" in st.models
    assert "tei" in st.corpus and "<TEI" in st.corpus["tei"]
