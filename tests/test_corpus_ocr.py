"""Tests for ``sv.pp.ocr_tei`` — the Tesseract-only OCR → TEI-P5 encoder.

These lock the *honesty contract*: the function has no Kraken/HTR path, so an
``engine=`` naming anything else must be recorded (not silently swapped for
Tesseract), and text-passthrough must still yield well-formed TEI when no OCR
engine is available.
"""
from __future__ import annotations

import warnings
from xml.dom import minidom

warnings.simplefilter("ignore")

import socialverse as sv  # noqa: E402


def test_ocr_tei_text_passthrough_yields_wellformed_tei():
    """Text pages (no image) encode straight to valid TEI regardless of engines."""
    st = sv.StudyState()
    st.write("sources", "scans", {"doc1": ["Anno Domini MCCCXLII", "second page"]})
    sv.pp.ocr_tei(st, titles={"doc1": "Chronica"})

    tei = st.corpus["tei"]["doc1"]
    # well-formed XML (raises if not) + the title/text made it in
    minidom.parseString(tei)
    assert "<title>Chronica</title>" in tei
    assert "Anno Domini MCCCXLII" in tei

    prov = st.evidence["provenance"]
    # no image + no engine used → text-passthrough, and no bogus "requested_engine"
    assert prov["engine"] == "text-passthrough"
    assert "requested_engine" not in prov


def test_ocr_tei_unimplemented_engine_is_recorded_not_silent():
    """engine='kraken' must surface an auditable note, never a silent Tesseract swap."""
    st = sv.StudyState()
    st.write("sources", "scans", {"ms": ["a manuscript line already transcribed"]})
    sv.pp.ocr_tei(st, engine="kraken", model="catmus-medieval.mlmodel")

    prov = st.evidence["provenance"]
    # the fallback is honest: engine is NOT reported as tesseract...
    assert prov["engine"] == "text-passthrough"
    # ...and the requested-but-unimplemented engine is captured with a note.
    assert prov["requested_engine"] == "kraken"
    assert "kraken" in prov["note"].lower()
    assert "tesseract-only" in prov["note"].lower()


def test_ocr_tei_tesseract_engine_kwarg_is_not_flagged():
    """Asking for the engine we actually implement adds no 'requested_engine' noise."""
    st = sv.StudyState()
    st.write("sources", "scans", {"d": ["plain text page"]})
    sv.pp.ocr_tei(st, engine="tesseract")

    prov = st.evidence["provenance"]
    assert "requested_engine" not in prov
    assert "note" not in prov
