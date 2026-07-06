"""socialverse.datasets — tiny, deterministic toy datasets for demos and tests."""
from ._toy import load_bib, load_corpus, load_did_panel, load_survey  # noqa: F401

__all__ = ["load_did_panel", "load_survey", "load_corpus", "load_bib"]
