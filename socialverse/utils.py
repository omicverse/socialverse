"""socialverse.utils — the OmicOS-facing query surface.

OmicOS's kernel calls ``ov.utils.registry_lookup(query, max_results=...)`` and
prints the returned string for the agent (the Expert prompt literally runs
``print(ov.utils.registry_lookup(query))``). This module exposes the *same two
functions with the same signatures and output format* over ``socialverse``'s
registry, so a domain-aware kernel can call ``sv.utils.registry_lookup`` for the
``humanities_social`` domain with no other changes — the agent reads an identical
"Found N matching functions … Requires … Produces … Example" layout it already
knows from the bio domain.
"""
from __future__ import annotations

from ._registry import registry
from ._slots import SLOTS

__all__ = ["registry_lookup", "registry_summary"]

_RULE = "  " + "─" * 76

# canonical analysis chains (derived from requires ↔ produces), shown in the summary
_CHAINS = [
    ("causal",       "ingest → declare_design → parallel_trends → did → event_study → forest"),
    ("complex survey", "ingest → declare_design → design_survey → survey_estimate → survey_dist"),
    ("qualitative",  "build_corpus → redact_pii → code_themes → trace_quotes → reflexive_memo → theme_map"),
    ("text/philology", "ocr_tei → build_corpus → philology_collate → tei_encode"),
    ("literature",   "search_free → zotero_bridge → citation_manage → verify_citations → manuscript_review"),
    ("governance (cross-cutting)", "data_use_check · ethics_check · redact_pii · ai_use_disclosure"),
]


def _friendly(full_name: str) -> str:
    """socialverse.tl._causal.did → sv.tl.did (the name an agent would write)."""
    parts = full_name.split(".")
    if len(parts) >= 3 and parts[0] == "socialverse":
        return f"sv.{parts[1]}.{parts[-1]}"
    return full_name


def _fmt_slots(slot_map: dict | None) -> str:
    """{'design': ['panel_id','time']} → \"design['panel_id'], design['time']\"."""
    if not slot_map:
        return ""
    out: list[str] = []
    for slot, keys in slot_map.items():
        if keys:
            out.extend(f"{slot}['{k}']" for k in keys)
        else:
            out.append(slot)
    return ", ".join(out)


def registry_lookup(query: str, max_results: int = 15) -> str:
    """Search the socialverse registry and format matches for an agent to read.

    Mirrors ``ov.utils.registry_lookup``: returns a markdown-ish string listing
    matching functions with their dependency contract (Requires / Produces /
    Must-run-first) and an example call. Query in Chinese, English, an abbreviation
    (e.g. ``DID``), or a backend name.
    """
    entries = registry.find(query, limit=int(max_results or 15))
    if not entries:
        return (f"No socialverse functions match {query!r}.\n"
                f"Try sv.utils.registry_summary() for the catalog, or a broader query "
                f"(e.g. 'did', '主题编码', 'survey', 'citation').")

    lines = [f"Found {len(entries)} matching functions:"]
    for i, e in enumerate(entries, 1):
        fname = _friendly(e["full_name"])
        lines.append(_RULE)
        lines.append(f"  [match {i}/{len(entries)}]")
        lines.append(f"  {fname}(state, **kwargs)")
        if e.get("description"):
            lines.append(f"    {e['description']}")
        pre = (e.get("prerequisites") or {}).get("functions") or []
        if pre:
            lines.append(f"    Must run first: {', '.join(pre)}")
        opt = (e.get("prerequisites") or {}).get("optional_functions") or []
        if opt:
            lines.append(f"    Recommended first: {', '.join(opt)}")
        req = _fmt_slots(e.get("requires"))
        if req:
            lines.append(f"    Requires: {req}")
        pro = _fmt_slots(e.get("produces"))
        if pro:
            lines.append(f"    Produces: {pro}")
        if e.get("key_tools"):
            lines.append(f"    Backend: {', '.join(e['key_tools'][:4])}")
        lines.append(f"    Tier: {e.get('tier', 'community')}  ·  auto_fix: {e.get('auto_fix')}")
        example = (e.get("examples") or [None])[0] or f"{fname}(state)"
        lines.append(f"    Example: {example}")
    lines.append(_RULE)
    lines.append("Plan a full chain with: sv.registry.resolve_plan('<function>')")
    return "\n".join(lines)


def registry_summary() -> str:
    """A high-level overview of the socialverse registry — the domain map an agent
    prints once at the start of a task. Mirrors ``ov.utils.registry_summary``."""
    cats = registry.list_functions()
    lines = [
        f"socialverse — social-science function registry ({len(registry)} functions).",
        "Query it before writing code — do NOT guess the API:",
        "  sv.utils.registry_lookup('<query>')      # find functions + their contracts",
        "  sv.registry.get_prerequisites('<fn>')    # what it requires / produces",
        "  sv.registry.resolve_plan('<target>')     # order the chain to reach a target",
        "",
        "StudyState slots (every requires/produces speaks in these — the AnnData analog):",
        "  " + ", ".join(SLOTS),
        "",
        "Functions by category:",
    ]
    for cat, funcs in cats.items():
        names = ", ".join(sorted(f.split(".")[-1] for f in funcs))
        lines.append(f"  [{cat}] {names}")
    lines.append("")
    lines.append("Typical chains (from requires ↔ produces):")
    for name, chain in _CHAINS:
        lines.append(f"  {name}: {chain}")
    return "\n".join(lines)
