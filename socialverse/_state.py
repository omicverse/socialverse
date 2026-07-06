"""``StudyState`` — the social-science analog of AnnData.

Where AnnData holds a single ``feature × sample`` matrix plus annotation slots,
``StudyState`` holds the heterogeneous artifacts of a social-science study in 12
typed slots (see :mod:`socialverse._slots`). It is deliberately *not* a data
matrix: social data is not commensurable (survey ≠ corpus ≠ network). The slots
are the shared **vocabulary** that the registry's ``requires`` / ``produces``
contracts speak in — the object exists to make dependencies checkable, not to
unify the data.

Every registered function that writes to the state also appends a provenance
record, so a completed analysis carries its own reproducible audit trail — the
"evidence spine" that is a first-class concern in social science.
"""
from __future__ import annotations

from typing import Any, Iterable

from ._slots import SLOTS, VALID_SLOTS


class Slot(dict):
    """A single StudyState slot — a plain dict with attribute access for keys.

    ``state.design['weights']`` and ``state.design.weights`` are equivalent.
    """

    __getattr__ = dict.get

    def __setattr__(self, key: str, value: Any) -> None:  # pragma: no cover - trivial
        self[key] = value

    def __repr__(self) -> str:  # pragma: no cover - display
        return f"Slot({', '.join(self.keys())})" if self else "Slot(∅)"


class StudyState:
    """Container for a social-science study, organized into 12 canonical slots.

    Examples
    --------
    >>> st = StudyState()
    >>> st.design['panel_id'] = 'firm_id'
    >>> st.has('design', 'panel_id')
    True
    >>> st.record('sv.pp.declare_design', {'panel_id': 'firm_id'}, produces={'design': ['panel_id']})
    """

    def __init__(self, **initial: Any) -> None:
        # one Slot per canonical name
        for name in SLOTS:
            object.__setattr__(self, name, Slot())
        #: append-only provenance ledger — the reproducibility spine
        object.__setattr__(self, "provenance", [])
        for slot, payload in initial.items():
            if slot not in VALID_SLOTS:
                raise ValueError(f"Unknown slot {slot!r}; valid: {sorted(VALID_SLOTS)}")
            getattr(self, slot).update(payload or {})

    # -- introspection -------------------------------------------------------
    def has(self, slot: str, key: str) -> bool:
        """True if ``state.<slot>[<key>]`` exists (and is not None)."""
        if slot not in VALID_SLOTS:
            raise ValueError(f"Unknown slot {slot!r}")
        return getattr(self, slot).get(key) is not None

    def missing(self, requires: dict[str, Iterable[str]]) -> list[tuple[str, str]]:
        """Return the ``(slot, key)`` pairs in ``requires`` that are absent.

        This is the check a resolver runs before invoking a function — the
        machine-readable equivalent of "does the data have what this step needs".
        """
        out: list[tuple[str, str]] = []
        for slot, keys in (requires or {}).items():
            for key in keys or []:
                if not self.has(slot, key):
                    out.append((slot, key))
        return out

    def satisfies(self, requires: dict[str, Iterable[str]]) -> bool:
        return not self.missing(requires)

    # -- mutation ------------------------------------------------------------
    def write(self, slot: str, key: str, value: Any) -> None:
        if slot not in VALID_SLOTS:
            raise ValueError(f"Unknown slot {slot!r}")
        getattr(self, slot)[key] = value

    def record(
        self,
        func: str,
        params: dict[str, Any] | None = None,
        *,
        produces: dict[str, Iterable[str]] | None = None,
        requires: dict[str, Iterable[str]] | None = None,
        note: str = "",
    ) -> None:
        """Append a provenance record for a function that just ran."""
        self.provenance.append(
            {
                "step": len(self.provenance) + 1,
                "function": func,
                "params": dict(params or {}),
                "requires": {k: list(v) for k, v in (requires or {}).items()},
                "produces": {k: list(v) for k, v in (produces or {}).items()},
                "note": note,
            }
        )

    # -- display -------------------------------------------------------------
    def populated(self) -> dict[str, list[str]]:
        """Map of non-empty slots to the keys they currently hold."""
        return {
            name: list(getattr(self, name).keys())
            for name in SLOTS
            if getattr(self, name)
        }

    def summary(self) -> str:
        lines = ["StudyState"]
        pop = self.populated()
        if not pop:
            lines.append("  (empty)")
        for name, keys in pop.items():
            lines.append(f"  {name}: {', '.join(keys)}")
        lines.append(f"  provenance: {len(self.provenance)} step(s)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - display
        pop = self.populated()
        inner = ", ".join(f"{k}[{len(v)}]" for k, v in pop.items()) or "empty"
        return f"StudyState({inner}; {len(self.provenance)} steps)"
