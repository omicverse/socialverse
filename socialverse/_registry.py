"""``socialverse.registry`` — a structured, dependency-annotated function registry.

This is the spine of the package, ported from the mechanism that makes omicverse's
agent capability work: every analysis function is registered with a machine-readable
contract — what it ``requires`` (which StudyState slots/keys must already exist),
what it ``produces`` (which it writes), which ``prerequisites`` (other functions)
should run first, and an ``auto_fix`` policy for unmet dependencies.

An agent (or a human, or :meth:`FunctionRegistry.resolve_plan`) can then **query**
the registry instead of guessing an API: "what functions exist for X", "what does
``did`` require", "how do I get from raw survey data to a design-weighted estimate".
That grounding — query, don't hallucinate — is the whole point.

The public query surface deliberately mirrors omicverse's registry
(``find`` / ``get_prerequisites`` / ``get_function`` / ``list_functions`` /
``export_registry``) so that OmicOS's ``registry_lookup`` tool can consume a
``socialverse`` registry with no changes.
"""
from __future__ import annotations

import functools
import inspect
from difflib import get_close_matches
from typing import Any, Callable, Iterable

from ._slots import SLOTS, validate_prerequisites, validate_slot_map
from ._state import StudyState


class RegistryError(Exception):
    """Raised when a registered function is called with unmet ``requires``."""


class RegistryEntry:
    """One registered function and its full dependency contract."""

    __slots__ = (
        "name", "full_name", "func", "aliases", "category", "description",
        "requires", "produces", "prerequisites", "auto_fix", "examples",
        "related", "languages", "key_tools", "tier", "skill",
    )

    def __init__(self, **kw: Any) -> None:
        for s in self.__slots__:
            setattr(self, s, kw.get(s))

    def as_dict(self, *, with_func: bool = False) -> dict[str, Any]:
        d = {s: getattr(self, s) for s in self.__slots__ if s != "func"}
        if with_func:
            d["func"] = self.func
        return d

    def __repr__(self) -> str:  # pragma: no cover - display
        r = "+".join(f"{k}:{','.join(v)}" for k, v in (self.requires or {}).items()) or "∅"
        p = "+".join(f"{k}:{','.join(v)}" for k, v in (self.produces or {}).items()) or "∅"
        return f"<{self.full_name} requires[{r}] produces[{p}] auto_fix={self.auto_fix}>"


class FunctionRegistry:
    """A queryable registry of dependency-annotated functions."""

    def __init__(self) -> None:
        self._entries: dict[str, RegistryEntry] = {}      # short_name -> entry
        self._by_full: dict[str, RegistryEntry] = {}      # full_name -> entry
        self._aliases: dict[str, str] = {}                # alias -> short_name
        # (slot, key) -> set of producing short_names; key '*' = produces slot at all
        self._producers: dict[tuple[str, str], set[str]] = {}

    # ------------------------------------------------------------------ register
    def register(
        self,
        func: Callable | None = None,
        *,
        name: str | None = None,
        aliases: Iterable[str] | None = None,
        category: str = "",
        description: str = "",
        requires: dict[str, Iterable[str]] | None = None,
        produces: dict[str, Iterable[str]] | None = None,
        prerequisites: dict[str, Iterable[str]] | None = None,
        auto_fix: str = "escalate",
        examples: Iterable[str] | None = None,
        related: Iterable[str] | None = None,
        languages: Iterable[str] | None = None,
        key_tools: Iterable[str] | None = None,
        tier: str = "community",
        skill: str = "",
        enforce: bool = True,
    ) -> Callable:
        """Decorator: register ``func`` with its dependency contract.

        ``requires`` / ``produces`` are ``{slot: [keys...]}`` validated against the
        12-slot vocabulary. ``prerequisites`` is ``{'functions': [...],
        'optional_functions': [...]}``. ``auto_fix`` ∈ {``'auto'``, ``'escalate'``,
        ``'none'``}: how :meth:`resolve_plan` treats an unmet requirement.

        If ``enforce`` (default), the wrapped callable checks ``requires`` against a
        :class:`StudyState` argument at call time and records provenance on success —
        making the contract *live*, not just metadata.
        """
        if auto_fix not in {"auto", "escalate", "none"}:
            raise ValueError(f"auto_fix must be auto|escalate|none, got {auto_fix!r}")
        req = validate_slot_map(requires, field="requires")
        pro = validate_slot_map(produces, field="produces")
        pre = validate_prerequisites(prerequisites)

        def deco(fn: Callable) -> Callable:
            short = name or fn.__name__
            module = getattr(fn, "__module__", "socialverse")
            full = f"{module}.{short}"

            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                state = _find_state(fn, args, kwargs)
                if enforce and state is not None and req:
                    missing = state.missing(req)
                    if missing:
                        raise RegistryError(_unmet_message(full, missing, self))
                result = fn(*args, **kwargs)
                if state is not None:
                    state.record(full, _call_params(fn, args, kwargs),
                                 requires=req, produces=pro)
                return result

            entry = RegistryEntry(
                name=short, full_name=full, func=wrapper,
                aliases=list(aliases or []), category=category,
                description=description or (inspect.getdoc(fn) or "").split("\n")[0],
                requires=req, produces=pro, prerequisites=pre, auto_fix=auto_fix,
                examples=list(examples or []), related=list(related or []),
                languages=list(languages or []), key_tools=list(key_tools or []),
                tier=tier, skill=skill,
            )
            self._add(entry)
            wrapper._sv_entry = entry  # type: ignore[attr-defined]
            return wrapper

        return deco(func) if callable(func) else deco

    def _add(self, entry: RegistryEntry) -> None:
        self._entries[entry.name.lower()] = entry
        self._by_full[entry.full_name.lower()] = entry
        for a in entry.aliases:
            self._aliases[a.lower()] = entry.name.lower()
        for slot, keys in (entry.produces or {}).items():
            if keys:
                # producer of specific named keys — only those keys are satisfied
                for k in keys:
                    self._producers.setdefault((slot, k), set()).add(entry.name.lower())
            else:
                # generic producer of the whole slot (empty key list)
                self._producers.setdefault((slot, "*"), set()).add(entry.name.lower())

    # ------------------------------------------------------------------ lookup
    def get(self, query: str) -> RegistryEntry | None:
        q = (query or "").lower()
        if q in self._entries:
            return self._entries[q]
        if q in self._by_full:
            return self._by_full[q]
        if q in self._aliases:
            return self._entries[self._aliases[q]]
        # tail of a full name, e.g. "did" for "socialverse.tl.did"
        tail = q.rsplit(".", 1)[-1]
        return self._entries.get(tail)

    def get_function(self, query: str) -> Callable | None:
        e = self.get(query)
        return e.func if e else None

    def find(self, query: str, threshold: float = 0.6, limit: int = 10) -> list[dict]:
        """Fuzzy + substring search (Chinese / English / abbreviation / tool name).

        Returns entry dicts sorted by relevance — the shape OmicOS's
        ``registry_lookup`` renders to the agent.
        """
        q = (query or "").lower().strip()
        if not q:
            return []
        results: list[RegistryEntry] = []
        seen: set[str] = set()

        def add(e: RegistryEntry | None) -> None:
            if e and e.name.lower() not in seen:
                seen.add(e.name.lower())
                results.append(e)

        add(self.get(q))
        for m in get_close_matches(q, list(self._entries), n=limit, cutoff=threshold):
            add(self._entries[m])
        for m in get_close_matches(q, list(self._aliases), n=limit, cutoff=threshold):
            add(self._entries[self._aliases[m]])
        for e in self._entries.values():
            hay = " ".join([
                e.name, e.full_name, e.description or "", e.category or "",
                " ".join(e.aliases or []), " ".join(e.key_tools or []),
                " ".join(e.examples or []), e.skill or "",
            ]).lower()
            if q in hay:
                add(e)
        return [e.as_dict() for e in results[:limit]]

    # ------------------------------------------------------------------ deps
    def _satisfiers(self, slot: str, key: str, exclude: str = "") -> set[str]:
        """Functions that satisfy a required ``(slot, key)``: exact key-producers
        plus any generic whole-slot producer. Empty ⇒ user-supplied input."""
        exact = self._producers.get((slot, key), set())
        generic = self._producers.get((slot, "*"), set())
        return (exact | generic) - ({exclude} if exclude else set())

    def get_prerequisites(self, name: str) -> dict[str, Any]:
        """Return the dependency contract of a function.

        Mirrors omicverse's ``get_prerequisites``: ``required_functions``,
        ``optional_functions``, ``requires``, ``produces``, ``auto_fix`` — plus the
        producers that would satisfy each required slot (the "how to get there").
        """
        e = self.get(name)
        if e is None:
            raise KeyError(f"No registered function matching {name!r}")
        satisfiers = {
            f"{slot}.{k}": sorted(self._satisfiers(slot, k, exclude=e.name.lower()))
            for slot, keys in (e.requires or {}).items() for k in (keys or [])
        }
        return {
            "function": e.full_name,
            "required_functions": list((e.prerequisites or {}).get("functions", [])),
            "optional_functions": list((e.prerequisites or {}).get("optional_functions", [])),
            "requires": e.requires,
            "produces": e.produces,
            "auto_fix": e.auto_fix,
            "satisfied_by": satisfiers,
        }

    def producers(self, slot: str, key: str = "*") -> list[str]:
        """Function short-names that produce ``state.<slot>[<key>]`` (or the slot)."""
        if key == "*":
            return sorted(self._producers.get((slot, "*"), set()))
        return sorted(self._satisfiers(slot, key))

    def resolve_plan(self, targets: str | Iterable[str],
                     state: StudyState | None = None) -> dict[str, Any]:
        """Compute an ordered plan of functions to reach ``targets``.

        Walks the dependency graph: to run a function whose ``requires`` are unmet,
        first run a producer of each missing slot (recursively), honoring
        ``prerequisites.functions``. Requirements that no function produces and that
        the ``state`` lacks are returned as ``needs_input`` (user must supply, e.g.
        ``estimand``). Auto-inserted steps whose downstream ``auto_fix == 'escalate'``
        are surfaced in ``escalations`` (a human should confirm them).

        This is the ``leiden → neighbors → pca`` resolution, ported to social science.
        """
        if isinstance(targets, str):
            targets = [targets]
        state = state or StudyState()
        plan: list[str] = []
        needs_input: list[dict] = []
        escalations: list[dict] = []
        placed: set[str] = set()

        def visit(fname: str, trail: tuple[str, ...]) -> None:
            e = self.get(fname)
            if e is None or e.name.lower() in placed:
                return
            if e.name.lower() in trail:
                return  # cycle guard
            trail = trail + (e.name.lower(),)
            # explicit function prerequisites first
            for pfn in (e.prerequisites or {}).get("functions", []):
                visit(pfn, trail)
            # then satisfy each required (slot, key)
            for slot, keys in (e.requires or {}).items():
                for k in keys or []:
                    if state.has(slot, k):
                        continue
                    prods = sorted(self._satisfiers(slot, k, exclude=e.name.lower()))
                    if not prods:
                        needs_input.append({"for": e.full_name, "slot": slot, "key": k})
                        continue
                    chosen = prods[0]
                    if e.auto_fix == "escalate":
                        escalations.append({
                            "for": e.full_name, "needs": f"{slot}.{k}",
                            "auto_insert": chosen,
                            "reason": "downstream auto_fix=escalate — confirm before running",
                        })
                    visit(chosen, trail)
            if e.name.lower() not in placed:
                placed.add(e.name.lower())
                plan.append(e.full_name)

        for t in targets:
            visit(t, ())

        return {
            "targets": list(targets),
            "plan": plan,
            "needs_input": needs_input,
            "escalations": escalations,
        }

    # ------------------------------------------------------------------ listing
    def list_functions(self, category: str | None = None) -> dict[str, list[str]]:
        """Group registered function full-names by category."""
        out: dict[str, list[str]] = {}
        for e in self._entries.values():
            if category and e.category != category:
                continue
            out.setdefault(e.category or "uncategorized", []).append(e.full_name)
        return {k: sorted(v) for k, v in sorted(out.items())}

    def categories(self) -> list[str]:
        return sorted({e.category for e in self._entries.values() if e.category})

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: str) -> bool:
        return self.get(name) is not None

    def manifest(self) -> dict[str, Any]:
        """Full JSON-serializable dump — for export and the OmicOS HTTP endpoint."""
        return {
            "slots": {k: v[0] for k, v in SLOTS.items()},
            "count": len(self._entries),
            "categories": self.categories(),
            "functions": [e.as_dict() for e in
                          sorted(self._entries.values(), key=lambda x: x.full_name)],
        }

    def export_registry(self, filepath: str | None = None) -> str:
        import json
        blob = json.dumps(self.manifest(), ensure_ascii=False, indent=2)
        if filepath:
            with open(filepath, "w", encoding="utf-8") as fh:
                fh.write(blob)
        return blob


# --------------------------------------------------------------------- helpers
def _find_state(fn: Callable, args: tuple, kwargs: dict) -> StudyState | None:
    if isinstance(kwargs.get("state"), StudyState):
        return kwargs["state"]
    for a in args:
        if isinstance(a, StudyState):
            return a
    return None


def _call_params(fn: Callable, args: tuple, kwargs: dict) -> dict:
    try:
        sig = inspect.signature(fn)
        bound = sig.bind_partial(*args, **kwargs)
        return {k: _short(v) for k, v in bound.arguments.items()
                if not isinstance(v, StudyState)}
    except Exception:  # pragma: no cover - defensive
        return {k: _short(v) for k, v in kwargs.items()}


def _short(v: Any) -> Any:
    r = repr(v)
    return v if len(r) <= 80 else f"{type(v).__name__}(…)"


def _unmet_message(full: str, missing: list[tuple[str, str]], reg: "FunctionRegistry") -> str:
    lines = [f"{full} cannot run — unmet requires:"]
    for slot, key in missing:
        prods = reg.producers(slot, key)
        hint = f" (produced by: {', '.join(prods)})" if prods else " (user-supplied input)"
        lines.append(f"  - {slot}.{key}{hint}")
    lines.append("Query registry.get_prerequisites(...) or registry.resolve_plan(...) to plan the chain.")
    return "\n".join(lines)


#: the process-wide singleton — import as ``from socialverse import registry``
registry = FunctionRegistry()

# module-level convenience mirrors (omicverse-style)
register = registry.register
find_function = registry.find
list_functions = registry.list_functions
get_prerequisites = registry.get_prerequisites
export_registry = registry.export_registry
