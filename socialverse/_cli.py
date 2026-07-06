"""socialverse command-line interface — a thin window onto the registry.

    socialverse find "did"          # what functions exist for DID?
    socialverse prereqs did         # what does it require / produce?
    socialverse plan sv.pl.forest   # order the chain to reach a target
    socialverse list                # all functions by category
    socialverse manifest --out reg.json
    socialverse demo                # run the causal chain end-to-end on toy data

This is the same query surface OmicOS's ``registry_lookup`` calls — here exposed
for humans on the terminal.
"""
from __future__ import annotations

import argparse
import json
import sys


def _reg():
    import socialverse as sv
    return sv.registry


def cmd_find(args) -> int:
    for r in _reg().find(args.query, limit=args.limit):
        req = "+".join(f"{k}:{','.join(v)}" for k, v in (r["requires"] or {}).items()) or "∅"
        pro = "+".join(f"{k}:{','.join(v)}" for k, v in (r["produces"] or {}).items()) or "∅"
        print(f"• {r['full_name']}  [{r['tier']}]  {req} → {pro}")
        if r["description"]:
            print(f"    {r['description']}")
    return 0


def cmd_prereqs(args) -> int:
    print(json.dumps(_reg().get_prerequisites(args.func), ensure_ascii=False, indent=2))
    return 0


def cmd_plan(args) -> int:
    plan = _reg().resolve_plan(args.target)
    print("plan:")
    for i, step in enumerate(plan["plan"], 1):
        print(f"  {i}. {step}")
    if plan["needs_input"]:
        print("needs_input (user must supply):")
        for n in plan["needs_input"]:
            print(f"  - {n['slot']}.{n['key']}  (for {n['for']})")
    if plan["escalations"]:
        print(f"escalations: {len(plan['escalations'])} step(s) need human confirmation")
    return 0


def cmd_list(args) -> int:
    for cat, funcs in _reg().list_functions(args.category).items():
        print(f"\n[{cat}]  ({len(funcs)})")
        for f in funcs:
            print(f"  {f}")
    return 0


def cmd_manifest(args) -> int:
    blob = _reg().export_registry(args.out)
    if not args.out:
        print(blob)
    else:
        print(f"wrote registry manifest → {args.out}")
    return 0


def cmd_demo(args) -> int:
    import socialverse as sv
    from socialverse import datasets

    st = sv.StudyState()
    st.write("estimand", "target", "ATT")        # user-declared estimand
    st.write("variables", "outcome", "y")         # user-declared outcome column
    df = datasets.load_did_panel()
    # follow the registry's own plan
    plan = sv.registry.resolve_plan("did")
    print("resolved plan:", " → ".join(s.split(".")[-1] for s in plan["plan"]))
    sv.pp.ingest(st, data=df)
    sv.pp.declare_design(st, panel_id="firm_id", time="year",
                         treatment="treat_post", first_treated="first_treated")
    sv.tl.parallel_trends(st)
    sv.tl.did(st)
    print("\n" + st.summary())
    print("\nATT estimate:", st.models.get("did"))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="socialverse", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("find", help="fuzzy-search the registry")
    f.add_argument("query"); f.add_argument("--limit", type=int, default=10)
    f.set_defaults(fn=cmd_find)

    pr = sub.add_parser("prereqs", help="show a function's dependency contract")
    pr.add_argument("func"); pr.set_defaults(fn=cmd_prereqs)

    pl = sub.add_parser("plan", help="resolve the chain to reach a target")
    pl.add_argument("target"); pl.set_defaults(fn=cmd_plan)

    ls = sub.add_parser("list", help="list functions by category")
    ls.add_argument("--category", default=None); ls.set_defaults(fn=cmd_list)

    mf = sub.add_parser("manifest", help="dump the full registry as JSON")
    mf.add_argument("--out", default=None); mf.set_defaults(fn=cmd_manifest)

    dm = sub.add_parser("demo", help="run the causal chain on toy data")
    dm.set_defaults(fn=cmd_demo)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
