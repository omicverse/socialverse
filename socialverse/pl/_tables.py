"""``sv.pl._tables`` — publication-grade regression tables.

Figures are only half of a results section; the other half is the multi-model
regression table (booktabs three-line style: coefficient with the standard error
beneath it in parentheses, significance stars, and fit rows for N / R² / fixed
effects). ``regtable`` turns any set of socialverse model results into that table in
LaTeX (booktabs), Markdown, or aligned plain text — so the numbers the reader sees
are generated from the same estimates, never re-typed.

Pure-Python string formatting; no backend required.
"""
from __future__ import annotations

from typing import Any

from .._registry import register
from .._state import StudyState


def _stars(p):
    if p is None:
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def _esc_latex(s):
    s = str(s)
    for a, b in (("\\", r"\textbackslash "), ("_", r"\_"), ("&", r"\&"), ("%", r"\%"),
                 ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde ")):
        s = s.replace(a, b)
    return s


def _esc_md(s):
    return str(s).replace("|", r"\|")


def _extract(model: dict) -> tuple[dict, dict]:
    """Best-effort ``{term: (coef, se, p)}`` + fit-stat dict from a socialverse model
    dict. Handles single-effect models (did/fect/dml ATT/ATE), a ``coefficients``
    mapping, and ``cate_linear`` interaction terms."""
    coefs: dict[str, tuple] = {}
    stats: dict[str, Any] = {}
    if not isinstance(model, dict):
        return coefs, stats

    # explicit coefficient table
    tbl = model.get("coefficients") or model.get("coefs")
    if isinstance(tbl, dict):
        for k, v in tbl.items():
            if isinstance(v, dict):
                coefs[k] = (v.get("coef", v.get("estimate")), v.get("se"), v.get("p"))
            elif isinstance(v, (list, tuple)) and len(v) >= 2:
                coefs[k] = (v[0], v[1], v[2] if len(v) > 2 else None)
    # single-effect estimators
    for key, label in (("att", "ATT"), ("ate", "ATE")):
        if model.get(key) is not None and label not in coefs:
            coefs[label] = (model[key], model.get("se"), model.get("p"))
    # linear-CATE interactions
    cl = model.get("cate_linear")
    if isinstance(cl, dict):
        for k, v in cl.items():
            if k.endswith("_se") or k == "intercept_se":
                continue
            se = cl.get(k + "_se") if k != "intercept" else cl.get("intercept_se")
            coefs[("CATE·" + k) if k != "intercept" else "CATE(mean)"] = (v, se, None)

    for s in ("n", "n_treated_obs", "n_units", "n_clusters", "r2", "outcome", "estimator"):
        if model.get(s) is not None:
            stats[s] = model[s]
    return coefs, stats


def _fmt(v, nd=3):
    return "" if v is None else f"{float(v):.{nd}f}"


@register(
    name="regtable",
    aliases=["回归表", "regression_table", "esttab", "outreg"],
    category="figure",
    tier="community",
    skill="social-science-figure",
    languages=["Python"],
    key_tools=[],
    description="出版级多模型回归表(booktabs 三线表/Markdown/纯文本):系数+括注SE+星号+N/R²/FE 行",
    requires={},
    produces={"artifacts": ["tables"]},
    auto_fix="none",
)
def regtable(state: StudyState, **kwargs: Any) -> StudyState:
    """Render a multi-model regression table from socialverse results.

    ``models=`` a list of ``(label, model_dict)`` tuples or bare model dicts; if
    omitted, every entry in ``state.models`` is used. ``format=`` one of ``"latex"``
    (booktabs), ``"markdown"``, ``"text"`` (default). Coefficients print with the SE in
    parentheses beneath and significance stars; a fit block lists N and estimator.
    """
    fmt = kwargs.get("format", "text")
    models = kwargs.get("models")
    if models is None:
        models = [(k, v) for k, v in state.models.items() if isinstance(v, dict)]
    else:
        norm = []
        for i, m in enumerate(models):
            if isinstance(m, tuple) and len(m) == 2:
                norm.append(m)
            elif isinstance(m, dict):
                norm.append((m.get("estimator") or f"({i + 1})", m))
        models = norm

    cols = []
    all_terms: list[str] = []
    for label, model in models:
        coefs, stats = _extract(model)
        for t in coefs:
            if t not in all_terms:
                all_terms.append(t)
        cols.append({"label": str(label), "coefs": coefs, "stats": stats})
    if not cols or not all_terms:
        state.write("artifacts", "tables", {"content": "", "note": "无可制表的模型结果"})
        return state

    headers = [c["label"] for c in cols]
    nd = int(kwargs.get("decimals", 3))

    def cell(c, term):
        if term not in c["coefs"]:
            return ("", "")
        coef, se, p = c["coefs"][term]
        if coef is None:
            return ("", "")  # no coefficient → no bare stars
        return (_fmt(coef, nd) + _stars(p), f"({_fmt(se, nd)})" if se is not None else "")

    n_row = ["N"] + [str(c["stats"].get("n") or c["stats"].get("n_units")
                     or c["stats"].get("n_treated_obs") or "") for c in cols]

    if fmt == "latex":
        hdr = [_esc_latex(h) for h in headers]
        lines = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
                 " & " + " & ".join(hdr) + r" \\", r"\midrule"]
        for term in all_terms:
            a = [cell(c, term)[0] for c in cols]
            b = [cell(c, term)[1] for c in cols]
            lines.append(f"{_esc_latex(term)} & " + " & ".join(a) + r" \\")
            lines.append(" & " + " & ".join(b) + r" \\")
        lines += [r"\midrule", " & ".join(n_row) + r" \\", r"\bottomrule",
                  r"\multicolumn{%d}{l}{\footnotesize *** p<0.01, ** p<0.05, * p<0.1} \\" % (len(cols) + 1),
                  r"\end{tabular}"]
        content = "\n".join(lines)
    elif fmt == "markdown":
        hdr = [_esc_md(h) for h in headers]
        lines = ["| | " + " | ".join(hdr) + " |",
                 "|" + "---|" * (len(cols) + 1)]
        for term in all_terms:
            a = [cell(c, term)[0] for c in cols]
            b = [cell(c, term)[1] for c in cols]
            lines.append(f"| {_esc_md(term)} | " + " | ".join(a) + " |")
            lines.append("| | " + " | ".join(b) + " |")
        lines.append("| " + " | ".join(n_row) + " |")
        lines.append("\n*** p<0.01, ** p<0.05, * p<0.1")
        content = "\n".join(lines)
    else:  # aligned plain text
        w = max([len(t) for t in all_terms] + [len("N")]) + 2
        cw = max(12, max(len(h) for h in headers) + 2)
        def row(label, vals):
            return label.ljust(w) + "".join(str(v).rjust(cw) for v in vals)
        out = [row("", headers), "-" * (w + cw * len(cols))]
        for term in all_terms:
            a = [cell(c, term)[0] for c in cols]
            b = [cell(c, term)[1] for c in cols]
            out.append(row(term, a))
            out.append(row("", b))
        out += ["-" * (w + cw * len(cols)), row("N", n_row[1:]),
                "*** p<0.01, ** p<0.05, * p<0.1"]
        content = "\n".join(out)

    state.write("artifacts", "tables", {
        "content": content, "format": fmt, "n_models": len(cols), "terms": all_terms,
        "note": "出版级回归表(系数+括注SE+显著性星号);正文数字应与本表一致",
    })
    return state


__all__ = ["regtable"]
