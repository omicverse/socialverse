"""Stata / R / SPSS command-name compatibility aliases.

Researchers coming from Stata, R, or SPSS look for methods by the *command name*
they already know (``xtreg``, ``lmer``, ``stcox``, ``svyglm``, ``FACTOR`` …). This
module attaches those names — prefixed ``py-`` to mark "the Python reimplementation
in socialverse" — as searchable registry aliases, so::

    sv.registry.get("py-lmer")            # -> sv.tl.multilevel
    sv.registry.find("stcox")             # fuzzy-matches py-stcox -> sv.tl.survival
    sv.utils.registry_lookup("py-svyglm") # -> sv.tl.survey_estimate

The function names themselves are unchanged (a hyphen isn't a legal Python
identifier, so these can only be aliases). This is a deliberate compatibility
*layer* kept separate from each function's native `@register` description — a
Rosetta stone mapping the three dominant social-science packages onto socialverse.

Each entry lists the ``py-<command>`` aliases for one socialverse function, with a
comment noting the source tool(s): (S)=Stata, (R)=R, (SPSS)=IBM SPSS Statistics.
Aliases are deduped across functions (an alias binds to exactly one function).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ._registry import FunctionRegistry

# socialverse short-name -> familiar command names from Stata / R / SPSS (py- prefixed)
PY_ALIASES: dict[str, list[str]] = {
    # ---- causal / quasi-experimental -------------------------------------
    "did": ["py-didregress", "py-xtdidregress", "py-csdid", "py-hdidregress",  # (S)
            "py-att_gt", "py-did"],                                            # (R) did::att_gt
    "event_study": ["py-eventdd", "py-event_plot", "py-eventstudyinteract",    # (S)
                    "py-sunab", "py-aggte"],                                    # (R) fixest::sunab, did::aggte
    "parallel_trends": ["py-pretrends"],                                       # (R) HonestDiD/pretrends
    "fect": ["py-fect", "py-ifect", "py-did_imputation", "py-didimputation",   # (R) fect / did_imputation
             "py-did2s", "py-did_multiplegt"],                                 # (R) did2s / DIDmultiplegt
    "rdd": ["py-rdrobust", "py-rdbwselect", "py-rddensity"],                   # (S/R) rdrobust
    "synthetic_control": ["py-synth", "py-synth_runner", "py-sdid",            # (S)
                          "py-gsynth", "py-augsynth", "py-synthdid"],          # (R)
    # ---- regression base (P0) --------------------------------------------
    "glm": ["py-regress", "py-logit", "py-probit", "py-poisson", "py-nbreg",   # (S)
            "py-glm", "py-lm",                                                  # (R)
            "py-REGRESSION", "py-GENLIN", "py-LOGISTIC"],                       # (SPSS)
    "mlogit": ["py-mlogit", "py-multinom", "py-NOMREG"],                       # (S/R/SPSS)
    "ologit": ["py-ologit", "py-oprobit", "py-polr", "py-PLUM"],               # (S/R/SPSS)
    "margins": ["py-margins", "py-marginsplot",                                # (S)
                "py-marginaleffects", "py-emmeans", "py-slopes"],              # (R)
    "iv_regress": ["py-ivregress", "py-ivreg2", "py-ivreg", "py-2SLS"],        # (S/R/SPSS)
    "psm": ["py-psmatch2", "py-teffects", "py-kmatch",                         # (S)
            "py-matchit", "py-weightit"],                                       # (R)
    "mediation": ["py-mediate", "py-med4way", "py-sgmediation",                # (S)
                  "py-mediation", "py-PROCESS"],                                # (R/SPSS Hayes PROCESS)
    # ---- complex survey ---------------------------------------------------
    "declare_design": ["py-svyset", "py-xtset", "py-tsset",                    # (S) design declaration
                       "py-svydesign", "py-declare_design",                    # (R) survey::svydesign
                       "py-CSPLAN"],                                           # (SPSS)
    "design_survey": ["py-alpha",                                             # (S) Cronbach alpha
                      "py-cronbach", "py-psych_alpha",                         # (R) psych::alpha
                      "py-RELIABILITY"],                                       # (SPSS)
    "survey_estimate": ["py-svy", "py-svy_mean", "py-svy_regress",             # (S) svy:
                        "py-svyglm", "py-svymean", "py-svytotal", "py-svyby",  # (R) survey
                        "py-CSGLM", "py-CSLOGISTIC", "py-CSDESCRIPTIVES"],     # (SPSS)
    # ---- psychometrics / SEM / IRT ---------------------------------------
    "cfa": ["py-cfa", "py-lavaan"],                                           # (R) lavaan::cfa; (S) sem
    "sem": ["py-sem", "py-gsem", "py-growth"],                                # (S) sem/gsem; (R) lavaan::sem/growth
    "irt": ["py-irt", "py-irt_2pl", "py-irt_grm",                             # (S) irt
            "py-mirt", "py-ltm", "py-grm", "py-rasch"],                        # (R) mirt/ltm
    "efa": ["py-factor", "py-pca",                                            # (S) factor/pca; (SPSS) FACTOR
            "py-fa", "py-principal", "py-fa_parallel"],                        # (R) psych::fa
    "reliability": ["py-alpha_full", "py-omega", "py-mcdonald",                # (R) psych::alpha/omega
                    "py-icc", "py-item_total"],
    "interrater": ["py-kappa", "py-kap", "py-kappam_fleiss",                   # (S) kap; (R) irr; (SPSS) KAPPA
                   "py-kripp_alpha", "py-krippalpha", "py-irr"],               # (R) irr::kripp.alpha
    # ---- multilevel / longitudinal ---------------------------------------
    "multilevel": ["py-mixed", "py-meglm", "py-melogit", "py-mepoisson",       # (S) mixed/me*
                   "py-xtreg", "py-reghdfe",                                    # (S) panel FE
                   "py-lmer", "py-glmer", "py-lme", "py-feols",                # (R) lme4/fixest
                   "py-GENLINMIXED"],                                         # (SPSS) MIXED == py-mixed

    "survival": ["py-stcox", "py-streg", "py-stset", "py-sts",                 # (S)
                 "py-coxph", "py-survfit", "py-survdiff", "py-Surv",           # (R) survival
                 "py-COXREG", "py-KM"],                                        # (SPSS)
    # ---- spatial ----------------------------------------------------------
    "spatial_autocorr": ["py-spatgsa", "py-spatlsa",                           # (S)
                         "py-moran_test", "py-localmoran", "py-geary_test"],   # (R) spdep
    "spatial_regression": ["py-spregress", "py-spxtregress",                   # (S)
                          "py-lagsarlm", "py-errorsarlm", "py-spautolm",       # (R) spatialreg
                          "py-sacsarlm"],
    # ---- networks ---------------------------------------------------------
    "build_network": ["py-nwcommands",                                        # (S)
                      "py-igraph", "py-graph_from_data_frame", "py-sna"],      # (R)
    "ergm": ["py-ergm", "py-btergm"],                                          # (R) statnet::ergm
    "saom": ["py-siena07", "py-RSiena"],                                       # (R) RSiena
    # ---- QCA --------------------------------------------------------------
    "qca": ["py-fuzzy",                                                        # (S)
            "py-truthTable", "py-minimize", "py-superSubset",                  # (R) QCA
            "py-calibrate", "py-cna"],
    # ---- demography -------------------------------------------------------
    "life_table": ["py-ltable",                                               # (S)
                   "py-lifetable", "py-lt_abridged"],                          # (R) demography
    "decomposition": ["py-oaxaca", "py-kitagawa"],                            # (S/R) oaxaca
    # ---- text / digital humanities ---------------------------------------
    "stylometry": ["py-stylo", "py-rolling_classify", "py-oppose"],            # (R) stylo
    "philology_collate": ["py-collatex"],                                      # external (CollateX)
    # ---- data prep --------------------------------------------------------
    "ingest": ["py-import", "py-use", "py-read_csv"],                          # (S/R) data import
    "build_corpus": ["py-corpus", "py-tokens", "py-dfm"],                      # (R) quanteda
    "redact_pii": ["py-sdcMicro"],                                            # (R) sdcMicro
    # ---- figures ----------------------------------------------------------
    "forest": ["py-coefplot", "py-forestplot"],                               # (S/R)
    "event_study_plot": ["py-iplot"],                                         # (R) fixest::iplot
    "manuscript_docx": ["py-esttab", "py-outreg2"],                           # (S) esttab/outreg2
    "dendrogram": ["py-cluster", "py-hclust"],                                # (S/R)
    "km_curve": ["py-sts_graph", "py-ggsurvplot", "py-stcurve"],              # (S/R)
    "moran_scatter": ["py-moran_plot"],                                       # (R)
    "rdd_plot": ["py-rdplot"],                                                # (R) rdrobust::rdplot
    # ---- literature -------------------------------------------------------
    "zotero_bridge": ["py-zotero"],                                          # Zotero
}


def apply(registry: "FunctionRegistry") -> int:
    """Attach every ``PY_ALIASES`` entry to its function in ``registry``.

    Safe to call more than once (idempotent) and safe when a function is absent
    (e.g. its optional backend failed to import) — unknown names are skipped.
    Returns the total number of newly added aliases.
    """
    total = 0
    for name, aliases in PY_ALIASES.items():
        total += registry.add_aliases(name, aliases)
    return total
