"""Humanities & social-science example datasets (broad-category loaders).

Small, deterministic, synthetic toy datasets with a KNOWN data-generating process —
each wired to a socialverse analysis function so a tutorial/agent can show the method
recovering the truth. Spans labor economics, political science, comparative sociology,
contentious politics, communication/content analysis, psychology panels, complex-survey
methods, and digital-humanities text.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

def load_wages(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Cross-sectional wage data — a Mincer earnings equation with a gender wage gap
    for Blinder-Oaxaca decomposition.

    Data-generating process (known ground truth)
    ---------------------------------------------
    ``log_wage`` is generated from a Mincer equation shared by both genders::

        log_wage = 1.50
                 + 0.080 * education           # return to schooling ~ 0.08 log-wage / yr
                 + 0.030 * experience          # concave experience profile ...
                 - 0.0005 * experience_sq      #   ... (negative quadratic term)
                 + 0.150 * union               # union wage premium
                 + sector_effect               # {tech:+0.20, finance:+0.15, public:0.0, retail:-0.10}
                 - 0.150 * female              # PURE unexplained gender gap (same coefficients)
                 + N(0, 0.35^2)                # idiosyncratic noise

    On top of the identical coefficients, women differ in *endowments* (composition):
    they have slightly less experience on average and are under-represented in the
    high-paying tech/finance sectors and in union jobs. So the raw mean gap
    ``mean(log_wage | female=0) - mean(log_wage | female=1)`` is LARGER than 0.15 and
    splits into:

      * an **explained / endowments** part (experience, union, sector composition), and
      * an **unexplained / coefficients** part ≈ **+0.15** (i.e. women's residual gap
        is about **-0.15** log-wage), which is the intercept ``-0.150 * female`` term.

    Ground truth to recover
    -----------------------
      * return to education  ≈ 0.080 log-wage per year of schooling
      * experience profile is concave (positive linear, negative quadratic)
      * union premium        ≈ 0.150
      * Oaxaca **unexplained** gap ≈ **+0.15** (men over women); women's residual ≈ -0.15
      * total raw gap > 0.15 because women also have worse endowments

    Columns
    -------
    wage : float
        Hourly wage in currency units, ``exp(log_wage)`` (level; for models on levels).
    log_wage : float
        Natural log of wage — the Oaxaca / Mincer outcome.
    female : int (0/1)
        Gender group column (1 = female). Oaxaca uses the LARGER value (female=1) as
        group "A"; the unexplained term flips sign accordingly.
    education : int
        Years of completed schooling (~8..20).
    experience : float
        Years of potential labor-market experience (>= 0).
    experience_sq : float
        ``experience ** 2`` (the quadratic Mincer term).
    union : int (0/1)
        Union membership (1 = member).
    sector : str (categorical)
        One of {"tech", "finance", "public", "retail"}.
    sector_tech, sector_finance, sector_retail : int (0/1)
        One-hot dummies for ``sector`` with ``public`` as the omitted baseline. These
        are numeric so Oaxaca / OLS can attribute the sector *composition* difference
        to the explained (endowments) part rather than to the female coefficient.

    Parameters
    ----------
    n : int
        Number of workers (rows).
    seed : int
        RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    # Gender group (balanced)
    female = rng.integers(0, 2, n)

    # Education: women slightly higher schooling on average (realistic; keeps the gap
    # from being purely explained by endowments).
    education = np.clip(
        np.round(rng.normal(13.0 + 0.3 * female, 2.5)), 8, 20
    ).astype(int)

    # Experience: women have somewhat less potential experience (endowment difference).
    experience = np.clip(
        rng.normal(18.0 - 3.0 * female, 8.0), 0, None
    ).round(3)
    experience_sq = (experience ** 2).round(3)

    # Union: men more likely to be union members (endowment difference).
    p_union = np.where(female == 1, 0.18, 0.30)
    union = (rng.uniform(size=n) < p_union).astype(int)

    # Sector: women under-represented in high-paying tech/finance (endowment difference).
    sectors = np.array(["tech", "finance", "public", "retail"])
    probs_male = np.array([0.35, 0.25, 0.20, 0.20])
    probs_female = np.array([0.15, 0.15, 0.35, 0.35])
    sector = np.empty(n, dtype=object)
    for i in range(n):
        p = probs_female if female[i] == 1 else probs_male
        sector[i] = rng.choice(sectors, p=p)
    sector_map = {"tech": 0.20, "finance": 0.15, "public": 0.0, "retail": -0.10}
    sector_effect = np.array([sector_map[s] for s in sector])

    # Mincer equation with an identical structure for both genders plus a pure
    # (unexplained) female intercept shift of -0.15.
    log_wage = (
        1.50
        + 0.080 * education
        + 0.030 * experience
        - 0.0005 * experience_sq
        + 0.150 * union
        + sector_effect
        - 0.150 * female
        + rng.normal(0.0, 0.35, n)
    )
    wage = np.exp(log_wage)

    return pd.DataFrame(
        {
            "wage": wage.round(4),
            "log_wage": log_wage.round(4),
            "female": female.astype(int),
            "education": education,
            "experience": experience,
            "experience_sq": experience_sq,
            "union": union,
            "sector": sector.astype(str),
            # one-hot dummies (baseline = "public") so sector composition is
            # attributable to endowments in Oaxaca / OLS.
            "sector_tech": (sector == "tech").astype(int),
            "sector_finance": (sector == "finance").astype(int),
            "sector_retail": (sector == "retail").astype(int),
        }
    )


def load_vote(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Individual survey of multi-party vote choice (political behavior).

    Simulated respondents choose one of **3 parties** — ``"left"``, ``"center"``,
    ``"right"`` — via a true multinomial-logit data-generating process. The nominal
    outcome column is ``party``; predictors are ``ideology`` (self-placement on a
    left-right scale, integer −3..3), plus ``age``, ``income``, ``education`` and a
    categorical ``region``.

    Ground truth (utility model, ``center`` is the reference/base category):

        U_left  = -1.10 * ideology + 0.0 * z_income + eps
        U_center= 0                                        (reference)
        U_right = +1.10 * ideology + 0.6 * z_income + eps

    where ``z_income`` is standardized income and ``eps`` is i.i.d. Gumbel noise.
    So **ideology strongly drives party**: a right-leaning respondent (ideology > 0)
    is far more likely to pick ``right``, a left-leaning one to pick ``left``.
    The recoverable truth for ``sv.tl.mlogit`` (base = ``center``):

        coef[ideology] for category "right"  ~ +1.10   (strongly positive)
        coef[ideology] for category "left"   ~ -1.10   (strongly negative)

    Columns
    -------
    party : str
        Nominal vote choice in {"left", "center", "right"} (the mlogit outcome).
    ideology : int
        Left-right self-placement, −3 (far left) .. +3 (far right). Main driver.
    income : float
        Annual income in thousands (roughly 20..120). Nudges toward "right".
    age : int
        Respondent age in years (18..85). No true effect (a nuisance covariate).
    education : int
        Years of schooling (8..20). No true effect (a nuisance covariate).
    region : str
        Categorical region in {"north", "south", "east", "west"}. No true effect.
    """
    rng = np.random.default_rng(seed)

    # --- covariates ---
    ideology = rng.integers(-3, 4, n)                    # -3..3 inclusive
    age = rng.integers(18, 86, n)
    education = rng.integers(8, 21, n)
    income = np.clip(rng.normal(70, 20, n), 20, 120)
    region = rng.choice(["north", "south", "east", "west"], n)

    z_income = (income - income.mean()) / income.std()

    # --- true multinomial-logit utilities (center = reference, utility 0) ---
    b_ideo = 1.10        # ground-truth ideology slope (right +, left -)
    b_inc = 0.60         # income mildly favors "right"
    u_left = -b_ideo * ideology + 0.0 * z_income
    u_center = np.zeros(n)
    u_right = b_ideo * ideology + b_inc * z_income

    # Gumbel-max trick: adding i.i.d. Gumbel noise and taking argmax
    # yields exact multinomial-logit choice probabilities.
    g = rng.gumbel(0.0, 1.0, size=(n, 3))
    utils = np.column_stack([u_left, u_center, u_right]) + g
    choice = utils.argmax(axis=1)
    party = np.array(["left", "center", "right"])[choice]

    return pd.DataFrame({
        "party": party,
        "ideology": ideology.astype(int),
        "income": income.round(2),
        "age": age.astype(int),
        "education": education.astype(int),
        "region": region,
    })


def load_values(seed: int = 0) -> pd.DataFrame:
    """Cross-national values survey: respondents nested in ~20 countries (a
    comparative-sociology, two-level design) for fitting a random-intercept
    multilevel model. Individual respondents (level 1) are nested within
    ``country`` (level 2); a numeric trust/values scale is the outcome.

    Ground-truth data-generating process
    -------------------------------------
    For country ``j`` with a random intercept ``u_j ~ Normal(0, sd_u)`` and
    respondent ``i``::

        trust_ij = 5.0 + u_j
                       + 0.40 * education_ij      # true positive education effect
                       - 0.010 * (age_ij - 45)    # mild age gradient
                       + 0.15 * gdp_pc_j_centered # country-level context effect
                       + e_ij,   e_ij ~ Normal(0, sd_e)

    with ``sd_u = 0.46`` (between-country intercept variance ~ 0.21) and
    ``sd_e = 1.00`` (residual variance ~ 1.0). This yields a target intraclass
    correlation ``ICC = sd_u**2 / (sd_u**2 + sd_e**2) ~ 0.15`` (in the
    0.10-0.15 band typical of cross-national attitude data), and a recoverable
    positive fixed effect of ``education`` of ``+0.40``.

    A multilevel model (``sv.tl.multilevel`` with ``groups="country"``,
    ``predictors=["education", "age"]``, ``outcome="trust"``) should recover
    the education slope ~ 0.40 and an ICC in the 0.10-0.15 range.

    Columns
    -------
    country : str
        Level-2 grouping / cluster id (the random-intercept factor), e.g. "C01".
        20 countries.
    respondent : int
        Within-country respondent index (level-1 unit id).
    trust : float
        Outcome - a numeric interpersonal-trust / values scale (higher = more
        trusting), roughly on a 0-10 range.
    age : float
        Respondent age in years (level-1 covariate).
    education : float
        Respondent education, in years of schooling (level-1 covariate; true
        positive effect on ``trust``).
    gdp_pc : float
        Country-level GDP per capita (in $1k), a level-2 covariate constant
        within each country.

    Parameters
    ----------
    seed : int
        Seed for ``numpy.random.default_rng``.
    """
    rng = np.random.default_rng(seed)

    n_countries = 20
    n_per = 60                       # respondents per country -> 1200 rows

    sd_u = 0.46                      # between-country intercept sd (ICC ~0.15)
    sd_e = 1.00                      # residual sd

    # country-level GDP per capita ($1k), and the country random intercept
    gdp_pc = rng.uniform(8.0, 60.0, n_countries)
    gdp_c = (gdp_pc - gdp_pc.mean()) / 10.0        # centered & scaled context var
    u = rng.normal(0.0, sd_u, n_countries)         # random intercepts

    rows = []
    for j in range(n_countries):
        for i in range(n_per):
            age = rng.uniform(18, 80)
            education = rng.uniform(6, 20)          # years of schooling
            trust = (
                5.0
                + u[j]
                + 0.40 * education                  # <-- ground-truth education effect
                - 0.010 * (age - 45.0)
                + 0.15 * gdp_c[j]
                + rng.normal(0.0, sd_e)
            )
            rows.append({
                "country": f"C{j + 1:02d}",
                "respondent": i,
                "trust": round(float(trust), 4),
                "age": round(float(age), 2),
                "education": round(float(education), 2),
                "gdp_pc": round(float(gdp_pc[j]), 2),
            })
    return pd.DataFrame(rows)


def load_protest(n_countries: int = 60, n_years: int = 12, seed: int = 0) -> pd.DataFrame:
    """Country-year panel of protest *event counts* from a known Poisson DGP.

    Each row is one country-year. The number of protests is drawn from a Poisson
    whose (log) mean is a linear function of covariates plus a population offset,
    so a Poisson GLM should recover the coefficients below.

    Ground-truth DGP
    ----------------
        log E[n_protests] = log_population            (offset, coefficient fixed at 1)
                          + beta0
                          + beta_dem  * democracy
                          + beta_gdp  * gdp_growth
                          + beta_urb  * urban_pct_std
    with

        beta0    = -9.5     (intercept, on the per-capita log scale)
        beta_dem = +0.60    (democracies protest MORE, holding size fixed)
        beta_gdp = -0.05    (per +1 percentage-point growth, protests FEWER)
        beta_urb = +0.30    (more urban -> more protests; on standardized urban_pct)

    ``urban_pct`` enters the DGP in *standardized* form (mean 0, sd 1); the raw
    ``urban_pct`` column (roughly 20-90) is provided for realism, so a GLM run on
    the raw column will recover a rescaled (smaller-magnitude) coefficient of the
    same positive sign. ``democracy`` (+) and ``gdp_growth`` (-) recover directly.

    Because the mean already contains ``log_population`` with a unit coefficient,
    the model is most faithfully fit with ``offset=log_population`` (an exposure
    term). With statsmodels this is passed via ``sm.GLM(..., offset=...)``; when
    fitting through ``sv.tl.glm`` without offset support, include ``log_population``
    as an ordinary predictor and its coefficient will recover to about +1.0.

    Columns
    -------
    country : str
        Country id, ``"C00".."C{n-1}"``.
    year : int
        Calendar year (2000 .. 2000 + n_years - 1).
    n_protests : int
        Outcome — non-negative integer count of protest events that year (Poisson).
    gdp_growth : float
        Annual real GDP growth, in percentage points (roughly -6 .. +10).
    democracy : int
        1 if the country-year is a democracy, else 0 (time-invariant per country).
    urban_pct : float
        Percent of population living in urban areas (roughly 20 .. 90).
    log_population : float
        Natural log of population (~11-15, i.e. ~60k-3M) — the recommended
        Poisson exposure ``offset``; recovers with coefficient ~+1.0 as a predictor.

    Parameters
    ----------
    n_countries : int
        Number of countries in the panel.
    n_years : int
        Number of years per country.
    seed : int
        RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    beta0 = -9.5
    beta_dem = 0.60
    beta_gdp = -0.05
    beta_urb = 0.30

    # Time-invariant country attributes
    democracy_c = (rng.uniform(size=n_countries) < 0.5).astype(int)
    urban_base_c = rng.uniform(20.0, 90.0, n_countries)          # raw urban %
    logpop_base_c = rng.uniform(11.0, 15.0, n_countries)          # log population (~60k-3M)

    rows = []
    for c in range(n_countries):
        for t in range(n_years):
            # mild within-country drift so covariates vary over time
            urban = np.clip(urban_base_c[c] + rng.normal(0, 1.5), 5.0, 99.0)
            log_pop = logpop_base_c[c] + rng.normal(0, 0.02)
            gdp_growth = rng.normal(2.5, 3.0)                     # % points
            rows.append({
                "country": f"C{c:02d}",
                "year": 2000 + t,
                "gdp_growth": round(float(gdp_growth), 3),
                "democracy": int(democracy_c[c]),
                "urban_pct": round(float(urban), 2),
                "log_population": round(float(log_pop), 4),
                "_urban_raw": float(urban),
                "_logpop_raw": float(log_pop),
                "_gdp_raw": float(gdp_growth),
                "_dem_raw": int(democracy_c[c]),
            })

    df = pd.DataFrame(rows)

    # Standardize urban for the DGP (documented above)
    u = df["_urban_raw"].to_numpy()
    urban_std = (u - u.mean()) / u.std()

    log_mu = (
        df["_logpop_raw"].to_numpy()                    # offset, unit coefficient
        + beta0
        + beta_dem * df["_dem_raw"].to_numpy()
        + beta_gdp * df["_gdp_raw"].to_numpy()
        + beta_urb * urban_std
    )
    mu = np.exp(log_mu)
    df["n_protests"] = rng.poisson(mu).astype(int)

    df = df.drop(columns=["_urban_raw", "_logpop_raw", "_gdp_raw", "_dem_raw"])
    return df[["country", "year", "n_protests", "gdp_growth",
               "democracy", "urban_pct", "log_population"]]


def load_coding(n_docs: int = 200, n_coders: int = 3, seed: int = 0) -> pd.DataFrame:
    """Content-analysis coding: ``n_docs`` documents each scored on the SAME nominal
    framing category (1..4) by ``n_coders`` independent coders — the classic
    inter-coder reliability setup a communication-research tutorial recovers with
    ``sv.tl.interrater``.

    Data-generating process (wide frame, one row per document):

    * Each document has a *true* framing category drawn from a mildly skewed
      1..4 distribution (probs [0.35, 0.30, 0.20, 0.15]) — realistic base rates
      that keep chance agreement non-trivial.
    * Each coder reports the true category with probability ``p_agree = 0.83``;
      otherwise they emit a uniformly-drawn *other* category (a confusion error).
      This ``p_agree`` is tuned so the chance-corrected agreement lands in the
      **moderate/substantial** band, i.e. the GROUND TRUTH is **Fleiss' κ ≈ 0.6**
      (Krippendorff's α tracks it, ≈ 0.6; pairwise Cohen κ on any two coders
      ≈ 0.6 as well). Percent (raw) agreement is higher, ≈ 0.68, as expected.

    Ground truth summary:
        - true category ∈ {1,2,3,4}, base rates [0.35, 0.30, 0.20, 0.15]
        - per-coder accuracy p_agree = 0.83
        - Fleiss κ ≈ 0.6 (moderate/substantial inter-coder agreement)
        - Krippendorff α (nominal) ≈ 0.6

    Columns
    -------
    doc_id : int
        Document identifier 0..n_docs-1.
    coder_1, coder_2, ..., coder_{n_coders} : int
        Each coder's assigned framing category (nominal 1..4) for that document.
        Pass these as ``raters=['coder_1', ...]`` to ``sv.tl.interrater``.
    true_label : int
        The latent true framing category (for reference / diagnostics only; not a
        rater column).

    Returns
    -------
    pandas.DataFrame
        ``n_docs`` rows; wide subjects × raters frame ready for
        ``sv.tl.interrater``.
    """
    rng = np.random.default_rng(seed)
    categories = np.array([1, 2, 3, 4])
    base_rates = np.array([0.35, 0.30, 0.20, 0.15])
    p_agree = 0.83

    true_label = rng.choice(categories, size=n_docs, p=base_rates)

    coder_cols: dict[str, np.ndarray] = {}
    for c in range(n_coders):
        assigned = np.empty(n_docs, dtype=int)
        for i in range(n_docs):
            if rng.uniform() < p_agree:
                assigned[i] = true_label[i]
            else:
                # a confusion error: pick a DIFFERENT category uniformly
                others = categories[categories != true_label[i]]
                assigned[i] = rng.choice(others)
        coder_cols[f"coder_{c + 1}"] = assigned

    data = {"doc_id": np.arange(n_docs)}
    data.update(coder_cols)
    data["true_label"] = true_label
    return pd.DataFrame(data)


def load_wellbeing(n_persons: int = 200, n_waves: int = 6, seed: int = 0) -> pd.DataFrame:
    """Individual x wave panel of subjective well-being (person-within-panel).

    A two-level panel for a random-intercept multilevel model: repeated ``wave``
    observations are nested within ``person_id``. Each person carries a large,
    stable random intercept, so the intraclass correlation (ICC) is high — most of
    the variance in ``life_satisfaction`` is *between* people, not within.

    Data-generating process (ground truth)::

        life_satisfaction_it = 6.0
                               + u_i                 # person random intercept
                               + 1.50 * income_it    # within-person income effect (log income)
                               - 1.20 * employed==0  # i.e. UNEMPLOYMENT penalty of 1.20
                               + 0.80 * married_it    # marriage bonus
                               + e_it                # residual

    where ``u_i ~ N(0, 1.6^2)`` (person random intercept, sd_u = 1.6) and
    ``e_it ~ N(0, 0.9^2)`` (within-person residual, sd_e = 0.9). This implies a
    high intraclass correlation on the raw intercept scale::

        ICC = sd_u^2 / (sd_u^2 + sd_e^2) = 1.6^2 / (1.6^2 + 0.9^2) ≈ 0.76

    Ground truth to recover
    -----------------------
    * income (log) fixed slope  = +1.50   (within-person positive effect)
    * unemployment penalty      = -1.20   (employed=1 raises satisfaction ~1.20 vs employed=0;
                                            equivalently the coefficient on ``employed`` is +1.20)
    * married bonus             = +0.80
    * ICC (person random intercept) ≈ 0.76  (high — driven by sd_u=1.6 vs sd_e=0.9)

    Columns
    -------
    person_id : int
        Level-2 grouping id (the individual). Pass as ``groups`` to multilevel.
    wave : int
        Panel time index 0..n_waves-1 (level-1 within-person occasion).
    life_satisfaction : float
        Numeric outcome (subjective well-being, roughly 0-10 scale).
    income : float
        Log income at that wave (predictor; positive within-person effect).
    employed : int
        1 = employed, 0 = unemployed (predictor; unemployment lowers satisfaction).
    married : int
        1 = married, 0 = not (predictor; marriage raises satisfaction).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_persons):
        u = rng.normal(0.0, 1.6)                       # person random intercept
        base_income = rng.normal(10.0, 0.5)            # person's mean log income
        for w in range(n_waves):
            # log income drifts around the person's baseline over waves
            income = base_income + 0.15 * w + rng.normal(0.0, 0.3)
            employed = int(rng.uniform() < 0.85)        # mostly employed
            married = int(rng.uniform() < 0.55)
            y = (6.0
                 + u
                 + 1.50 * (income - 10.0)               # center income so intercept is interpretable
                 - 1.20 * (1 - employed)                # unemployment penalty
                 + 0.80 * married
                 + rng.normal(0.0, 0.9))                 # within-person residual
            rows.append({
                "person_id": i,
                "wave": w,
                "life_satisfaction": round(float(y), 4),
                "income": round(float(income), 4),
                "employed": employed,
                "married": married,
            })
    return pd.DataFrame(rows)


def load_complex_survey(seed: int = 0) -> pd.DataFrame:
    """Complex-sample individual-level survey data for design-based estimation.

    A stratified, clustered health survey: individuals nested in PSUs (primary
    sampling units) nested in strata, each individual carrying a sampling
    ``weight``. Because high-prevalence regions are DELIBERATELY oversampled
    (given *small* weights) while low-prevalence regions are undersampled (given
    *large* weights), the naive unweighted sample mean of ``hypertension`` is
    biased UPWARD; the design-weighted (Horvitz-Thompson) prevalence recovers the
    true population value.

    Data-generating process
    ------------------------
    - 4 strata (region A/B/C/D). Region-specific TRUE prevalences are
      ``{A: 0.10, B: 0.20, C: 0.35, D: 0.50}`` and TRUE population shares are
      ``{A: 0.40, B: 0.30, C: 0.20, D: 0.10}`` -- so the design-weighted
      population prevalence is the share-weighted mean:
      ``0.40*0.10 + 0.30*0.20 + 0.20*0.35 + 0.10*0.50 = 0.220``.
    - Each individual's ``hypertension`` (0/1) is a Bernoulli draw at that
      region's true prevalence, mildly nudged by ``age`` (older => slightly
      higher risk); the age tilt is centered on the sample mean age so each
      region's mean stays exactly at ``base`` while a design-based WLS
      ``hypertension ~ age`` still recovers the small positive age slope
      (true ~ +0.004 per year).
    - Sampling is intentionally disproportionate to the population: high-prevalence
      strata are oversampled, so each individual's ``weight`` = (population share
      of its stratum) / (sample share of its stratum), i.e. inverse selection
      probability. Weighting UNDOES the oversampling and returns the truth.

    GROUND TRUTH
    ------------
    - Design-weighted prevalence of ``hypertension`` ~ **0.220**
      (recovered by ``sv.tl.survey_estimate`` as the weighted ``const``).
    - Naive unweighted sample mean is materially HIGHER (~0.33) because
      high-prevalence strata are oversampled -- the weighted vs unweighted gap is
      the whole point of the design-based estimate.
    - Design-based age slope on hypertension ~ **+0.004** per year of age.

    Columns
    -------
    hypertension : int (0/1)
        Binary health outcome (the survey estimand). Also usable as a continuous
        0/1 outcome for the weighted mean / WLS.
    weight : float
        Sampling weight = inverse selection probability (design weight). Sums to
        approximately the population size N; larger weight => each respondent
        represents more people.
    stratum : str
        Design stratum id (``"S_A"``..``"S_D"``) -- the sampling stratification.
    psu : int
        Primary sampling unit id (cluster), unique across the whole frame; used
        for cluster-robust variance in ``survey_estimate``.
    region : str
        Region label (``"A"``..``"D"``), aligned 1:1 with ``stratum``.
    age : int
        Respondent age in years (18-85); weak positive driver of the outcome.
    """
    rng = np.random.default_rng(seed)

    regions = ["A", "B", "C", "D"]
    true_prev = {"A": 0.10, "B": 0.20, "C": 0.35, "D": 0.50}
    pop_share = {"A": 0.40, "B": 0.30, "C": 0.20, "D": 0.10}

    # Disproportionate sample: oversample the high-prevalence strata so the naive
    # mean is biased and weighting is required to recover the truth.
    n_sample = {"A": 300, "B": 300, "C": 400, "D": 400}
    N_pop = 100_000  # notional finite population the weights blow up to

    n_psu_per_stratum = 10
    age_beta = 0.004  # true per-year effect on hypertension probability

    rows = []
    psu_counter = 0
    for reg in regions:
        n = n_sample[reg]
        # population count vs sample count -> design weight (inverse prob of selection)
        pop_count = pop_share[reg] * N_pop
        weight = pop_count / n  # each sampled unit represents this many people

        # assign individuals to PSUs (clusters) within the stratum
        psu_ids = np.arange(psu_counter, psu_counter + n_psu_per_stratum)
        psu_counter += n_psu_per_stratum
        psu_assign = rng.integers(0, n_psu_per_stratum, n)

        age = rng.integers(18, 86, n)
        base = true_prev[reg]
        # Center the age tilt on the sample's own mean age so each region's mean
        # probability stays exactly at ``base`` (the age effect is orthogonal to
        # the intercept), while the per-year slope remains recoverable.
        p = np.clip(base + age_beta * (age - age.mean()), 0.01, 0.99)
        y = (rng.uniform(size=n) < p).astype(int)

        for i in range(n):
            rows.append({
                "hypertension": int(y[i]),
                "weight": round(float(weight), 4),
                "stratum": f"S_{reg}",
                "psu": int(psu_ids[psu_assign[i]]),
                "region": reg,
                "age": int(age[i]),
            })

    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_speeches(seed: int = 0) -> pd.DataFrame:
    """A small labelled speech corpus for corpus building + text classification.

    Data-generating process
    ------------------------
    Each row is one short *document* (1-3 English sentences) attributed to an
    ``author`` in a given ``year`` and tagged with a two-class ``label`` (party:
    ``"blue"`` vs ``"red"``). The generative process is a simple label-conditioned
    bag-of-words: every document is stitched together from three pools of tokens —

    * a shared **neutral** vocabulary drawn by *both* labels, plus
    * a set of **marker** words that are strongly over-represented in one label.

    Ground truth
    ------------
    The label is *recoverable from the vocabulary*: a handful of marker words are
    emitted almost exclusively under one label, so their per-label frequency
    (or a bag-of-words classifier) reveals the association.

    * ``"blue"`` markers (appear in ~blue docs, rare in red):
      ``healthcare, workers, climate, union, equality``
    * ``"red"``  markers (appear in ~red docs, rare in blue):
      ``tax, border, freedom, faith, business``
    * neutral words (roughly equal across labels):
      ``today, country, people, future, together, believe, nation, work``

    A marker word lands in a same-label document with probability ~0.9 and in an
    opposite-label document with probability ~0.1, so the marker-word rate is
    ~9x higher in its own label. Labels are balanced ~50/50. ``year`` is drawn
    uniformly from 2000-2020 and carries no signal; ``author`` is one of six
    fictional names nested within label (three per label) and is informative only
    through the label it belongs to.

    Columns
    -------
    doc_id : str    stable document identifier (``"doc0000"`` ...), matches the
                    ``{doc_id: text}`` mapping consumed by ``build_corpus``.
    text   : str    1-3 sentence English speech excerpt (the raw document).
    label  : str    party tag, ``"blue"`` or ``"red"`` (the class to recover).
    author : str    fictional speaker, nested within ``label``.
    year   : int    year of the speech, 2000-2020 (noise; no label signal).

    Returns
    -------
    pandas.DataFrame with columns ``[doc_id, text, label, author, year]``,
    ``n=120`` rows. Feed the ``{doc_id: text}`` mapping to
    ``sv.pp.build_corpus`` (via ``sources['corpora']``); use ``label`` as the
    supervised target for stylometry / qualitative-coding / classification.
    """
    rng = np.random.default_rng(seed)

    neutral = ["today", "country", "people", "future", "together",
               "believe", "nation", "work"]
    markers = {
        "blue": ["healthcare", "workers", "climate", "union", "equality"],
        "red": ["tax", "border", "freedom", "faith", "business"],
    }
    authors = {
        "blue": ["A. Rivera", "J. Chen", "M. Okafor"],
        "red": ["R. Blackwood", "T. Hollis", "S. Vance"],
    }

    n = 120
    labels = np.array(["blue", "red"])[rng.integers(0, 2, n)]

    rows = []
    for i in range(n):
        lab = str(labels[i])
        other = "red" if lab == "blue" else "blue"
        author = str(rng.choice(authors[lab]))
        year = int(rng.integers(2000, 2021))

        n_sent = int(rng.integers(1, 4))  # 1-3 sentences
        sentences = []
        for _ in range(n_sent):
            toks = []
            # 2-4 neutral words per sentence
            for w in rng.choice(neutral, size=int(rng.integers(2, 5)), replace=True):
                toks.append(str(w))
            # same-label markers land with high prob, opposite-label markers rarely
            for w in markers[lab]:
                if rng.uniform() < 0.9:
                    toks.append(w)
            for w in markers[other]:
                if rng.uniform() < 0.1:
                    toks.append(w)
            rng.shuffle(toks)
            if toks:
                sent = " ".join(toks)
                sentences.append(sent[0].upper() + sent[1:] + ".")
        if not sentences:  # guarantee non-empty document
            sentences = ["Today the people work together."]
        text = " ".join(sentences)

        rows.append({
            "doc_id": f"doc{i:04d}",
            "text": text,
            "label": lab,
            "author": author,
            "year": year,
        })

    return pd.DataFrame(rows, columns=["doc_id", "text", "label", "author", "year"])


__all__ = [
    'load_wages', 'load_vote', 'load_values', 'load_protest', 'load_coding', 'load_wellbeing', 'load_complex_survey', 'load_speeches',
]
