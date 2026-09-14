import numpy as np
import pandas as pd
from scipy import stats, optimize

"""
Track B — synthetic UPI ticket-size generation.

Calibration source: docs/rbi_dpss_upi_stats.md (RBI DPSS / NPCI secondary
reporting). Fitting work validated in notebooks/03_synthetic_upi.ipynb
before landing here — see that notebook for the full derivation, the
rejected P2P alternative, and the validation plot.

Both P2M and P2P ticket sizes are modeled as log-normal, per the stats doc's
own guidance (heavily right-skewed, log-normal/gamma fit much better than
normal). This is a calibrated-to-aggregates fit, not learned from individual-
level microdata — RBI/NPCI don't publish that. Document this limitation
explicitly wherever these functions get cited (matches the research
proposal's Section 12.1 framing).
"""

def fit_lognormal_mean_percentile(target_mean, target_pctile_value, pctile, root="larger"):
    """
    Closed-form solve for log-normal (mu, sigma) given a target mean and a
    target percentile.

    For X ~ LogNormal(mu, sigma):
        mean            = exp(mu + sigma^2/2)                  ... (1)
        p-th percentile = exp(mu + sigma*z_p), z_p = Phi^-1(p)  ... (2)
    (1)-(2) gives a quadratic in sigma: 0.5*sigma^2 - z_p*sigma - ln(mean/pctile_value) = 0

    The ratio target_pctile_value/target_mean is only achievable up to
    exp(0.5*z_p^2) (attained at sigma == z_p). Below that ratio there are two
    real roots (a smaller-sigma/less-skewed one and a larger-sigma/more-skewed
    one, symmetric around z_p); at it, one root; above it, none (raises).

    Returns (mu, sigma).
    """
    z_p = stats.norm.ppf(pctile)
    c = np.log(target_mean / target_pctile_value)
    discriminant = z_p**2 + 2 * c

    if discriminant < 0:
        max_ratio = np.exp(0.5 * z_p**2)
        actual_ratio = target_pctile_value / target_mean
        raise ValueError(
            f"Infeasible: target ratio (pctile/mean) = {actual_ratio:.2f} exceeds "
            f"the max achievable {max_ratio:.2f} for p={pctile} (z={z_p:.4f}). "
            f"No real (mu, sigma) satisfies both constraints."
        )

    sqrt_disc = np.sqrt(discriminant)
    sigma = (z_p + sqrt_disc) if root == "larger" else (z_p - sqrt_disc)
    mu = np.log(target_mean) - sigma**2 / 2
    return mu, sigma


def truncated_lognormal_percentile(mu, sigma, cap, p):
    """
    F_trunc^-1(p) for X ~ LogNormal(mu, sigma) truncated to X <= cap - i.e. the
    p-th percentile of the CONDITIONAL distribution given X<=cap, not of the
    unconditional one. Needed because re-solving mu alone to fix a truncated
    mean shifts every percentile of the untruncated distribution (including
    p86) right along with it - checking dist.ppf(0.86) on the shifted
    (mu, sigma) is the wrong quantity; this is the right one.

    P(X<=x | X<=cap) = p  <=>  Phi((ln x - mu)/sigma) = p * Phi(z_cap)
    """
    z_cap = (np.log(cap) - mu) / sigma
    inner = p * stats.norm.cdf(z_cap)
    if inner >= 1:
        return np.inf
    z_p = stats.norm.ppf(inner)
    return np.exp(mu + sigma * z_p)


def fit_p2m_params(cap=None):
    """
    P2M ticket size: both constraints are real RBI-derived figures
    (ATS=Rs.659, 86th percentile=Rs.500 - see stats doc Sec. 2-3), so this is
    an exact solve, not an assumption. The other algebraic root gives a
    negative sigma (invalid), so mu=3.630972, sigma=2.391548 is the unique
    valid solution when cap=None (untruncated), confirmed in
    notebooks/03_synthetic_upi.ipynb.

    NOTE - implied median is ~Rs.38 (cap=None case): forcing one log-normal
    through both the mean and the 86th percentile simultaneously pushes over
    half the distribution below Rs.38, with a long tail pulling the mean up
    to 659. This is a direct, unavoidable consequence of the two constraints,
    not an extra assumption layered on top - flag it wherever this fit is
    cited (e.g. thesis methodology) so it isn't a surprise later.

    cap: BUG FOUND AND FIXED - cap=None (the original design) is only safe in
    percentile terms (p99.9~Rs.61,170, well under every published ceiling),
    but a log-normal's tail is unbounded, and at production scale (11.7M P2M
    draws in the full 307,505-applicant run) the realized max was
    Rs.1.24 CRORE - ~12x even the highest cited P2M ceiling. Confirmed the
    naive fix (resample-above-cap using the untruncated mu/sigma unchanged)
    is not good enough either: even a 0.05%-of-draws resample fraction drags
    the sample mean down 18% (659->538, cap=Rs.1L) - the same
    heavy-tail-carries-disproportionate-mean-weight effect as the original
    P2P bug. A mu-only re-solve (mirroring the P2P fix) doesn't work here
    either, because P2M has TWO real constraints, not one: re-solving mu
    alone to hit the truncated mean drags the truncated p86 off-target too
    (tested: p86 drifted to Rs.632 at cap=Rs.1L when only mu moved).

    Fix: when cap is given, jointly re-solve BOTH mu and sigma via
    scipy.optimize.fsolve so the TRUNCATED distribution hits both
    E[X|X<=cap]=659 and the truncated 86th percentile=500 simultaneously
    (see truncated_lognormal_mean and truncated_lognormal_percentile).
    Converges cleanly for any reasonable cap, e.g. cap=300,000 (this
    project's default, see generate_p2m_amounts) gives mu=3.530206,
    sigma=2.485975, P(X>cap)=0.013% pre-resample.

    Pass cap=None to get the original untruncated exact fit (e.g. for
    reference/analysis); pass the same cap used by the amount generator
    downstream to get parameters consistent with what will actually be
    sampled and resampled.
    """
    if cap is None:
        return fit_lognormal_mean_percentile(
            target_mean=659, target_pctile_value=500, pctile=0.86, root="larger"
        )

    mu0, sigma0 = fit_lognormal_mean_percentile(
        target_mean=659, target_pctile_value=500, pctile=0.86, root="larger"
    )

    def equations(params):
        mu, sigma = params
        eq1 = truncated_lognormal_mean(mu, sigma, cap) - 659
        eq2 = truncated_lognormal_percentile(mu, sigma, cap, 0.86) - 500
        return [eq1, eq2]

    (mu, sigma), infodict, ier, msg = optimize.fsolve(equations, x0=[mu0, sigma0], full_output=True, xtol=1e-12)
    if ier != 1:
        raise RuntimeError(f"fit_p2m_params: truncated joint solve did not converge for cap={cap}: {msg}")
    return mu, sigma


def truncated_lognormal_mean(mu, sigma, cap):
    """
    E[X | X <= cap] for X ~ LogNormal(mu, sigma). No closed form for mu given
    this - fit_p2p_params solves it numerically. Derivation: with
    z_b = (ln(cap) - mu) / sigma,
        E[X; X<=cap] = exp(mu + sigma^2/2) * Phi(z_b - sigma)   (shift lemma)
        P(X<=cap)    = Phi(z_b)
        E[X|X<=cap]  = E[X; X<=cap] / P(X<=cap)
    """
    z_b = (np.log(cap) - mu) / sigma
    return np.exp(mu + sigma**2 / 2) * stats.norm.cdf(z_b - sigma) / stats.norm.cdf(z_b)


def fit_p2p_params(cap=100_000, target_mean=2812, sigma_p2m=None, bracket=(0.1, 15.0)):
    """
    P2P ticket size: only one real RBI-derived figure exists (ATS=Rs.2812,
    stats doc Sec. 2) - no published percentile to pin down a shape. The
    Rs.1,00,000 P2P cap is a regulatory ceiling, not a percentile statistic;
    treating it as one was tested in notebooks/03_synthetic_upi.ipynb at
    p99/99.5/99.9/99.99 and rejected - p99/99.5 are mathematically infeasible
    for any log-normal, and p99.9/99.99 force the median down to Rs.0.1-0.03,
    which is absurd for person-to-person transfers.

    ASSUMPTION (flagged, not RBI-sourced): borrow P2M's fitted sigma (shape)
    on the reasoning that both are UPI ticket-size distributions off the same
    payment rail. Document this explicitly as a calibrated-not-learned
    limitation wherever cited.

    mu is then solved so that E[X | X <= cap] == target_mean, NOT the
    untruncated mean == target_mean. Rs.2812 is RBI's real-world observed
    ATS, already produced under the live Rs.1,00,000 cap - matching it to the
    untruncated mean (the original approach) leaves the post-cap mean
    ~26% short after clipping/resampling. Solved numerically via
    scipy.optimize.brentq (no closed form for mu here), confirmed in
    notebooks/03_synthetic_upi.ipynb: mu=5.753896 hits E[X|X<=cap]=2812.00
    exactly, vs. an untruncated theoretical mean of ~5506 for that same mu
    (expected - extra mass has to be pushed up to compensate for what
    truncation removes).

    Pass sigma_p2m explicitly to avoid silently refitting P2M twice in a
    pipeline; defaults to fit_p2m_params()'s sigma if omitted.
    """
    if sigma_p2m is None:
        _, sigma_p2m = fit_p2m_params()

    sigma = sigma_p2m
    f = lambda mu: truncated_lognormal_mean(mu, sigma, cap) - target_mean
    lo, hi = bracket
    if not (f(lo) < 0 < f(hi)):
        raise ValueError(
            f"bracket {bracket} does not contain a root for cap={cap}, "
            f"target_mean={target_mean}, sigma={sigma}"
        )
    mu = optimize.brentq(f, lo, hi, xtol=1e-10)
    return mu, sigma


def generate_p2m_amounts(n, cap=300_000, random_state=None):
    """
    Sample n P2M transaction amounts from the fitted log-normal, by
    RESAMPLING any draw above cap rather than clipping it (same reasoning as
    P2P: clipping creates an artificial point-mass spike at exactly cap).

    cap defaults to Rs.3,00,000 - the MIDPOINT of the stats doc's general P2M
    range (Rs.1-5 lakh depending on category, Sec. 5). No single figure is
    published; the doc's other number (up to Rs.10 lakh/DAY for select
    verified merchants as of Sept 2025) is a daily aggregate limit, not a
    per-transaction one, so it isn't comparable and wasn't used to anchor
    this. Document this as a flagged assumption wherever cap is cited, same
    pattern as the P2P cap.

    cap=None (the ORIGINAL design) is a confirmed BUG, not a valid option to
    reach for casually: it only checked that p99.9 (~Rs.61,170) sits under
    every ceiling, but a log-normal's tail is unbounded, and at production
    scale (11.7M draws) the realized max was Rs.1.24 CRORE. See
    fit_p2m_params's docstring for the full diagnosis, including why a
    mu-only re-solve doesn't work (breaks the p86 constraint) and why the
    fix needs a joint (mu, sigma) re-solve under truncation.

    Calls fit_p2m_params(cap=cap) (not cap=None) so the sampled distribution
    is the truncation-corrected one - its untruncated mean/p86 no longer
    equal 659/500 by themselves, but the post-resample values do.
    """
    mu, sigma = fit_p2m_params(cap=cap)
    dist = stats.lognorm(s=sigma, scale=np.exp(mu))
    rng = np.random.default_rng(random_state)

    samples = dist.rvs(size=n, random_state=rng)
    n_resampled = 0
    if cap is not None:
        mask = samples > cap
        while mask.any():
            n_resampled += mask.sum()
            samples[mask] = dist.rvs(size=mask.sum(), random_state=rng)
            mask = samples > cap
        print(f"P2M: resampled {n_resampled} draw(s) ({n_resampled/n*100:.4f}% of n) that exceeded "
              f"cap={cap}; final mean={samples.mean():.1f} (target ATS=659), "
              f"max={samples.max():.1f} (< cap, no point mass)")

    return samples


def generate_p2p_amounts(n, cap=100_000, random_state=None):
    """
    Sample n P2P transaction amounts from the fitted log-normal (see
    fit_p2p_params for the shape-borrowed-from-P2M assumption and the
    truncated-mean correction), by RESAMPLING any draw above cap rather than
    clipping it to cap.

    cap defaults to Rs.1,00,000 - the confirmed, unchanged P2P per-transaction
    ceiling (stats doc Sec. 5). fit_p2p_params solves mu so E[X|X<=cap] (i.e.
    the mean of exactly this resampling scheme) equals the real-world ATS of
    Rs.2812 - so this function's output mean should land near 2812, not the
    ~26% short. Confirmed in notebooks/03_synthetic_upi.ipynb: sample mean
    2807.97 vs target 2812 on a 307,505-draw trial (~0.15% off), max well
    under cap, zero draws exactly at cap.

    Resampling (redraw until <= cap) was chosen over clipping (truncate to
    exactly cap) because clipping creates an artificial point-mass spike at
    exactly Rs.1,00,000 - a visible artifact in any histogram of the
    synthetic data. Uses a np.random.Generator internally so repeated
    redraws don't reset to the same seed state.
    """
    mu, sigma = fit_p2p_params(cap=cap)
    dist = stats.lognorm(s=sigma, scale=np.exp(mu))
    rng = np.random.default_rng(random_state)

    samples = dist.rvs(size=n, random_state=rng)
    n_resampled = 0
    mask = samples > cap
    while mask.any():
        n_resampled += mask.sum()
        samples[mask] = dist.rvs(size=mask.sum(), random_state=rng)
        mask = samples > cap

    print(f"P2P: resampled {n_resampled} draw(s) ({n_resampled/n*100:.3f}% of n) that exceeded "
          f"cap={cap}; final mean={samples.mean():.1f} (target ATS=2812), "
          f"max={samples.max():.1f} (< cap, no point mass)")

    return samples


def income_percentile_multiplier(income, lo=0.7, hi=1.3):
    """
    Per-applicant transaction-COUNT multiplier from income percentile rank,
    for correlating UPI transaction frequency with AMT_INCOME_TOTAL. Applied
    to lam_month (see generate_transaction_counts / generate_upi_transactions),
    NOT to ticket-size amounts - RBI's ATS/percentile figures are national
    aggregates with no income-conditional breakdown, so conditioning amounts
    on income would be a wholly new unranked assumption with zero anchor;
    frequency at least has the existing lam_month income-anchoring precedent
    (fit_count_params) to extend.

    multiplier(pctile) = lo + (hi - lo) * pctile, LINEAR in percentile rank
    (continuous, not decile bins - avoids arbitrary bin edges). At pctile=0.5
    this is exactly 1.0 for any lo/hi, so the population MEDIAN stays anchored
    at the already-validated lam_month=20 / ~1.98x-median-income-ratio finding
    (fit_count_params) by construction - this only redistributes count
    WITHIN the population, it doesn't re-anchor it.

    lo=0.7, hi=1.3 (+-30% at the extremes) is a FLAGGED, UNRANKED ASSUMPTION -
    one level further than lam_month itself, since no RBI or NPCI source ties
    individual income to individual transaction frequency at all. Chosen to be
    modest enough not to push turnover-to-income ratios to implausible
    extremes at the income tails (checked by decile in
    notebooks/03_synthetic_upi.ipynb) while still giving a downstream model
    real income-linked signal to find.

    Deliberately NOT extended to REGION_RATING_CLIENT (or any other
    Home Credit feature): the stats doc explicitly states RBI does not
    publish district/region-tier transaction granularity at all - unlike
    income, there is no calibration precedent to extend, so a region
    multiplier would be pure invention. REGION_RATING_CLIENT is also a likely
    audited attribute in this project's Phase 5 fairness audit; injecting a
    synthetic region correlation into the data-generation step now would
    predetermine that audit's findings rather than letting them emerge from
    whatever the model actually learns from real features. Revisit only if a
    real reference dataset surfaces tying region to payment behavior.

    income: array-like, same length/order as the sk_id_curr passed to the
    caller (positional alignment is the caller's responsibility - see
    generate_upi_transactions).
    """
    income = np.asarray(income, dtype=float)
    pctile = pd.Series(income).rank(pct=True, method="average").to_numpy()
    return lo + (hi - lo) * pctile


def fit_count_params(lam_month=20, n_months=3, var_mean_ratio=2.0):
    """
    Negative Binomial parameters (n, p in scipy/numpy convention: mean =
    n*(1-p)/p, var = n*(1-p)/p^2) for an applicant's TOTAL transaction count
    (P2M+P2P combined) over n_months.

    lam_month=20 (transactions/month) is an INCOME-ANCHORED ASSUMPTION, not an
    independent frequency source - RBI/NPCI publish no individual-level
    transaction-frequency figure at all (stats doc's own "what RBI does NOT
    publish" section). A websearch for active UPI users came back with a wide,
    inconsistent range (500M-839M active users against 15.33B monthly
    transactions nationally in FY24-25 -> ~18-31, up to ~40, txns/user/month) -
    too indirect to call a calibration, only a rough band.

    lam_month=20 was instead chosen by checking its IMPLIED CONSEQUENCE against
    train_fe's own AMT_INCOME_TOTAL: at the 0.635/0.365 P2M/P2P mix, blended
    ATS = Rs.1,444.84/txn, so lambda=20/month implies median simulated monthly
    UPI turnover of ~Rs.24,339 against train_fe's median monthly income of
    Rs.12,262.50 (AMT_INCOME_TOTAL/12) - a ~2.0x ratio. Read as "money recycles
    through UPI about twice a month," a plausible velocity for bill-splitting/
    frequent small transfers without claiming an implausible one. This is a
    plausibility check against an existing feature, not a fit to an
    independent source - same calibrated-not-learned framing as the rest of
    Track B; document it as such wherever lam_month is cited.

    NOTE - mean vs. median diverge substantially here: a NAIVE mean-based
    estimate (lambda * blended_ATS) gives a ~2.36x ratio, but the actual
    simulated MEDIAN monthly turnover (confirmed by Monte Carlo, N=307,505,
    notebooks/03_synthetic_upi.ipynb) lands at ~1.98x - even summing ~60
    transactions over 3 months, the per-transaction log-normal's heavy right
    tail (sigma~2.39) is fat enough that the sum's median still sits well
    below its mean. Always compare median-to-median or mean-to-mean, never mix
    the two, when validating turnover figures against income.

    var_mean_ratio=2.0 is an unvalidated placeholder (no data constrains
    count-dispersion either) - confirmed by direct check that the median
    turnover ratio is essentially insensitive to it (1.97x-1.99x across
    var_mean_ratio in [1.5, 3.0]), so this parameter matters far less than
    lam_month and doesn't need the same scrutiny.

    n_months=3 is an explicit, visible default (not a hidden constant) -
    matches the short recent-history window typical of alt-credit-scoring
    behavioral features; pass a different value to use a longer/shorter
    window.

    lam_month may be a scalar (population-level, the default) OR an array of
    per-applicant values (e.g. from lam_month=20*income_percentile_multiplier(...),
    see generate_upi_transactions) - every line below is elementwise-safe for
    an array lam_month; p_nb reduces to a constant (1/var_mean_ratio) regardless,
    only n_nb varies per applicant, and rng.negative_binomial broadcasts an
    array n against a scalar p natively.
    """
    mean_count = lam_month * n_months
    var_count = var_mean_ratio * mean_count
    p_nb = mean_count / var_count
    n_nb = mean_count * p_nb / (1 - p_nb)
    return n_nb, p_nb


def generate_transaction_counts(n_applicants, lam_month=20, n_months=3, var_mean_ratio=2.0, random_state=None):
    """
    Draw each applicant's total transaction count (P2M+P2P combined) over
    n_months from the Negative Binomial fitted by fit_count_params.

    lam_month may be a scalar (independent of any applicant feature, the
    default) or a per-applicant array (see income_percentile_multiplier /
    generate_upi_transactions) to correlate count with income. Deliberately
    NOT correlated with REGION_RATING_CLIENT or any other Home Credit feature
    here - see income_percentile_multiplier's docstring for why region tier
    specifically stays excluded (no RBI calibration precedent, and it's a
    likely fairness-audit attribute downstream).
    """
    n_nb, p_nb = fit_count_params(lam_month=lam_month, n_months=n_months, var_mean_ratio=var_mean_ratio)
    rng = np.random.default_rng(random_state)
    return rng.negative_binomial(n_nb, p_nb, size=n_applicants)


def split_p2m_p2p_counts(counts, p_p2m=0.635, random_state=None):
    """
    Split each applicant's total transaction count into (n_p2m, n_p2p) via
    independent per-transaction Bernoulli(p_p2m) draws - equivalently,
    n_p2m ~ Binomial(n_txn, p_p2m) per applicant.

    p_p2m=0.635 is the midpoint of the stats doc's 63-64% P2M-share-of-volume
    figure (Sec. 4) - a REAL RBI/NPCI-reported national aggregate, unlike
    lam_month. Applied per-transaction rather than as a fixed ratio per
    applicant because the 63/64% figure is a national volume share, not a
    documented individual trait - a per-transaction Bernoulli is the direct,
    parameter-free translation of an aggregate proportion into individual
    draws (recovers ~63/37 in aggregate by the law of large numbers) and
    naturally produces applicant-to-applicant heterogeneity from sampling
    variance alone, without inventing a second, ungrounded parameter for
    "how much applicants vary in their personal P2M-affinity" that a fixed or
    hierarchical per-applicant ratio would require.

    Returns (n_p2m, n_p2p), both same shape as counts.
    """
    counts = np.asarray(counts)
    rng = np.random.default_rng(random_state)
    n_p2m = rng.binomial(counts, p_p2m)
    n_p2p = counts - n_p2m
    return n_p2m, n_p2p


def generate_applicant_turnover(n_applicants, lam_month=20, n_months=3, p_p2m=0.635,
                                 var_mean_ratio=2.0, cap_p2m=300_000, cap_p2p=100_000,
                                 income=None, income_lo=0.7, income_hi=1.3, random_state=None):
    """
    Convenience/validation wrapper: draw each applicant's total transaction
    count, split into P2M/P2P, sample amounts for each, and sum to a single
    per-applicant turnover figure over n_months.

    This is the function used to income-anchor lam_month (see
    fit_count_params) and to produce the Monte Carlo validation numbers in
    notebooks/03_synthetic_upi.ipynb. Not itself part of the eventual merge
    step - that needs per-transaction rows (see generate_upi_transactions),
    not just a summed total - but useful for any future plausibility re-check
    against train_fe features.

    cap_p2m defaults to Rs.3,00,000 (see generate_p2m_amounts) - the original
    lam_month=20 income-anchoring was validated before this cap existed
    (cap=None was a bug, see fit_p2m_params); re-confirmed after the fix that
    the median turnover ratio is unchanged (both P2M constraints are
    re-solved to hit the same 659/500 targets under truncation, so aggregate
    behavior barely moves - only the impossible tail is removed).

    income: optional, array-like aligned 1:1 with the n_applicants slots (same
    convention as generate_upi_transactions). When given, per-applicant
    lam_month is scaled by income_percentile_multiplier(income, income_lo,
    income_hi) - see that function's docstring for the full reasoning. Used
    here to produce the by-income-decile turnover-ratio validation table in
    notebooks/03_synthetic_upi.ipynb before wiring the correlation into
    generate_upi_transactions.
    """
    rng = np.random.default_rng(random_state)
    lam_effective = (lam_month * income_percentile_multiplier(income, lo=income_lo, hi=income_hi)
                      if income is not None else lam_month)
    counts = generate_transaction_counts(n_applicants, lam_month=lam_effective, n_months=n_months,
                                          var_mean_ratio=var_mean_ratio, random_state=rng)
    n_p2m, n_p2p = split_p2m_p2p_counts(counts, p_p2m=p_p2m, random_state=rng)

    turnovers = np.zeros(n_applicants)
    total_p2m = n_p2m.sum()
    total_p2p = n_p2p.sum()
    if total_p2m > 0:
        p2m_amounts = generate_p2m_amounts(total_p2m, cap=cap_p2m, random_state=rng)
        np.add.at(turnovers, np.repeat(np.arange(n_applicants), n_p2m), p2m_amounts)
    if total_p2p > 0:
        p2p_amounts = generate_p2p_amounts(total_p2p, cap=cap_p2p, random_state=rng)
        np.add.at(turnovers, np.repeat(np.arange(n_applicants), n_p2p), p2p_amounts)

    return turnovers


def generate_upi_transactions(sk_id_curr, lam_month=20, n_months=3, p_p2m=0.635,
                               var_mean_ratio=2.0, cap_p2m=300_000, cap_p2p=100_000,
                               income=None, income_lo=0.7, income_hi=1.3, random_state=None):
    """
    Generate one row per synthetic UPI transaction for the given applicant IDs
    - the per-transaction table needed for the eventual merge onto train_fe
    (unlike generate_applicant_turnover, which only returns a summed total).

    income: optional, array-like of AMT_INCOME_TOTAL values aligned 1:1 BY
    POSITION with sk_id_curr (caller's responsibility - e.g. both pulled from
    the same row-drop-fixed application_train.csv in the same order). When
    given, each applicant's lam_month is scaled by
    income_percentile_multiplier(income, income_lo, income_hi) before drawing
    their transaction count, correlating UPI frequency with income while
    leaving ticket-size amounts unconditional (see that function's docstring
    for the full reasoning, including why REGION_RATING_CLIENT is deliberately
    NOT correlated here). income=None (default) reproduces the original
    income-INDEPENDENT behavior exactly - backward compatible with every
    earlier validation in notebooks/03_synthetic_upi.ipynb.

    NOT built with SDV, despite CLAUDE.md's original Track B roadmap naming
    it. SDV's synthesizers (GaussianCopula, CTGAN, multi-table HMA) all work
    by fitting a model to a REAL sample and learning its column distributions/
    correlations/cardinality from that data. There is no real UPI microdata
    here to fit on - every distribution in this module is calibrated directly
    from RBI aggregate figures (or income-anchored, see fit_count_params), not
    learned from rows. The only way to use SDV here would be to generate a
    seed dataset from these same closed-form generators, fit an SDV
    synthesizer to that seed, then sample from SDV instead - strictly worse
    than sampling directly, since it only adds approximation error and an
    opaque model in between the audited calibration and the output, for zero
    benefit. Referential integrity (which SK_ID_CURR owns which rows) is
    exact by construction here (sk_id_curr is repeated by its own generated
    count), not inferred - so SDV's multi-table cardinality-learning
    machinery has nothing to add either. Revisit if a real dataset with
    genuine joint structure (e.g. income/region correlated with payment
    behavior) ever surfaces for the deferred income/region-tier merge step -
    that is the scenario SDV is actually built for.

    Columns:
        SK_ID_CURR   int32        real applicant ID, repeated by its count
        TXN_TYPE     category     'P2M' or 'P2P'
        AMOUNT       float32      transaction amount (Rs.)
        MONTH_INDEX  int8         1..n_months, uniform-random within the
                                   window (RBI's growth-trend note, stats doc
                                   Sec. 6, is a multi-year national trend, not
                                   something meaningful to simulate inside a
                                   single applicant's n_months window, so no
                                   within-window trend is imposed)

    Scale note: at the locked defaults (lam_month=20, n_months=3) this is
    ~60 rows/applicant, e.g. ~18.4M rows for all 307,505 train_fe applicants.
    Sorted by SK_ID_CURR on return. Caller should save as Parquet, not CSV,
    at that scale.
    """
    sk_id_curr = np.asarray(sk_id_curr)
    n_applicants = len(sk_id_curr)
    rng = np.random.default_rng(random_state)

    if income is not None:
        income = np.asarray(income)
        assert len(income) == n_applicants, "income must align 1:1 with sk_id_curr"
        lam_effective = lam_month * income_percentile_multiplier(income, lo=income_lo, hi=income_hi)
    else:
        lam_effective = lam_month

    counts = generate_transaction_counts(n_applicants, lam_month=lam_effective, n_months=n_months,
                                          var_mean_ratio=var_mean_ratio, random_state=rng)
    n_p2m, n_p2p = split_p2m_p2p_counts(counts, p_p2m=p_p2m, random_state=rng)
    n_zero = (counts == 0).sum()
    if n_zero > 0:
        print(f"NOTE: {n_zero} applicant(s) drew 0 transactions and will have no rows below")

    total_p2m = int(n_p2m.sum())
    total_p2p = int(n_p2p.sum())

    p2m_amounts = generate_p2m_amounts(total_p2m, cap=cap_p2m, random_state=rng) if total_p2m > 0 else np.array([])
    p2p_amounts = generate_p2p_amounts(total_p2p, cap=cap_p2p, random_state=rng) if total_p2p > 0 else np.array([])

    ids = np.concatenate([np.repeat(sk_id_curr, n_p2m), np.repeat(sk_id_curr, n_p2p)])
    types = np.concatenate([np.full(total_p2m, "P2M"), np.full(total_p2p, "P2P")])
    amounts = np.concatenate([p2m_amounts, p2p_amounts])
    months = rng.integers(1, n_months + 1, size=total_p2m + total_p2p)

    df = pd.DataFrame({
        "SK_ID_CURR": ids.astype(np.int32),
        "TXN_TYPE": pd.Categorical(types, categories=["P2M", "P2P"]),
        "AMOUNT": amounts.astype(np.float32),
        "MONTH_INDEX": months.astype(np.int8),
    })
    df = df.sort_values("SK_ID_CURR", kind="mergesort").reset_index(drop=True)

    print(f"generate_upi_transactions: {len(df):,} rows for {n_applicants:,} applicants "
          f"({len(df)/n_applicants:.1f} txns/applicant avg), "
          f"P2M/P2P split = {(types=='P2M').mean()*100:.1f}%/{(types=='P2P').mean()*100:.1f}%, "
          f"memory = {df.memory_usage(deep=True).sum()/1e6:.1f} MB")

    return df
