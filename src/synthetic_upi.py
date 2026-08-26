import numpy as np
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


def fit_p2m_params():
    """
    P2M ticket size: both constraints are real RBI-derived figures
    (ATS=Rs.659, 86th percentile=Rs.500 - see stats doc Sec. 2-3), so this is
    an exact solve, not an assumption. The other algebraic root gives a
    negative sigma (invalid), so mu=3.630972, sigma=2.391548 is the unique
    valid solution, confirmed in notebooks/03_synthetic_upi.ipynb.

    NOTE - implied median is ~Rs.38: forcing one log-normal through both the
    mean and the 86th percentile simultaneously pushes over half the
    distribution below Rs.38, with a long tail pulling the mean up to 659.
    This is a direct, unavoidable consequence of the two constraints, not an
    extra assumption layered on top - flag it wherever this fit is cited
    (e.g. thesis methodology) so it isn't a surprise later.
    """
    return fit_lognormal_mean_percentile(
        target_mean=659, target_pctile_value=500, pctile=0.86, root="larger"
    )


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


def generate_p2m_amounts(n, cap=None, random_state=None):
    """
    Sample n P2M transaction amounts from the fitted log-normal
    (see fit_p2m_params for the derivation and the ~Rs.38-median caveat).

    cap: optional hard ceiling to clip at (Rs). The stats doc's general P2M
    ceiling is ambiguous (Rs.1-5 lakh depending on category, or up to
    Rs.10 lakh/day for select verified merchants as of Sept 2025 - Sec. 5),
    so no single value is applied by default. In practice the fitted
    distribution's own p99.9 (~Rs.61,170) already sits below every published
    ceiling, so clipping is optional here, unlike P2P.
    """
    mu, sigma = fit_p2m_params()
    dist = stats.lognorm(s=sigma, scale=np.exp(mu))
    samples = dist.rvs(size=n, random_state=random_state)

    if cap is not None:
        n_clipped = (samples > cap).sum()
        samples = np.clip(samples, None, cap)
        print(f"P2M: clipped {n_clipped} ({n_clipped/n*100:.4f}%) draws to cap={cap}")

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

    Independent of income/region tier BY DESIGN at this stage - see
    generate_transaction_counts.
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

    Deliberately INDEPENDENT of any Home Credit applicant feature (income,
    region tier, etc.) at this stage - no RBI source ties frequency to either,
    so imposing a functional form now would stack an unranked assumption on
    top of the already-flagged lam_month one. Correlation with income/region
    tier is deferred to the SDV merge step in the Track B roadmap (CLAUDE.md),
    which is the right place to inject that dependency structure, not here.
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
                                 var_mean_ratio=2.0, cap_p2p=100_000, random_state=None):
    """
    Convenience/validation wrapper: draw each applicant's total transaction
    count, split into P2M/P2P, sample amounts for each, and sum to a single
    per-applicant turnover figure over n_months.

    This is the function used to income-anchor lam_month (see
    fit_count_params) and to produce the Monte Carlo validation numbers in
    notebooks/03_synthetic_upi.ipynb. Not itself part of the eventual merge
    step - that will need per-transaction rows (for SDV), not just a summed
    total - but useful for any future plausibility re-check against
    train_fe features.
    """
    rng = np.random.default_rng(random_state)
    counts = generate_transaction_counts(n_applicants, lam_month=lam_month, n_months=n_months,
                                          var_mean_ratio=var_mean_ratio, random_state=rng)
    n_p2m, n_p2p = split_p2m_p2p_counts(counts, p_p2m=p_p2m, random_state=rng)

    turnovers = np.zeros(n_applicants)
    total_p2m = n_p2m.sum()
    total_p2p = n_p2p.sum()
    if total_p2m > 0:
        p2m_amounts = generate_p2m_amounts(total_p2m, random_state=rng)
        np.add.at(turnovers, np.repeat(np.arange(n_applicants), n_p2m), p2m_amounts)
    if total_p2p > 0:
        p2p_amounts = generate_p2p_amounts(total_p2p, cap=cap_p2p, random_state=rng)
        np.add.at(turnovers, np.repeat(np.arange(n_applicants), n_p2p), p2p_amounts)

    return turnovers
