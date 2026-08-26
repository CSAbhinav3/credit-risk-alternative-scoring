# RBI DPSS / NPCI UPI Statistics — Calibration Reference for Synthetic Data (Track B)

Compiled from RBI's half-yearly Payment Systems Report and NPCI product statistics, via secondary reporting (RBI/NPCI primary PDFs are not directly web-fetchable — see Sourcing Note at bottom). Use this to define marginal distributions for the synthetic UPI feature generation step.

## 1. Overall scale (context, not directly used in per-applicant simulation)
- FY 2025–26: ~24,162 crore (241.62 billion) UPI transactions/year, ~₹314 lakh crore value
- H1 CY2025 (Jan–Jun): 106.36 billion transactions, ₹143.35 lakh crore value
- Digital payments = 99.7% of all payment volume, 97.5% of value (CY2024)
- UPI = ~84–85.5% of India's total retail digital payment volume (FY25 / H2 2025)

## 2. Average Transaction Size (ATS) — key calibration input
| Segment | ATS | Period |
|---|---|---|
| Overall UPI | ₹1,300–1,348 | FY25–26 / H1 2025 |
| P2P (person-to-person) | ₹2,812 | H1 2023, rising trend |
| P2M (person-to-merchant) | ₹659 | H1 2023, falling trend (was ₹839 H1 2022) |

**Use for:** the log-normal / gamma parameters of your `AMT_TRANSACTION` marginal. The P2P/P2M gap is large and directionally stable across years (P2M consistently ~4x smaller ticket than P2P) — worth modeling as two separate distributions rather than one blended one, if your feature set distinguishes P2P vs P2M-like behavior.

## 3. Ticket size distribution (skew)
- 86% of P2M transaction volume falls in the ₹0–500 band
- Implies a strongly right-skewed distribution — most transactions are small, a long tail of larger ones pulls the mean up well above the median
- **Use for:** don't use a normal distribution for transaction amounts; log-normal or gamma is a much closer fit to this shape

## 4. P2P vs P2M split — shifting over time
| Metric | ~2020 | ~2023 | Jul 2025 |
|---|---|---|---|
| P2M share of volume | ~39% | ~56% | ~63–64% |
| P2M share of value | ~13% | — | ~29% |

**Use for:** if your synthetic feature set includes a P2P/P2M mix ratio per applicant, calibrate around the ~63–64% P2M / ~36–37% P2P volume split as the current-era baseline (2025), not the older 39%/61% split — the shift itself is also citable in your thesis as evidence of India's payments ecosystem maturing toward everyday merchant use.

## 5. Transaction limits (hard ceilings — use to bound your simulated amount distribution)
- P2P: capped at ₹1,00,000 (₹1 lakh) per transaction — unchanged
- P2M: revised Sept 2025, up to ₹10,00,000 (₹10 lakh)/day for select verified merchant categories; general P2M historically ₹1–5 lakh depending on category
- **Use for:** truncate/clip your synthetic amount distribution at these ceilings so no simulated transaction is structurally impossible

## 6. Growth trend (for time-series realism if your synthetic data spans multiple months)
- UPI volume CAGR ~43% over the last 5 years; value CAGR ~17% (RBI, 2025 report)
- Reflects strong month-on-month volume growth with falling average ticket size (more small transactions, not just more money moving)

## What RBI does NOT publish at the individual/district granularity you'd want
No RBI DPSS source found in this search gives:
- District-level or income-tier-level transaction frequency/value breakdowns
- Individual-level transaction consistency, merchant-category diversity (Shannon entropy), or savings-behavior proxies

These are the features your research proposal's Section 5.2.1 table describes (transaction frequency, merchant category diversity, income regularity, etc.) — none of them exist in RBI aggregate publications at that granularity. This is expected and matches what your proposal's Section 12.1 already frames as a known limitation: **the synthetic data is calibrated to match plausible aggregate statistics (ATS, P2P/P2M split, ticket-size skew), not learned from real individual-level microdata.** Document this explicitly in the methodology section, as already planned.

One more real data source worth checking for geography: NPCI's June 2025 statistics release reportedly added **state-wise usage data** and merchant-category-wise transaction counts for the first time — this wasn't confirmed in detail in this pass and is worth a dedicated follow-up search if you want a real geographic reference point instead of resorting to your own region-tier assumptions from the Home Credit EDA.

## Sourcing note
RBI's own report PDFs (rbi.org.in, half-yearly Payment Systems Report) and NPCI's product statistics page were not directly fetched in this pass — the numbers above come from secondary reporting (Business Standard, IBEF, PIB press releases, SBI Research, Worldline Digital Payments Report) that cite those primary sources directly. For a citation in the actual thesis, go to the primary RBI/NPCI report and cite it directly rather than the secondary source — worth doing before final submission, not urgent now.

## Suggested next action for Claude Code
Once this file is in the repo (e.g. `docs/rbi_dpss_upi_stats.md`), the next concrete step is defining marginal distributions in code — something like:

```python
# Draft starting point — refine with Claude Code, validate before use
import numpy as np

# P2M-dominant ticket size: log-normal, ATS ~659, heavily right-skewed, 86% under 500
p2m_amount = np.random.lognormal(mean=..., sigma=..., size=n)  # fit params to hit ATS=659, 86th pct=500

# P2P: log-normal, ATS ~2812, capped at 100000
p2p_amount = np.random.lognormal(mean=..., sigma=..., size=n)

# Per-applicant P2M/P2P transaction count mix ~63/37 split
```
Have Claude Code fit the actual log-normal parameters (mean, sigma) that hit both the ATS and the 86%-under-₹500 constraint simultaneously — that's a two-constraint fit, not a guess, and should go through the same pre-check → scratch cell → validate pattern as everything else in this project.
