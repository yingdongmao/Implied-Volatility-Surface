# Implied-Volatility-Surface — state variables & their risk premia

Sample code to extract, from an interpolated firm-level **implied-volatility (IV)
surface**, the four option-implied **state variables** of Carr and Wu (2016,
2020) and the **risk premia** they carry — including the *return risk premium
implied by the volatility surface*, `RRP^v`.

The four state variables (all reported at the 30-day horizon) are

| symbol | name | what it governs on the surface |
|--------|------|--------------------------------|
| `σ` (sigma) | instantaneous volatility of the underlying return | overall level of the term structure |
| `μ` (mu)    | risk-neutral drift rate of implied volatility       | slope of the term structure (this is the volatility-risk channel) |
| `ρ` (rho)   | correlation rate between return and IV innovations  | slope / asymmetry of the volatility smile |
| `ω` (omega) | volatility of implied volatility ("vol-of-vol")     | curvature of the volatility smile |

This is the empirical machinery behind the job-market paper *"Return Risk
Premium implied by the IV surface around Earnings Announcement Days."* The repo
ships a **minimal, dependency-light, runnable** version of the estimator (no WRDS
required for the demo) so the method can be read and reused.

---

## The model in one identity

Following Carr and Wu, the underlying and the BMS implied volatility of every
contract on it follow, under the risk-neutral measure `ℚ`,

```
dS_t / S_t   = σ_t dW_t
dI_t / I_t   = μ_t dt + ω_t dZ_t ,        E[dW_t dZ_t] = ρ_t dt
```

No-dynamic-arbitrage (Carr–Wu 2016, Prop. 1), after dividing the pricing PDE by
cash gamma `B_SS S²`, links the *observed* IV surface to the state variables
contemporaneously — no time-series calibration needed:

```
I(K,τ)²  =  [ 2τ μ_t I(K,τ)²  +  σ_t² ]          ← term structure
          + [ 2 ρ_t ω_t σ_t z₊  +  ω_t² z₊ z₋ ]   ← volatility smile
```

with the Carr–Wu moneyness variables

```
z₊ = ln(K/S) + ½ I² τ        z₋ = ln(K/S) − ½ I² τ        x = z₊ / (I √τ)
```

Because the surface at a single instant pins down `(σ, μ, ρ, ω)`, the estimator
is well suited to event studies (e.g. earnings announcements) where information
is incorporated quickly.

---

## How the state variables are calculated

`src/iv_surface.py` → `src/state_variables.py`. Three steps, all at the 30-day
horizon, using the smoothed surface on the grid `τ ∈ {20, 30, 40}` days and
`x ∈ {−2, −1.5, …, +2}`.

**Step 0 — interpolate the surface.** For one stock-day, the raw option quotes
are smoothed onto the `(τ, x)` grid with a Nadaraya–Watson (Gaussian-kernel)
regression in `(x, ln τ)`. Each contract is weighted by an out-of-the-money
delta weight `N(z₋ / (I√τ))` (calls) / `1 − N(·)` (puts), so deep-ITM and
illiquid legs do not drive the fit; bandwidths use Silverman's rule
`h = (4/3)^(1/5) · sd / n^(1/5)`.

**Step 1 — drift `μ` from the term structure.** Set `z₊ = 0` (the ATM, `x = 0`
column). The identity collapses to `I(S,τ)² = 2τ μ I(S,τ)² + σ²`, linear in `τ`,
so from the two ATM maturities `τ₁ = 20/365`, `τ₂ = 40/365`:

```
        I²(τ₂) − I²(τ₁)
μ_t  = ----------------------------
       2 ( I²(τ₂) τ₂ − I²(τ₁) τ₁ )
```

**Step 2 — instantaneous vol `σ`.** From the same ATM term-structure relation at
`τ₁`:

```
σ_t = sqrt( I²(τ₁) · ( 1 − 2 τ₁ μ_t ) )
```

**Step 3 — `ρ` and `ω` from the smile.** Subtract the ATM level and regress, at
the 30-day maturity over `−1 ≤ x ≤ 1`, the squared deviation on the smile basis
(constraining the vol-of-vol coefficient to be non-negative):

```
I(x,τ)² − I(ATM,τ)²  =  b₁ · (2 z₊)  +  b₂ · (z₊ z₋)
```

then

```
ω_t = sqrt(b₂)            (since b₂ = ω²)
ρ_t = b₁ / (σ_t · ω_t)    (since b₁ = ρ σ ω)
```

The non-negativity constraint on `b₂` is enforced by
`src/constrained_linear_regression.py` (a small coordinate-descent solver),
which keeps `ω` real when interpolation noise would otherwise push the
unconstrained estimate slightly negative.

---

## How the risk premia are calculated

`src/risk_premia.py`. A Taylor expansion of the BMS price shows that the
delta-hedged, cash-gamma-scaled return of **every** contract `i` on a stock is
the *same* four-factor model:

```
r_i  =  (dB_i − B_S dS) / (B_SS S²)  =  β_i' f_t

β_i  = [ ½ ,  I_i² τ_i ,  ½ z₊z₋ ,  z₊ ]              ← option loadings (Greeks)
f_t  = [ (dS/S)² − σ² ,                               gamma / "jump"  premium
         dI/I − μ ,                                   VOLATILITY RISK PREMIUM (VRP)
         (dI/I)² − ω² ,                               vol-of-vol      premium
         dS·dI/(S·I) − ρσω ]                          covariance      premium
```

So a **single cross-sectional OLS** on date `t` of delta-hedged option returns
onto the loadings recovers all four realized risk premia `f_t` at once. The
loading on **Vega** (`I² τ`) is the **volatility risk premium**
`VRP_t = E^P[dI/I] − μ_t`: because it sits on a distinct option characteristic,
it is a *clean* VRP that does not mix in the gamma/jump term the way raw
delta-hedged straddle returns do.

### Return risk premium implied by the surface

The headline object links the volatility risk premium to the stock's **return**
risk premium (paper, *"Return risk premium implied from volatility surface"*):

```
RRP^v_t  =  VRP_t · ( σ_t / ω_t ) · ρ_t
                     └─────────────┘
                    ESV: elasticity of stock price
                    w.r.t. implied-vol changes
```

`ρ σ / ω` is the elasticity of the stock price with respect to implied-volatility
changes (ESV) — equivalently `cov(dS/S, dI/I) / var(dI/I)`. Intuitively, VRP is
the *price* of one unit of the volatility hedge and ESV the *quantity* needed to
hedge the stock; their product is the share of the return risk premium that the
IV surface can identify. `return_risk_premium()` implements this.

---

## Repository layout

```
src/
  iv_surface.py                   step 1: raw quotes -> smooth IV surface (kernel regression)
  state_variables.py              step 2: IV surface -> (σ, μ, ρ, ω)
  risk_premia.py                  step 3: delta-hedged returns -> f_t, VRP, and RRP^v
  constrained_linear_regression.py  non-negativity-constrained smile regression
examples/
  synthetic_demo.py               full pipeline on fabricated data — NO WRDS needed
  run_from_wrds.py                reference driver on real OptionMetrics data (needs WRDS)
requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt

# 1. Run the whole pipeline on synthetic data (no credentials, ~1s):
python examples/synthetic_demo.py

# 2. Reproduce on real OptionMetrics data (needs a WRDS account):
export WRDS_USERNAME=your_username        # Windows: set WRDS_USERNAME=...
python examples/run_from_wrds.py
```

`synthetic_demo.py` prints the smoothed surface, the four state variables, the
realized risk premia, and `RRP^v`, so you can confirm the install end-to-end.

## Data

The estimator consumes, per option contract on a stock-day: BMS implied
volatility, strike, underlying price, days to maturity, and call/put flag (plus
`t+1` option and underlying values for the risk-premia regression). In the paper
these come from **OptionMetrics** (US equity options, 1996–2021) for S&P 500
member stocks, pulled via **WRDS**. American-option IVs are used to reprice the
matching European contract, per Carr–Wu. Standard liquidity / no-arbitrage
filters apply (maturity 5–90 days, IV > 0.01, positive bid/offer, valid prices);
each firm-day regression needs ≥ 8 contracts spanning ≥ 2 maturities.

## References

- Carr, P. and L. Wu (2016). *Analyzing volatility risk and risk premium in
  option contracts: A new theory.* Journal of Financial Economics.
- Carr, P. and L. Wu (2020). *Option profit and loss attribution and pricing:
  A new framework.* Journal of Finance.
- Mao, Y. *Return Risk Premium implied by the IV surface around Earnings
  Announcement Days* (job-market paper).

This repository contains illustrative sample code; the full reproduction
pipeline (universe construction, filters, panel assembly, and the empirical
tests around earnings announcements) lives with the paper.
