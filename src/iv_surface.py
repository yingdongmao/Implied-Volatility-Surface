"""
iv_surface.py
=============
Step 1 of the pipeline: build a smooth implied-volatility (IV) surface for one
stock on one day from its raw option quotes.

The surface is a small grid over

    * maturity  tau  in {20, 30, 40} calendar days, and
    * standardized moneyness  x  in {-2, -1.5, ..., +2}   (9 points),

estimated by a Nadaraya-Watson (Gaussian-kernel) regression of observed
Black-Merton-Scholes (BMS) implied volatilities onto that grid.  Following
Carr and Wu (2016, 2020), each option is weighted by an OTM weight so that the
surface is driven by the most informative, actively traded contracts.

Conventions (Carr-Wu)
---------------------
For a contract with strike K, spot S, BMS implied vol I and maturity tau (years):

    log-moneyness        k   = ln(K / S)
    convexity-adj.       z_+ = k + 1/2 I^2 tau
    moneyness            z_- = k - 1/2 I^2 tau
    standardized         x   = z_+ / (I * sqrt(tau))

The returned surface feeds ``state_variables.extract_state_variables``.

This module is deliberately dependency-light (numpy / pandas / scipy) and has
no WRDS or I/O side effects, so it can be unit-tested on synthetic data
(see ``examples/synthetic_demo.py``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

# Grid definition (matches the paper: 20/30/40-day maturities, x in [-2, 2]).
MATURITIES_DAYS = (20, 30, 40)
X_GRID = np.arange(-2.0, 2.5, 0.5)          # -2, -1.5, ..., 2   -> 9 points
MIN_OTM_WEIGHT = 0.2                         # drop deep-ITM contributions
MIN_CONTRACTS = 5                            # need enough quotes to smooth


def _silverman_bandwidth(values: np.ndarray, n: int) -> float:
    """Rule-of-thumb bandwidth  h = (4/3)^(1/5) * sd / n^(1/5)  (Silverman)."""
    return (4.0 / 3.0) ** (1.0 / 5.0) * np.std(values, ddof=1) / n ** (1.0 / 5.0)


def build_surface(quotes: pd.DataFrame,
                  maturities_days=MATURITIES_DAYS,
                  x_grid=X_GRID) -> np.ndarray:
    """
    Smooth a single stock-day's quotes into an IV surface.

    Parameters
    ----------
    quotes : DataFrame with one row per option contract and columns
        iv          BMS implied volatility (decimal, e.g. 0.30)
        strike      strike price K
        spot        underlying price S
        maturity    calendar days to expiration
        option_type 'C' or 'P'
    maturities_days, x_grid : the output grid.

    Returns
    -------
    surface : ndarray, shape (len(maturities_days), len(x_grid))
        Interpolated IV at each (tau, x) node; np.nan where it cannot be
        estimated.  Row order follows ``maturities_days`` (20, 30, 40);
        the ATM column is the one where x == 0.
    """
    nm, nx = len(maturities_days), len(x_grid)
    surface = np.full((nm, nx), np.nan)

    df = quotes.dropna(subset=['iv']).copy()
    df = df[df['iv'] > 0]
    if len(df) < MIN_CONTRACTS:
        return surface

    df['tau'] = df['maturity'] / 365.0
    df['lm'] = np.log(df['tau'])                       # smoothing in log-maturity
    df['k'] = np.log(df['strike'] / df['spot'])
    df['z_plus'] = df['k'] + 0.5 * df['iv'] ** 2 * df['tau']
    df['z_minus'] = df['k'] - 0.5 * df['iv'] ** 2 * df['tau']
    df['x'] = df['z_plus'] / (df['iv'] * np.sqrt(df['tau']))

    # OTM (delta-based) weight: N(z_-/(I sqrt(tau))) for calls, 1 - that for puts.
    w = norm.cdf(df['z_minus'] / (df['iv'] * np.sqrt(df['tau'])))
    w = np.where(df['option_type'].str.upper() == 'P', 1.0 - w, w)
    w = np.where(w < MIN_OTM_WEIGHT, 0.0, w)           # ignore deep-ITM legs
    df['w_otm'] = w
    df = df[df['w_otm'] > 0]

    n = len(df)
    if n < MIN_CONTRACTS:
        return surface

    h_x = _silverman_bandwidth(df['x'].values, n)
    h_m = _silverman_bandwidth(df['lm'].values, n)
    if not np.isfinite(h_x) or not np.isfinite(h_m) or h_x == 0 or h_m == 0:
        return surface

    log_mat = np.log(np.asarray(maturities_days) / 365.0)
    for m in range(nm):
        wm = df['w_otm'].values * np.exp(-0.5 * ((df['lm'].values - log_mat[m]) / h_m) ** 2)
        for j in range(nx):
            wk = wm * np.exp(-0.5 * ((df['x'].values - x_grid[j]) / h_x) ** 2)
            denom = wk.sum()
            if denom > 0:
                surface[m, j] = np.sum(wk * df['iv'].values) / denom
    return surface
