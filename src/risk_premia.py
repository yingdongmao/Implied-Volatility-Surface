"""
risk_premia.py
==============
Step 3 of the pipeline: the *risk premia carried by the state variables*, and
the return risk premium they imply.

Idea (Carr-Wu 2020; paper Section "Risk premia carried by state variables").
A Taylor expansion of the BMS price gives the delta-hedged, cash-gamma-scaled
option return as a factor model with the *same* four factors for every contract
i written on a stock:

    r_i = (dB_i - B_S dS) / (B_SS S^2)
        = beta_i' f_t ,

    beta_i = [ 1/2 ,  I_i^2 tau_i ,  1/2 z_{i+} z_{i-} ,  z_{i+} ]'          (loadings)
    f_t    = [ (dS/S)^2 - sigma^2 ,  dI/I - mu ,  (dI/I)^2 - omega^2 ,
               dS dI /(S I) - rho sigma omega ]'                             (realized premia)

So a single cross-sectional regression on date t of delta-hedged returns onto
the loadings recovers the four realized risk premia f_t at once:

    intercept  ->  1/2 * [ (dS/S)^2 - sigma^2 ]   gamma / "jump" risk premium
    I^2 tau    ->  dI/I - mu                       VOLATILITY RISK PREMIUM (VRP)
    1/2 z+ z-  ->  (dI/I)^2 - omega^2              vol-of-vol risk premium
    z+         ->  dS dI/(S I) - rho sigma omega   covariance risk premium

Among the four, the loading on Vega ( I^2 tau ) is the **volatility risk
premium** VRP_t = E^P[dI/I] - mu_t: a clean measure that does not mix in the
gamma/jump term, because it loads on a distinct option characteristic.

Return risk premium implied by the surface (paper Section "Return risk premium
implied from volatility surface"):

    RRP^v_t = VRP_t * (sigma_t / omega_t) * rho_t

i.e. the volatility risk premium times the elasticity of the stock price with
respect to implied-volatility changes (ESV = rho * sigma / omega).  This is the
component of the stock's return risk premium that the IV surface can identify.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm


# --------------------------------------------------------------------------- #
# Black-Merton-Scholes Greeks (forward / European, zero financing assumed)
# --------------------------------------------------------------------------- #
def _d1(spot, strike, iv, tau):
    return (np.log(spot / strike) + 0.5 * iv ** 2 * tau) / (iv * np.sqrt(tau))


def bms_delta(spot, strike, iv, tau, option_type):
    d1 = _d1(spot, strike, iv, tau)
    call = norm.cdf(d1)
    return np.where(np.asarray(option_type) == 'C', call, call - 1.0)


def cash_gamma(spot, strike, iv, tau):
    """Dollar gamma  B_SS S^2 = S n(d1) / (I sqrt(tau))."""
    d1 = _d1(spot, strike, iv, tau)
    return spot * norm.pdf(d1) / (iv * np.sqrt(tau))


# --------------------------------------------------------------------------- #
# Cross-sectional estimation of f_t
# --------------------------------------------------------------------------- #
def estimate_risk_premia(quotes: pd.DataFrame, min_contracts: int = 8) -> pd.Series:
    """
    Recover the realized risk premia f_t for one stock on one date.

    Parameters
    ----------
    quotes : DataFrame, one row per contract, columns
        iv          BMS implied volatility at t (decimal)
        strike      strike K
        spot        underlying price S at t
        spot_next   underlying price S at t+1 (next interval/day)
        value       (European) option value at t
        value_next  (European) option value at t+1
        tau         maturity in years
        option_type 'C' / 'P'
    min_contracts : minimum effective contracts to attempt the regression.

    Returns
    -------
    pandas.Series indexed
        ['gamma_rp', 'vrp', 'vov_rp', 'cov_rp', 'R2', 'nobs']
    where 'vrp' is the volatility risk premium (loading on Vega).
    """
    idx = ['gamma_rp', 'vrp', 'vov_rp', 'cov_rp', 'R2', 'nobs']
    df = quotes.dropna(subset=['iv', 'value', 'value_next',
                               'spot', 'spot_next', 'tau', 'strike']).copy()
    df = df[(df['iv'] > 0) & (df['tau'] > 0)]
    if len(df) < min_contracts:
        return pd.Series([np.nan] * len(idx), index=idx)

    iv, S, K, tau = df['iv'].values, df['spot'].values, df['strike'].values, df['tau'].values

    # delta-hedged P&L scaled by cash gamma  ->  r_i
    dC = df['value_next'].values - df['value'].values
    dS = df['spot_next'].values - df['spot'].values
    delta = bms_delta(S, K, iv, tau, df['option_type'].values)
    cg = cash_gamma(S, K, iv, tau)
    r = (dC - delta * dS) / cg

    # factor loadings  beta_i = [1/2, I^2 tau, 1/2 z+ z-, z+]
    z_plus = np.log(K / S) + 0.5 * iv ** 2 * tau
    z_minus = np.log(K / S) - 0.5 * iv ** 2 * tau
    vega = iv ** 2 * tau
    volga = 0.5 * z_plus * z_minus
    vanna = z_plus

    X = sm.add_constant(np.stack((vega, volga, vanna), axis=1))   # const = 1/2 factor
    good = np.isfinite(r) & np.isfinite(X).all(axis=1)
    if good.sum() < min_contracts:
        return pd.Series([np.nan] * len(idx), index=idx)

    res = sm.OLS(r[good], X[good]).fit()
    b = res.params
    # intercept loads on the constant 1/2 factor -> multiply by 2 for the premium
    return pd.Series([2.0 * b[0], b[1], b[2], b[3], res.rsquared, int(res.nobs)], index=idx)


# --------------------------------------------------------------------------- #
# Return risk premium implied by the surface
# --------------------------------------------------------------------------- #
def return_risk_premium(vrp, sigma, omega, rho):
    """
    RRP^v = VRP * (sigma / omega) * rho.

    VRP is the volatility risk premium (loading on Vega from
    ``estimate_risk_premia``); (sigma, omega, rho) are state variables from
    ``state_variables.extract_state_variables``.  rho * sigma / omega is the
    elasticity of the stock price w.r.t. implied-volatility changes (ESV).
    """
    return vrp * (sigma / omega) * rho
