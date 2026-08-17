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


def european_value(spot, strike, iv, tau, option_type):
    """
    European (Black) option value on the underlying `spot`, zero financing.

        call = U N(d1) - K N(d2),   put = call - U + K   (put-call parity)
        d1   = (ln(U/K) + I^2 tau / 2) / (I sqrt(tau)),   d2 = d1 - I sqrt(tau)

    OptionMetrics reports the implied volatility of the AMERICAN contract (from
    its binomial tree).  Following Carr-Wu, that IV is used here to price the
    matching EUROPEAN contract, which is the object the factor structure below
    is derived for.  Discounting is omitted because the underlying passed in is
    the forward (see ``estimate_risk_premia``), so the value is already
    expressed in time-t+tau currency; the delta-hedged, cash-gamma-scaled ratio
    is invariant to a common discount factor anyway.

    This is deliberately NOT left to the caller: pricing the option value is
    where independent replications of this pipeline diverge most often.
    """
    d1 = _d1(spot, strike, iv, tau)
    d2 = d1 - iv * np.sqrt(tau)
    call = spot * norm.cdf(d1) - strike * norm.cdf(d2)
    return np.where(np.asarray(option_type) == 'C', call, call - spot + strike)


# --------------------------------------------------------------------------- #
# Cross-sectional estimation of f_t
# --------------------------------------------------------------------------- #
def estimate_risk_premia(quotes: pd.DataFrame, min_contracts: int = 8) -> pd.Series:
    """
    Recover the realized risk premia f_t for one stock on one date.

    Parameters
    ----------
    quotes : DataFrame, one row per contract, ALL of
        iv          BMS implied volatility at t     (decimal)
        iv_next     BMS implied volatility at t+1   (SAME contract)
        spot        underlying at t                 (forward; see below)
        spot_next   underlying at t+1
        tau         maturity in years at t
        tau_next    maturity in years at t+1        (the contract ages)
        strike      strike K
        option_type 'C' / 'P'
    min_contracts : minimum effective contracts to attempt the regression.

    The option value at t and t+1 is priced INSIDE this function with
    ``european_value``; it is not an input.  Pass the FORWARD as ``spot`` /
    ``spot_next`` (OptionMetrics ``fwdprd``) so no discounting is needed.

    The regression carries an INTERCEPT: the gamma factor loads on the constant
    1/2 for every contract, so without it the gamma P&L is forced into the other
    three coefficients and the volatility risk premium is badly biased.

    Returns
    -------
    pandas.Series indexed
        ['gamma_rp', 'vrp', 'vov_rp', 'cov_rp', 'R2', 'nobs']
    where 'vrp' is the volatility risk premium (loading on Vega).
    """
    idx = ['gamma_rp', 'vrp', 'vov_rp', 'cov_rp', 'R2', 'nobs']
    need = ['iv', 'iv_next', 'spot', 'spot_next', 'tau', 'tau_next',
            'strike', 'option_type']
    missing = [c for c in need if c not in quotes.columns]
    if missing:
        raise KeyError("estimate_risk_premia needs column(s) %s" % missing)

    df = quotes.dropna(subset=[c for c in need if c != 'option_type']).copy()
    df = df[(df['iv'] > 0) & (df['iv_next'] > 0) &
            (df['tau'] > 0) & (df['tau_next'] > 0)]
    if len(df) < min_contracts:
        return pd.Series([np.nan] * len(idx), index=idx)

    iv, S, K, tau = df['iv'].values, df['spot'].values, df['strike'].values, df['tau'].values
    cp = df['option_type'].values

    # European value of the SAME contract at t and at t+1: the underlying, the
    # implied volatility AND the remaining maturity all move.
    value = european_value(S, K, iv, tau, cp)
    value_next = european_value(df['spot_next'].values, K, df['iv_next'].values,
                                df['tau_next'].values, cp)

    # delta-hedged P&L scaled by cash gamma  ->  r_i
    dC = value_next - value
    dS = df['spot_next'].values - S
    delta = bms_delta(S, K, iv, tau, cp)
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
