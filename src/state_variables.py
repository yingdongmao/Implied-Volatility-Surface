"""
state_variables.py
==================
Step 2 of the pipeline: recover the four Carr-Wu (2020) state variables from a
smoothed IV surface produced by ``iv_surface.build_surface``.

The four state variables (all at the 30-day horizon) are

    sigma_t : instantaneous volatility of the underlying return
    mu_t    : risk-neutral drift rate of implied volatility
    rho_t   : correlation rate between return and IV innovations
    omega_t : volatility of implied volatility ("vol-of-vol")

Identity used (Carr-Wu no-dynamic-arbitrage condition, divided by cash gamma):

    I(K, tau)^2 = [ 2 tau mu_t I(K,tau)^2 + sigma_t^2 ]                      (term structure)
                + [ 2 rho_t omega_t sigma_t z_+ + omega_t^2 z_+ z_- ]        (smile)

Estimation (see Appendix "Estimating state variables" of the paper):

1. Term structure / drift.  Set z_+ = 0 (the ATM, x = 0 column).  Then
   I(S,tau)^2 = 2 tau mu_t I(S,tau)^2 + sigma_t^2 is linear in tau, so with two
   ATM maturities tau_1 = 20/365, tau_2 = 40/365:

       mu_t = [ I(tau_2)^2 - I(tau_1)^2 ]
              -----------------------------------------
              2 [ I(tau_2)^2 tau_2 - I(tau_1)^2 tau_1 ]

2. Instantaneous vol.  From the ATM term-structure relation at tau_1:

       sigma_t = sqrt( I(tau_1)^2 * (1 - 2 tau_1 mu_t) )

3. Smile / (rho, omega).  At the 30-day maturity, regress the squared deviation
   from the ATM level on the smile basis (constraining the vol-of-vol to be
   non-negative):

       I(x,tau)^2 - I(ATM,tau)^2 = b1 * (2 z_+) + b2 * (z_+ z_-)

   then   omega_t = sqrt(b2),   rho_t = b1 / (sigma_t * omega_t).

Note that on the smoothed grid, by construction z_+ = x * I * sqrt(tau) and
z_- = z_+ - I^2 tau, so the smile basis is recovered directly from the surface
(the underlying price S cancels out).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from iv_surface import MATURITIES_DAYS, X_GRID
from constrained_linear_regression import ConstrainedLinearRegression

STATE_VARIABLES = ("sigma", "mu", "rho", "omega")


def extract_state_variables(surface: np.ndarray,
                            maturities_days=MATURITIES_DAYS,
                            x_grid=X_GRID,
                            report_maturity_days: int = 30) -> pd.Series:
    """
    Map a smoothed IV surface to (sigma, mu, rho, omega) at the 30-day horizon.

    Parameters
    ----------
    surface : ndarray (n_maturities, n_x) from ``iv_surface.build_surface``.
    maturities_days, x_grid : the grid the surface was built on.
    report_maturity_days : maturity at which the smile (rho, omega) is read.

    Returns
    -------
    pandas.Series indexed
        ['sigma', 'mu', 'rho', 'omega', 'b1_p', 'b2_p', 'R2']
    (NaN throughout if the surface is too sparse to estimate).
    """
    idx = ['sigma', 'mu', 'rho', 'omega', 'b1_p', 'b2_p', 'R2']
    try:
        x_grid = np.asarray(x_grid)
        atm = int(np.where(np.isclose(x_grid, 0.0))[0][0])          # ATM column (x = 0)
        m20 = maturities_days.index(20) if isinstance(maturities_days, tuple) \
            else list(maturities_days).index(20)
        m40 = list(maturities_days).index(40)
        m_rep = list(maturities_days).index(report_maturity_days)
        tau1, tau2 = 20.0 / 365.0, 40.0 / 365.0

        I20_sq = surface[m20, atm] ** 2
        I40_sq = surface[m40, atm] ** 2

        # 1. drift rate of implied volatility (term-structure slope)
        mu = (I40_sq - I20_sq) / (2.0 * (I40_sq * tau2 - I20_sq * tau1))

        # 2. instantaneous volatility (term-structure level at tau_1)
        sigma = np.sqrt(I20_sq * (1.0 - 2.0 * tau1 * mu))

        # 3. smile regression at the report maturity over -1 <= x <= 1
        smile_cols = np.where((x_grid >= -1.0) & (x_grid <= 1.0))[0]
        iv_row = surface[m_rep, smile_cols]
        tau_rep = report_maturity_days / 365.0
        # On the smoothed grid:  z_+ = x I sqrt(tau),  z_- = z_+ - I^2 tau
        z_plus = x_grid[smile_cols] * iv_row * np.sqrt(tau_rep)
        z_minus = z_plus - iv_row ** 2 * tau_rep

        Y = iv_row ** 2 - surface[m_rep, atm] ** 2
        # basis columns: (2 z_+) for the rho-term, (z_+ z_-) for the omega^2-term
        Xmat = np.stack((2.0 * z_plus, z_plus * z_minus), axis=1)

        model = ConstrainedLinearRegression(fit_intercept=False)
        min_coef = np.array([-np.inf, 0.0])     # omega^2 >= 0
        max_coef = np.array([np.inf, np.inf])
        model.fit(Xmat, Y, min_coef=min_coef, max_coef=max_coef)
        b1, b2 = model.coef_[0], model.coef_[1]

        # diagnostics from an unconstrained OLS on the same design
        ols = sm.OLS(Y, Xmat).fit()

        omega = np.sqrt(b2)                      # b2 = omega^2
        rho = b1 / (sigma * omega)               # b1 = rho * sigma * omega

        out = pd.Series([sigma, mu, rho, omega,
                         ols.pvalues[0], ols.pvalues[1], ols.rsquared], index=idx)
        return out.replace([-np.inf, np.inf], np.nan)
    except Exception:
        return pd.Series([np.nan] * len(idx), index=idx)
