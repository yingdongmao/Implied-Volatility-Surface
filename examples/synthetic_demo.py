"""
synthetic_demo.py
=================
End-to-end smoke test that runs WITHOUT WRDS or any market data.

It fabricates a plausible single-stock-day option chain, then runs the three
pipeline steps:

    1. iv_surface.build_surface         raw quotes      -> smooth IV surface
    2. state_variables.extract_...      IV surface      -> sigma, mu, rho, omega
    3. risk_premia.estimate_risk_premia quotes (t, t+1) -> realized risk premia
       risk_premia.return_risk_premium  VRP + state vars -> RRP^v

Run:  python examples/synthetic_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from iv_surface import build_surface
from state_variables import extract_state_variables
from risk_premia import estimate_risk_premia, return_risk_premium


def fake_chain(spot=100.0, seed=0):
    """A small, smiley, upward-sloping option chain across 3 maturities."""
    rng = np.random.default_rng(seed)
    rows = []
    for mat in (15, 25, 35, 45):                         # days
        tau = mat / 365.0
        for strike in np.arange(80, 121, 2.5):
            x = np.log(strike / spot) / (0.30 * np.sqrt(tau))
            # base level + upward term structure + convex smile + noise
            iv = 0.28 + 0.04 * tau * 12 + 0.05 * x ** 2 - 0.01 * x
            iv += rng.normal(0, 0.003)
            rows.append((iv, strike, spot, mat,
                         'C' if strike >= spot else 'P'))
    return pd.DataFrame(rows, columns=['iv', 'strike', 'spot', 'maturity', 'option_type'])


def main():
    quotes = fake_chain()
    print(f"raw quotes: {len(quotes)} contracts\n")

    # 1. IV surface
    surface = build_surface(quotes)
    print("smoothed IV surface (rows=20/30/40d, cols=x=-2..2):")
    print(np.round(surface, 4), "\n")

    # 2. state variables
    sv = extract_state_variables(surface)
    print("state variables (30-day):")
    print(sv.round(4), "\n")

    # 3. realized risk premia + RRP^v
    #    Build the t+1 leg of the SAME contracts.  All three inputs must move:
    #    the underlying, the implied volatility, and the remaining maturity.
    #    (A t+1 leg with the IV held fixed would set dI/I = 0 and the volatility
    #    risk premium -- the whole point of the exercise -- would be zero by
    #    construction.)  ``estimate_risk_premia`` prices the European contract
    #    at both dates itself.
    q = quotes.copy()
    q['tau'] = q['maturity'] / 365.0
    q['tau_next'] = q['tau'] - 1.0 / 365.0            # one day passes
    q['spot_next'] = q['spot'] * 1.01                 # +1% underlying move
    q['iv_next'] = q['iv'] * 0.98                     # -2% implied-vol move

    rp = estimate_risk_premia(q)
    print("realized risk premia carried by state variables:")
    print(rp.round(6), "\n")

    rrp_v = return_risk_premium(rp['vrp'], sv['sigma'], sv['omega'], sv['rho'])
    print(f"RRP^v  = VRP * (sigma/omega) * rho = {rrp_v:.6f}")


if __name__ == "__main__":
    main()
