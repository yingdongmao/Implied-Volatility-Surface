"""
run_risk_premia_from_wrds.py
============================
Companion to ``run_from_wrds.py``: that script produces the four STATE
VARIABLES from the IV surface; this one produces the four RISK PREMIA carried
by them, on real OptionMetrics data.

    gamma_rp   (dS/S)^2 - sigma^2         gamma / "jump" premium
    vrp        dI/I - mu                  VOLATILITY RISK PREMIUM
    vov_rp     (dI/I)^2 - omega^2         vol-of-vol premium
    cov_rp     dS dI/(S I) - rho sigma omega   covariance premium

Merging this CSV with ``run_from_wrds.py``'s on (secid, date) gives
``RRP^v = vrp * (sigma/omega) * rho`` via ``risk_premia.return_risk_premium``.

Requires a WRDS account with OptionMetrics access and ``pip install wrds``.

The three choices this script pins down
---------------------------------------
1. CONTRACT level.  The regression runs across the day's actual option
   contracts (not across nodes of the smoothed surface), which is what the
   factor structure in ``risk_premia`` is derived for.
2. EUROPEAN value, priced inside ``risk_premia.european_value`` from the
   contract's own OptionMetrics IV -- at t with (S_t, I_t, tau) and at t+1 with
   (S_t+1, I_t+1, tau_next).  All three move; the contract really ages.
3. INTERCEPT retained.  The gamma factor loads on the constant 1/2 for every
   contract; dropping the intercept forces that P&L into the remaining
   coefficients and badly biases the volatility risk premium.

Matching t to t+1
-----------------
Contracts are linked on ``optionid``, which survives the strike re-listing that
follows a stock split (matching on strike does not), and the next quote must
fall on the stock's next trading day.

Run:  python examples/run_risk_premia_from_wrds.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from risk_premia import estimate_risk_premia

WRDS_USERNAME = os.environ.get("WRDS_USERNAME", "your_wrds_username")
SECIDS = [101594]                    # e.g. 101594 = AAPL
YEARS = [2005, 2008]
MIN_CONTRACTS = 8                    # contracts needed for the day's regression
OUTFILE = "risk_premia.csv"

# 'forward' matches the production pipeline (no discounting needed, see
# european_value).  'close' changes the premia only marginally.
UNDERLYING = "forward"


# --------------------------------------------------------------------------- #
# WRDS pulls
# --------------------------------------------------------------------------- #
def pull_options(conn, year, secids):
    """Option chain: 2-90 day maturity, positive spread, valid IV."""
    in_list = "(" + ",".join(str(int(s)) for s in secids) + ")"
    op = conn.raw_sql(f"""
        select a.secid, a.optionid, a.date, a.exdate,
               a.cp_flag as option_type,
               a.strike_price/1000.0 as strike,
               a.impl_volatility as iv,
               b.close as spot
        from optionm.opprcd{year} a
        left join optionm.secprd{year} b
               on a.secid = b.secid and a.date = b.date
        where a.secid in {in_list}
          and a.exdate - a.date >= 2 and a.exdate - a.date <= 90
          and a.best_bid > 0 and a.best_offer > a.best_bid
          and a.impl_volatility is not null and a.impl_volatility > 0.01
        order by a.secid, a.date
    """, date_cols=['date', 'exdate'])
    op[['strike', 'iv', 'spot']] = op[['strike', 'iv', 'spot']].apply(
        pd.to_numeric, errors='coerce')
    return op


def pull_forwards(conn, year, secids):
    """OptionMetrics forward price, one per (secid, date, expiration)."""
    in_list = "(" + ",".join(str(int(s)) for s in secids) + ")"
    f = conn.raw_sql(f"""
        select secid, date, expiration as exdate, forwardprice
        from optionm.fwdprd{year} where secid in {in_list}
    """, date_cols=['date', 'exdate'])
    f['forwardprice'] = pd.to_numeric(f['forwardprice'], errors='coerce')
    return f.drop_duplicates(subset=['secid', 'date', 'exdate'])


# --------------------------------------------------------------------------- #
# t -> t+1 contract matching
# --------------------------------------------------------------------------- #
def build_panel(op, underlying=UNDERLYING):
    """Attach each contract's t+1 leg and return the frame
    ``estimate_risk_premia`` expects."""
    op = op.copy()
    op['underlying'] = op['forwardprice'] if underlying == 'forward' else op['spot']

    # the stock's own calendar: each date -> its next trading date
    cal = op[['secid', 'date']].drop_duplicates().sort_values(['secid', 'date'])
    cal['next_date'] = cal.groupby('secid')['date'].shift(-1)
    op = op.merge(cal, on=['secid', 'date'], how='left')

    op = op.sort_values(['secid', 'optionid', 'date'])
    g = op.groupby(['secid', 'optionid'])
    # the contract's next quote must BE the stock's next trading day
    consecutive = g['date'].shift(-1).eq(op['next_date'])
    op['iv_next'] = np.where(consecutive, g['iv'].shift(-1), np.nan)
    op['spot_next'] = np.where(consecutive, g['underlying'].shift(-1), np.nan)

    op['spot'] = op['underlying']
    op['tau'] = (op['exdate'] - op['date']).dt.days / 365.0
    op['tau_next'] = (op['exdate'] - op['next_date']).dt.days / 365.0
    return op


# --------------------------------------------------------------------------- #
def main():
    import wrds
    conn = wrds.Connection(wrds_username=WRDS_USERNAME)
    try:
        frames = []
        for year in YEARS:
            op = pull_options(conn, year, SECIDS)
            fw = pull_forwards(conn, year, SECIDS)
            op = op.merge(fw, on=['secid', 'date', 'exdate'], how='left')
            print(f"  {year}: {len(op):,} contract-days")
            frames.append(op)
    finally:
        conn.close()

    panel = build_panel(pd.concat(frames, ignore_index=True))
    print("contracts with a usable t+1 leg: %.1f%%" % (100 * panel['iv_next'].notna().mean()))

    rows = []
    for (secid, date), chain in panel.groupby(['secid', 'date']):
        rp = estimate_risk_premia(chain, min_contracts=MIN_CONTRACTS)
        rp['secid'], rp['date'] = secid, date
        rp['n_contracts'] = len(chain)
        rows.append(rp)

    out = pd.DataFrame(rows)
    out = out[['secid', 'date', 'n_contracts', 'gamma_rp', 'vrp', 'vov_rp',
               'cov_rp', 'R2', 'nobs']].sort_values(['secid', 'date'])
    out.to_csv(OUTFILE, index=False, float_format='%.6f')

    print(f"\nrisk-premium panel: {len(out):,} stock-days, "
          f"{out['vrp'].notna().sum():,} with an estimate")
    print(out.head(10).to_string(index=False))
    print(f"\nVRP: mean={out['vrp'].mean():.4f}  median={out['vrp'].median():.4f}  "
          f"share negative={100 * (out['vrp'] < 0).mean():.1f}%")
    print(f"\nsaved -> {OUTFILE}")


if __name__ == "__main__":
    main()
