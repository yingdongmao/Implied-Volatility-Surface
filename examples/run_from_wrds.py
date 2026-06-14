"""
run_from_wrds.py
================
Reference driver that reproduces the state variables on REAL data from
OptionMetrics via WRDS.  Requires:

    * a WRDS account with OptionMetrics access (set WRDS_USERNAME below),
    * `pip install wrds` and a one-time `~/.pgpass` (created on first connect).

It pulls one year of equity-option records for a handful of secids, builds the
IV surface per (secid, date), and extracts (sigma, mu, rho, omega).  Scale it up
to the full S&P 500 universe by widening `SECIDS` / the year range.

Run:  python examples/run_from_wrds.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from iv_surface import build_surface
from state_variables import extract_state_variables

WRDS_USERNAME = os.environ.get("WRDS_USERNAME", "your_wrds_username")
YEAR = 2015
SECIDS = [101594, 102796, 106203]          # e.g. AAPL, MSFT, ... (illustrative secids)


def pull_year(conn, year, secids):
    """OptionMetrics opprcd join secprd, filtered to 2-90 day, valid-IV contracts."""
    in_list = "(" + ",".join(str(int(s)) for s in secids) + ")"
    sql = f"""
        select a.secid, a.date, a.exdate,
               a.cp_flag as option_type,
               (a.exdate - a.date) as maturity,
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
    """
    op = conn.raw_sql(sql, date_cols=['date', 'exdate'])
    num = ['maturity', 'strike', 'iv', 'spot']
    op[num] = op[num].apply(pd.to_numeric, errors='coerce')
    return op


def main():
    import wrds
    conn = wrds.Connection(wrds_username=WRDS_USERNAME)
    try:
        data = pull_year(conn, YEAR, SECIDS)
    finally:
        conn.close()
    print(f"pulled {len(data):,} option records for {YEAR}")

    out = []
    for (secid, date), chain in data.groupby(['secid', 'date']):
        surface = build_surface(chain)
        sv = extract_state_variables(surface)
        sv['secid'] = secid
        sv['date'] = date
        out.append(sv)

    panel = pd.DataFrame(out).dropna(subset=['sigma', 'mu', 'rho', 'omega'])
    print(f"\nstate-variable panel: {len(panel):,} firm-days")
    print(panel[['secid', 'date', 'sigma', 'mu', 'rho', 'omega']].head(10).to_string(index=False))
    panel.to_pickle(f"state_variables_{YEAR}.pickle")
    print(f"\nsaved -> state_variables_{YEAR}.pickle")


if __name__ == "__main__":
    main()
