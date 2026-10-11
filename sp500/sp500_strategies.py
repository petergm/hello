"""
S&P 500 futures – backtest af veldokumenterede regler (ingen parameteroptimering).

    pip install yfinance pandas numpy
    python sp500_strategies.py fetch      # henter ^GSPC, ^SP500TR, ^IRX, ES=F, ^VIX til data/
    python sp500_strategies.py backtest   # skriver data/backtest_summary.txt

Futures-afkast ≈ S&P 500 total return − T-bill-rente (futures-prisen indeholder carry).
Afkast er "excess return" pr. enhed notional; renter på marginkontoen er ikke medregnet.

Strategier (signal ved månedsafslutning, handles fra næste dag):
- Buy & hold                1 kontrakt-enhed hele tiden.
- SMA10                     Lang hvis indeks > gennemsnit af seneste 10 månedslukninger, ellers flad
                            (Faber 2007, "A Quantitative Approach to Tactical Asset Allocation").
- TSMOM12                   Lang hvis 12-måneders excess return > 0, ellers flad
                            (Moskowitz, Ooi & Pedersen 2012, "Time series momentum").
- Vol-target 15%            Vægt = 15% / realiseret vol (seneste 21 dage), maks 2x
                            (Moreira & Muir 2017, "Volatility-managed portfolios").
- SMA10 + vol-target        SMA10-signal × vol-vægt.
Omkostninger: 3 bp pr. handlet enhed + 1 bp pr. enhed ved hvert kvartalsvist roll.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
TICKERS = {"GSPC": "^GSPC", "SP500TR": "^SP500TR", "IRX": "^IRX", "ES": "ES=F", "VIX": "^VIX"}
COST_TRADE = 0.0003
COST_ROLL = 0.0001
VOL_TARGET = 0.15
MAX_LEV = 2.0


# ---------------------------------------------------------------- data

def fetch():
    import yfinance as yf
    DATA.mkdir(parents=True, exist_ok=True)
    for name, tk in TICKERS.items():
        h = yf.download(tk, period="max", interval="1d", progress=False, auto_adjust=False)
        s = h["Close"].squeeze().dropna() if len(h) else pd.Series(dtype=float)
        if not len(s):
            print(f"{tk}: ingen data")
            continue
        s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
        s.rename("close").rename_axis("date").to_frame().to_csv(DATA / f"{name}.csv", float_format="%.6g")
        print(f"{tk:10s} {len(s):6d} dage  {s.index[0].date()} – {s.index[-1].date()}")


def load(name):
    p = DATA / f"{name}.csv"
    if not p.exists():
        return pd.Series(dtype=float)
    s = pd.read_csv(p, index_col=0, parse_dates=True)["close"].dropna()
    return s[~s.index.duplicated(keep="last")].sort_index()


def daily_returns(total_return=True):
    """Daglige futures-lignende excess returns og indeksniveau til signaler."""
    px = load("GSPC")
    irx = load("IRX").reindex(px.index).ffill()
    rf = (irx / 100 / 252).fillna(0.0)                  # T-bill fra 1960; før det sættes rf = 0
    if total_return:
        tr = load("SP500TR")
        px = px[px.index >= tr.index[0]]
        ret = tr.reindex(px.index).ffill().pct_change()
    else:
        ret = px.pct_change()                             # kun kursafkast (uden udbytte)
    ex = (ret - rf.reindex(ret.index)).dropna()
    return ex, px.reindex(ex.index)


# ---------------------------------------------------------------- strategier

def month_ends(idx):
    s = pd.Series(idx, index=idx)
    return s.groupby(idx.to_period("M")).last().values


def weights(ex, px, sma_n=10, mom_n=12, vol_target=VOL_TARGET):
    me = month_ends(ex.index)
    px_m = px.loc[me]
    exidx_m = (1 + ex).cumprod().loc[me]
    sma = (px_m > px_m.rolling(sma_n).mean()).astype(float).where(px_m.rolling(sma_n).count() == sma_n)
    mom = (exidx_m.pct_change(mom_n) > 0).astype(float).where(exidx_m.pct_change(mom_n).notna())
    rv = ex.rolling(21).std() * np.sqrt(252)
    volw = (vol_target / rv.loc[me]).clip(upper=MAX_LEV)
    w = pd.DataFrame({
        "Buy & hold": 1.0,
        f"SMA{sma_n}": sma,
        f"TSMOM{mom_n}": mom,
        f"Vol-target {vol_target:.0%}": volw,
        f"SMA{sma_n} + vol-target": sma * volw,
    }, index=me)
    w = w.dropna()
    # vægt sat ved månedsafslutning gælder fra næste handelsdag
    return w.reindex(ex.index).ffill().shift(1).dropna()


def strategy_returns(ex, w):
    dw = w.diff().abs().fillna(w.abs())
    roll_days = ex.index.to_series().groupby(ex.index.to_period("Q")).last().values
    roll = pd.Series(0.0, index=ex.index)
    roll.loc[roll.index.isin(roll_days)] = COST_ROLL
    r = w.mul(ex.reindex(w.index), axis=0) - dw * COST_TRADE - w.abs().mul(roll.reindex(w.index), axis=0)
    return r


# ---------------------------------------------------------------- statistik

def stats(r, bench=None):
    m = (1 + r).groupby(r.index.to_period("M")).prod() - 1
    yrs = len(r) / 252
    cagr = (1 + r).prod() ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    cum = (1 + r).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    out = {"CAGR": f"{cagr:+.1%}", "vol": f"{vol:.1%}", "Sharpe": round(r.mean() * 252 / vol, 2),
           "max DD": f"{dd:.0%}", "Calmar": round(cagr / -dd, 2) if dd < 0 else np.nan,
           "værste md.": f"{m.min():.0%}"}
    if bench is not None:
        b = (1 + bench).groupby(bench.index.to_period("M")).prod() - 1
        X = np.column_stack([np.ones(len(m)), b.reindex(m.index).values])
        y = m.values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * resid.var(ddof=2))
        out["alpha/år"] = f"{beta[0]*12:+.1%}"
        out["alpha t"] = round(beta[0] / se[0], 2)
    return out


def table(R):
    bench = R["Buy & hold"]
    return pd.DataFrame({c: stats(R[c], None if c == "Buy & hold" else bench) for c in R}).T


def by_decade(R):
    g = R.groupby((R.index.year // 10) * 10)
    sh = g.apply(lambda d: d.mean() * 252 / (d.std() * np.sqrt(252)))
    sh.index = [f"{d}'erne" for d in sh.index]
    return sh.round(2)


def crashes(R):
    episodes = {"1987 krak": ("1987-08-25", "1987-12-04"), "Dotcom": ("2000-03-24", "2002-10-09"),
                "Finanskrise": ("2007-10-09", "2009-03-09"), "Covid": ("2020-02-19", "2020-03-23"),
                "2022": ("2022-01-03", "2022-10-12")}
    out = {}
    for k, (a, b) in episodes.items():
        d = R.loc[a:b]
        if len(d) > 5:
            out[k] = ((1 + d).prod() - 1).map(lambda x: f"{x:+.0%}")
    return pd.DataFrame(out)


def sensitivity(ex, px):
    rows = {}
    for n in [6, 8, 10, 12, 14]:
        r = strategy_returns(ex, weights(ex, px, sma_n=n, mom_n=n))
        rows[f"{n} mdr."] = {f"SMA Sharpe": round(r.iloc[:, 1].mean() / r.iloc[:, 1].std() * np.sqrt(252), 2),
                             f"TSMOM Sharpe": round(r.iloc[:, 2].mean() / r.iloc[:, 2].std() * np.sqrt(252), 2)}
    for vt in [0.10, 0.15, 0.20]:
        r = strategy_returns(ex, weights(ex, px, vol_target=vt))
        rows[f"vol {vt:.0%}"] = {"Vol-target Sharpe": round(r.iloc[:, 3].mean() / r.iloc[:, 3].std() * np.sqrt(252), 2),
                                 "SMA+vol Sharpe": round(r.iloc[:, 4].mean() / r.iloc[:, 4].std() * np.sqrt(252), 2)}
    return pd.DataFrame(rows).T


def backtest():
    pd.set_option("display.width", 200)
    lines = []

    ex, px = daily_returns(total_return=True)
    R = strategy_returns(ex, weights(ex, px))
    lines += [f"=== A: Total return − T-bill (≈ ES-futures), {R.index[0].date()} – {R.index[-1].date()} ===",
              table(R).to_string(), "", "Sharpe pr. årti:", by_decade(R).to_string(), "",
              "Afkast i krak:", crashes(R).to_string(), "",
              "Parameterfølsomhed (Sharpe; artiklernes valg: 10/12 mdr., 15% vol):",
              sensitivity(ex, px).to_string(), ""]
    R.to_csv(DATA / "daily_returns_A.csv", float_format="%.6g")

    ex2, px2 = daily_returns(total_return=False)
    R2 = strategy_returns(ex2, weights(ex2, px2))
    lines += [f"=== B: Lang historik, kun kursafkast (uden udbytte; T-bill fra 1960), "
              f"{R2.index[0].date()} – {R2.index[-1].date()} ===",
              table(R2).to_string(), "", "Sharpe pr. årti:", by_decade(R2).to_string(), "",
              "Afkast i krak:", crashes(R2).to_string(), ""]

    # aktuelt signal
    w = weights(ex, px)
    me = month_ends(ex.index)[-1]
    lines.append(f"Seneste signal (beregnet {pd.Timestamp(me).date()}; endeligt ved månedsafslutning):")
    lines.append(w.iloc[-1].round(2).to_string())

    text = "\n".join(lines)
    print(text)
    (DATA / "backtest_summary.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "backtest"
    {"fetch": fetch, "backtest": backtest}[cmd]()
