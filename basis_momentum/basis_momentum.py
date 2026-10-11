"""
Basis-momentum (Boons & Prado, Journal of Finance 2019) – dagens rangering af råvarer.

    pip install yfinance pandas numpy
    python basis_momentum.py fetch   # henter kontraktkurser fra Yahoo og gemmer dem i data/contracts/
    python basis_momentum.py rank    # rangerer ud fra de gemte kurser (henter fra Yahoo, hvis de mangler)

Yahoo sletter kurser for udløbne kontrakter. Derfor gemmer 'fetch' hver kontrakts dagskurser i en CSV
og fletter nye data ind i de gamle, så historikken overlever udløbet. GitHub-workflowen
.github/workflows/basis-momentum.yml kører 'fetch' + 'rank' hver hverdag og committer resultatet.

Metode
- Hver månedsafslutning t: C1 = første kontrakt med levering >= t + 2 måneder (undgår udløb/first notice),
  C2 = næste listede kontrakt efter C1.
- Månedligt afkast på SAMME kontrakt fra t til t+1 (ingen falske roll-spring).
- BM = prod(1 + R_C1) - prod(1 + R_C2) over de seneste 12 måneder.
- Strategi: lang de højeste BM, short de laveste (artiklen: top-4 / bund-4).
Bemærk: Ikke investeringsrådgivning. Tjek kontraktspecifikationer og likviditet hos din mægler.
"""
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

HERE = Path(__file__).resolve().parent
STORE = HERE / "data" / "contracts"
LOOKBACK = 12

MCODE = "FGHJKMNQUVXZ"  # jan..dec
# symbol: (Yahoo-suffix, listede leveringsmåneder, navn)
UNIVERSE = {
    "CL": (".NYM", "FGHJKMNQUVXZ", "Råolie WTI"),
    "HO": (".NYM", "FGHJKMNQUVXZ", "Fyringsolie"),
    "RB": (".NYM", "FGHJKMNQUVXZ", "Benzin RBOB"),
    "NG": (".NYM", "FGHJKMNQUVXZ", "Naturgas"),
    "PL": (".NYM", "FJNV",         "Platin"),
    "PA": (".NYM", "HMUZ",         "Palladium"),
    "HG": (".CMX", "HKNUZ",        "Kobber"),
    "GC": (".CMX", "GJMQVZ",       "Guld"),
    "SI": (".CMX", "HKNUZ",        "Sølv"),
    "ZC": (".CBT", "HKNUZ",        "Majs"),
    "ZS": (".CBT", "FHKNQUX",      "Sojabønner"),
    "ZW": (".CBT", "HKNUZ",        "Hvede"),
    "KE": (".CBT", "HKNUZ",        "KC-hvede"),
    "ZL": (".CBT", "FHKNQUVZ",     "Sojaolie"),
    "ZM": (".CBT", "FHKNQUVZ",     "Sojaskrå"),
    "LE": (".CME", "GJMQVZ",       "Levende kvæg"),
    "HE": (".CME", "GJKMNQVZ",     "Svin"),
    "GF": (".CME", "FHJKQUVX",     "Fodringskvæg"),
    "KC": (".NYB", "HKNUZ",        "Kaffe"),
    "SB": (".NYB", "HKNV",         "Sukker"),
    "CT": (".NYB", "HKNVZ",        "Bomuld"),
    "CC": (".NYB", "HKNUZ",        "Kakao"),
}


def ticker(sym, year, month):
    suf = UNIVERSE[sym][0]
    return f"{sym}{MCODE[month-1]}{str(year)[-2:]}{suf}"


def listed_after(sym, asof, n=2):
    """De n første listede kontrakter (år, måned) med levering >= asof + 2 måneder."""
    months = [MCODE.index(c) + 1 for c in UNIVERSE[sym][1]]
    first = (asof.year * 12 + asof.month - 1) + 2
    out = []
    k = first
    while len(out) < n and k < first + 48:
        y, m = divmod(k, 12)
        if m + 1 in months:
            out.append((y, m + 1))
        k += 1
    return out


def nearby_pair(sym, asof):
    """C1, C2 (år, måned) ved månedsafslutning 'asof'."""
    return listed_after(sym, asof, 2)


# ---------------------------------------------------------------- datalager

def store_path(tk):
    return STORE / f"{tk.replace('.', '_')}.csv"


def load_daily(tk):
    p = store_path(tk)
    if not p.exists():
        return pd.Series(dtype=float)
    s = pd.read_csv(p, index_col=0, parse_dates=True)["close"].dropna()
    return s[~s.index.duplicated(keep="last")].sort_index()


def download_daily(tk, period="2y"):
    if yf is None:
        return pd.Series(dtype=float)
    try:
        h = yf.download(tk, period=period, interval="1d", progress=False, auto_adjust=False)
    except Exception:
        return pd.Series(dtype=float)
    if not len(h):
        return pd.Series(dtype=float)
    s = h["Close"].squeeze().dropna()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    return s


def save_daily(tk, new):
    """Flet nye kurser ind i de gemte; nye værdier vinder ved samme dato."""
    old = load_daily(tk)
    s = pd.concat([old, new])
    s = s[~s.index.duplicated(keep="last")].sort_index()
    STORE.mkdir(parents=True, exist_ok=True)
    s.rename("close").rename_axis("date").to_frame().to_csv(store_path(tk), float_format="%.6g")
    return len(s) - len(old)


def needed_contracts(sym, today):
    """Alle kontrakter, der indgår i BM for de seneste LOOKBACK måneder, plus de næste der kommer i brug."""
    last = pd.Period(today, "M") - 1
    tks = set()
    for k in range(LOOKBACK + 1, -1, -1):
        t0 = last - k
        for y, m in listed_after(sym, t0.to_timestamp(how="end"), 2):
            tks.add(ticker(sym, y, m))
    for y, m in listed_after(sym, pd.Timestamp(today), 8):     # længere ude = længere historik til backtest
        tks.add(ticker(sym, y, m))
    return sorted(tks)


def fetch(today):
    if yf is None:
        raise SystemExit("Installer først: pip install yfinance pandas numpy")
    ok = miss = 0
    for sym in UNIVERSE:
        for tk in needed_contracts(sym, today):
            s = download_daily(tk, period="max")
            if len(s):
                added = save_daily(tk, s)
                ok += 1
                print(f"{tk:12s} {len(s):4d} dage hentet, {added:+d} nye  ({s.index[0].date()} – {s.index[-1].date()})")
            else:
                miss += 1
                print(f"{tk:12s} ingen data hos Yahoo" + (" (gemt historik bruges)" if store_path(tk).exists() else ""))
    print(f"\n{ok} kontrakter hentet, {miss} uden data.")


# ---------------------------------------------------------------- beregning

_cache = {}


def monthly_closes(tk):
    if tk not in _cache:
        s = load_daily(tk)
        if not len(s):
            s = download_daily(tk, period="3y")
        _cache[tk] = s.groupby(s.index.to_period("M")).last() if len(s) else pd.Series(dtype=float)
    return _cache[tk]


def basis_momentum(sym, today):
    """BM over de seneste LOOKBACK måneder. Returnerer (bm, antal måneder, note).
    Mangler de ældste måneder, beregnes BM over de seneste sammenhængende måneder (markeres som delvis)."""
    last = pd.Period(today, "M") - 1                       # seneste afsluttede måned
    r1, r2 = [], []
    gap = None
    for k in range(1, LOOKBACK + 1):                       # baglæns fra seneste måned
        t1 = last - k + 1
        t0 = t1 - 1
        (y1, m1), (y2, m2) = nearby_pair(sym, t0.to_timestamp(how="end"))
        a, b = monthly_closes(ticker(sym, y1, m1)), monthly_closes(ticker(sym, y2, m2))
        try:
            r1.append(a[t1] / a[t0] - 1)
            r2.append(b[t1] / b[t0] - 1)
        except KeyError:
            gap = f"mangler {ticker(sym,y1,m1).split('.')[0]}/{ticker(sym,y2,m2).split('.')[0]} i {t0}–{t1}"
            break
    n = len(r1)
    if n == 0:
        return np.nan, 0, gap
    bm = np.prod(1 + np.array(r1)) - np.prod(1 + np.array(r2))
    mom = np.prod(1 + np.array(r1)) - 1
    note = f"{n}m-afkast front {mom*100:+.1f}%"
    if gap:
        note += f"; {gap}"
    return bm, n, note


def basis_momentum_proxy(sym, today):
    """Proxy, når udløbne kontrakter mangler: BM på DAGENS C1/C2-par, holdt fast over de seneste
    LOOKBACK måneder. Samme formel, men med længere løbetid tidligt i perioden end i artiklen."""
    last = pd.Period(today, "M") - 1
    t0 = last - LOOKBACK
    (y1, m1), (y2, m2) = nearby_pair(sym, pd.Timestamp(today))
    a, b = monthly_closes(ticker(sym, y1, m1)), monthly_closes(ticker(sym, y2, m2))
    try:
        return (a[last] / a[t0]) - (b[last] / b[t0])
    except KeyError:
        return np.nan


def rank(today, out=None):
    rows = []
    for sym in UNIVERSE:
        bm, n, note = basis_momentum(sym, today)
        proxy = basis_momentum_proxy(sym, today)
        (y1, m1), (y2, m2) = nearby_pair(sym, pd.Timestamp(today))
        rows.append(dict(symbol=sym, navn=UNIVERSE[sym][2],
                         BM_pct=None if np.isnan(bm) else round(bm * 100, 2), mdr=n,
                         proxy_pct=None if np.isnan(proxy) else round(proxy * 100, 2),
                         front=ticker(sym, y1, m1).split(".")[0], anden=ticker(sym, y2, m2).split(".")[0],
                         note=note))
    df = pd.DataFrame(rows)
    full = df[df.mdr == LOOKBACK].sort_values("BM_pct", ascending=False).reset_index(drop=True)
    part = df[(df.mdr > 0) & (df.mdr < LOOKBACK)].sort_values("BM_pct", ascending=False)
    miss = df[df.mdr == 0]

    n = len(full)
    k = min(4, n // 2) if n >= 16 else n // 4
    full["signal"] = "-"
    if k:
        full.loc[:k - 1, "signal"] = "LANG"
        full.loc[n - k:, "signal"] = "SHORT"

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 80)
    lines = [f"Basis-momentum pr. {today}  (data til og med {pd.Period(today, 'M') - 1})", ""]
    if n:
        lines.append(full[["signal", "symbol", "navn", "BM_pct", "front", "anden", "note"]].to_string(index=False))
        lines.append(f"\n{n} råvarer med fulde {LOOKBACK} måneder. Lang top-{k}, short bund-{k}, "
                     "ligevægtet, rebalancér månedligt.")
    else:
        lines.append(f"Ingen råvarer har endnu fulde {LOOKBACK} måneders kontrakthistorik.")
        px = df.dropna(subset=["proxy_pct"]).sort_values("proxy_pct", ascending=False).reset_index(drop=True)
        m = len(px)
        kp = min(4, m // 2) if m >= 16 else m // 4
        px["signal"] = "-"
        if kp:
            px.loc[:kp - 1, "signal"] = "LANG"
            px.loc[m - kp:, "signal"] = "SHORT"
        lines.append(f"\nPROXY (indtil historikken er bygget op): BM på dagens front/2. kontrakt holdt fast i "
                     f"{LOOKBACK} måneder. Ikke artiklens præcise signal.")
        lines.append(px[["signal", "symbol", "navn", "proxy_pct", "front", "anden"]].to_string(index=False))
        lines.append(f"\n{m} råvarer med proxy. Lang top-{kp}, short bund-{kp}.")
    if len(part):
        lines.append(f"\nDelvis historik (BM over færre end {LOOKBACK} måneder – kun vejledende, ikke artiklens signal):")
        lines.append(part[["symbol", "navn", "BM_pct", "mdr", "front", "anden", "note"]].to_string(index=False))
    if len(miss):
        lines.append("\nUden data:")
        lines.append(miss[["symbol", "navn", "note"]].to_string(index=False))
    lines.append("\nSpread-version (lavere margin): i LANG-råvarer køb front/sælg 2. kontrakt, omvendt i SHORT.")
    text = "\n".join(lines)
    print(text)
    if out:
        Path(out).write_text(text + "\n", encoding="utf-8")
        df.to_csv(Path(out).with_suffix(".csv"), index=False)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "rank"
    today = dt.date.today()
    if cmd == "fetch":
        fetch(today)
    elif cmd == "rank":
        rank(today, sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        raise SystemExit("Brug: python basis_momentum.py [fetch|rank [outfil]]")


if __name__ == "__main__":
    main()
