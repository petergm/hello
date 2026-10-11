"""
Backtest af basis-momentum på de gemte kontraktkurser i data/contracts/.

    python backtest.py [lookback-måneder ...]      # standard: 12 6 3
    MIN_N=12 python backtest.py                    # kræv mindst 12 råvarer pr. måned (standard)

Begrænsning: Yahoo har kun kurser for kontrakter, der stadig handles. Ved hver månedsafslutning t
vælges derfor C1 = den nærmeste kontrakt med levering >= t + 2 måneder, som har kurser for hele
perioden t-L .. t+1, og C2 = den næste sådanne kontrakt. Tidligt i perioden er det kontrakter med
lang løbetid (ikke artiklens nærmeste kontrakter), så resultatet er en tilnærmelse.

Strategi pr. måned: rangér på BM_t = prod(1+R_C1) - prod(1+R_C2) over L måneder,
lang top-k / short bund-k ligevægtet (k = 4, eller n//4 ved færre end 16 råvarer).
Afkast t -> t+1 måles på C1 (samme kontrakt), ingen omkostninger.
Spread-versionen holder C1-C2 i stedet (lang front/short 2. i LANG, omvendt i SHORT).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from basis_momentum import MCODE, STORE, UNIVERSE, load_daily

HERE = Path(__file__).resolve().parent
MIN_N = int(os.environ.get("MIN_N", 12))   # færre råvarer giver for tynd en rangering


def load_contracts():
    """{sym: [(leveringsmåned som Period, månedlige lukkekurser), ...]} sorteret efter levering."""
    out = {sym: [] for sym in UNIVERSE}
    for p in STORE.glob("*.csv"):
        name = p.stem.split("_")[0]                      # fx CLZ26
        sym, code, yy = name[:-3], name[-3], name[-2:]
        if sym not in UNIVERSE or code not in MCODE:
            continue
        s = load_daily(p.stem.replace("_", "."))
        if len(s) < 20:
            continue
        m = s.groupby(s.index.to_period("M")).last()
        out[sym].append((pd.Period(year=2000 + int(yy), month=MCODE.index(code) + 1, freq="M"), m))
    for sym in out:
        out[sym].sort(key=lambda x: x[0])
    return out


def pair_at(contracts, t, L):
    """Nærmeste to kontrakter med levering >= t+2 og kurser for alle måneder t-L .. t+1."""
    need = pd.period_range(t - L, t + 1, freq="M")
    ok = [(d, m) for d, m in contracts if d >= t + 2 and need.isin(m.index).all()]
    return ok[:2] if len(ok) >= 2 else None


def run(L, data):
    months = sorted({p for c in data.values() for _, m in c for p in m.index})
    last = pd.Period(pd.Timestamp.today(), "M") - 1      # seneste afsluttede måned
    rows, picks = [], []
    for t in months:
        if t + 1 > last:
            break
        sig = {}
        for sym, cs in data.items():
            pr = pair_at(cs, t, L)
            if pr is None:
                continue
            (d1, a), (d2, b) = pr
            bm = a[t] / a[t - L] - b[t] / b[t - L]
            r1 = a[t + 1] / a[t] - 1
            r2 = b[t + 1] / b[t] - 1
            sig[sym] = (bm, r1, r2, d1, d2)
        n = len(sig)
        if n < max(4, MIN_N):
            continue
        k = min(4, n // 2) if n >= 16 else max(1, n // 4)
        order = sorted(sig, key=lambda s: sig[s][0], reverse=True)
        longs, shorts = order[:k], order[-k:]
        ls = np.mean([sig[s][1] for s in longs]) - np.mean([sig[s][1] for s in shorts])
        spread = np.mean([sig[s][1] - sig[s][2] for s in longs]) - np.mean([sig[s][1] - sig[s][2] for s in shorts])
        ew = np.mean([v[1] for v in sig.values()])
        rows.append(dict(maaned=str(t + 1), n=n, LS=ls, spread=spread, ew_lang=ew))
        avg_mat = np.mean([(v[3] - t).n for v in sig.values()])
        picks.append(dict(formation=str(t), lang=" ".join(longs), short=" ".join(shorts),
                          gns_mdr_til_levering_C1=round(avg_mat, 1)))
    return pd.DataFrame(rows), pd.DataFrame(picks)


def stats(r):
    r = r.dropna()
    if not len(r):
        return dict(mdr=0)
    ann = (1 + r).prod() ** (12 / len(r)) - 1
    vol = r.std(ddof=1) * np.sqrt(12) if len(r) > 1 else np.nan
    cum = (1 + r).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 and r.std() > 0 else np.nan
    return dict(mdr=len(r), samlet=f"{cum.iloc[-1]-1:+.1%}", aarlig=f"{ann:+.1%}", vol=f"{vol:.1%}",
                sharpe=round(r.mean() * 12 / vol, 2) if vol else np.nan, t_stat=round(t, 2),
                hit=f"{(r > 0).mean():.0%}", max_dd=f"{dd:.1%}")


def main():
    lookbacks = [int(x) for x in sys.argv[1:]] or [12, 6, 3]
    data = load_contracts()
    print(f"Mindst {MIN_N} råvarer pr. måned.")
    print(f"Kontrakter i datalageret: {sum(len(v) for v in data.values())} "
          f"({', '.join(f'{s}:{len(v)}' for s, v in data.items())})\n")
    lines = []
    for L in lookbacks:
        res, picks = run(L, data)
        lines.append(f"=== Lookback {L} måneder ===")
        if not len(res):
            lines.append("For lidt historik til denne lookback.\n")
            continue
        lines.append(f"Periode {res.maaned.iloc[0]} – {res.maaned.iloc[-1]}, "
                     f"{res.n.min()}–{res.n.max()} råvarer pr. måned, "
                     f"gns. {picks.gns_mdr_til_levering_C1.mean():.0f} mdr. til levering for C1")
        st = pd.DataFrame({c: stats(res[c]) for c in ["LS", "spread", "ew_lang"]}).T
        st.index = ["Lang/short (C1)", "Spread (C1-C2)", "Ligevægtet lang alle"]
        lines.append(st.to_string())
        yr = res.assign(aar=res.maaned.str[:4]).groupby("aar")[["LS", "spread", "ew_lang"]].apply(lambda d: (1 + d).prod() - 1)
        lines.append("\nPr. år:\n" + yr.map(lambda x: f"{x:+.1%}").to_string())
        lines.append("")
        res.to_csv(HERE / "data" / f"backtest_L{L}.csv", index=False)
        picks.to_csv(HERE / "data" / f"backtest_L{L}_positioner.csv", index=False)
    text = "\n".join(lines)
    print(text)
    (HERE / "data" / "backtest_summary.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
