#!/usr/bin/env python3
"""
Vad skiljer vinnande trades från förlorande?

1. Tar 20 års OMXS30-data och gör väldigt många trades: slumpköp (samma exit för alla:
   stop 2 ATR under, mål 2R, max 20 dagar) plus alla metodernas signaler.
2. Mäter för varje trade hur läget såg ut på köpdagen: trend, RSI, MACD, medelvärden,
   volym, candle, avstånd till toppen, index, sektor …
3. För varje faktor: snitt-R i fem grupper (lägst → högst). Bra faktorer ger tydlig skillnad
   både på gamla år (träning, före 2018) och nya år (test, 2018–).
4. "Facit i hand": hur skiljer sig de stora vinnarna (≥ +1,5R) från de som stoppades ut?
5. Kombinationer av de bästa faktorerna – och hur långt de räcker mot 40 % per år.
Resultat: results/faktorer.json
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import metoder as M
import omx30_skanner as sk

M.AR = 20
M.CACHE = Path("cache20")
TEST_FRAN = "2018-01-01"
M.METODER["slump"]["p"]["var_n"] = (5, "tätare slumpköp för analysen")


def egenskaper(d, idx_close, idx_trend):
    c, h, l, o = d["Close"], d["High"], d["Low"], d["Open"]
    ic = idx_close.reindex(d.index).ffill()
    f = pd.DataFrame(index=d.index)
    f["RSI"] = d["rsi"]
    f["MACD-histogram"] = d["hist"] / c * 100
    f["MACD-histogram stiger"] = (d["hist"] > d["hist"].shift(1)).astype(float)
    f["MACD över nollan"] = (d["macd"] > 0).astype(float)
    f["Kurs mot 20-dagars %"] = (c / d["ma20"] - 1) * 100
    f["Kurs mot 50-dagars %"] = (c / d["ma50"] - 1) * 100
    f["Kurs mot 200-dagars %"] = (c / d["ma200"] - 1) * 100
    f["50-dagars över 200-dagars"] = (d["ma50"] > d["ma200"]).astype(float)
    f["Fas 2 (kurs>20>50)"] = d["fas2"].astype(float)
    f["Utveckling 5 dagar %"] = (c / c.shift(5) - 1) * 100
    f["Utveckling 20 dagar %"] = (c / c.shift(20) - 1) * 100
    f["Utveckling 60 dagar %"] = (c / c.shift(60) - 1) * 100
    f["Under årshögsta %"] = (1 - c / h.rolling(250, min_periods=120).max()) * 100
    f["Volym mot snitt"] = d["vol_x"]
    f["Rörlighet (ATR %)"] = d["atr"] / c * 100
    f["Grön candle"] = (c > o).astype(float)
    f["Stängning i dagens spann (0=botten,1=topp)"] = ((c - l) / (h - l).replace(0, np.nan))
    f["Index utveckling 20 dagar %"] = (ic / ic.shift(20) - 1) * 100
    f["Index mot 200-dagars %"] = (ic / ic.rolling(200).mean() - 1) * 100
    f["Aktie starkare än index 60 dagar %"] = f["Utveckling 60 dagar %"] - (ic / ic.shift(60) - 1) * 100
    return f


KATEG = ["MACD-histogram stiger", "MACD över nollan", "50-dagars över 200-dagars", "Fas 2 (kurs>20>50)", "Grön candle"]


def E(lista):
    return round(float(np.mean(lista)), 3) if len(lista) else None


def main():
    idx = M.ladda(sk.INDEX_TICKER)
    idx_close, idx_trend = idx["Close"], M.trend(idx["Close"])
    rader = []
    for i, (namn, ticker) in enumerate(sk.BOLAG.items()):
        df = M.ladda(ticker)
        if df is None:
            continue
        d = M.forbered(df)
        feat = egenskaper(d, idx_close, idx_trend)
        for m in M.METODER:
            for t in M.simulera(namn, m, d, idx_close, idx_trend, fro=i):
                r = feat.loc[pd.Timestamp(t["datum"])].to_dict()
                r.update({"metod": m, "R": t["R"], "datum": t["datum"], "ut_datum": t["ut_datum"], "bolag": namn,
                          "Index-läge": t["index_lage"], "Aktiens trend": t["aktie_trend"],
                          "Sektor": M.SEKTOR.get(namn, "Övrigt"), "R/R mot motstånd": t.get("rr_mot"),
                          "orsak": t["orsak"]})
                rader.append(r)
    df = pd.DataFrame(rader)
    df["test"] = df["datum"] >= TEST_FRAN
    start, slut = df["datum"].min(), df["datum"].max()
    ar_tot = (pd.Timestamp(slut) - pd.Timestamp(start)).days / 365.25
    ar_tran = (pd.Timestamp(TEST_FRAN) - pd.Timestamp(start)).days / 365.25
    ar_test = ar_tot - ar_tran
    slump = df[df.metod == "slump"].copy()
    print(f"{len(df)} trades, varav {len(slump)} slumpköp, {start} – {slut}")

    # ---- 1. faktor för faktor (slumpköp = renaste svaret på "vad gör en trade bra")
    faktorer = []
    kolumner = [c for c in slump.columns if c not in ("metod", "R", "datum", "ut_datum", "bolag", "test", "orsak")]
    tran, test = slump[~slump.test], slump[slump.test]
    for k in kolumner:
        x = slump[k]
        grupper = []
        if not pd.api.types.is_numeric_dtype(x) or k in KATEG:
            for v in sorted(x.dropna().unique(), key=str):
                a, b = tran[tran[k] == v].R, test[test[k] == v].R
                if len(a) >= 50:
                    grupper.append({"grupp": str(v) if k not in KATEG else ("ja" if v == 1 else "nej"),
                                    "n": int(len(a)), "E_tran": E(a), "n_test": int(len(b)), "E_test": E(b),
                                    "traff": round(float((a > 0).mean() * 100), 1)})
        else:
            try:
                kv = tran[k].quantile([0, .2, .4, .6, .8, 1]).to_numpy().copy()
            except Exception:
                continue
            kv[0], kv[-1] = -np.inf, np.inf
            namn5 = ["lägst", "låg", "mitten", "hög", "högst"]
            for j in range(5):
                a = tran[(tran[k] > kv[j]) & (tran[k] <= kv[j + 1])].R
                b = test[(test[k] > kv[j]) & (test[k] <= kv[j + 1])].R
                lo = "" if j == 0 else f"{kv[j]:.1f}"
                hi = "" if j == 4 else f"{kv[j + 1]:.1f}"
                grupper.append({"grupp": f"{namn5[j]} ({lo}–{hi})", "n": int(len(a)), "E_tran": E(a),
                                "n_test": int(len(b)), "E_test": E(b), "traff": round(float((a > 0).mean() * 100), 1) if len(a) else None})
        g = [x for x in grupper if x["E_tran"] is not None and x["E_test"] is not None and x["n_test"] >= 30]
        if len(g) < 2:
            continue
        bast, samst = max(g, key=lambda x: x["E_tran"]), min(g, key=lambda x: x["E_tran"])
        spann_tran = bast["E_tran"] - samst["E_tran"]
        spann_test = bast["E_test"] - samst["E_test"]
        faktorer.append({"faktor": k, "grupper": grupper, "spann_tran": round(spann_tran, 3),
                         "spann_test": round(spann_test, 3), "haller": bool(spann_test > 0.05),
                         "bast": bast["grupp"], "samst": samst["grupp"]})
    faktorer.sort(key=lambda x: -(min(x["spann_tran"], x["spann_test"]) if x["haller"] else -1))

    # ---- 2. facit i hand: stora vinnare mot utstoppade
    vinn, forl = slump[slump.R >= 1.5], slump[slump.orsak == "stop"]
    facit = []
    for k in kolumner:
        if not pd.api.types.is_numeric_dtype(slump[k]):
            continue
        a, b, alla = vinn[k].dropna(), forl[k].dropna(), slump[k].dropna()
        sd = alla.std()
        if not sd or not np.isfinite(sd):
            continue
        facit.append({"faktor": k, "vinnare": round(float(a.mean()), 2), "forlorare": round(float(b.mean()), 2),
                      "skillnad_i_sd": round(float((a.mean() - b.mean()) / sd), 2)})
    facit.sort(key=lambda x: -abs(x["skillnad_i_sd"]))

    # ---- 3. kombinationer av de bästa villkoren (valda på träningsåren, kontrollerade på teståren)
    villkor = []
    for fk in faktorer[:8]:
        if not fk["haller"]:
            continue
        k = fk["faktor"]
        bast = max([g for g in fk["grupper"] if g["E_tran"] is not None], key=lambda g: g["E_tran"])
        if not pd.api.types.is_numeric_dtype(slump[k]) or k in KATEG:
            v = bast["grupp"]
            val = (1.0 if v == "ja" else 0.0) if k in KATEG else v
            villkor.append((f"{k} = {v}", lambda s, k=k, val=val: s[k] == val))
        else:
            kv = tran[k].quantile([0, .2, .4, .6, .8, 1]).to_numpy().copy()
            j = [g["grupp"] for g in fk["grupper"]].index(bast["grupp"])
            # ta bästa gruppen plus grannen åt samma håll för att inte bli för smalt
            if j >= 3:
                gr = kv[3]
                villkor.append((f"{k} över {gr:.1f}", lambda s, k=k, gr=gr: s[k] > gr))
            elif j <= 1:
                gr = kv[2]
                villkor.append((f"{k} under {gr:.1f}", lambda s, k=k, gr=gr: s[k] <= gr))
            else:
                a_, b_ = kv[1], kv[4]
                villkor.append((f"{k} mellan {a_:.1f} och {b_:.1f}", lambda s, k=k, a_=a_, b_=b_: (s[k] > a_) & (s[k] <= b_)))
    kombos = []
    for n in (1, 2, 3):
        for combo in itertools.combinations(villkor, n):
            mask = np.ones(len(df), dtype=bool)
            for _, fn in combo:
                mask &= fn(df).fillna(False).to_numpy()
            for urval, sub in (("slumpköp", df[mask & (df.metod == "slump").to_numpy()]),
                               ("alla metoders signaler", df[mask & (df.metod != "slump").to_numpy()])):
                a, b = sub[~sub.test], sub[sub.test]
                if len(a) < 80 or len(b) < 30:
                    continue
                kombos.append({"villkor": [c[0] for c in combo], "urval": urval, "n_tran": int(len(a)), "E_tran": E(a.R),
                               "n_test": int(len(b)), "E_test": E(b.R), "per_ar_test": round(len(b) / ar_test, 1),
                               "traff_test": round(float((b.R > 0).mean() * 100), 1)})
    kombos.sort(key=lambda x: -x["E_tran"])
    topp = kombos[:12]

    # handelsbar version av de tre bästa (max 2 öppna, max 2 nya per vecka) på teståren
    for x in topp[:6]:
        mask = np.ones(len(df), dtype=bool)
        for namn in x["villkor"]:
            fn = dict(villkor)[namn]
            mask &= fn(df).fillna(False).to_numpy()
        urv = df[mask & ((df.metod == "slump") if x["urval"] == "slumpköp" else (df.metod != "slump")).to_numpy() & df.test.to_numpy()]
        tr = sorted(urv[["datum", "ut_datum", "bolag", "R", "R/R mot motstånd"]].rename(columns={"R/R mot motstånd": "rr_mot"}).to_dict("records"),
                    key=lambda t: t["datum"])
        for t in tr:
            t["rr_mot"] = None if pd.isna(t["rr_mot"]) else t["rr_mot"]
        val = M.portfolj(tr, 2, 2)
        R_ar = sum(t["R"] for t in val) / ar_test
        x["handelbar_test"] = {"trades_per_ar": round(len(val) / ar_test, 1), "E": E([t["R"] for t in val]),
                               "R_per_ar": round(R_ar, 1), "procent_per_ar_2pct": round(R_ar * 2, 1)}

    res = {"period": f"{start} – {slut}", "test_fran": TEST_FRAN, "antal": int(len(df)), "antal_slump": int(len(slump)),
           "slump_E_tran": E(tran.R), "slump_E_test": E(test.R),
           "faktorer": faktorer, "facit": facit, "kombinationer": topp,
           "krav_40": "40 % per år med 2 % risk = 20R per år, t.ex. 40 trades à +0,5R eller 80 trades à +0,25R."}
    Path("results").mkdir(exist_ok=True)
    (Path("results") / "faktorer.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("Bästa faktorer:")
    for f in faktorer[:10]:
        print(f"  {f['faktor']:45} tränings-spann {f['spann_tran']:+.2f}  test {f['spann_test']:+.2f}  bäst {f['bast']}")
    print("Facit:")
    for f in facit[:8]:
        print(f"  {f['faktor']:45} vinnare {f['vinnare']}  förlorare {f['forlorare']}  ({f['skillnad_i_sd']:+.2f} sd)")
    print("Kombinationer:")
    for x in topp[:8]:
        print(f"  {x['urval']:22} {' + '.join(x['villkor'])}  träning {x['E_tran']:+.2f} ({x['n_tran']})  test {x['E_test']:+.2f} ({x['n_test']}) {x.get('handelbar_test','')}")


if __name__ == "__main__":
    main()
