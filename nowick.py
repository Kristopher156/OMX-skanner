#!/usr/bin/env python3
"""
"No wick"-strategin, testad på dagsgraf (93 svenska bolag, 10 år).

Lång: en grön candle utan nedre wick (öppning = lägsta). Öppningskursen blir en nivå.
När kursen senare kommer tillbaka ner och testar nivån köper man där.
Kort: en röd candle utan övre wick (öppning = högsta). När kursen kommer tillbaka upp
till nivån blankar man.

Jämförelse: exakt samma upplägg men med vanliga candles som HAR wick. Om "no wick" är
speciell ska den vara tydligt bättre än jämförelsen.
Resultat: results/nowick.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import metoder as M
import omx30_skanner as sk

KOSTNAD = 0.25 / 100
TOL = 0.001          # wick högst 0,1 % av kursen räknas som "ingen wick"
VANTA = 10           # nivån måste testas inom så många dagar
MAX_DAGAR = 10
TEST_FRAN = "2022-01-01"


def trades(d, sida, utan_wick, stop_atr, mal, idx_lage):
    O, H, L, C = (d[k].to_numpy(float) for k in ["Open", "High", "Low", "Close"])
    A = d["atr"].to_numpy(float)
    n = len(C)
    ut = []
    fri = 0
    for t in range(1, n - 2):
        if t < fri or not np.isfinite(A[t]):
            continue
        if sida == "lang":
            if not C[t] > O[t]:
                continue
            wick = (O[t] - L[t]) / O[t]
        else:
            if not C[t] < O[t]:
                continue
            wick = (H[t] - O[t]) / O[t]
        if utan_wick and wick > TOL:
            continue
        if not utan_wick and wick <= TOL:
            continue
        niva = O[t]
        risk = stop_atr * A[t]
        # vänta på test av nivån
        e_dag = None
        for j in range(t + 1, min(n, t + 1 + VANTA)):
            if (sida == "lang" and L[j] <= niva) or (sida == "kort" and H[j] >= niva):
                e_dag = j
                break
        if e_dag is None:
            continue
        j = e_dag
        if sida == "lang":
            e = min(O[j], niva)
            stop, malpris = niva - risk, (H[t] if mal == "topp" else niva + mal * risk)
        else:
            e = max(O[j], niva)
            stop, malpris = niva + risk, (L[t] if mal == "topp" else niva - mal * risk)
        r = None
        orsak = None
        # inköpsdagen: bara stop räknas (försiktigt antagande)
        if (sida == "lang" and L[j] <= stop) or (sida == "kort" and H[j] >= stop):
            utpris, orsak = stop, "stop"
        else:
            utpris = None
            for k in range(j + 1, min(n, j + 1 + MAX_DAGAR)):
                if sida == "lang":
                    if L[k] <= stop:
                        utpris, orsak = min(O[k], stop), "stop"; break
                    if H[k] >= malpris:
                        utpris, orsak = max(O[k], malpris), "mål"; break
                else:
                    if H[k] >= stop:
                        utpris, orsak = max(O[k], stop), "stop"; break
                    if L[k] <= malpris:
                        utpris, orsak = min(O[k], malpris), "mål"; break
            if utpris is None:
                k = min(n - 1, j + MAX_DAGAR)
                utpris, orsak = C[k], "tid"
            j = k
        rp = (utpris - e) if sida == "lang" else (e - utpris)
        R = rp / risk - (e + utpris) * KOSTNAD / risk
        ut.append({"datum": str(d.index[t].date()), "R": float(R), "orsak": orsak,
                   "lage": idx_lage.asof(d.index[t]), "test": str(d.index[t].date()) >= TEST_FRAN})
        fri = j + 1
    return ut


def stat(l, ar):
    r = np.array([x["R"] for x in l])
    if not len(r):
        return {"n": 0}
    return {"n": int(len(r)), "per_ar": round(len(r) / ar, 1), "E": round(float(r.mean()), 3),
            "traff": round(float((r > 0).mean() * 100), 1),
            "osakerhet": round(float(1.96 * r.std(ddof=1) / np.sqrt(len(r))), 3)}


def main():
    idx = M.ladda(sk.INDEX_TICKER)
    idx_lage = M.trend(idx["Close"])
    data = {}
    for namn, t in sk.BOLAG.items():
        df = M.ladda(t)
        if df is not None:
            data[namn] = M.forbered(df)
    ar = 10.0
    res = []
    for sida in ("lang", "kort"):
        for stop_atr in (0.5, 1.0):
            for mal in (1.0, 2.0, "topp"):
                for utan in (True, False):
                    alla = []
                    for d in data.values():
                        alla += trades(d, sida, utan, stop_atr, mal, idx_lage)
                    rad = {"sida": sida, "stop_atr": stop_atr, "mal": mal, "utan_wick": utan,
                           "alla": stat(alla, ar), "test": stat([x for x in alla if x["test"]], 4.9),
                           "lage": {l: stat([x for x in alla if x["lage"] == l], ar) for l in ("upp", "sidled", "ned")}}
                    res.append(rad)
                    a = rad["alla"]
                    print(f"{sida:5} stop {stop_atr} ATR mål {mal!s:5} {'UTAN wick' if utan else 'med wick '}  "
                          f"n={a.get('n')} E={a.get('E')} ±{a.get('osakerhet')} träff {a.get('traff')} test E={rad['test'].get('E')}  "
                          f"upp {rad['lage']['upp'].get('E')} sid {rad['lage']['sidled'].get('E')} ned {rad['lage']['ned'].get('E')}")
    Path("results").mkdir(exist_ok=True)
    Path("results/nowick.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
