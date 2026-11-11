#!/usr/bin/env python3
"""
"No wick"-strategin som daytrading: 1-timmesgraf (ca 2 år) och 15-minutersgraf (ca 60 dagar).

Samma regler som nowick.py, men allt sker inom samma handelsdag:
nivån måste testas samma dag och positionen stängs senast dagens sista candle.
Två kostnadsnivåer: 0,25 % per affär (Avanza Mini för små belopp) och 0,05 % (bästa fall).
Jämförelse: samma upplägg med vanliga candles som HAR wick.
Resultat: results/nowick_intra.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import omx30_skanner as sk

TOL = 0.001


def hamta(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    ut = pd.DataFrame({k: sk.kolumn(df, k) for k in ["Open", "High", "Low", "Close"]}).dropna()
    ut.index = pd.to_datetime(ut.index)
    if ut.index.tz is not None:
        ut.index = ut.index.tz_convert("Europe/Stockholm").tz_localize(None)
    if len(ut) < 100:
        return None
    pc = ut["Close"].shift()
    tr = pd.concat([ut["High"] - ut["Low"], (ut["High"] - pc).abs(), (ut["Low"] - pc).abs()], axis=1).max(axis=1)
    ut["atr"] = tr.rolling(14).mean()
    return ut


def trades(d, sida, utan_wick, stop_atr, mal, kostnad):
    O, H, L, C, A = (d[k].to_numpy(float) for k in ["Open", "High", "Low", "Close", "atr"])
    dag = np.array([x.date() for x in d.index])
    n = len(C)
    ut, fri = [], 0
    for t in range(1, n - 2):
        if t < fri or not np.isfinite(A[t]) or A[t] <= 0:
            continue
        if sida == "lang":
            if not C[t] > O[t]:
                continue
            wick = (O[t] - L[t]) / O[t]
        else:
            if not C[t] < O[t]:
                continue
            wick = (H[t] - O[t]) / O[t]
        if utan_wick != (wick <= TOL):
            continue
        niva, risk = O[t], stop_atr * A[t]
        e_dag = None
        j = t + 1
        while j < n and dag[j] == dag[t]:
            if (sida == "lang" and L[j] <= niva) or (sida == "kort" and H[j] >= niva):
                e_dag = j
                break
            j += 1
        if e_dag is None:
            continue
        j = e_dag
        if sida == "lang":
            e = min(O[j], niva)
            stop, malpris = niva - risk, (H[t] if mal == "topp" else niva + mal * risk)
        else:
            e = max(O[j], niva)
            stop, malpris = niva + risk, (L[t] if mal == "topp" else niva - mal * risk)
        if (sida == "lang" and L[j] <= stop) or (sida == "kort" and H[j] >= stop):
            utpris, k = stop, j
        else:
            utpris, k = None, j + 1
            while k < n and dag[k] == dag[t]:
                if sida == "lang":
                    if L[k] <= stop:
                        utpris = min(O[k], stop); break
                    if H[k] >= malpris:
                        utpris = max(O[k], malpris); break
                else:
                    if H[k] >= stop:
                        utpris = max(O[k], stop); break
                    if L[k] <= malpris:
                        utpris = min(O[k], malpris); break
                k += 1
            if utpris is None:  # stäng vid dagens sista candle
                k -= 1
                utpris = C[k]
        rp = (utpris - e) if sida == "lang" else (e - utpris)
        ut.append(float(rp / risk - (e + utpris) * kostnad / risk))
        fri = k + 1
    return ut


def stat(r):
    r = np.array(r)
    if not len(r):
        return {"n": 0}
    return {"n": int(len(r)), "E": round(float(r.mean()), 3), "traff": round(float((r > 0).mean() * 100), 1),
            "osakerhet": round(float(1.96 * r.std(ddof=1) / np.sqrt(len(r))), 3)}


def main():
    res = []
    for interval, period in (("60m", "730d"), ("15m", "60d")):
        data = [d for d in (hamta(t, period, interval) for t in sk.BOLAG.values()) if d is not None]
        print(f"{interval}: {len(data)} bolag")
        for kostnad in (0.25 / 100, 0.05 / 100):
            for sida in ("lang", "kort"):
                for stop_atr in (0.5, 1.0):
                    for mal in (1.0, 2.0, "topp"):
                        for utan in (True, False):
                            alla = []
                            for d in data:
                                alla += trades(d, sida, utan, stop_atr, mal, kostnad)
                            s = stat(alla)
                            res.append({"graf": interval, "kostnad_pct": kostnad * 100, "sida": sida,
                                        "stop_atr": stop_atr, "mal": mal, "utan_wick": utan, **s})
                            print(f"{interval} kostn {kostnad*100:.2f}% {sida:5} stop {stop_atr} mål {mal!s:5} "
                                  f"{'UTAN wick' if utan else 'med wick '} n={s.get('n')} E={s.get('E')} "
                                  f"±{s.get('osakerhet')} träff {s.get('traff')}")
    Path("results").mkdir(exist_ok=True)
    Path("results/nowick_intra.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
