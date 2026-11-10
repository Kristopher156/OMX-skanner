#!/usr/bin/env python3
"""
Indexfilter: ta bara trades när S&P 500 (eller OMXS30) ligger över sitt 200-dagars EMA.
Läser results/metoder_trades.json (skapas av metoder.py i samma körning) och jämför
med och utan filter. Resultat: results/indexfilter.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import metoder as M

INDEX = {"S&P 500": "^GSPC", "OMXS30": "^OMX"}


def over_ema(ticker):
    df = yf.download(ticker, period="12y", interval="1d", progress=False, auto_adjust=True)
    c = df["Close"]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    c = c.dropna()
    c.index = pd.to_datetime(c.index).tz_localize(None) if c.index.tz is not None else pd.to_datetime(c.index)
    return c > c.ewm(span=200, adjust=False, min_periods=200).mean()


def sammanfatta(namn, lista, ar):
    r = np.array([x["R"] for x in lista])
    te = [x["R"] for x in lista if x["test"]]
    ut = {"namn": namn, "n": int(len(r))}
    if len(r):
        ut.update({"E": round(float(r.mean()), 3), "traff": round(float((r > 0).mean() * 100), 1),
                   "E_test": round(float(np.mean(te)), 3) if te else None})
    return ut


def main():
    trades = json.loads(Path("results/metoder_trades.json").read_text(encoding="utf-8"))
    trades = sorted([x for x in trades if x["metod"] != "slump"], key=lambda x: x["datum"])
    filt = {k: over_ema(t) for k, t in INDEX.items()}
    for x in trades:
        dt = pd.Timestamp(x["datum"])
        for k, s in filt.items():
            v = s.asof(dt)
            x[k] = bool(v) if v == v else None
    forsta, sista = trades[0]["datum"], trades[-1]["ut_datum"]
    ar = (pd.Timestamp(sista) - pd.Timestamp(forsta)).days / 365.25

    def krav(l):
        l = [x for x in l if x["metod"] in M.LAGE_METODER.get(x["index_lage"], [])]
        return [x for x in l if (x["rr_mot"] is None or x["rr_mot"] >= 2) and (x["vol_x"] is None or x["vol_x"] >= 1)]

    varianter = {
        "Inget indexfilter": lambda x: True,
        "Bara när S&P 500 > EMA 200": lambda x: x["S&P 500"] is True,
        "Bara när OMXS30 > EMA 200": lambda x: x["OMXS30"] is True,
        "Bara när båda > EMA 200": lambda x: x["S&P 500"] is True and x["OMXS30"] is True,
        "Bara när S&P 500 UNDER EMA 200 (kontroll)": lambda x: x["S&P 500"] is False,
    }
    res = {"period": f"{forsta} – {sista}", "per_metod": {}, "kravna": [], "handelbar": []}
    for m in sorted(set(x["metod"] for x in trades)):
        l = [x for x in trades if x["metod"] == m]
        res["per_metod"][m] = [sammanfatta(v, [x for x in l if f(x)], ar) for v, f in varianter.items()]
    for v, f in varianter.items():
        alla = [x for x in trades if f(x)]
        res["kravna"].append(sammanfatta(v + " · alla signaler", alla, ar))
        k = krav(alla)
        res["kravna"].append(sammanfatta(v + " · kräsna", k, ar))
        for mo in (2, 4):
            val = M.portfolj(k, mo, 2)
            per = {}
            for x in val:
                per[x["ut_datum"][:4]] = per.get(x["ut_datum"][:4], 0) + x["R"] * 2
            res["handelbar"].append({"namn": f"{v} · max {mo} samtidigt", "trades_per_ar": round(len(val) / ar, 1),
                                     "E": round(float(np.mean([x["R"] for x in val])), 3) if val else None,
                                     "procent_per_ar": round(sum(x["R"] for x in val) / ar * 2, 1),
                                     "bast": round(max(per.values()), 1) if per else None,
                                     "samst": round(min(per.values()), 1) if per else None,
                                     "per_ar": {y: round(v_, 1) for y, v_ in sorted(per.items())}})
    Path("results/indexfilter.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for x in res["handelbar"]:
        print(x["namn"], x["trades_per_ar"], x["E"], x["procent_per_ar"], x["bast"], x["samst"])


if __name__ == "__main__":
    main()
