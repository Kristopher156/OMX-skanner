#!/usr/bin/env python3
"""
Svep: plockar ut historiska signaler (100 per metod + slumpköp som kontroll) och sparar
grafen fram till signaldagen + vad som hände efteråt, så att du kan svepa ja/nej blint.

Resultat: results/svep.json (används av svep-sidan).
"""
import json
import random

import numpy as np
import pandas as pd

import metoder as M
import omx30_skanner as sk

PER_METOD = 100
FORE = 120     # dagar före signalen som visas
EFTER = 60     # max dagar efter som visas när svaret avslöjas
FRO = 2026


def r2(x, dec=2):
    return [None if not np.isfinite(v) else round(float(v), dec) for v in x]


def main():
    idx = M.ladda(sk.INDEX_TICKER)
    idx_close = idx["Close"]
    idx_trend = M.trend(idx_close)
    data, trades = {}, []
    for i, (namn, ticker) in enumerate(sk.BOLAG.items()):
        df = M.ladda(ticker)
        if df is None:
            continue
        d = M.forbered(df)
        data[namn] = d
        for m in M.METODER:
            trades += M.simulera(namn, m, d, idx_close, idx_trend, fro=i)

    rng = random.Random(FRO)
    fall = []
    for m in M.METODER:
        t = [x for x in trades if x["metod"] == m]
        rng.shuffle(t)
        valda = []
        for x in t:
            d = data[x["bolag"]]
            pos = d.index.get_loc(pd.Timestamp(x["datum"]))
            if pos >= FORE:
                valda.append((x, pos))
            if len(valda) >= PER_METOD:
                break
        for x, pos in valda:
            d = data[x["bolag"]]
            a, b = pos - FORE + 1, min(len(d), pos + 1 + EFTER)
            ut_pos = d.index.get_loc(pd.Timestamp(x["ut_datum"]))
            bas = float(d["Close"].iloc[pos])
            n = lambda s: r2(s.iloc[a:b].to_numpy() / bas * 100, 1)
            ic = idx_close.reindex(d.index).ffill()
            ib = float(ic.iloc[pos])
            mal_R = M.P(x["metod"]).get("mal_R")
            risk = x["kop"] - x["stop"]
            fall.append({
                "m": x["metod"], "b": x["bolag"], "dt": x["datum"], "ut": x["ut_datum"], "R": x["R"],
                "orsak": x["orsak"], "dagar": x["dagar"], "lage": x["index_lage"], "trend": x["aktie_trend"],
                "rr": x.get("rr_mot"), "risk": x["risk_pct"],
                "stop": round(x["stop"] / bas * 100, 2), "kop": round(x["kop"] / bas * 100, 2),
                "salj": round(x["salj"] / bas * 100, 2),
                "mal": round((x["kop"] + mal_R * risk) / bas * 100, 2) if mal_R else None,
                "sig": FORE - 1, "utIdx": int(ut_pos - a),
                "o": n(d["Open"]), "h": n(d["High"]), "l": n(d["Low"]), "c": n(d["Close"]),
                "ma50": n(d["ma50"]), "ma200": n(d["ma200"]),
                "v": r2(d["vol_x"].iloc[a:b].to_numpy(), 1),
                "rsi": r2(d["rsi"].iloc[a:b].to_numpy(), 0),
                "ml": r2(d["macd"].iloc[a:b].to_numpy() / bas * 100, 2),
                "ms": r2(d["macd_sig"].iloc[a:b].to_numpy() / bas * 100, 2),
                "ix": r2(ic.iloc[a:b].to_numpy() / ib * 100, 1),
            })
    rng.shuffle(fall)
    for i, f in enumerate(fall):
        f["id"] = i
    M.Path("results").mkdir(exist_ok=True)
    (M.Path("results") / "svep.json").write_text(json.dumps({
        "metoder": {m: {"namn": M.METODER[m]["namn"], "kort": M.METODER[m]["kort"]} for m in M.METODER},
        "fall": fall,
    }, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{len(fall)} fall sparade.")


if __name__ == "__main__":
    main()
