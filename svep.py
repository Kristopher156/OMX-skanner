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
FORE = 260     # dagar före signalen (ett år, så att 1-årsvy och veckograf fungerar)
EFTER = 80     # dagar efter signalen som kan spelas upp
FRO = 2026


def r2(x, dec=2):
    return [None if not np.isfinite(v) else round(float(v), dec) for v in x]


EFTER_SASONG = 130
SASONG_AR = list(range(2016, 2026))


def bygg_fall(x, d, pos, a, b, bolag_nr, idx_close, trail=None, datum=False):
    ut_pos = d.index.get_loc(pd.Timestamp(x["ut_datum"]))
    bas = float(d["Close"].iloc[pos])
    n = lambda s: r2(s.iloc[a:b].to_numpy() / bas * 100, 1)
    ic = idx_close.reindex(d.index).ffill()
    ib = float(ic.iloc[pos])
    mal_R = M.P(x["metod"]).get("mal_R")
    risk = x["kop"] - x["stop"]
    wk = d.index[a:b].isocalendar()
    f = {
        "id": f"{bolag_nr[x['bolag']]:02d}{x['datum'].replace('-', '')}",
        "w": [int(y) * 100 + int(v) for y, v in zip(wk["year"], wk["week"])],
        "m": x["metod"], "b": x["bolag"], "dt": x["datum"], "ut": x["ut_datum"], "R": x["R"],
        "orsak": x["orsak"], "dagar": x["dagar"], "lage": x["index_lage"], "trend": x["aktie_trend"],
        "rr": x.get("rr_mot"), "risk": x["risk_pct"], "fas2": x.get("fas2"), "fas2plus": x.get("fas2plus"),
        "stop": round(x["stop"] / bas * 100, 2), "kop": round(x["kop"] / bas * 100, 2),
        "salj": round(x["salj"] / bas * 100, 2),
        "mal": round((x["kop"] + mal_R * risk) / bas * 100, 2) if mal_R else None,
        "sig": int(pos - a), "utIdx": int(ut_pos - a),
        "o": n(d["Open"]), "h": n(d["High"]), "l": n(d["Low"]), "c": n(d["Close"]),
        "ma50": n(d["ma50"]), "ma200": n(d["ma200"]),
        "v": r2(d["vol_x"].iloc[a:b].to_numpy(), 1),
        "rsi": r2(d["rsi"].iloc[a:b].to_numpy(), 0),
        "ml": r2(d["macd"].iloc[a:b].to_numpy() / bas * 100, 2),
        "ms": r2(d["macd_sig"].iloc[a:b].to_numpy() / bas * 100, 2),
        "ix": r2(ic.iloc[a:b].to_numpy() / ib * 100, 1),
    }
    if f["rr"] is not None:
        f["mot"] = round((x["kop"] + f["rr"] * risk) / bas * 100, 2)
    if trail:
        f["Rt"], f["dagarT"] = trail
    if datum:
        f["d"] = [int(t.strftime("%Y%m%d")) for t in d.index[a:b]]
    return f


def main():
    idx = M.ladda(sk.INDEX_TICKER)
    idx_close = idx["Close"]
    idx_trend = M.trend(idx_close)
    data, trades, trail = {}, [], {}
    for i, (namn, ticker) in enumerate(sk.BOLAG.items()):
        df = M.ladda(ticker)
        if df is None:
            continue
        d = M.forbered(df)
        data[namn] = d
        for m in M.METODER:
            trades += M.simulera(namn, m, d, idx_close, idx_trend, fro=i)
            if m != "slump":
                for t in M.simulera(namn, m, d, idx_close, idx_trend, fro=i, exit_typ="be_trail"):
                    trail[(namn, m, t["datum"])] = (t["R"], t["dagar"])

    rng = random.Random(FRO)
    fall = []
    bolag_nr = {n: i for i, n in enumerate(sk.BOLAG)}
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
            fall.append(bygg_fall(x, d, pos, a, b, bolag_nr, idx_close, trail.get((x["bolag"], m, x["datum"]))))
    rng.shuffle(fall)
    M.Path("results").mkdir(exist_ok=True)
    metoder = {m: {"namn": M.METODER[m]["namn"], "kort": M.METODER[m]["kort"]} for m in M.METODER}

    def skriv(namn, obj):
        (M.Path("results") / namn).write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    for m in M.METODER:
        skriv(f"svep_{m}.json", {"metoder": metoder, "fall": [f for f in fall if f["m"] == m]})
    blandat = []
    for m in M.METODER:
        blandat += [f for f in fall if f["m"] == m][:25]
    rng.shuffle(blandat)
    skriv("svep_blandat.json", {"metoder": metoder, "fall": blandat})
    print(f"{len(fall)} fall sparade, {len(blandat)} i blandat.")

    # ---- Säsonger: alla kräsna signaler under ett kalenderår, i datumordning
    def krasen(x):
        return (x["metod"] in M.LAGE_METODER.get(x["index_lage"], [])
                and (x["rr_mot"] is None or x["rr_mot"] >= M.K("min_rr"))
                and (x["vol_x"] is None or x["vol_x"] >= M.K("min_volym")))

    riktiga = sorted([x for x in trades if x["metod"] != "slump"], key=lambda x: x["datum"])
    oversikt = []
    for ar in SASONG_AR:
        kand = [x for x in riktiga if x["datum"][:4] == str(ar) and krasen(x)]
        sfall = []
        for x in kand:
            d = data[x["bolag"]]
            pos = d.index.get_loc(pd.Timestamp(x["datum"]))
            if pos < FORE:
                continue
            a, b = pos - FORE + 1, min(len(d), pos + 1 + EFTER_SASONG)
            sfall.append(bygg_fall(x, d, pos, a, b, bolag_nr, idx_close, trail.get((x["bolag"], x["metod"], x["datum"])), datum=True))
        ic = idx_close[(idx_close.index >= f"{ar}-01-01") & (idx_close.index <= f"{ar}-12-31")]
        it = idx_trend.reindex(ic.index)

        def regel(lista, Rkey):
            val = M.portfolj([dict(y, R=y[Rkey]) for y in lista], 2, 2)
            return round(sum(v["R"] for v in val) * 2, 1), len(val)

        tr_lista = [dict(x, Rt=trail.get((x["bolag"], x["metod"], x["datum"]), (x["R"], 0))[0]) for x in kand]
        r_std, n_std = regel(tr_lista, "R")
        r_tr, n_tr = regel(tr_lista, "Rt")
        index_ar = round(float(ic.iloc[-1] / ic.iloc[0] - 1) * 100, 1)
        skriv(f"sasong_{ar}.json", {
            "ar": ar, "metoder": metoder, "fall": sfall,
            "index": {"d": [int(t.strftime("%Y%m%d")) for t in ic.index], "c": r2(ic.to_numpy(), 1),
                      "lage": [None if v is None or v != v else v for v in it.tolist()]},
            "regel": {"standard": r_std, "standard_n": n_std, "slapande": r_tr, "slapande_n": n_tr},
            "index_procent": index_ar,
        })
        oversikt.append({"ar": ar, "signaler": len(sfall), "index": index_ar, "regel": r_std, "regel_slapande": r_tr})
        print(f"Säsong {ar}: {len(sfall)} kräsna signaler, index {index_ar} %, regeln {r_std} % / släpande {r_tr} %")
    skriv("sasonger.json", oversikt)


if __name__ == "__main__":
    main()
