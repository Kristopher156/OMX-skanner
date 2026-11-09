#!/usr/bin/env python3
"""
Storbolag och dippköp: vad hade hänt om man sparat i "säkra" jättar och fyllt på vid dippar?

Tre sätt att lägga samma pengar (1 000 per månad):
  * Månadssparande: köp varje månad oavsett pris.
  * Bara dippar: spara pengarna och köp bara när aktien ligger minst X % under årshögsta.
  * Månadssparande + extra vid dipp: hälften varje månad, hälften sparas och läggs i när det är dipp.

Två perioder:
  * I dag: Meta, Alphabet, Microsoft, Apple, Amazon, Nvidia sedan 2014 (de vi VET blev vinnare).
  * År 2000: dåtidens "säkra" jättar – Microsoft, Cisco, Intel, General Electric, IBM – 2000–2012.
Resultat: results/bigtech.json. Priser i dollar inklusive utdelningar.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

PERIODER = {
    "idag": {"namn": "Dagens jättar 2014–2026", "start": "2013-01-01", "fran": "2014-01-01", "slut": None,
             "bolag": {"Meta": "META", "Alphabet": "GOOGL", "Microsoft": "MSFT", "Apple": "AAPL", "Amazon": "AMZN", "Nvidia": "NVDA"}},
    "ar2000": {"namn": "Jättarna år 2000, 2000–2012", "start": "1999-01-01", "fran": "2000-01-01", "slut": "2012-12-31",
               "bolag": {"Microsoft": "MSFT", "Cisco": "CSCO", "Intel": "INTC", "General Electric": "GE", "IBM": "IBM"}},
}
INDEX = {"S&P 500 (indexfond)": "SPY"}
MANAD = 1000.0
DIPPAR = [10, 20, 30]


def manadsdata(ticker, start, slut):
    df = yf.download(ticker, start=start, end=slut, interval="1d", progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    c = df["Close"]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    c = c.dropna()
    hog = c.rolling(252, min_periods=120).max()
    m = pd.DataFrame({"pris": c, "fall": (1 - c / hog) * 100}).resample("MS").first().dropna()
    return m


def kor(m, fran, regel, dipp=None):
    m = m[m.index >= fran]
    aktier, kassa, insatt, varde_hist, vantade = 0.0, 0.0, 0.0, [], 0
    for _, r in m.iterrows():
        insatt += MANAD
        if regel == "manad":
            aktier += MANAD / r.pris
        elif regel == "dipp":
            kassa += MANAD
            if r.fall >= dipp:
                aktier += kassa / r.pris
                kassa = 0.0
            else:
                vantade += 1
        elif regel == "mix":
            aktier += MANAD / 2 / r.pris
            kassa += MANAD / 2
            if r.fall >= dipp:
                aktier += kassa / r.pris
                kassa = 0.0
        varde_hist.append(aktier * r.pris + kassa)
    v = np.array(varde_hist)
    topp = np.maximum.accumulate(v)
    ar = len(m) / 12
    slut = float(v[-1])
    # årlig avkastning på pengarna (pengaviktad), enkel bisektion
    flows = [-MANAD] * len(m)

    def npv(r):
        return sum(f / (1 + r) ** (i / 12) for i, f in enumerate(flows)) + slut / (1 + r) ** (len(m) / 12)

    lo, hi = -0.9, 2.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if npv(mid) > 0 else (lo, mid)
    return {
        "insatt": round(insatt), "varde": round(slut), "ganger": round(slut / insatt, 2),
        "arlig": round(lo * 100, 1), "kassa_kvar": round(kassa),
        "storsta_fall": round(float(((topp - v) / topp).max() * 100), 1),
        "vantade_manader": vantade, "ar": round(ar, 1),
    }


def main():
    ut = {}
    for pk, per in PERIODER.items():
        alla = dict(per["bolag"], **INDEX)
        res, korg = {}, {}
        for namn, t in alla.items():
            m = manadsdata(t, per["start"], per["slut"])
            if m is None:
                continue
            rad = {"månadssparande": kor(m, per["fran"], "manad")}
            for d in DIPPAR:
                rad[f"bara dippar {d} %"] = kor(m, per["fran"], "dipp", d)
                rad[f"månad + extra vid {d} %"] = kor(m, per["fran"], "mix", d)
            res[namn] = rad
            if t not in INDEX.values():
                korg[namn] = rad
        # korgen: lika mycket i varje bolag (summa av bolagen, skalat till 1 000 per månad)
        if korg:
            n = len(korg)
            sk = {}
            for regel in next(iter(korg.values())):
                ins = sum(r[regel]["insatt"] for r in korg.values()) / n
                var = sum(r[regel]["varde"] for r in korg.values()) / n
                sk[regel] = {"insatt": round(ins), "varde": round(var), "ganger": round(var / ins, 2),
                             "arlig_snitt_bolag": round(float(np.mean([r[regel]["arlig"] for r in korg.values()])), 1),
                             "storsta_fall_snitt": round(float(np.mean([r[regel]["storsta_fall"] for r in korg.values()])), 1)}
            res["Korg (lika i alla)"] = sk
        ut[pk] = {"namn": per["namn"], "resultat": res}
        print(per["namn"])
        for namn, rad in res.items():
            print(f"  {namn:22}", " | ".join(f"{k}: {v['ganger']}x" for k, v in rad.items()))
    Path("results").mkdir(exist_ok=True)
    (Path("results") / "bigtech.json").write_text(json.dumps(ut, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
