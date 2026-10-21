#!/usr/bin/env python3
"""
Liknande lägen i historiken: vad hände när en OMXS30-aktie var lika nedtryckt som nu?

Ett "läge" = RSI(14) som lägst under RSI_MAX de senaste 5 dagarna OCH kursen minst
FALL_MIN % under sin högsta kurs de senaste 20 dagarna. Första dagen räknas, sedan
väntar vi 20 handelsdagar innan samma aktie kan ge ett nytt läge.

För varje läge mäts, från stängningskursen den dagen:
  * avkastning efter 5, 10, 20 och 40 handelsdagar
  * om kursen nådde MAL_UPP % upp inom 40 dagar – och om den först föll STOP_NER %
Resultatet sparas i results/analoger.json.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import backtest as bt
import omx30_skanner as sk

AR = 10
RSI_MAX = 23
FALL_MIN = 15
MAL_UPP = 16.0     # 173 -> ~200
STOP_NER = 6.4     # 173 -> ~162
BANKER = {"Nordea", "SEB A", "Handelsbanken A", "Swedbank A"}


def ladda(ticker):
    df = yf.download(ticker, period=f"{AR}y", interval="1d", progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None
    ut = pd.DataFrame({k: sk.kolumn(df, k) for k in ["Open", "High", "Low", "Close"]}).dropna()
    return ut if len(ut) > 300 else None


def lagen(namn, df):
    c, h, l = df["Close"], df["High"], df["Low"]
    r = sk.rsi(c)
    fall = (c / h.rolling(20).max() - 1) * 100
    villkor = ((r.rolling(5).min() <= RSI_MAX) & (fall <= -FALL_MIN)).to_numpy()
    cn, hn, ln = c.to_numpy(), h.to_numpy(), l.to_numpy()
    ut, nasta = [], 0
    for t in np.flatnonzero(villkor):
        if t < nasta or t + 40 >= len(cn):
            continue
        nasta = t + 20
        p = cn[t]
        fram_h, fram_l = hn[t + 1:t + 41], ln[t + 1:t + 41]
        upp = np.flatnonzero(fram_h >= p * (1 + MAL_UPP / 100))
        ner = np.flatnonzero(fram_l <= p * (1 - STOP_NER / 100))
        forst_upp = upp[0] if upp.size else None
        forst_ner = ner[0] if ner.size else None
        ut.append({
            "bolag": namn, "datum": str(df.index[t].date()), "kurs": round(float(p), 2),
            "r5": (cn[t + 5] / p - 1) * 100, "r10": (cn[t + 10] / p - 1) * 100,
            "r20": (cn[t + 20] / p - 1) * 100, "r40": (cn[t + 40] / p - 1) * 100,
            "max_upp": (fram_h.max() / p - 1) * 100, "max_ner": (fram_l.min() / p - 1) * 100,
            "nadde_mal": forst_upp is not None,
            "stop_forst": forst_ner is not None and (forst_upp is None or forst_ner < forst_upp),
            "mal_forst": forst_upp is not None and (forst_ner is None or forst_upp < forst_ner),
            "dagar_till_mal": int(forst_upp + 1) if forst_upp is not None else None,
        })
    return ut


def sammanfatta(lista):
    if not lista:
        return None
    d = pd.DataFrame(lista)
    return {
        "antal": len(d),
        "median_5d": d.r5.median(), "median_10d": d.r10.median(), "median_20d": d.r20.median(), "median_40d": d.r40.median(),
        "andel_upp_20d": (d.r20 > 0).mean() * 100,
        "median_max_upp_40d": d.max_upp.median(), "median_max_ner_40d": d.max_ner.median(),
        "andel_nadde_mal": d.nadde_mal.mean() * 100,
        "andel_mal_fore_stop": d.mal_forst.mean() * 100,
        "andel_stop_fore_mal": d.stop_forst.mean() * 100,
        "median_dagar_till_mal": d.dagar_till_mal.dropna().median() if d.nadde_mal.any() else None,
    }


def main():
    alla = []
    for namn, ticker in sk.BOLAG.items():
        try:
            df = ladda(ticker)
            if df is not None:
                alla += lagen(namn, df)
        except Exception as e:
            print(f"  Fel för {namn}: {e}")
    banker = [x for x in alla if x["bolag"] in BANKER]
    nordea = [x for x in alla if x["bolag"] == "Nordea"]
    res = {
        "villkor": {"rsi_max": RSI_MAX, "fall_min": FALL_MIN, "mal_upp": MAL_UPP, "stop_ner": STOP_NER, "ar": AR},
        "alla": sammanfatta(alla), "banker": sammanfatta(banker), "nordea": sammanfatta(nordea),
        "nordea_lagen": nordea, "bank_lagen": banker,
    }
    bt.spara_json("analoger.json", res)
    print(f"{len(alla)} lägen, varav {len(banker)} banker och {len(nordea)} Nordea.")


if __name__ == "__main__":
    main()
