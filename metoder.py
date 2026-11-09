#!/usr/bin/env python3
"""
Metoder – testar dina swingtrading-metoder på 10 års historik och visar dagens signaler.

Sidan (site/metoder.html) visar:
  * Marknaden: är OMXS30 i upptrend, sidled eller nedtrend, och hur mycket styr indexet aktierna?
  * Sektorer: vilka branscher är starkare eller svagare än index just nu.
  * Dagens signaler, metod för metod, med stop, mål, antal aktier och varningar.
  * Backtest: hur varje metod har gått sedan 2016 – totalt, per marknadsläge och på nya år (test).
  * Alla bolag: trend, RSI, volym, candlestick, hur mycket de följer index, rapportdatum.

Finjustera en metod: ändra siffrorna under "p" i METODER nedan (eller be Claude göra det).
Varje siffra har en förklaring som också syns på sidan.

Viktigt: allt bygger bara på data som fanns den dagen (ingen titt in i framtiden).
Köp sker på nästa dags öppning. Courtage + spread dras av. Verkligheten blir ändå
nästan alltid lite sämre än historiken.
"""
import json
import math
import os
from datetime import date, datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import omx30_skanner as sk

AR = 10                      # så många års historik testas
TEST_FRAN = "2024-01-01"     # trades från detta datum = "test" (nya år som reglerna inte formats efter)
KOSTNAD_PROCENT = 0.25       # courtage + spread per köp och per sälj, i procent
MAX_RISK_PROCENT = 15        # signaler där stoppen ligger längre bort än så hoppas över
RAPPORT_VARNING_DAGAR = 15   # varna om rapport inom så många dagar
CACHE = Path("cache")

# ---------------------------------------------------------------------------
# Sektorer (för att se vilka branscher som går bra)
# ---------------------------------------------------------------------------
SEKTOR = {
    "Nordea": "Banker", "SEB A": "Banker", "Handelsbanken A": "Banker", "Swedbank A": "Banker",
    "ABB": "Industri", "Alfa Laval": "Industri", "Assa Abloy B": "Industri", "Atlas Copco A": "Industri",
    "Atlas Copco B": "Industri", "Epiroc A": "Industri", "Sandvik": "Industri", "SKF B": "Industri",
    "Volvo B": "Industri", "Addtech B": "Industri", "Lifco B": "Industri", "Nibe B": "Industri",
    "Saab B": "Försvar",
    "Ericsson B": "Teknik & telekom", "Tele2 B": "Teknik & telekom", "Telia": "Teknik & telekom",
    "Hexagon B": "Teknik & telekom",
    "AstraZeneca": "Hälsa", "Getinge B": "Hälsa",
    "H&M B": "Konsument", "Essity B": "Konsument", "Evolution": "Konsument",
    "Boliden": "Råvaror & skog", "SCA B": "Råvaror & skog",
    "Investor B": "Investmentbolag & finans", "EQT": "Investmentbolag & finans",
    "Skanska B": "Bygg",
}

# ---------------------------------------------------------------------------
# Metoderna. "p" = parametrar du kan finjustera: (värde, förklaring)
# ---------------------------------------------------------------------------
METODER = {
    "dipp": {
        "namn": "Dippköp: RSI + MACD",
        "kort": "Köper svaghet: aktien är översåld och MACD börjar vända upp.",
        "typ": "Motrörelse (köper när det fallit)",
        "bast": "Rekyl i upptrend eller sidledes marknad",
        "regler": [
            "RSI har varit under {rsi_max} någon gång de senaste {rsi_dagar} dagarna.",
            "MACD-histogrammet är under noll men har stigit {vander_dagar} dagar i rad (vändningen).",
            "Stop {stop_marginal} % under lägsta kursen de senaste {botten_dagar} dagarna.",
            "Sälj vid {mal_R} R vinst, vid stoppen eller efter {max_dagar} handelsdagar.",
        ],
        "p": {
            "rsi_max": (30, "RSI ska ha varit under detta (lägre = färre men mer nedtryckta lägen)"),
            "rsi_dagar": (5, "inom så många dagar"),
            "vander_dagar": (2, "antal dagar i rad som histogrammet ska ha stigit"),
            "botten_dagar": (20, "stoppen läggs under lägsta kursen inom så många dagar"),
            "stop_marginal": (1.0, "procent under bottnen"),
            "mal_R": (2.0, "vinstmål i R (2 = dubbla risken)"),
            "max_dagar": (20, "längsta tid i traden"),
        },
    },
    "macd_vecka": {
        "namn": "Långsiktig MACD (veckograf)",
        "kort": "MACD på veckodata – motsvarar att titta på 1–3-årsgrafen. Färre men längre trades.",
        "typ": "Trendföljande (följer en ny uppgång länge)",
        "bast": "När en längre nedgång vänder till uppgång",
        "regler": [
            "Varje vecka: MACD-linjen (12, 26, 9 på veckostängningar) korsar upp över signallinjen.",
            "Korsningen ska ske under nollinjen (alltså efter en nedgång): {under_noll} (1 = ja, 0 = nej).",
            "Stop {stop_marginal} % under lägsta kursen de senaste {stop_veckor} veckorna.",
            "Inget fast mål: sälj när vecko-MACD korsar ner igen, vid stoppen eller efter {max_dagar} handelsdagar.",
        ],
        "p": {
            "under_noll": (1, "1 = bara korsningar under nollan (efter nedgång), 0 = alla korsningar"),
            "stop_veckor": (8, "stoppen läggs under lägsta kursen så många veckor bakåt"),
            "stop_marginal": (1.0, "procent under bottnen"),
            "max_dagar": (250, "längsta tid i traden (250 ≈ ett år)"),
        },
    },
    "darvas": {
        "namn": "Darvas-box / utbrott i upptrend",
        "kort": "Köper styrka: aktie nära toppen har vilat i en smal låda och bryter upp på hög volym (expansion).",
        "typ": "Utbrott (köper när det börjar stiga)",
        "bast": "Upptrend, aktier som gör nya toppar",
        "regler": [
            "Aktien är i upptrend (kurs över 200-dagars och 50-dagars över 200-dagars): {krav_upptrend} (1 = krav).",
            "Kursen ligger högst {nara_topp} % under årets högsta.",
            "De senaste {box_dagar} dagarna har kursen rört sig i en låda som är högst {box_max} % hög.",
            "I dag stänger kursen över lådans tak på minst {vol_faktor} gånger normal volym.",
            "Stop {stop_marginal} % under lådans botten. Sälj vid {mal_R} R, stoppen eller efter {max_dagar} dagar.",
        ],
        "p": {
            "krav_upptrend": (1, "1 = bara aktier i upptrend"),
            "nara_topp": (10, "max avstånd från årshögsta i procent"),
            "box_dagar": (15, "lådans längd i handelsdagar"),
            "box_max": (10, "lådans höjd i procent (lägre = tightare)"),
            "vol_faktor": (1.5, "volym i dag jämfört med 50-dagars snitt"),
            "stop_marginal": (1.0, "procent under lådans botten"),
            "mal_R": (2.0, "vinstmål i R"),
            "max_dagar": (30, "längsta tid i traden"),
        },
    },
    "bas": {
        "namn": "Bas efter nedgång (ackumulation → utbrott)",
        "kort": "Efter ett fall rör sig aktien sidledes mellan tydligt stöd och motstånd, och bryter sedan upp över motståndet.",
        "typ": "Utbrott ur en bas",
        "bast": "När en nedgång har bottnat ur",
        "regler": [
            "Aktien har först fallit minst {fall_min} % från sin topp de senaste {fall_dagar} dagarna före basen.",
            "Sedan {bas_dagar} dagar sidledes: skillnaden mellan högsta (motstånd) och lägsta (stöd) är högst {bas_max} %.",
            "I dag stänger kursen över motståndet på minst {vol_faktor} gånger normal volym.",
            "Stop {stop_marginal} % under stödet. Sälj vid {mal_R} R, stoppen eller efter {max_dagar} dagar.",
        ],
        "p": {
            "fall_min": (15, "hur mycket aktien ska ha fallit innan basen, i procent"),
            "fall_dagar": (60, "fallet letas inom så många dagar före basen"),
            "bas_dagar": (20, "basens längd i handelsdagar"),
            "bas_max": (12, "basens höjd i procent"),
            "vol_faktor": (1.3, "volym i dag jämfört med 50-dagars snitt"),
            "stop_marginal": (1.0, "procent under stödet"),
            "mal_R": (2.0, "vinstmål i R"),
            "max_dagar": (30, "längsta tid i traden"),
        },
    },
    "rekyl": {
        "namn": "Rekyl i upptrend",
        "kort": "Aktien är i upptrend, har backat ner mot 50-dagars medelvärdet och visar en stark dag uppåt.",
        "typ": "Köper en tillfällig dipp i en uppgång",
        "bast": "Tydlig upptrend",
        "regler": [
            "Aktien är i upptrend (kurs över 200-dagars, 50-dagars över 200-dagars, 200-dagars stiger).",
            "Någon av de senaste {rekyl_dagar} dagarna var kursen nere inom {narhet} % från 50-dagars.",
            "RSI är under {rsi_max} (en riktig rekyl, inte toppen).",
            "I dag stänger kursen över gårdagens högsta och över öppningen (styrkedag).",
            "Stop {stop_marginal} % under lägsta kursen de senaste 5 dagarna. Sälj vid {mal_R} R, stoppen eller efter {max_dagar} dagar.",
        ],
        "p": {
            "rekyl_dagar": (3, "inom så många dagar ska kursen ha nått ner mot 50-dagars"),
            "narhet": (2, "hur nära 50-dagars, i procent"),
            "rsi_max": (55, "RSI ska vara under detta"),
            "stop_marginal": (1.0, "procent under bottnen"),
            "mal_R": (2.0, "vinstmål i R"),
            "max_dagar": (20, "längsta tid i traden"),
        },
    },
    "engulfing": {
        "namn": "Bullish engulfing efter fall",
        "kort": "Efter ett fall kommer en röd candle och sedan en större grön som helt täcker den (köparna tar över).",
        "typ": "Candlestick-vändning",
        "bast": "Nära stöd efter en nedgång, gärna med lågt RSI",
        "regler": [
            "Kursen har fallit minst {fall_min} % de senaste {fall_dagar} dagarna.",
            "Gårdagen var röd. I dag är grön och kroppen täcker gårdagens kropp helt.",
            "RSI i går var under {rsi_max}.",
            "Volymen i dag är minst {vol_faktor} gånger normal.",
            "Stop {stop_marginal} % under de två dagarnas lägsta. Sälj vid {mal_R} R, stoppen eller efter {max_dagar} dagar.",
        ],
        "p": {
            "fall_min": (5, "fallet innan, i procent"),
            "fall_dagar": (5, "fallet mäts över så många dagar"),
            "rsi_max": (40, "RSI i går ska vara under detta"),
            "vol_faktor": (1.0, "volym i dag jämfört med 50-dagars snitt"),
            "stop_marginal": (1.0, "procent under lägsta"),
            "mal_R": (2.0, "vinstmål i R"),
            "max_dagar": (15, "längsta tid i traden"),
        },
    },
    "slump": {
        "namn": "Kontroll: slumpmässiga köp",
        "kort": "Köper på slumpade dagar med samma sorts stop och mål. En metod som inte slår denna tillför inget.",
        "typ": "Jämförelse",
        "bast": "–",
        "regler": [
            "Ungefär var {var_n}:e handelsdag per aktie köps på måfå.",
            "Stop {atr_ggr} × ATR (normal daglig rörelse) under kursen. Sälj vid {mal_R} R, stoppen eller efter {max_dagar} dagar.",
        ],
        "p": {
            "var_n": (20, "i snitt en slumpad köpdag per så många dagar"),
            "atr_ggr": (2.0, "stoppens avstånd i ATR"),
            "mal_R": (2.0, "vinstmål i R"),
            "max_dagar": (20, "längsta tid i traden"),
        },
    },
}


# ---------------------------------------------------------------------------
# Kräsna urvalet: bara signaler som klarar alla krav, och bara så många du hinner handla.
# ---------------------------------------------------------------------------
# Vilka metoder som används i vilket marknadsläge (OMXS30). Bygger på backtesten:
# utbrott funkar i upptrend, dippköp när index går sidled eller ner, vecko-MACD efter nedgång.
LAGE_METODER = {
    "upp": ["darvas", "bas"],
    "sidled": ["dipp", "bas", "darvas"],
    "ned": ["dipp", "bas", "macd_vecka"],
}
KRASEN = {
    "min_rr": (2.0, "R/R mot närmaste motstånd ska vara minst så här (aktier på nya toppar saknar motstånd och godkänns)"),
    "min_volym": (1.0, "volymen på signaldagen minst så många gånger snittet (någon köper på riktigt)"),
    "rapport_dagar": (15, "ingen rapport inom så många dagar (bara i dag – historiska rapportdatum saknas)"),
    "max_oppna": (2, "max antal öppna trades samtidigt"),
    "max_per_vecka": (1, "max nya trades per vecka – den med bäst R/R väljs"),
}


def K(namn):
    return KRASEN[namn][0]


def P(metod):
    return {k: v[0] for k, v in METODER[metod]["p"].items()}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def ladda(ticker):
    fil = CACHE / ("m_" + ticker.replace("^", "_") + ".pkl")
    if fil.exists() and date.fromtimestamp(fil.stat().st_mtime) == date.today():
        return pd.read_pickle(fil)
    df = yf.download(ticker, period=f"{AR + 1}y", interval="1d", progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    kol = {k: sk.kolumn(df, k) for k in ["Open", "High", "Low", "Close"]}
    try:
        kol["Volume"] = sk.kolumn(df, "Volume")
    except Exception:
        pass
    ut = pd.DataFrame(kol).dropna(subset=["Open", "High", "Low", "Close"])
    if "Volume" not in ut:
        ut["Volume"] = np.nan
    ut.index = pd.to_datetime(ut.index)
    if ut.index.tz is not None:
        ut.index = ut.index.tz_localize(None)
    if len(ut) < 300:
        return None
    CACHE.mkdir(exist_ok=True)
    ut.to_pickle(fil)
    return ut


def trend(c):
    """upp / sidled / ned för varje dag (samma regel för index och aktier)."""
    ma50, ma200 = c.rolling(50).mean(), c.rolling(200).mean()
    t = pd.Series("sidled", index=c.index, dtype=object)
    t[(c > ma200) & (ma50 > ma200) & (ma200 > ma200.shift(20))] = "upp"
    t[(c < ma200) & (ma50 < ma200)] = "ned"
    t[ma200.isna()] = None
    return t


def forbered(df):
    d = df.copy()
    c = d["Close"]
    d["rsi"] = sk.rsi(c)
    d["macd"], d["macd_sig"], d["hist"] = sk.macd(c)
    d["ma20"], d["ma50"], d["ma200"] = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    # "Fas 2" (stabil upptrend): kurs över 20-dagars och 20-dagars över 50-dagars. "Fas 2+" = även 50 över 200.
    d["fas2"] = (c > d["ma20"]) & (d["ma20"] > d["ma50"])
    d["fas2plus"] = d["fas2"] & (d["ma50"] > d["ma200"])
    d["vol50"] = d["Volume"].rolling(50).mean().shift(1)
    d["vol_x"] = d["Volume"] / d["vol50"]
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - c.shift()).abs(), (d["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    d["trend"] = trend(c)
    return d


# ---------------------------------------------------------------------------
# Detektorer: returnerar (signal, stop, exit) som dagliga serier.
# signal = True på signaldagens stängning, exit = True när metoden säger sälj (nästa öppning)
# ---------------------------------------------------------------------------
def sig_dipp(d, p):
    h = d["hist"]
    vander = h < 0
    for i in range(int(p["vander_dagar"])):
        vander &= h.shift(i) > h.shift(i + 1)
    sig = (d["rsi"].rolling(int(p["rsi_dagar"])).min() < p["rsi_max"]) & vander
    stop = d["Low"].rolling(int(p["botten_dagar"])).min() * (1 - p["stop_marginal"] / 100)
    return sig, stop, None


def vecko(d):
    """Veckodata (fredag) + vilket dagligt datum som är veckans sista handelsdag."""
    dag = d.index.to_series()
    w = pd.DataFrame({
        "Close": d["Close"].resample("W-FRI").last(),
        "Low": d["Low"].resample("W-FRI").min(),
        "sista": dag.resample("W-FRI").last(),
    }).dropna()
    w["macd"], w["sig"], w["hist"] = sk.macd(w["Close"])
    return w


def sig_macd_vecka(d, p):
    w = vecko(d)
    upp = (w["macd"] > w["sig"]) & (w["macd"].shift(1) <= w["sig"].shift(1))
    if p["under_noll"]:
        upp &= w["macd"] < 0
    ner = (w["macd"] < w["sig"]) & (w["macd"].shift(1) >= w["sig"].shift(1))
    wstop = w["Low"].rolling(int(p["stop_veckor"])).min() * (1 - p["stop_marginal"] / 100)
    sig = pd.Series(False, index=d.index)
    exitv = pd.Series(False, index=d.index)
    stop = pd.Series(np.nan, index=d.index)
    sig.loc[w.loc[upp, "sista"].values] = True
    exitv.loc[w.loc[ner, "sista"].values] = True
    stop.loc[w["sista"].values] = wstop.values
    return sig, stop, exitv


def i_upptrend(d):
    return (d["Close"] > d["ma200"]) & (d["ma50"] > d["ma200"])


def sig_darvas(d, p):
    n = int(p["box_dagar"])
    tak = d["High"].shift(1).rolling(n).max()
    golv = d["Low"].shift(1).rolling(n).min()
    sig = ((tak / golv - 1) * 100 <= p["box_max"]) & (d["Close"] > tak) & (d["vol_x"] >= p["vol_faktor"])
    sig &= d["Close"] >= d["High"].rolling(250, min_periods=200).max() * (1 - p["nara_topp"] / 100)
    if p["krav_upptrend"]:
        sig &= i_upptrend(d)
    return sig, golv * (1 - p["stop_marginal"] / 100), None


def sig_bas(d, p):
    n = int(p["bas_dagar"])
    mot = d["High"].shift(1).rolling(n).max()
    stod = d["Low"].shift(1).rolling(n).min()
    topp_fore = d["High"].shift(n + 1).rolling(int(p["fall_dagar"])).max()
    sig = ((mot / stod - 1) * 100 <= p["bas_max"]) & ((1 - stod / topp_fore) * 100 >= p["fall_min"])
    sig &= (d["Close"] > mot) & (d["vol_x"] >= p["vol_faktor"])
    return sig, stod * (1 - p["stop_marginal"] / 100), None


def sig_rekyl(d, p):
    upp = i_upptrend(d) & (d["ma200"] > d["ma200"].shift(20))
    nara = (d["Low"] <= d["ma50"] * (1 + p["narhet"] / 100)).rolling(int(p["rekyl_dagar"])).max() > 0
    stark = (d["Close"] > d["High"].shift(1)) & (d["Close"] > d["Open"])
    sig = upp & nara & stark & (d["rsi"] < p["rsi_max"])
    stop = d["Low"].rolling(5).min() * (1 - p["stop_marginal"] / 100)
    return sig, stop, None


def engulfing(d):
    o, c, o1, c1 = d["Open"], d["Close"], d["Open"].shift(1), d["Close"].shift(1)
    return (c1 < o1) & (c > o) & (o <= c1) & (c >= o1) & ((c - o) > (o1 - c1))


def sig_engulfing(d, p):
    fall = (1 - d["Close"].shift(1) / d["Close"].shift(1 + int(p["fall_dagar"]))) * 100 >= p["fall_min"]
    sig = engulfing(d) & fall & (d["rsi"].shift(1) < p["rsi_max"]) & (d["vol_x"].fillna(1) >= p["vol_faktor"])
    stop = pd.concat([d["Low"], d["Low"].shift(1)], axis=1).min(axis=1) * (1 - p["stop_marginal"] / 100)
    return sig, stop, None


def sig_slump(d, p, fro):
    rng = np.random.default_rng(fro)
    sig = pd.Series(rng.random(len(d)) < 1 / p["var_n"], index=d.index) & d["atr"].notna()
    return sig, d["Close"] - p["atr_ggr"] * d["atr"], None


DETEKTOR = {"dipp": sig_dipp, "macd_vecka": sig_macd_vecka, "darvas": sig_darvas, "bas": sig_bas,
            "rekyl": sig_rekyl, "engulfing": sig_engulfing}


def signaler(metod, d, fro=0):
    p = P(metod)
    if metod == "slump":
        return sig_slump(d, p, fro)
    return DETEKTOR[metod](d, p)


# ---------------------------------------------------------------------------
# Simulering
# ---------------------------------------------------------------------------
def simulera(namn, metod, d, idx_close, idx_trend, fro=0):
    p = P(metod)
    sig, stop, exitv = signaler(metod, d, fro)
    sig = sig.fillna(False).to_numpy(dtype=bool)
    st = stop.to_numpy(dtype=float)
    ex = exitv.fillna(False).to_numpy(dtype=bool) if exitv is not None else None
    O, H, L, C = (d[k].to_numpy(dtype=float) for k in ["Open", "High", "Low", "Close"])
    V = d["vol_x"].to_numpy(dtype=float)
    piv = pivottoppar(H)
    dagar = d.index
    mal_R = p.get("mal_R")
    max_d = int(p["max_dagar"])
    k = KOSTNAD_PROCENT / 100
    trades, fri_fran = [], 0
    for t in np.flatnonzero(sig):
        if t < fri_fran or t + 1 >= len(C) or not np.isfinite(st[t]):
            continue
        e = O[t + 1]
        s = st[t]
        if s >= e or (e - s) / e * 100 > MAX_RISK_PROCENT:
            continue
        risk = e - s
        mal = e + mal_R * risk if mal_R else None
        ut, orsak, j = None, None, t + 1
        while j < len(C):
            if L[j] <= s:
                ut, orsak = min(O[j], s) if j > t + 1 else s, "stop"
                break
            if mal is not None and H[j] >= mal:
                ut, orsak = max(O[j], mal) if j > t + 1 else mal, "mål"
                break
            if ex is not None and ex[j] and j + 1 < len(C):
                j += 1
                ut, orsak = O[j], "sälj-signal"
                break
            if j - (t + 1) >= max_d:
                ut, orsak = C[j], "tid"
                break
            j += 1
        if ut is None:
            continue  # traden pågår fortfarande – räknas inte
        fri_fran = j + 1
        R = (ut - e) / risk - (e + ut) * k / risk
        mot = motstand_vid(H, piv, t, e)
        d0, d1 = dagar[t], dagar[j]
        i0 = idx_close.asof(d0)
        i1 = idx_close.asof(d1)
        trades.append({
            "metod": metod, "bolag": namn, "datum": str(d0.date()), "ut_datum": str(d1.date()),
            "kop": round(float(e), 2), "stop": round(float(s), 2), "salj": round(float(ut), 2),
            "R": round(float(R), 3), "dagar": int(j - t), "orsak": orsak,
            "risk_pct": round(float(risk / e * 100), 2),
            "rr_mot": round(float((mot - e) / risk), 2) if mot else None,
            "vol_x": round(float(V[t]), 2) if np.isfinite(V[t]) else None,
            "index_lage": idx_trend.asof(d0) if idx_trend is not None else None,
            "aktie_trend": d["trend"].iloc[t],
            "fas2": bool(d["fas2"].iloc[t]), "fas2plus": bool(d["fas2plus"].iloc[t]),
            "index_utv": round(float((i1 / i0 - 1) * 100), 2) if i0 and i1 and i0 == i0 and i1 == i1 else None,
            "test": str(d0.date()) >= TEST_FRAN,
        })
    return trades


PIVOT_K = 3          # en topp = högst av 3 dagar före och efter
MOTSTAND_BAKAT = 180  # motstånd letas så många dagar bakåt


def pivottoppar(H, k=PIVOT_K):
    """True där dagens högsta är högst i fönstret k dagar före och efter (bekräftas först k dagar senare)."""
    s = pd.Series(H)
    return (s == s.rolling(2 * k + 1, center=True).max()).to_numpy()


def motstand_vid(H, piv, t, kurs, k=PIVOT_K):
    """Närmaste motstånd ovanför kursen som var känt dag t. None = ingen topp ovanför (nya toppar)."""
    a, b = max(0, t - MOTSTAND_BAKAT), t - k + 1
    if b <= a:
        return None
    h = H[a:b][piv[a:b]]
    h = h[h > kurs * 1.005]
    return float(h.min()) if h.size else None


def klarar(x, lage=None):
    """Vilka krav en signal inte klarar (tom lista = kräsen signal)."""
    fel = []
    lage = lage if lage is not None else x["index_lage"]
    if x["metod"] not in LAGE_METODER.get(lage, []):
        fel.append(f"fel metod när index är i {LAGE_NAMN.get(lage, '–').lower()}")
    if x.get("rr_mot") is not None and x["rr_mot"] < K("min_rr"):
        fel.append(f"R/R mot motstånd {_f(x['rr_mot'], 1)} (under {_f(K('min_rr'), 1)})")
    if x.get("vol_x") is not None and x["vol_x"] < K("min_volym"):
        fel.append(f"låg volym ({_f(x['vol_x'], 1)}×)")
    return fel


def rang(x):
    return x["rr_mot"] if x.get("rr_mot") is not None else 3.0


def portfolj(trades, max_oppna, max_per_vecka):
    """Det du faktiskt hinner handla: max X öppna samtidigt och max Y nya per vecka, bästa R/R först."""
    valda = []
    veckor = {}
    for x in trades:
        veckor.setdefault(pd.Timestamp(x["datum"]).strftime("%G-%V"), []).append(x)
    for v in sorted(veckor):
        nya = 0
        for x in sorted(veckor[v], key=lambda x: (-rang(x), x["datum"])):
            if max_per_vecka and nya >= max_per_vecka:
                break
            oppna = [o for o in valda if o["datum"] <= x["datum"] < o["ut_datum"]]
            if len(oppna) >= max_oppna or any(o["bolag"] == x["bolag"] for o in oppna):
                continue
            valda.append(x)
            nya += 1
    return valda


def statistik(lista, ar=None):
    if not lista:
        return {"n": 0}
    r = np.array([x["R"] for x in lista])
    vinst, forl = r[r > 0], r[r <= 0]
    i_rad, mx = 0, 0
    for x in sorted(lista, key=lambda x: x["datum"]):
        i_rad = i_rad + 1 if x["R"] <= 0 else 0
        mx = max(mx, i_rad)
    return {
        "n": int(len(r)),
        "per_ar": round(len(r) / ar, 1) if ar else None,
        "traff": round(float((r > 0).mean() * 100), 1),
        "vinst_R": round(float(vinst.mean()), 2) if len(vinst) else 0.0,
        "forlust_R": round(float(forl.mean()), 2) if len(forl) else 0.0,
        "E": round(float(r.mean()), 3),
        "osakerhet": round(float(1.96 * r.std(ddof=1) / math.sqrt(len(r))), 3) if len(r) > 1 else None,
        "summa_R": round(float(r.sum()), 1),
        "dagar": round(float(np.median([x["dagar"] for x in lista])), 0),
        "forlust_i_rad": int(mx),
    }


# ---------------------------------------------------------------------------
# Dagens läge: candles, beta, katalysatorer, motstånd
# ---------------------------------------------------------------------------
def candle_text(d):
    r = d.iloc[-1]
    kropp = abs(r["Close"] - r["Open"])
    spann = r["High"] - r["Low"]
    snitt = (d["Close"] - d["Open"]).abs().rolling(20).mean().iloc[-1]
    ut = []
    if bool(engulfing(d).iloc[-1]):
        ut.append("Möjlig bullish engulfing")
    if spann > 0 and kropp / spann < 0.1:
        ut.append("Doji (osäkert)")
    if r["Close"] > r["Open"] and snitt and kropp > 2 * snitt and (r["High"] - r["Close"]) < 0.25 * spann:
        ut.append("Stor grön candle")
    return ut


def beta_korr(d, idx_close, dagar=250):
    a = d["Close"].pct_change()
    b = idx_close.reindex(d.index).pct_change()
    x = pd.concat([a, b], axis=1).dropna().iloc[-dagar:]
    if len(x) < 60:
        return None, None
    korr = float(x.iloc[:, 0].corr(x.iloc[:, 1]))
    beta = float(x.iloc[:, 0].cov(x.iloc[:, 1]) / x.iloc[:, 1].var())
    return round(beta, 2), round(korr, 2)


def narmaste_motstand(d, k=3, bakat=180):
    h = d["High"].to_numpy()
    kurs = d["Close"].iloc[-1]
    n = len(h)
    kand = []
    for i in range(max(k, n - bakat), n - k):
        if h[i] == h[i - k:i + k + 1].max() and h[i] > kurs * 1.005:
            kand.append(h[i])
    return float(min(kand)) if kand else None


def katalysatorer(namn, ticker):
    """Nästa rapport och nästa x-utdelningsdag (Yahoo). None om okänt."""
    idag = date.today()
    rapport = sk.nasta_rapport(namn, ticker)
    exdiv = None
    try:
        cal = yf.Ticker(ticker).calendar
        x = cal.get("Ex-Dividend Date") if isinstance(cal, dict) else None
        if x is not None:
            x = x.date() if hasattr(x, "date") and callable(x.date) else x
            if isinstance(x, date) and x >= idag:
                exdiv = x
    except Exception:
        pass
    return rapport, exdiv


# ---------------------------------------------------------------------------
# Sidan
# ---------------------------------------------------------------------------
def _f(v, dec=2, plus=False):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–"
    s = f"{v:+.{dec}f}" if plus else f"{v:.{dec}f}"
    return s.replace(".", ",").replace("-", "−")


def _e_cell(s):
    if not s or s.get("n", 0) == 0:
        return '<td class="na">–</td>'
    e, o, n = s["E"], s.get("osakerhet") or 0, s["n"]
    klass = "pos" if e - o > 0 else ("neg" if e + o < 0 else ("svpos" if e > 0 else "svneg"))
    if n < 30:
        klass += " fa"
    return f'<td class="{klass}"><b>{_f(e, 2, True)}</b><small>{n} st</small></td>'


def _krav(x):
    if x.get("krasen"):
        return '<p class="ok">✓ Klarar alla krav – kräsen signal</p>'
    fel = x.get("fel") or []
    return f'<p class="nej">Inte kräsen: {escape(", ".join(fel))}</p>' if fel else ""


def _rad(x, extra=""):
    a, t = x["alla"], x["test"]
    if not a.get("n"):
        return f'<tr><td class="mn">{escape(x["namn"])}</td><td colspan="7" class="na">Inga trades</td></tr>'
    pct = f'<td><b>{_f(x["procent_per_ar"], 0, True)} %</b></td>' if x.get("handelbar") else '<td class="na">–</td>'
    return (f'<tr><td class="mn">{escape(x["namn"])}</td><td>{_f(a["per_ar"], 0)}</td><td>{_f(a["traff"], 0)} %</td>'
            f'{_e_cell(a)}{_e_cell(t)}{pct}{extra}</tr>')


def _taggar(lista):
    return "".join(f'<i class="{c}">{escape(t)}</i>' for c, t in lista)


LAGE_NAMN = {"upp": "Upptrend", "sidled": "Sidled", "ned": "Nedtrend", None: "–"}


def bygg_sida(res, fil):
    mk, idag = res["marknad"], res["idag"]
    ordning = list(METODER)

    # --- Backtest-tabell
    rader = []
    for m in ordning:
        s = res["stat"][m]
        a = s["alla"]
        rader.append(
            f'<tr><td class="mn"><a href="#m-{m}">{escape(METODER[m]["namn"])}</a></td>'
            f'{_e_cell(a)}<td>{_f(a.get("per_ar"), 0) if a.get("n") else "–"}</td>'
            f'<td>{_f(a.get("traff"), 0) + " %" if a.get("n") else "–"}</td>'
            f'{_e_cell(s["test"])}{_e_cell(s["lage"]["upp"])}{_e_cell(s["lage"]["sidled"])}{_e_cell(s["lage"]["ned"])}'
            f'{_e_cell(s["index_upp"])}{_e_cell(s["index_ner"])}</tr>')
    backtest = "".join(rader)

    # --- Dagens signaler
    kort = []
    for m in ordning:
        if m == "slump":
            continue
        lista = idag["signaler"].get(m, [])
        if lista:
            inner = "".join(
                f'<div class="sig" data-kurs="{x["kurs"]}" data-stop="{x["stop"]}">'
                f'<div class="sig-top"><b>{escape(x["bolag"])}</b><span>{_f(x["kurs"])} kr</span></div>'
                f'<div class="sig-nums"><span>Stop <b>{_f(x["stop"])}</b> ({_f(-x["risk_pct"], 1)} %)</span>'
                f'<span>Mål 2R <b>{_f(x["mal"])}</b></span>'
                f'<span>Närmaste motstånd <b>{_f(x["motstand"]) if x["motstand"] else "–"}</b>'
                f'{" · R/R " + _f(x["rr_mot"], 1) if x["rr_mot"] else ""}</span>'
                f'<span>Antal för 2 %: <b class="antal">–</b></span></div>'
                f'<div class="tags">{_taggar(x["taggar"])}</div>'
                f'{_krav(x)}</div>'
                for x in lista)
        else:
            inner = '<p class="empty">Ingen signal i dag.</p>'
        kort.append(f'<div class="card"><h3>{escape(METODER[m]["namn"])}</h3><p class="muted">{escape(METODER[m]["kort"])}</p>{inner}</div>')

    # --- Metodbeskrivningar
    besk = []
    for m in ordning:
        info, p, s = METODER[m], P(m), res["stat"][m]
        regler = "".join(f"<li>{escape(r.format(**{k: _f(v, 1 if isinstance(v, float) else 0) for k, v in p.items()}))}</li>" for r in info["regler"])
        params = "".join(f'<tr><td><code>{k}</code></td><td>{_f(v[0], 1 if isinstance(v[0], float) else 0)}</td><td>{escape(v[1])}</td></tr>' for k, v in info["p"].items())
        a = s["alla"]
        detalj = (f'Träff {_f(a.get("traff"), 0)} % · snittvinst {_f(a.get("vinst_R"), 2, True)}R · snittförlust {_f(a.get("forlust_R"), 2, True)}R · '
                  f'osäkerhet ±{_f(a.get("osakerhet"), 2)}R · median {_f(a.get("dagar"), 0)} dagar · längsta förlustsvit {a.get("forlust_i_rad", "–")}') if a.get("n") else "Inga trades."
        orsaker = s.get("orsaker", {})
        ors = " · ".join(f"{k} {v} %" for k, v in orsaker.items())
        besk.append(f'''<details id="m-{m}"><summary>{escape(info["namn"])} <span class="muted">– {escape(info["typ"])}</span></summary>
<p>{escape(info["kort"])} <b>Funkar bäst:</b> {escape(info["bast"])}.</p>
<ol>{regler}</ol>
<p class="muted">{detalj}<br>Hur traderna slutade: {ors or "–"}</p>
<div class="tablewrap"><table><tr><th>Parameter</th><th>Värde</th><th>Vad den gör</th></tr>{params}</table></div></details>''')

    # --- Sektorer
    sek = "".join(
        f'<tr><td>{escape(x["sektor"])}</td><td>{x["antal"]}</td><td class="{"pos" if x["rel_1m"] > 0 else "neg"}">{_f(x["rel_1m"], 1, True)}</td>'
        f'<td class="{"pos" if x["rel_3m"] > 0 else "neg"}">{_f(x["rel_3m"], 1, True)}</td><td>{x["upptrend"]} av {x["antal"]}</td></tr>'
        for x in idag["sektorer"])

    # --- Alla bolag
    bol = "".join(
        f'<tr><td>{escape(x["bolag"])}</td><td>{escape(x["sektor"])}</td><td>{_f(x["kurs"])}</td>'
        f'<td class="t-{x["trend"]}">{LAGE_NAMN.get(x["trend"], "–")}</td><td>{_f(x["rsi"], 0)}</td>'
        f'<td>{_f(x["vol_x"], 1)}×</td><td>{escape(", ".join(x["candle"])) or "–"}</td>'
        f'<td>{_f(x["beta"], 2)}</td><td>{_f(x["korr"] * 100 if x["korr"] is not None else None, 0)} %</td>'
        f'<td>{x["rapport"] or "okänt"}</td><td>{escape(", ".join(METODER[m]["namn"].split(":")[0] for m in x["metoder"])) or "–"}</td></tr>'
        for x in idag["bolag"])

    kr = res.get("krasna_idag", [])
    krasna_html = "".join(
        f'<div class="card"><div class="sig-top"><b>{escape(x["bolag"])}</b><span>{_f(x["kurs"])} kr</span></div>'
        f'<p class="muted">{escape(METODER[x["metod"]]["namn"])}</p>'
        f'<div class="sig" data-kurs="{x["kurs"]}" data-stop="{x["stop"]}"><div class="sig-nums">'
        f'<span>Stop <b>{_f(x["stop"])}</b> ({_f(-x["risk_pct"], 1)} %)</span><span>Mål 2R <b>{_f(x["mal"])}</b></span>'
        f'<span>Motstånd <b>{_f(x["motstand"]) if x["motstand"] else "inget (ny topp)"}</b>{" · R/R " + _f(x["rr_mot"], 1) if x["rr_mot"] else ""}</span>'
        f'<span>Antal för 2 %: <b class="antal">–</b></span></div><div class="tags">{_taggar(x["taggar"])}</div></div></div>'
        for x in kr) or '<p class="empty">Ingen kräsen signal i dag. Vänta – det är så metoden ska fungera.</p>'
    tratt_html = "".join(_rad(x) for x in res["tratt"])
    takt_html = "".join(_rad(x) for x in res["takt"])
    sista = res["tratt"][-1].get("per_kalenderar", {})
    per_ar_html = " · ".join(f"{y}: {_f(v, 1, True)}R" for y, v in sista.items()) or "–"
    lage = mk["lage"]
    html = f"""<!doctype html>
<html lang="sv"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Metoder och marknad</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,700&family=Instrument+Serif&display=swap">
<style>
:root{{--bg:#F4F1EA;--card:#FFFFFF;--fg:#1B1A17;--muted:#6B665C;--line:#E2DDD2;--up:#2F7A3E;--up-bg:#DCEBD7;
--down:#A3321C;--down-bg:#F6DCD5;--warn:#8A5A0B;--warn-bg:#F4E4C4;--blue:#2F4BD8;--blue-bg:#D9E2F7;color-scheme:light}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161513;--card:#211F1C;--fg:#EEEAE2;--muted:#A8A295;--line:#33302B;
--up:#8FD19C;--up-bg:#1F3322;--down:#F09A86;--down-bg:#3D211B;--warn:#EBC890;--warn-bg:#3A2C17;--blue:#9DB0FF;--blue-bg:#1F2A4A;color-scheme:dark}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font-family:'DM Sans',system-ui,sans-serif;font-size:15px;line-height:1.45}}
.wrap{{max-width:1100px;margin:0 auto;padding:32px 16px 48px;display:flex;flex-direction:column;gap:30px}}
h1{{margin:0;font-family:'Instrument Serif',Georgia,serif;font-weight:400;font-size:clamp(36px,6vw,54px);line-height:1}}
h2{{margin:0 0 6px;font-size:21px}} h3{{margin:0;font-size:17px}}
.eyebrow{{font-size:13px;color:var(--muted);letter-spacing:.04em;text-transform:uppercase}}
.muted{{color:var(--muted);font-size:14px;margin:4px 0 0}}
a{{color:var(--blue)}}
.lead{{max-width:72ch;color:var(--muted);margin:6px 0 0}}
.market{{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:18px;display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px}}
.big{{font-family:'Instrument Serif',Georgia,serif;font-size:34px;line-height:1.05}}
.l-upp{{color:var(--up)}}.l-ned{{color:var(--down)}}.l-sidled{{color:var(--warn)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:16px;display:flex;flex-direction:column;gap:10px;min-width:0}}
.sig{{border-top:1px solid var(--line);padding-top:10px;display:flex;flex-direction:column;gap:6px}}
.sig-top{{display:flex;justify-content:space-between;font-size:16px}}
.sig-nums{{display:grid;grid-template-columns:1fr 1fr;gap:4px 10px;font-size:13px;color:var(--muted)}}
.sig-nums b{{color:var(--fg);font-variant-numeric:tabular-nums}}
.tags{{display:flex;flex-wrap:wrap;gap:6px}}
.tags i{{font-style:normal;font-size:12px;padding:3px 8px;border-radius:999px;background:var(--bg);color:var(--muted)}}
.tags i.bra{{background:var(--up-bg);color:var(--up)}}.tags i.varn{{background:var(--warn-bg);color:var(--warn)}}.tags i.dalig{{background:var(--down-bg);color:var(--down)}}
.empty{{margin:0;color:var(--muted);font-size:14px;padding:10px 12px;border:1px dashed var(--line);border-radius:12px}}
.konto{{display:flex;align-items:center;gap:8px;flex-wrap:wrap;font-size:14px}}
.konto input{{width:110px;padding:6px 8px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);font:inherit}}
.tablewrap{{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:18px}}
table{{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap}}
th{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;vertical-align:bottom}}
td small{{display:block;font-size:11px;color:var(--muted);font-weight:400}}
td.mn{{white-space:normal;min-width:180px}}
td.pos{{background:var(--up-bg);color:var(--up)}}td.neg{{background:var(--down-bg);color:var(--down)}}
td.svpos{{color:var(--up)}}td.svneg{{color:var(--down)}}td.fa{{opacity:.55}}td.na{{color:var(--muted)}}
.t-upp{{color:var(--up)}}.t-ned{{color:var(--down)}}.t-sidled{{color:var(--warn)}}
details{{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:14px 18px}}
details .tablewrap{{border:none;border-radius:0}}
summary{{cursor:pointer;font-weight:600}}
ol{{padding-left:20px;margin:8px 0}}
code{{font-size:13px;background:var(--bg);padding:1px 5px;border-radius:5px}}
.legend{{display:flex;flex-wrap:wrap;gap:8px 16px;font-size:13px;color:var(--muted)}}
.legend span{{display:inline-flex;align-items:center;gap:6px}}
.legend i{{display:inline-block;width:14px;height:14px;border-radius:4px}}
.intro{{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:18px}}
.intro p{{margin:6px 0;max-width:80ch}}
p.ok{{margin:0;font-size:13px;color:var(--up);font-weight:600}}
p.nej{{margin:0;font-size:13px;color:var(--muted)}}
footer{{color:var(--muted);font-size:13px;max-width:80ch}}
</style></head>
<body><div class="wrap">
<header>
  <span class="eyebrow">Data till {escape(res["datum"])} · backtest {escape(res["period"])} · {res["antal_bolag"]} bolag</span>
  <h1>Metoder och marknad</h1>
  <p class="lead">Marknad → sektor → aktie. Först vilket läge börsen är i, sedan vilka branscher som är starka, sedan vilka aktier som ger signal – och hur varje metod faktiskt har gått historiskt. <a href="./">← Tillbaka till skanningen</a></p>
</header>

<section class="intro">
<h2>Så läser du siffrorna</h2>
<p><b>R = det du riskerar i en trade.</b> Med 10 000 kr och 2 % risk är 1R = 200 kr (avståndet ner till stoppen × antal aktier).</p>
<p><b>Snitt-R +0,20</b> betyder att du i snitt tjänar 0,2 × 200 kr = <b>40 kr per trade</b> efter courtage. Vissa trades ger +2R (400 kr), andra −1R (−200 kr) – snittet över många trades blir +0,2R.
Gör du 25 sådana trades per år blir det 25 × 0,2 = 5R = <b>ungefär +10 % på kontot</b> (5 × 2 %).</p>
<p><b>±-siffran</b> är osäkerheten. +0,20 ±0,30 kan lika gärna vara noll – det har inte hänt tillräckligt många gånger för att veta. Grön bakgrund = plus även med osäkerheten räknad.</p>
</section>

<section>
<h2>Kräsna signaler i dag</h2>
<p class="muted">Bara signaler som klarar alla krav: rätt metod för marknadsläget, R/R minst {_f(K("min_rr"), 1)} mot närmaste motstånd, volym minst {_f(K("min_volym"), 1)}× snittet och ingen rapport inom {int(K("rapport_dagar"))} dagar.
Historiskt kommer ungefär {_f(res["tratt"][3]["alla"].get("per_ar"), 0)} sådana per år – oftast är svaret "ingen i dag", och det är meningen.</p>
<div class="grid">{krasna_html}</div>
</section>

<section>
<h2>Från alla signaler till kräsna</h2>
<p class="muted">Varje rad lägger till ett krav. Så ser du vad varje krav gör med resultatet. Sista raden är det du faktiskt hinner handla, och bara där går det att räkna procent per år (vid 2 % risk per trade, utan ränta-på-ränta).</p>
<div class="tablewrap"><table><tr><th>Steg</th><th>Trades<br>per år</th><th>Träff</th><th>Snitt-R</th><th>Test<br>{TEST_FRAN[:4]}–</th><th>≈ % per år</th></tr>{tratt_html}</table></div>
<p class="muted">År för år (sista raden, summa R): {per_ar_html}</p>
</section>

<section>
<h2>Hur ofta ska du handla?</h2>
<p class="muted">Samma kräsna signaler, olika takt. Max {int(K("max_oppna"))} öppna samtidigt i alla rader.</p>
<div class="tablewrap"><table><tr><th>Takt</th><th>Trades<br>per år</th><th>Träff</th><th>Snitt-R</th><th>Test<br>{TEST_FRAN[:4]}–</th><th>≈ % per år</th></tr>{takt_html}</table></div>
</section>

<section>
<h2>1. Marknaden</h2>
<div class="market">
  <div><span class="eyebrow">OMXS30 just nu</span><div class="big l-{lage}">{LAGE_NAMN[lage]}</div>
  <p class="muted">Index {_f(mk["kurs"], 0)} · {_f(mk["over_200"], 1, True)} % mot 200-dagars · senaste månaden {_f(mk["manad"], 1, True)} %</p></div>
  <div><span class="eyebrow">Bolag i upptrend</span><div class="big">{mk["andel_upp"]} av {mk["antal"]}</div>
  <p class="muted">Ju fler, desto bredare uppgång. Under hälften = svag marknad även om index ser okej ut.</p></div>
  <div><span class="eyebrow">Hur mycket styr index?</span><div class="big">{_f(mk["r2"], 0)} %</div>
  <p class="muted">I snitt förklaras så stor del av en OMXS30-akties dagliga rörelser av index (senaste året). Snittbeta {_f(mk["beta"], 2)}: när index rör sig 1 % rör sig aktien i snitt {_f(mk["beta"], 1)} %.</p></div>
</div>
</section>

<section>
<h2>2. Sektorer</h2>
<p class="muted">Utveckling jämfört med index. Plus = starkare än OMXS30. Leta helst köp i sektorer som är starka.</p>
<div class="tablewrap"><table><tr><th>Sektor</th><th>Bolag</th><th>1 mån mot index</th><th>3 mån mot index</th><th>I upptrend</th></tr>{sek}</table></div>
</section>

<section>
<h2>3. Dagens signaler per metod</h2>
<div class="konto">Tradingkonto: <input id="konto" type="number" min="0" step="100" value="10000"> kr · risk 2 % = <b id="risk">200</b> kr per trade</div>
<div class="grid">{''.join(kort)}</div>
</section>

<section>
<h2>4. Hur metoderna har gått</h2>
<p class="muted">Snitt-R per trade efter courtage, sedan {escape(res["period"].split("–")[0])}. +0,30 betyder att du i snitt tjänat 0,3 gånger risken per trade. Siffran under är antal trades.
"Test" = bara trades från {TEST_FRAN[:4]} och framåt. Marknadsläge = hur OMXS30 låg när köpet gjordes. "OMX steg/föll" = vad index gjorde medan traden pågick.</p>
<div class="legend"><span><i style="background:var(--up-bg)"></i>säkert plus</span><span><i style="background:var(--down-bg)"></i>säkert minus</span>
<span>färgad text = kan vara slumpen</span><span>blek = under 30 trades, säger lite</span></div>
<div class="tablewrap"><table>
<tr><th>Metod</th><th>Snitt-R<br>alla</th><th>Trades<br>per år</th><th>Träff</th><th>Test<br>{TEST_FRAN[:4]}–</th><th>Index<br>upptrend</th><th>Index<br>sidled</th><th>Index<br>nedtrend</th><th>OMX steg<br>under traden</th><th>OMX föll<br>under traden</th></tr>
{backtest}</table></div>
</section>

<section>
<h2>5. Metoderna och deras inställningar</h2>
<p class="muted">Klicka för regler och parametrar. Vill du finjustera – säg till exempel "ändra darvas box_max till 8".</p>
{''.join(besk)}
</section>

<section>
<h2>6. Alla bolag i dag</h2>
<div class="tablewrap"><table><tr><th>Bolag</th><th>Sektor</th><th>Kurs</th><th>Trend</th><th>RSI</th><th>Volym</th><th>Candle i dag</th><th>Beta</th><th>Följer index</th><th>Rapport</th><th>Signal</th></tr>{bol}</table></div>
<p class="muted">Volym = dagens volym jämfört med 50-dagars snitt. Följer index = korrelation senaste året (100 % = rör sig exakt som index). Candle-mönster är ledtrådar, inget facit.</p>
</section>

<footer>Allt räknas bara på data som fanns den dagen. Köp på nästa dags öppning, {KOSTNAD_PROCENT:.2f} % courtage + spread per affär är avdraget. Bara dagens OMXS30-bolag testas, vilket gör historiken något för snäll (bolag som rasat ur index saknas).
Ett verktyg för att hitta kandidater, inte köpråd.</footer>
</div>
<script>
(function(){{
  var inp=document.getElementById('konto'),out=document.getElementById('risk');
  try{{var s=localStorage.getItem('konto');if(s)inp.value=s;}}catch(e){{}}
  function rakna(){{
    var k=parseFloat(inp.value)||0,r=k*0.02;out.textContent=Math.round(r);
    document.querySelectorAll('.sig').forEach(function(el){{
      var kurs=parseFloat(el.dataset.kurs),st=parseFloat(el.dataset.stop),a=el.querySelector('.antal');
      var n=kurs>st?Math.floor(r/(kurs-st)):0;a.textContent=n+' st ('+Math.round(n*kurs)+' kr)';
    }});
    try{{localStorage.setItem('konto',inp.value);}}catch(e){{}}
  }}
  inp.addEventListener('input',rakna);rakna();
}})();
</script></body></html>"""
    fil.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
def main():
    idx = ladda(sk.INDEX_TICKER)
    if idx is None:
        raise SystemExit("Ingen indexdata")
    idx_close = idx["Close"]
    idx_trend = trend(idx_close)

    alla_trades, idag_bolag, idag_sig = [], [], {m: [] for m in METODER}
    data = {}
    for i, (namn, ticker) in enumerate(sk.BOLAG.items()):
        try:
            df = ladda(ticker)
            if df is None:
                print(f"  Ingen data: {namn}")
                continue
            d = forbered(df)
            data[namn] = d
            for m in METODER:
                alla_trades += simulera(namn, m, d, idx_close, idx_trend, fro=i)
        except Exception as e:
            print(f"  Fel för {namn}: {e}")

    start = min(d.index[0] for d in data.values())
    ar = max((max(d.index[-1] for d in data.values()) - start).days / 365.25 - 1, 1)  # första året går åt till 200-dagars
    test_ar = max((max(d.index[-1] for d in data.values()) - pd.Timestamp(TEST_FRAN)).days / 365.25, 0.5)

    stat = {}
    for m in METODER:
        t = [x for x in alla_trades if x["metod"] == m]
        n = len(t) or 1
        stat[m] = {
            "alla": statistik(t, ar),
            "trana": statistik([x for x in t if not x["test"]]),
            "test": statistik([x for x in t if x["test"]], test_ar),
            "lage": {l: statistik([x for x in t if x["index_lage"] == l]) for l in ["upp", "sidled", "ned"]},
            "aktie_trend": {l: statistik([x for x in t if x["aktie_trend"] == l]) for l in ["upp", "sidled", "ned"]},
            "fas2": {"ja": statistik([x for x in t if x["fas2"]]), "nej": statistik([x for x in t if not x["fas2"]]),
                     "plus": statistik([x for x in t if x["fas2plus"]])},
            "index_upp": statistik([x for x in t if (x["index_utv"] or 0) > 0]),
            "index_ner": statistik([x for x in t if (x["index_utv"] or 0) <= 0]),
            "orsaker": {o: round(sum(1 for x in t if x["orsak"] == o) / n * 100) for o in ["mål", "stop", "sälj-signal", "tid"]
                        if any(x["orsak"] == o for x in t)},
        }

    # --- Kräsna urvalet: steg för steg från alla signaler till det du faktiskt handlar
    riktiga = sorted([x for x in alla_trades if x["metod"] != "slump"], key=lambda x: x["datum"])
    s1 = [x for x in riktiga if x["metod"] in LAGE_METODER.get(x["index_lage"], [])]
    s2 = [x for x in s1 if x["rr_mot"] is None or x["rr_mot"] >= K("min_rr")]
    s3 = [x for x in s2 if x["vol_x"] is None or x["vol_x"] >= K("min_volym")]
    s4 = portfolj(s3, int(K("max_oppna")), int(K("max_per_vecka")))

    def steg(namn, lista, handelbar=False):
        a = statistik(lista, ar)
        out = {"namn": namn, "alla": a, "trana": statistik([x for x in lista if not x["test"]]),
               "test": statistik([x for x in lista if x["test"]], test_ar), "handelbar": handelbar}
        if handelbar and a.get("n"):
            out["R_per_ar"] = round(a["summa_R"] / ar, 1)
            out["procent_per_ar"] = round(a["summa_R"] / ar * 2, 1)
            out["test_R_per_ar"] = round(out["test"].get("summa_R", 0) / test_ar, 1) if out["test"].get("n") else None
            per_ar = {}
            for x in lista:
                per_ar.setdefault(x["datum"][:4], []).append(x["R"])
            out["per_kalenderar"] = {y: round(sum(v), 1) for y, v in sorted(per_ar.items())}
        return out

    tratt = [
        steg("Alla signaler från alla metoder", riktiga),
        steg("+ bara metoden som passar marknadsläget", s1),
        steg(f"+ R/R mot motstånd minst {_f(K('min_rr'), 1)}", s2),
        steg(f"+ volym minst {_f(K('min_volym'), 1)}× snittet", s3),
        steg(f"+ max {int(K('max_oppna'))} öppna, {int(K('max_per_vecka'))} ny per vecka (det du hinner handla)", s4, True),
    ]
    takt = [steg(t, portfolj(s3, int(K("max_oppna")), v), True) for t, v in
            [("Max 1 ny trade per vecka", 1), ("Max 2 nya per vecka", 2), ("Ingen veckogräns (bara max 2 öppna)", 0)]]
    takt.append(steg("Jämförelse: alla signaler, max 1 per vecka, inga krav", portfolj(riktiga, int(K("max_oppna")), 1), True))
    takt.append(steg("Jämförelse: slumpköp, max 1 per vecka", portfolj(
        sorted([x for x in alla_trades if x["metod"] == "slump"], key=lambda x: x["datum"]), int(K("max_oppna")), 1), True))

    # --- Marknaden och bolagen i dag
    betas, korrs, upp = [], [], 0
    for namn, d in data.items():
        beta, korr = beta_korr(d, idx_close)
        betas.append(beta), korrs.append(korr)
        tr = d["trend"].iloc[-1]
        upp += tr == "upp"
        metoder_idag = []
        for m in METODER:
            if m == "slump":
                continue
            sig, stop, _ = signaler(m, d)
            if bool(sig.fillna(False).iloc[-1]) and np.isfinite(stop.iloc[-1]) and stop.iloc[-1] < d["Close"].iloc[-1]:
                metoder_idag.append(m)
        rapport, exdiv = (None, None)
        if metoder_idag or os.environ.get("ALLA_RAPPORTER"):
            rapport, exdiv = katalysatorer(namn, sk.BOLAG[namn])
        kurs = float(d["Close"].iloc[-1])
        candle = candle_text(d)
        rad = {
            "bolag": namn, "sektor": SEKTOR.get(namn, "Övrigt"), "kurs": round(kurs, 2), "trend": tr,
            "rsi": float(d["rsi"].iloc[-1]), "vol_x": float(d["vol_x"].iloc[-1]) if d["vol_x"].iloc[-1] == d["vol_x"].iloc[-1] else None,
            "candle": candle, "beta": beta, "korr": korr,
            "rapport": rapport.isoformat() if rapport else None, "metoder": metoder_idag,
            "manad": float(d["Close"].iloc[-1] / d["Close"].iloc[-22] - 1) * 100,
            "kvartal": float(d["Close"].iloc[-1] / d["Close"].iloc[-64] - 1) * 100,
        }
        idag_bolag.append(rad)
        for m in metoder_idag:
            sig, stop, _ = signaler(m, d)
            s = float(stop.iloc[-1])
            risk = kurs - s
            mot = narmaste_motstand(d)
            taggar = [("bra" if tr == "upp" else ("dalig" if tr == "ned" else "varn"), f"Aktien: {LAGE_NAMN[tr]}")]
            if rad["vol_x"]:
                taggar.append(("bra" if rad["vol_x"] >= 1.5 else "", f"Volym {_f(rad['vol_x'], 1)}× snittet"))
            taggar += [("bra", c) if "engulfing" in c or "grön" in c else ("varn", c) for c in candle]
            if korr is not None and korr >= 0.7:
                taggar.append(("varn", f"Följer index starkt ({_f(korr * 100, 0)} %)"))
            if rapport:
                dgr = (rapport - date.today()).days
                taggar.append(("dalig" if dgr <= RAPPORT_VARNING_DAGAR else "", f"Rapport om {dgr} dagar ({rapport:%d/%m})"))
            else:
                taggar.append(("", "Rapportdatum okänt – kolla"))
            if exdiv:
                taggar.append(("varn", f"Utdelning (x-dag) {exdiv:%d/%m}"))
            post = {
                "metod": m, "bolag": namn, "kurs": round(kurs, 2), "stop": round(s, 2), "risk_pct": round(risk / kurs * 100, 1),
                "mal": round(kurs + 2 * risk, 2), "motstand": round(mot, 2) if mot else None,
                "rr_mot": round((mot - kurs) / risk, 1) if mot and risk > 0 else None, "taggar": taggar,
                "vol_x": rad["vol_x"],
            }
            fel = klarar(post, idx_trend.iloc[-1])
            if rapport and (rapport - date.today()).days <= K("rapport_dagar"):
                fel.append(f"rapport om {(rapport - date.today()).days} dagar")
            post["fel"] = fel
            post["krasen"] = not fel
            idag_sig[m].append(post)

    ik = float(idx_close.iloc[-1])
    k2 = [k for k in korrs if k is not None]
    b2 = [b for b in betas if b is not None]
    mk = {
        "lage": idx_trend.iloc[-1], "kurs": ik,
        "over_200": (ik / float(idx_close.rolling(200).mean().iloc[-1]) - 1) * 100,
        "manad": (ik / float(idx_close.iloc[-22]) - 1) * 100,
        "andel_upp": int(upp), "antal": len(data),
        "r2": float(np.mean([k * k for k in k2]) * 100) if k2 else None,
        "beta": float(np.mean(b2)) if b2 else None,
    }
    idx_1m = (ik / float(idx_close.iloc[-22]) - 1) * 100
    idx_3m = (ik / float(idx_close.iloc[-64]) - 1) * 100
    sektorer = []
    for s in sorted(set(SEKTOR.get(r["bolag"], "Övrigt") for r in idag_bolag)):
        g = [r for r in idag_bolag if r["sektor"] == s]
        sektorer.append({"sektor": s, "antal": len(g), "upptrend": sum(r["trend"] == "upp" for r in g),
                         "rel_1m": float(np.mean([r["manad"] for r in g])) - idx_1m,
                         "rel_3m": float(np.mean([r["kvartal"] for r in g])) - idx_3m})
    sektorer.sort(key=lambda x: -x["rel_3m"])
    idag_bolag.sort(key=lambda r: (-len(r["metoder"]), r["bolag"]))

    slut = max(d.index[-1] for d in data.values())
    res = {
        "datum": f"{slut:%Y-%m-%d}", "period": f"{start + pd.Timedelta(days=365):%Y}–{slut:%Y}", "antal_bolag": len(data),
        "marknad": mk, "stat": stat, "idag": {"signaler": idag_sig, "bolag": idag_bolag, "sektorer": sektorer},
        "parametrar": {m: P(m) for m in METODER},
        "tratt": tratt, "takt": takt,
        "krasna_idag": sorted([x for l in idag_sig.values() for x in l if x["krasen"]], key=lambda x: -(x["rr_mot"] or 3)),
    }
    Path("results").mkdir(exist_ok=True)
    (Path("results") / "metoder.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (Path("results") / "metoder_trades.json").write_text(json.dumps(alla_trades, ensure_ascii=False, default=str), encoding="utf-8")

    mapp = Path("site") if os.environ.get("GITHUB_ACTIONS") == "true" else Path(".")
    mapp.mkdir(exist_ok=True)
    bygg_sida(res, mapp / "metoder.html")

    print(f"{len(alla_trades)} trades totalt. Index: {mk['lage']}.")
    for x in tratt + takt:
        a, t = x["alla"], x["test"]
        print(f"  {x['namn'][:60]:60} n={a.get('n',0):5} E={a.get('E',0):+.2f} ±{a.get('osakerhet') or 0:.2f} "
              f"test E={t.get('E',0):+.2f} (n={t.get('n',0)}) %/år={x.get('procent_per_ar')} {x.get('per_kalenderar','')}")
    for m in METODER:
        a, t = stat[m]["alla"], stat[m]["test"]
        print(f"  {m:11} n={a.get('n',0):5}  E={a.get('E',0):+.2f}R ±{a.get('osakerhet') or 0:.2f}  "
              f"test E={t.get('E',0):+.2f} (n={t.get('n',0)})  "
              f"upp {stat[m]['lage']['upp'].get('E',0):+.2f} sid {stat[m]['lage']['sidled'].get('E',0):+.2f} "
              f"ned {stat[m]['lage']['ned'].get('E',0):+.2f}")


if __name__ == "__main__":
    main()
