#!/usr/bin/env python3
"""
Djupanalys: kan stöd och motstånd ge dig minst 1:2 i R/R?

Bara det du redan använder: RSI, MACD, stöd och motstånd, stop-loss, mål och R/R.
Varje strategi mäts med samma formler:

  p   = träffprocent (andel trades med vinst)
  W   = snittvinst i R för vinsttrades
  L   = snittförlust i R för förlusttrades
  R/R = W / L                       (realiserad, alltså vad du faktiskt fick)
  E   = p·W − (1 − p)·L             (förväntat resultat per trade, i R)
  p*  = L / (W + L)                 (träffprocent som krävs för att gå jämnt ut)

Strategin tjänar pengar när p > p*, alltså när E > 0.

Stöd och motstånd hittas som tydliga botten- och toppunkter i grafen: en dag vars lägsta kurs
är lägre än de 3 dagarna före och efter (stöd), eller vars högsta är högre (motstånd).
En punkt räknas först när de 3 dagarna efter har passerat, så testet "ser" aldrig framtiden.

Kör:  python analys.py   (i GitHub: site/analys.html)
"""
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
import omx30_skanner as sk

AR = bt.AR
TRANING_AR = 2
K = 3                # dagar på varje sida för en topp/botten
FONSTER = 120        # hur långt bakåt stöd och motstånd letas (handelsdagar)
NARA_STOD = 4.0      # kursen får ligga högst så många % över stödet vid signal
MAXDAGAR = 40


def forbered(df: pd.DataFrame):
    c, h, l, o = df["Close"], df["High"], df["Low"], df["Open"]
    r = sk.rsi(c)
    linje, signal, hist = sk.macd(c)
    hist_min10 = hist.rolling(10).min()
    w = 2 * K + 1
    return {
        "o": o.to_numpy(), "h": h.to_numpy(), "l": l.to_numpy(), "c": c.to_numpy(), "datum": df.index,
        "linje": linje.to_numpy(), "signal": signal.to_numpy(),
        "pl": (l == l.rolling(w, center=True).min()).to_numpy(),
        "ph": (h == h.rolling(w, center=True).max()).to_numpy(),
        "vand": ((hist > hist.shift(1)) & (hist.shift(1) > hist.shift(2)) & (r > r.shift(1)) & (hist_min10 < 0)),
        "rsi_min5": r.rolling(5).min(),
        "dyk": (c.rolling(5).min() / h.rolling(20).max() - 1) * 100,
        "kraft": (hist - hist_min10) >= 0.3 * hist_min10.abs(),
        "botten5": (l.rolling(5).min() * (1 - sk.STOP_MARGINAL)).to_numpy(),
        "botten20": (l.rolling(20).min() * (1 - sk.STOP_MARGINAL)).to_numpy(),
        "status": sk.signalserie(c, l, h)["status"],
    }


def farsk(s: pd.Series):
    s = s.fillna(False).astype(bool)
    return np.flatnonzero((s & ~s.shift(1, fill_value=False)).to_numpy())


def stod_motstand(b, t):
    lo, hi = max(0, t - FONSTER), t - K
    if hi <= lo:
        return None, None
    i = np.arange(lo, hi + 1)
    c = b["c"][t]
    st = b["l"][i][b["pl"][i] & (b["l"][i] < c)]
    mo = b["h"][i][b["ph"][i] & (b["h"][i] > c * 1.005)]
    return (float(st.max()) if st.size else None), (float(mo.min()) if mo.size else None)


# ---------------------------------------------------------------------------
# Strategierna: varje funktion ger (signaldag, stop, mål eller None)
# ---------------------------------------------------------------------------
def s_nuvarande(b):
    return [(t, b["botten20"][t], None) for t in farsk(b["status"].isin([sk.TIDIG, sk.KORS]))]


def s_strikt(b):
    s = b["vand"] & (b["rsi_min5"] < 30) & (b["dyk"] <= -8) & b["kraft"]
    return [(t, b["botten5"][t], None) for t in farsk(s)]


def s_stod(b, rr_min, strikt=False):
    s = b["vand"] & (b["rsi_min5"] < (30 if strikt else 35))
    if strikt:
        s = s & (b["dyk"] <= -8) & b["kraft"]
    ut = []
    for t in farsk(s):
        stod, mot = stod_motstand(b, t)
        if stod is None or mot is None:
            continue
        c = b["c"][t]
        if (c - stod) / c * 100 > NARA_STOD:
            continue                                    # inte nära stödet
        stop = stod * (1 - sk.STOP_MARGINAL)
        if c <= stop or (mot - c) / (c - stop) < rr_min:
            continue                                    # klarar inte R/R-kravet vid signalen
        ut.append((t, stop, mot))
    return ut


def simulera(b, poster, utgang, start):
    o, h, l, c = b["o"], b["h"], b["l"], b["c"]
    linje, sign, datum = b["linje"], b["signal"], b["datum"]
    n, ut, ledig = len(c), [], 0
    for t, stop0, mal_abs in sorted(poster):
        if t >= n - 1 or t < ledig or datum[t] < start:
            continue
        entry = o[t + 1]
        if not (entry > stop0 > 0):
            continue
        risk = entry - stop0
        mal = {"2R": entry + 2 * risk, "3R": entry + 3 * risk, "motstand": mal_abs}.get(utgang)
        if utgang == "motstand" and (mal is None or mal <= entry):
            continue
        stop, hogsta, slut, pris = stop0, entry, None, None
        for d in range(t + 1, min(n, t + 1 + MAXDAGAR)):
            if l[d] <= stop:
                pris, slut = (o[d] if o[d] < stop else stop), d
                break
            if mal is not None and h[d] >= mal:
                pris, slut = (o[d] if o[d] > mal else mal), d
                break
            if utgang == "macd" and d > t + 1 and linje[d] < sign[d] and linje[d - 1] >= sign[d - 1]:
                pris, slut = c[d], d
                break
            hogsta = max(hogsta, h[d])
            if utgang in ("trail", "macd", "motstand_trail") and hogsta >= entry + risk:
                stop = max(stop, entry)
                if utgang in ("trail", "motstand_trail"):
                    stop = max(stop, hogsta - risk)
        if slut is None:
            if t + MAXDAGAR > n - 1:
                continue
            slut, pris = t + MAXDAGAR, c[t + MAXDAGAR]
        ut.append({"Köpdag": datum[t + 1], "Säljdag": datum[slut], "R": (pris - entry) / risk, "Dagar": slut - t})
        ledig = slut + 1
    return ut


def matt(tr, ar):
    if not tr:
        return None
    R = np.array([x["R"] for x in tr])
    vinst, forl = R[R > 0], R[R <= 0]
    p = len(vinst) / len(R)
    W = float(vinst.mean()) if len(vinst) else 0.0
    L = float(-forl.mean()) if len(forl) else 0.0
    ordn = sorted(tr, key=lambda x: x["Säljdag"])
    cum = np.cumsum([x["R"] for x in ordn])
    dd = float((np.maximum.accumulate(np.concatenate([[0], cum])) - np.concatenate([[0], cum])).max())
    langst, nu = 0, 0
    for x in ordn:
        nu = nu + 1 if x["R"] <= 0 else 0
        langst = max(langst, nu)
    return {
        "n": len(R), "per_ar": len(R) / ar, "p": p * 100, "W": W, "L": L,
        "rr": (W / L) if L > 0 else float("inf"), "E": float(R.mean()),
        "p_krav": (L / (W + L) * 100) if (W + L) > 0 else float("nan"),
        "pf": (vinst.sum() / -forl.sum()) if forl.sum() < 0 else float("inf"),
        "dd": dd, "forlust_i_rad": langst, "dagar": float(np.mean([x["Dagar"] for x in tr])),
    }


STRATEGIER = [
    ("nuv", "Nuvarande skanner", "Grön eller blå signal, stop under 20-dagars botten, sälj när MACD korsar ner.", lambda b: s_nuvarande(b), "macd"),
    ("strikt2", "Strikt dyk, mål +2R", "RSI under 30, dyk minst 8 %, kraftig MACD-vändning. Stop under dykets botten. Mål +2R.", s_strikt, "2R"),
    ("strikt3", "Strikt dyk, mål +3R", "Samma köp som ovan. Mål +3R.", s_strikt, "3R"),
    ("striktT", "Strikt dyk, släpande stop", "Samma köp. Stop till köpkurs vid +1R, sedan 1R under högsta kursen.", s_strikt, "trail"),
    ("sr15", "Stöd/motstånd, R/R ≥ 1,5", "MACD vänder upp nära ett stöd (RSI under 35). Stop under stödet, mål vid närmaste motstånd. Köp bara om R/R ≥ 1,5.", lambda b: s_stod(b, 1.5), "motstand"),
    ("sr2", "Stöd/motstånd, R/R ≥ 2", "Som ovan, men köp bara om R/R till motståndet är minst 2.", lambda b: s_stod(b, 2.0), "motstand"),
    ("sr3", "Stöd/motstånd, R/R ≥ 3", "Som ovan, men minst 3.", lambda b: s_stod(b, 3.0), "motstand"),
    ("sr2T", "Stöd/motstånd R/R ≥ 2, släpande", "Samma köp som R/R ≥ 2, men i stället för att sälja vid motståndet: släpande stop.", lambda b: s_stod(b, 2.0), "trail"),
    ("srS2", "Stöd/motstånd + strikt dyk, R/R ≥ 2", "Kombination: strikt dyk (RSI under 30, dyk 8 %, kraftig vändning) nära ett stöd, R/R ≥ 2 till motståndet.", lambda b: s_stod(b, 2.0, strikt=True), "motstand"),
]


def main():
    idag = date.today()
    start = pd.Timestamp(idag - timedelta(days=365 * AR))
    grans = pd.Timestamp(idag - timedelta(days=365 * (AR - TRANING_AR)))
    data = {}
    for namn, ticker in sk.BOLAG.items():
        try:
            df = bt.ladda(ticker)
            if df is not None:
                data[namn] = forbered(df)
        except Exception as e:
            print(f"  Fel för {namn}: {e}")
    print(f"Data för {len(data)} bolag.")

    res = []
    for kod, namn, text, fn, utgang in STRATEGIER:
        alla = []
        for b in data.values():
            alla += simulera(b, fn(b), utgang, start)
        tr = [t for t in alla if t["Köpdag"] < grans]
        te = [t for t in alla if t["Köpdag"] >= grans]
        res.append({"kod": kod, "namn": namn, "text": text,
                    "alla": matt(alla, AR), "traning": matt(tr, TRANING_AR), "test": matt(te, AR - TRANING_AR)})
        a = res[-1]["alla"]
        if a:
            print(f"{namn}: n={a['n']} p={a['p']:.0f}% W={a['W']:.2f} L={a['L']:.2f} RR={a['rr']:.2f} E={a['E']:+.2f}")

    bt.spara_json("analys.json", {"period": [str(start.date()), str(grans.date()), str(idag)], "bolag": len(data), "strategier": res})
    fil = Path("site") / "analys.html" if os.environ.get("GITHUB_ACTIONS") == "true" else Path(__file__).with_name("analys.html")
    fil.parent.mkdir(exist_ok=True)
    spara(fil, res, start, grans, idag, len(data))
    print(f"Sparat: {fil}")


def spara(fil, res, start, grans, idag, n_bolag):
    f, rr = bt._f, bt._r
    def cell(m, nyckel, fmt):
        if not m:
            return "<td>–</td>"
        v = m[nyckel]
        if fmt == "R":
            return f"<td class='{'up' if v > 0 else 'down'}'>{rr(v)}</td>"
        if fmt == "rr":
            return f"<td class='{'up' if v >= 2 else ''}'>1:{f(v, 2)}</td>"
        if fmt == "%":
            return f"<td>{f(v, 0)} %</td>"
        return f"<td>{f(v, fmt)}</td>"
    rader = ""
    for r in res:
        a, te = r["alla"], r["test"]
        rader += (f"<tr><td class='wrap'><b>{r['namn']}</b><br><span class='muted'>{r['text']}</span></td>"
                  + cell(a, "per_ar", 0) + cell(a, "p", "%") + cell(a, "p_krav", "%") + cell(a, "W", 2) + cell(a, "L", 2)
                  + cell(a, "rr", "rr") + cell(a, "E", "R") + cell(te, "E", "R") + cell(a, "dd", 1) + cell(a, "forlust_i_rad", 0) + "</tr>")
    html = f"""<!doctype html>
<html lang="sv"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Djupanalys av strategin</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,700&family=Instrument+Serif&display=swap">
<style>
:root{{--bg:#F4F1EA;--card:#FFFFFF;--fg:#1B1A17;--muted:#6B665C;--line:#E2DDD2;--up:#2F7A3E;--down:#A3321C;--accent:#2F4BD8;--hl:#EEF2FD;color-scheme:light}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161513;--card:#211F1C;--fg:#EEEAE2;--muted:#A8A295;--line:#33302B;--up:#8FD19C;--down:#F09A86;--accent:#9DB0FF;--hl:#1F2A4A;color-scheme:dark}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font-family:'DM Sans',system-ui,sans-serif;font-size:15px;line-height:1.45}}
.page{{max-width:1100px;margin:0 auto;padding:32px 20px 48px;display:flex;flex-direction:column;gap:28px}}
h1{{margin:0;font-family:'Instrument Serif',Georgia,serif;font-weight:400;font-size:clamp(34px,6vw,50px);line-height:1}}
h2{{margin:0 0 6px;font-size:21px}}
p{{margin:0 0 10px;max-width:75ch}}
.muted{{color:var(--muted)}}
a{{color:var(--accent);font-weight:600;text-decoration:none}}
.box{{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:18px}}
.formel{{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:14px;background:var(--hl);border-radius:10px;padding:10px 12px;margin:6px 0}}
.up{{color:var(--up);font-weight:600}}.down{{color:var(--down);font-weight:600}}
.tablewrap{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums;background:var(--card)}}
th,td{{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);white-space:nowrap;vertical-align:top}}
td.wrap{{white-space:normal;min-width:260px}}
th{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}}
</style></head><body><div class="page">
<header><a href="index.html">← Till skanningen</a>
<h1>Djupanalys: stöd, motstånd och R/R</h1>
<p class="muted">{n_bolag} bolag · {start:%Y-%m-%d} – {idag:%Y-%m-%d} · test = {grans:%Y-%m-%d} och framåt · uppdaterad {datetime.now():%Y-%m-%d}</p></header>
<section class="box"><h2>Formlerna</h2>
<div class="formel">E = p · W − (1 − p) · L</div>
<p>E är vad du tjänar i snitt per trade, mätt i R. p är träffprocenten, W snittvinsten och L snittförlusten.</p>
<div class="formel">p* = L / (W + L)</div>
<p>p* är träffprocenten du behöver för att gå jämnt ut. Med realiserad R/R 1:2 (W = 2, L = 1) räcker 33 %. Med 1:1 krävs 50 %.</p>
<div class="formel">Avkastning per år ≈ E × antal trades per år × 2 %</div>
<p>Med din regel om 2 % risk per trade.</p></section>
<section><h2>Jämförelse</h2>
<p class="muted">R/R här är den <b>realiserade</b>, alltså snittvinst delat med snittförlust – inte den planerade. "Test E" är det sista året, som reglerna inte "sett". Störst nedgång = värsta raset i R från en topp, i följd.</p>
<div class="tablewrap"><table>
<tr><th>Strategi</th><th>Per år</th><th>Träff p</th><th>Krävs p*</th><th>Snittvinst W</th><th>Snittförlust L</th><th>R/R</th><th>E per trade</th><th>Test E</th><th>Störst nedgång (R)</th><th>Förluster i rad</th></tr>
{rader}</table></div></section>
<section><h2>Begränsningar</h2><p class="muted">Courtage, spread och rapportregeln är inte med. Bolagen är dagens OMXS30. Stöd och motstånd är räknade med en enkel regel – en människa som tittar på grafen kan välja annorlunda. Historik är ingen garanti.</p></section>
</div></body></html>"""
    fil.write_text(html, encoding="utf-8")




# ---------------------------------------------------------------------------
# Max en ny trade i veckan, och granskning av enskilda bolag och datum
# ---------------------------------------------------------------------------
def sim_en(b, t, stop0, mal_abs, utgang):
    """En enda trade från signaldag t. Utgång 'halv' = hälften säljs vid +2R, resten med släpande stop."""
    o, h, l, c = b["o"], b["h"], b["l"], b["c"]
    linje, sign, datum = b["linje"], b["signal"], b["datum"]
    n = len(c)
    if t >= n - 1:
        return None
    entry = o[t + 1]
    if not (entry > stop0 > 0):
        return None
    risk = entry - stop0
    mal = {"2R": entry + 2 * risk, "3R": entry + 3 * risk, "motstand": mal_abs}.get(utgang)
    if utgang == "motstand" and (mal is None or mal <= entry):
        return None
    stop, hogsta, delvis = stop0, entry, None
    for d in range(t + 1, min(n, t + 1 + MAXDAGAR)):
        if l[d] <= stop:
            pris = o[d] if o[d] < stop else stop
            r = (pris - entry) / risk
            r = 0.5 * delvis + 0.5 * r if delvis is not None else r
            return {"Köpdag": datum[t + 1], "Säljdag": datum[d], "R": r, "Dagar": d - t}
        if mal is not None and h[d] >= mal:
            pris = o[d] if o[d] > mal else mal
            return {"Köpdag": datum[t + 1], "Säljdag": datum[d], "R": (pris - entry) / risk, "Dagar": d - t}
        if utgang == "halv" and delvis is None and h[d] >= entry + 2 * risk:
            delvis = 2.0
        if utgang == "macd" and d > t + 1 and linje[d] < sign[d] and linje[d - 1] >= sign[d - 1]:
            return {"Köpdag": datum[t + 1], "Säljdag": datum[d], "R": (c[d] - entry) / risk, "Dagar": d - t}
        hogsta = max(hogsta, h[d])
        if utgang in ("trail", "macd", "halv") and hogsta >= entry + risk:
            stop = max(stop, entry)
            if utgang in ("trail", "halv"):
                stop = max(stop, hogsta - risk)
    if t + MAXDAGAR > n - 1:
        return None
    d = t + MAXDAGAR
    r = (c[d] - entry) / risk
    r = 0.5 * delvis + 0.5 * r if delvis is not None else r
    return {"Köpdag": datum[t + 1], "Säljdag": datum[d], "R": r, "Dagar": d - t}


def planerad_rr(b, t, stop):
    _, mot = stod_motstand(b, t)
    c = b["c"][t]
    if mot is None or c <= stop:
        return None
    return (mot - c) / (c - stop)


VECKA = [
    ("Nuvarande skanner", lambda b: s_nuvarande(b), "macd"),
    ("Strikt dyk, mål +2R", s_strikt, "2R"),
    ("Strikt dyk, halva vid +2R, resten släpande", s_strikt, "halv"),
    ("Stöd/motstånd R/R ≥ 2, mål vid motståndet", lambda b: s_stod(b, 2.0), "motstand"),
    ("Stöd/motstånd R/R ≥ 2, halva vid +2R, resten släpande", lambda b: s_stod(b, 2.0), "halv"),
    ("Stöd/motstånd R/R ≥ 3, mål vid motståndet", lambda b: s_stod(b, 3.0), "motstand"),
]


def en_i_veckan(data, start, grans):
    """Välj högst EN ny trade per vecka: den signal som har högst planerad R/R till närmaste motstånd."""
    ut = []
    for namn, fn, utgang in VECKA:
        kandidater = []
        for bolag, b in data.items():
            for t, stop, mal in fn(b):
                if t >= len(b["c"]) - 1 or b["datum"][t] < start:
                    continue
                rr = planerad_rr(b, t, stop)
                kandidater.append((b["datum"][t], bolag, t, stop, mal, rr if rr is not None else -1))
        veckor = {}
        for k in kandidater:
            nyckel = k[0].isocalendar()[:2]
            if nyckel not in veckor or k[5] > veckor[nyckel][5]:
                veckor[nyckel] = k
        trades = []
        for dag, bolag, t, stop, mal, rr in sorted(veckor.values()):
            tr = sim_en(data[bolag], t, stop, mal, utgang)
            if tr:
                tr["Bolag"] = bolag
                tr["Planerad RR"] = rr
                trades.append(tr)
        tr_d = [x for x in trades if x["Köpdag"] < grans]
        te_d = [x for x in trades if x["Köpdag"] >= grans]
        ut.append({"namn": namn, "alla": matt(trades, AR), "traning": matt(tr_d, TRANING_AR),
                   "test": matt(te_d, AR - TRANING_AR),
                   "senaste": [{k: (str(v.date()) if hasattr(v, "date") else v) for k, v in x.items()} for x in trades[-8:]]})
    return ut


GRANSKA = {
    "Nordea": ("2026-10-06", None),
    "Addtech B": ("2026-10-01", "2026-10-14"),
    "EQT": ("2026-09-24", "2026-10-07"),
}


def granska(data):
    ut = {}
    for bolag, (fran, till) in GRANSKA.items():
        b = data.get(bolag)
        if b is None:
            continue
        rader = []
        for t, d in enumerate(b["datum"]):
            if d < pd.Timestamp(fran) or (till and d > pd.Timestamp(till)):
                continue
            stod, mot = stod_motstand(b, t)
            c = float(b["c"][t])
            stop_sr = stod * (1 - sk.STOP_MARGINAL) if stod else None
            rr = (mot - c) / (c - stop_sr) if (stod and mot and c > stop_sr) else None
            vand, kraft = bool(b["vand"].iloc[t]), bool(b["kraft"].iloc[t])
            rsi5, dyk = float(b["rsi_min5"].iloc[t]), float(b["dyk"].iloc[t])
            rader.append({
                "datum": str(d.date()), "kurs": round(c, 2), "rsi_min5": round(rsi5, 1), "dyk_%": round(dyk, 1),
                "macd_vander": vand, "kraftig": kraft, "skannern": b["status"].iloc[t],
                "stod": round(stod, 2) if stod else None, "motstand": round(mot, 2) if mot else None,
                "rr_till_motstand": round(rr, 2) if rr else None,
                "nara_stod": bool(stod and (c - stod) / c * 100 <= NARA_STOD),
                "strikt_dyk": bool(vand and rsi5 < 30 and dyk <= -8 and kraft),
                "sr_rr2": bool(vand and rsi5 < 35 and stod and (c - stod) / c * 100 <= NARA_STOD and rr and rr >= 2),
            })
        ut[bolag] = rader
    return ut


def main_extra():
    idag = date.today()
    start = pd.Timestamp(idag - timedelta(days=365 * AR))
    grans = pd.Timestamp(idag - timedelta(days=365 * (AR - TRANING_AR)))
    data = {}
    for namn, ticker in sk.BOLAG.items():
        try:
            df = bt.ladda(ticker)
            if df is not None:
                data[namn] = forbered(df)
        except Exception as e:
            print(f"  Fel för {namn}: {e}")
    bt.spara_json("vecka.json", en_i_veckan(data, start, grans))
    bt.spara_json("granskning.json", granska(data))
    print("Sparat vecka.json och granskning.json")


if __name__ == "__main__":
    main()
    main_extra()
