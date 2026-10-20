#!/usr/bin/env python3
"""
OMXS30-skanner – hittar bolag där RSI har varit lågt och MACD börjar vända upp.

Så kör du:
  1. Installera Python (python.org) om du inte har det.
  2. I terminalen:   pip install yfinance pandas
  3. Kör:            python omx30_skanner.py
     (helst efter 17:30, när dagens stängningskurser finns)

Resultatet skrivs ut i terminalen och sparas som omx30_skanning.html,
som öppnas i webbläsaren automatiskt.

Det här är ett verktyg för att hitta kandidater, inte köpråd.
Titta alltid själv på grafen, rapportdatum och R/R innan du köper.
"""
import json
import os
import sys
import webbrowser
from html import escape
from urllib import request as urlrequest
from datetime import date, datetime
from pathlib import Path

try:
    import pandas as pd
    import yfinance as yf
except ImportError:
    sys.exit("Installera först:  pip install yfinance pandas")

# ---------------------------------------------------------------------------
# OMXS30-bolagen (Yahoo-tickers). Listan ändras två gånger per år –
# kolla aktuell lista på Nasdaqs sida och lägg till eller ta bort här.
# ---------------------------------------------------------------------------
BOLAG = {
    "ABB": "ABB.ST",
    "Addtech B": "ADDT-B.ST",
    "Alfa Laval": "ALFA.ST",
    "Assa Abloy B": "ASSA-B.ST",
    "AstraZeneca": "AZN.ST",
    "Atlas Copco A": "ATCO-A.ST",
    "Atlas Copco B": "ATCO-B.ST",
    "Boliden": "BOL.ST",
    "Epiroc A": "EPI-A.ST",
    "EQT": "EQT.ST",
    "Ericsson B": "ERIC-B.ST",
    "Essity B": "ESSITY-B.ST",
    "Evolution": "EVO.ST",
    "Getinge B": "GETI-B.ST",
    "H&M B": "HM-B.ST",
    "Handelsbanken A": "SHB-A.ST",
    "Hexagon B": "HEXA-B.ST",
    "Investor B": "INVE-B.ST",
    "Lifco B": "LIFCO-B.ST",
    "Nibe B": "NIBE-B.ST",
    "Nordea": "NDA-SE.ST",
    "Saab B": "SAAB-B.ST",
    "Sandvik": "SAND.ST",
    "SCA B": "SCA-B.ST",
    "SEB A": "SEB-A.ST",
    "Skanska B": "SKA-B.ST",
    "SKF B": "SKF-B.ST",
    "Swedbank A": "SWED-A.ST",
    "Tele2 B": "TEL2-B.ST",
    "Telia": "TELIA.ST",
    "Volvo B": "VOLV-B.ST",
}

# ---------------------------------------------------------------------------
# Dina regler – ändra fritt
# ---------------------------------------------------------------------------
RSI_LAG = 40           # "lågt" RSI: under detta någon gång senaste 5 dagarna
RSI_OVERSALD = 35      # under detta just nu = översåld (bevakningslistan)
KORS_DAGAR = 2         # MACD-korsning räknas om den skett senaste så många dagar
DAGAR_BOTTEN = 20      # senaste bottnen (för stop-loss) letas inom så många dagar
DAGAR_TOPP = 40        # senaste toppen (som mål) letas inom så många dagar
STOP_MARGINAL = 0.01   # stop läggs 1 % under bottnen
MIN_RR = 2.0           # ditt minimum för R/R
MIN_DAGAR_RAPPORT = 15 # din regel: minst så många dagar kvar till rapport vid köp

# Rapportdatum som du vet själv (vinner över det Yahoo säger). Format "ÅÅÅÅ-MM-DD".
RAPPORTDATUM = {
    "Addtech B": "2026-10-22",
}

# Kategorier, i den ordning de visas
TIDIG = "1. MACD vänder upp (tidigt)"
KORS = "2. MACD-korsning"
OVERSALD = "3. Översålda – bevaka"
PRIO = {TIDIG: 0, KORS: 1, OVERSALD: 2, "": 9}
MIN_BOLAG_DATA = 20    # färre bolag med data än så = något är fel, behåll förra sidan


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    upp = delta.clip(lower=0)
    ner = -delta.clip(upper=0)
    snitt_upp = upp.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    snitt_ner = ner.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = snitt_upp / snitt_ner
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series):
    linje = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = linje.ewm(span=9, adjust=False).mean()
    return linje, signal, linje - signal


def kolumn(df: pd.DataFrame, namn: str) -> pd.Series:
    s = df[namn]
    if isinstance(s, pd.DataFrame):  # nyare yfinance kan ge flernivåkolumner
        s = s.iloc[:, 0]
    return s.dropna()


INDEX_TICKER = "^OMX"   # OMX Stockholm 30 på Yahoo


def marknad():
    """Läget för OMXS30-indexet: upp-, ned- eller sidledes trend."""
    try:
        df = yf.download(INDEX_TICKER, period="2y", interval="1d", progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        close = kolumn(df, "Close")
        if len(close) < 210:
            return None
        ma50, ma200 = close.rolling(50).mean(), close.rolling(200).mean()
        kurs, m50, m200 = float(close.iloc[-1]), float(ma50.iloc[-1]), float(ma200.iloc[-1])
        if kurs > m200 and m50 > m200:
            lage = "upp"
        elif kurs < m200 and m50 < m200:
            lage = "ned"
        else:
            lage = "sidled"
        return {
            "kurs": kurs, "ma50": m50, "ma200": m200, "lage": lage,
            "manad": (kurs / float(close.iloc[-22]) - 1) * 100,
            "rsi": float(rsi(close).iloc[-1]),
            "spark": [float(x) for x in close.iloc[-126:]],
            "ma200_spark": [float(x) for x in ma200.iloc[-126:]],
        }
    except Exception as e:
        print(f"Kunde inte hämta indexet: {e}")
        return None


def nasta_rapport(namn: str, ticker: str):
    """Nästa rapportdatum: först din egen lista, annars Yahoo. None om okänt."""
    idag = date.today()
    if namn in RAPPORTDATUM:
        try:
            d = datetime.strptime(RAPPORTDATUM[namn], "%Y-%m-%d").date()
            if d >= idag:
                return d
        except ValueError:
            pass
    try:
        cal = yf.Ticker(ticker).calendar
        datum = None
        if isinstance(cal, dict):
            datum = cal.get("Earnings Date")
        elif hasattr(cal, "loc"):
            datum = list(cal.loc["Earnings Date"])
        if datum is None:
            return None
        if not isinstance(datum, (list, tuple)):
            datum = [datum]
        framtida = []
        for x in datum:
            x = x.date() if hasattr(x, "date") and callable(x.date) else x
            if isinstance(x, date) and x >= idag:
                framtida.append(x)
        return min(framtida) if framtida else None
    except Exception:
        return None


def avanza_lank(namn: str):
    """Letar upp bolagets sida på Avanza. None om det inte går (då används TradingView)."""
    try:
        body = json.dumps({"query": namn, "searchFilter": {"types": ["STOCK"]},
                           "pagination": {"from": 0, "size": 8}}).encode("utf-8")
        req = urlrequest.Request("https://www.avanza.se/_api/search/filtered-search", data=body,
                                 headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
                                 method="POST")
        svar = json.loads(urlrequest.urlopen(req, timeout=10).read().decode("utf-8"))
        for hit in svar.get("hits", []):
            oid = hit.get("orderBookId")
            plats = str(hit.get("marketPlaceName") or "")
            if oid and ("Stockholm" in plats or not plats):
                slug = hit.get("urlSlugName") or "aktie"
                return f"https://www.avanza.se/aktier/om-aktien.html/{oid}/{slug}"
    except Exception:
        pass
    return None


def signalserie(close: pd.Series, low: pd.Series, high: pd.Series) -> pd.DataFrame:
    """Signalerna för varje dag i historiken – samma regler som i dagens skanning.
    Varje rad bygger bara på data fram till och med den dagen (används även av backtest.py)."""
    r = rsi(close)
    linje, signal, hist = macd(close)
    rsi_min5 = r.rolling(5).min()
    hist_vander = (hist < 0) & (hist > hist.shift(1)) & (hist.shift(1) > hist.shift(2))
    kors = (linje > signal) & (linje.shift(1) <= signal.shift(1))
    kors_upp = kors.copy()
    for i in range(1, KORS_DAGAR):
        kors_upp = kors_upp | kors.shift(i, fill_value=False)
    stop = low.rolling(DAGAR_BOTTEN).min() * (1 - STOP_MARGINAL)
    topp = high.rolling(DAGAR_TOPP).max()
    risk = close - stop
    rr = (topp - close) / risk.where(risk > 0)

    status = pd.Series("", index=close.index, dtype=object)
    status[(r < RSI_OVERSALD)] = OVERSALD
    status[hist_vander & (rsi_min5 < RSI_LAG) & (r >= r.shift(1))] = TIDIG
    status[kors_upp & (rsi_min5 < RSI_LAG + 10)] = KORS
    return pd.DataFrame({"rsi": r, "rsi_min5": rsi_min5, "linje": linje, "signal": signal, "hist": hist,
                         "status": status, "stop": stop, "topp": topp, "rr": rr})


def analysera(namn: str, ticker: str):
    df = yf.download(ticker, period="9mo", interval="1d", progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None
    close, low, high = kolumn(df, "Close"), kolumn(df, "Low"), kolumn(df, "High")
    if len(close) < 60:
        return None

    s = signalserie(close, low, high)
    linje, hist = s["linje"], s["hist"]
    kurs = float(close.iloc[-1])
    rsi_nu = float(s["rsi"].iloc[-1])
    rsi_min5 = float(s["rsi_min5"].iloc[-1])
    h0, h1 = float(hist.iloc[-1]), float(hist.iloc[-2])
    stop = float(s["stop"].iloc[-1])
    topp = float(s["topp"].iloc[-1])
    rr = float(s["rr"].iloc[-1]) if s["rr"].iloc[-1] == s["rr"].iloc[-1] else float("nan")
    status = s["status"].iloc[-1]

    return {
        "Bolag": namn,
        "Status": status,
        "Kurs": round(kurs, 2),
        "RSI": round(rsi_nu, 1),
        "RSI min 5d": round(rsi_min5, 1),
        "MACD-hist": "stiger" if h0 > h1 else "faller",
        "MACD-linje": "stiger" if linje.iloc[-1] > linje.iloc[-2] else "faller",
        "Stop (under botten)": round(stop, 2),
        "Mål (senaste topp)": round(topp, 2),
        "R/R": round(rr, 2) if rr == rr else None,
        "R/R ≥ min": "ja" if rr == rr and rr >= MIN_RR else "nej",
        "_ticker": ticker,
        "_spark": [round(float(x), 2) for x in close.iloc[-126:]],
        "_spark_datum": [d.strftime("%Y-%m-%d") for d in close.index[-126:]],
        "_spark_hog": [round(float(x), 2) for x in high.iloc[-126:]],
        "_datum": close.index[-1].strftime("%Y-%m-%d"),
        "_namn_sok": namn,
    }


# ---------------------------------------------------------------------------
# Resultatsidan
# ---------------------------------------------------------------------------
KATEGORIER = [
    (TIDIG, "tidig", "MACD vänder upp",
     f"Histogrammet är under noll men har stigit två dagar i rad, och RSI har varit under {RSI_LAG}. Din tidigaste signal."),
    (KORS, "kors", "MACD-korsning",
     f"MACD-linjen har korsat upp genom signallinjen de senaste {KORS_DAGAR} dagarna."),
    (OVERSALD, "oversald", "Översålda – bevaka",
     f"RSI under {RSI_OVERSALD}. Står det ↗ stiger på MACD har vändningen börjat – en dag till så hamnar den i grönt. ↘ faller betyder vänta."),
]


def _tal(v, dec=2):
    if v is None or (isinstance(v, float) and v != v):
        return "–"
    return f"{v:,.{dec}f}".replace(",", " ").replace(".", ",")


def _spark_svg(punkter, stop):
    if not punkter or len(punkter) < 2:
        return ""
    w, h, pad = 300, 60, 4
    lo, hi = min(min(punkter), stop), max(punkter)
    span = (hi - lo) or 1
    y = lambda v: h - pad - (v - lo) / span * (h - 2 * pad)
    steg = w / (len(punkter) - 1)
    linje = " ".join(f"{'M' if i == 0 else 'L'}{i * steg:.1f} {y(v):.1f}" for i, v in enumerate(punkter))
    yta = f"{linje} L{w} {h} L0 {h} Z"
    ys = y(stop)
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" preserveAspectRatio="none" aria-hidden="true">'
            f'<path d="{yta}" class="spark-area"></path>'
            f'<path d="{linje}" class="spark-line"></path>'
            f'<line x1="0" x2="{w}" y1="{ys:.1f}" y2="{ys:.1f}" class="spark-stop"></line></svg>')


MANADER = ["jan", "feb", "mar", "apr", "maj", "jun", "jul", "aug", "sep", "okt", "nov", "dec"]


def _graf_6m(punkter, datum, stop, mal):
    """6-månadersgraf i Avanza-stil: yta, prisskala till höger, månader under, stop och mål inritade."""
    if not punkter or len(punkter) < 2:
        return ""
    w, h = 600, 200
    lo = min(min(punkter), stop if stop == stop else min(punkter))
    hi = max(max(punkter), mal if mal == mal else max(punkter))
    pad = (hi - lo) * 0.06 or 1
    lo, hi = lo - pad, hi + pad
    yp = lambda v: (hi - v) / (hi - lo) * 100            # procent uppifrån
    steg = w / (len(punkter) - 1)
    linje = " ".join(f"{'M' if i == 0 else 'L'}{i * steg:.1f} {yp(v) * h / 100:.1f}" for i, v in enumerate(punkter))
    yta = f"{linje} L{w} {h} L0 {h} Z"
    upp = punkter[-1] >= punkter[0]
    andring = (punkter[-1] / punkter[0] - 1) * 100
    # rutnät + prisetiketter (4 nivåer)
    nivaer = [v for v in (lo + (hi - lo) * k / 4 for k in range(1, 4)) if abs(yp(v) - yp(punkter[-1])) > 8]
    rutnat = "".join(f'<line x1="0" x2="{w}" y1="{yp(v) * h / 100:.1f}" y2="{yp(v) * h / 100:.1f}" class="g-grid"></line>' for v in nivaer)
    etiketter = "".join(f'<span class="g-y" style="top:{yp(v):.1f}%">{_tal(v, 0 if v >= 100 else 1)}</span>' for v in nivaer)
    # månadsetiketter
    manader, forra = [], None
    for i, d in enumerate(datum):
        m = d[5:7]
        if forra is not None and m != forra:
            manader.append(f'<span class="g-x" style="left:{i / (len(datum) - 1) * 100:.1f}%">{MANADER[int(m) - 1]}</span>')
        forra = m
    linjer = ""
    extra = ""
    if stop == stop:
        linjer += f'<line x1="0" x2="{w}" y1="{yp(stop) * h / 100:.1f}" y2="{yp(stop) * h / 100:.1f}" class="g-stop"></line>'
        extra += f'<span class="g-tag stop" style="top:{yp(stop):.1f}%">Stop {_tal(stop, 1)}</span>'
    if mal == mal:
        linjer += f'<line x1="0" x2="{w}" y1="{yp(mal) * h / 100:.1f}" y2="{yp(mal) * h / 100:.1f}" class="g-mal"></line>'
        extra += f'<span class="g-tag mal" style="top:{yp(mal):.1f}%">Mål {_tal(mal, 1)}</span>'
    extra += f'<span class="g-tag nu {"up" if upp else "down"}" style="top:{yp(punkter[-1]):.1f}%">{_tal(punkter[-1], 1)}</span>'
    return f"""<div class="g">
  <div class="g-head"><span>6 mån</span><span class="{'up' if upp else 'down'}">{'+' if andring >= 0 else '−'}{_tal(abs(andring), 1)} %</span></div>
  <div class="g-plot">
    <svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" aria-hidden="true">{rutnat}<path d="{yta}" class="g-area {'up' if upp else 'down'}"></path>{linjer}<path d="{linje}" class="g-line {'up' if upp else 'down'}"></path></svg>
    {etiketter}{extra}
  </div>
  <div class="g-xaxis">{''.join(manader)}</div>
</div>"""


def _kort(r, klass):
    namn = escape(str(r["Bolag"]))
    tv = str(r.get("_ticker", "")).replace(".ST", "").replace("-", "_")
    rsi_v = r["RSI"] if r["RSI"] == r["RSI"] else 0
    rr = r["R/R"]
    rr_ok = rr is not None and rr == rr and rr >= MIN_RR
    pil = "↗ stiger" if r["MACD-hist"] == "stiger" else "↘ faller"
    linje_pil = "↗ stiger" if r.get("MACD-linje") == "stiger" else "↘ faller"
    return f"""<article class="card {klass}">
  <div class="card-top">
    <div><h3>{namn}</h3><span class="ticker">{escape(str(r.get('_ticker', '')))}</span></div>
    <div class="price">{_tal(r['Kurs'])} <small>kr</small></div>
  </div>
  {_graf_6m(r.get('_spark') or [], r.get('_spark_datum') or [], r['Stop (under botten)'], r['Mål (senaste topp)'])}
  <div class="stats">
    <div class="stat"><span class="lbl">RSI</span><span class="val">{_tal(r['RSI'], 1)}</span>
      <span class="bar"><i style="width:{max(0, min(100, rsi_v)):.0f}%"></i><b style="left:{RSI_OVERSALD}%"></b></span></div>
    <div class="stat"><span class="lbl">Histogram</span><span class="val {'up' if r['MACD-hist'] == 'stiger' else 'down'}">{pil}</span></div>
    <div class="stat"><span class="lbl">MACD-linje</span><span class="val {'up' if r.get('MACD-linje') == 'stiger' else 'down'}">{linje_pil}</span></div>
    <div class="stat"><span class="lbl">Stop</span><span class="val">{_tal(r['Stop (under botten)'])}</span></div>
    <div class="stat"><span class="lbl">Mål</span><span class="val">{_tal(r['Mål (senaste topp)'])}</span></div>
    <div class="stat"><span class="lbl">R/R</span><span class="val rr {'ok' if rr_ok else 'low'}">{_tal(rr)}</span></div>
  </div>
  {_rapport_rad(r)}
  {_lank(r, tv)}
</article>"""


LAGE_TEXT = {
    "upp": ("Upptrend", "Medvind för dina köp. Dippköp i en marknad som trendar uppåt brukar fungera bäst."),
    "ned": ("Nedtrend", "Motvind. Fler signaler misslyckas när hela marknaden faller – ta mindre positioner och håll hårt på 2R och stop."),
    "sidled": ("Sidled / osäker", "Ingen tydlig trend. Följ reglerna och var noga med stop."),
}


def _marknad_html(mk):
    if not mk:
        return '<section class="market"><p class="empty">Kunde inte hämta OMXS30-indexet i dag.</p></section>'
    rubrik, text = LAGE_TEXT[mk["lage"]]
    pts, ma = mk["spark"], mk["ma200_spark"]
    w, h, pad = 600, 90, 6
    alla = [v for v in pts + ma if v == v]
    lo, hi = min(alla), max(alla)
    span = (hi - lo) or 1
    y = lambda v: h - pad - (v - lo) / span * (h - 2 * pad)
    steg = w / (len(pts) - 1)
    linje = " ".join(f"{'M' if i == 0 else 'L'}{i * steg:.1f} {y(v):.1f}" for i, v in enumerate(pts))
    ma_pts = [(i, v) for i, v in enumerate(ma) if v == v]
    ma_linje = " ".join(f"{'M' if j == 0 else 'L'}{i * steg:.1f} {y(v):.1f}" for j, (i, v) in enumerate(ma_pts))
    mot_ma = (mk["kurs"] / mk["ma200"] - 1) * 100
    tecken = lambda v: ("+" if v >= 0 else "−") + _tal(abs(v), 1) + " %"
    return f"""<section class="market lage-{mk['lage']}">
  <div class="market-top">
    <div><span class="lbl">Marknaden just nu · OMXS30</span><h2>{rubrik}</h2><p>{text}</p></div>
    <div class="market-nums">
      <div class="stat"><span class="lbl">Index</span><span class="val">{_tal(mk['kurs'], 0)}</span></div>
      <div class="stat"><span class="lbl">1 månad</span><span class="val {'up' if mk['manad'] >= 0 else 'down'}">{tecken(mk['manad'])}</span></div>
      <div class="stat"><span class="lbl">Mot 200-dagars</span><span class="val {'up' if mot_ma >= 0 else 'down'}">{tecken(mot_ma)}</span></div>
      <div class="stat"><span class="lbl">RSI</span><span class="val">{_tal(mk['rsi'], 0)}</span></div>
    </div>
  </div>
  <svg class="market-spark" viewBox="0 0 {w} {h}" preserveAspectRatio="none" aria-hidden="true">
    <path d="{ma_linje}" class="ma-line"></path><path d="{linje}" class="idx-line"></path></svg>
  <p class="legend">Senaste 6 månaderna. Heldragen linje = index, streckad = 200-dagars medelvärde. Ligger index över den streckade linjen räknas det som upptrend.</p>
</section>"""


GRAF_JS = ""


def _rapport_rad(r):
    dagar, datum = r.get("_rapport_dagar"), r.get("_rapport_datum")
    if dagar is None:
        return '<div class="report unknown">Rapportdatum okänt – kolla själv innan köp</div>'
    if dagar < MIN_DAGAR_RAPPORT:
        return (f'<div class="report warn">⚠ Rapport om {dagar} {"dag" if dagar == 1 else "dagar"} ({datum}) '
                f'– för nära, din regel är minst {MIN_DAGAR_RAPPORT}</div>')
    return f'<div class="report ok">Rapport om {dagar} dagar ({datum})</div>'


def _lank(r, tv):
    if r.get("_avanza"):
        return f'<a class="chart" href="{escape(r["_avanza"])}" target="_blank" rel="noopener">Öppna i Avanza →</a>'
    return (f'<a class="chart" href="https://www.tradingview.com/chart/?symbol=OMXSTO:{escape(tv)}" '
            f'target="_blank" rel="noopener">Öppna graf (TradingView) →</a>')


def spara_html(rader: list, fil: Path, mk=None):
    nu = datetime.now()
    datum = next((r.get("_datum") for r in rader if r.get("_datum")), f"{nu:%Y-%m-%d}")
    sektioner, chips = [], []
    for status, klass, rubrik, text in KATEGORIER:
        grupp = [r for r in rader if r["Status"] == status]
        chips.append(f'<span class="chip {klass}"><b>{len(grupp)}</b> {rubrik.split(" –")[0].lower()}</span>')
        innehall = "".join(_kort(r, klass) for r in grupp) or '<p class="empty">Inga bolag i den här kategorin i dag.</p>'
        sektioner.append(f'<section><div class="sec-head"><h2><span class="dot {klass}"></span>{rubrik}</h2>'
                         f'<p>{text}</p></div><div class="grid">{innehall}</div></section>')
    ovriga = [r for r in rader if not r["Status"]]
    ovr_rader = "".join(
        f"<tr><td>{escape(str(r['Bolag']))}</td><td>{_tal(r['Kurs'])}</td><td>{_tal(r['RSI'], 1)}</td>"
        f"<td>{r['MACD-hist']}</td><td>{_tal(r['R/R'])}</td></tr>" for r in ovriga)
    html = f"""<!doctype html>
<html lang="sv"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OMXS30-skanning {datum}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,700&family=Instrument+Serif&display=swap">
<style>
:root{{--bg:#F4F1EA;--card:#FFFFFF;--fg:#1B1A17;--muted:#6B665C;--line:#E2DDD2;
--tidig:#2F7A3E;--tidig-bg:#DCEBD7;--kors:#2F4BD8;--kors-bg:#D9E2F7;--oversald:#8A5A0B;--oversald-bg:#F4E4C4;
--up:#2F7A3E;--down:#A3321C;color-scheme:light}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161513;--card:#211F1C;--fg:#EEEAE2;--muted:#A8A295;--line:#33302B;
--tidig:#8FD19C;--tidig-bg:#1F3322;--kors:#9DB0FF;--kors-bg:#1F2A4A;--oversald:#EBC890;--oversald-bg:#3A2C17;
--up:#8FD19C;--down:#F09A86;color-scheme:dark}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font-family:'DM Sans',system-ui,sans-serif;font-size:15px;line-height:1.45}}
.wrap{{max-width:1100px;margin:0 auto;padding:32px 20px 48px;display:flex;flex-direction:column;gap:32px}}
header{{display:flex;flex-direction:column;gap:10px}}
.eyebrow{{font-size:13px;color:var(--muted);letter-spacing:.04em;text-transform:uppercase}}
h1{{margin:0;font-family:'Instrument Serif',Georgia,serif;font-weight:400;font-size:clamp(36px,6vw,54px);line-height:1}}
.chips{{display:flex;flex-wrap:wrap;gap:8px}}
.chip{{padding:6px 12px;border-radius:999px;font-size:14px}}
.chip.tidig{{background:var(--tidig-bg);color:var(--tidig)}}.chip.kors{{background:var(--kors-bg);color:var(--kors)}}.chip.oversald{{background:var(--oversald-bg);color:var(--oversald)}}
section{{display:flex;flex-direction:column;gap:14px}}
.sec-head h2{{margin:0;font-size:20px;display:flex;align-items:center;gap:10px}}
.sec-head p{{margin:4px 0 0;color:var(--muted);font-size:14px;max-width:70ch}}
.dot{{width:12px;height:12px;border-radius:50%;display:inline-block}}
.dot.tidig{{background:var(--tidig)}}.dot.kors{{background:var(--kors)}}.dot.oversald{{background:var(--oversald)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:16px;display:flex;flex-direction:column;gap:12px;min-width:0}}
.card-top{{display:flex;justify-content:space-between;align-items:flex-start;gap:8px}}
.card h3{{margin:0;font-size:17px}}
.ticker{{font-size:12px;color:var(--muted)}}
.price{{font-size:22px;font-weight:700;font-variant-numeric:tabular-nums;white-space:nowrap}}
.price small{{font-size:13px;font-weight:500;color:var(--muted)}}
.spark{{width:100%;height:60px;display:block}}
.spark-line{{fill:none;stroke-width:2;vector-effect:non-scaling-stroke}}
.spark-area{{stroke:none;opacity:.14}}
.spark-stop{{stroke:var(--down);stroke-width:1;stroke-dasharray:4 4;vector-effect:non-scaling-stroke}}
.tidig .spark-line{{stroke:var(--tidig)}}.tidig .spark-area{{fill:var(--tidig)}}
.kors .spark-line{{stroke:var(--kors)}}.kors .spark-area{{fill:var(--kors)}}
.oversald .spark-line{{stroke:var(--oversald)}}.oversald .spark-area{{fill:var(--oversald)}}
.stats{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px 14px}}
.stat{{display:flex;flex-direction:column;gap:2px}}
.lbl{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}}
.val{{font-size:15px;font-weight:600;font-variant-numeric:tabular-nums}}
.val.up{{color:var(--up)}}.val.down{{color:var(--down)}}
.rr.ok{{color:var(--up)}}.rr.low{{color:var(--muted)}}
.bar{{position:relative;height:6px;border-radius:3px;background:var(--line);margin-top:4px}}
.bar i{{position:absolute;inset:0 auto 0 0;border-radius:3px;background:var(--fg);opacity:.6}}
.bar b{{position:absolute;top:-3px;width:2px;height:12px;background:var(--down)}}
.chart{{align-self:flex-start;color:var(--kors);font-weight:600;font-size:14px;text-decoration:none}}
.chart:hover{{text-decoration:underline}}
.report{{font-size:13px;padding:8px 10px;border-radius:10px}}
.report.ok{{background:var(--bg);color:var(--muted)}}
.report.warn{{background:var(--oversald-bg);color:var(--down);font-weight:600}}
.report.unknown{{background:var(--bg);color:var(--muted);font-style:italic}}
.empty{{margin:0;color:var(--muted);font-size:14px;padding:14px 16px;border:1px dashed var(--line);border-radius:14px}}
details{{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:14px 18px}}
summary{{cursor:pointer;font-weight:600}}
.tablewrap{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;margin-top:12px;font-size:14px;font-variant-numeric:tabular-nums}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}}
th{{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}}
footer{{color:var(--muted);font-size:13px}}
.g{{display:flex;flex-direction:column;gap:4px}}
.g-head{{display:flex;justify-content:space-between;font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}}
.g-head .up{{color:var(--up);font-weight:600}}.g-head .down{{color:var(--down);font-weight:600}}
.g-plot{{position:relative;height:150px;margin-right:46px}}
.g-plot svg{{position:absolute;inset:0;width:100%;height:100%;overflow:visible}}
.g-grid{{stroke:var(--line);stroke-width:1;vector-effect:non-scaling-stroke}}
.g-line{{fill:none;stroke-width:2;vector-effect:non-scaling-stroke}}
.g-line.up{{stroke:var(--up)}}.g-line.down{{stroke:var(--down)}}
.g-area{{stroke:none;opacity:.12}}.g-area.up{{fill:var(--up)}}.g-area.down{{fill:var(--down)}}
.g-stop{{stroke:var(--down);stroke-width:1;stroke-dasharray:4 4;vector-effect:non-scaling-stroke}}
.g-mal{{stroke:var(--up);stroke-width:1;stroke-dasharray:2 4;vector-effect:non-scaling-stroke}}
.g-y{{position:absolute;right:-46px;transform:translateY(-50%);font-size:10px;color:var(--muted);font-variant-numeric:tabular-nums}}
.g-tag{{position:absolute;right:-46px;transform:translateY(-50%);font-size:10px;font-weight:700;padding:1px 4px;border-radius:4px;white-space:nowrap;font-variant-numeric:tabular-nums}}
.g-tag.nu.up{{background:var(--up);color:var(--card)}}.g-tag.nu.down{{background:var(--down);color:var(--card)}}
.g-tag.stop{{right:4px;color:var(--down);background:var(--card);opacity:.9}}
.g-tag.mal{{right:4px;color:var(--up);background:var(--card);opacity:.9}}
.g-xaxis{{position:relative;height:14px;margin-right:46px}}
.g-x{{position:absolute;transform:translateX(-50%);font-size:10px;color:var(--muted)}}
.links{{display:flex;flex-wrap:wrap;gap:6px 18px}}
.toplink{{align-self:flex-start;color:var(--kors);font-weight:600;font-size:14px;text-decoration:none}}
.toplink:hover{{text-decoration:underline}}
.market{{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:18px;gap:12px}}
.market-top{{display:flex;flex-wrap:wrap;justify-content:space-between;gap:16px}}
.market-top > div:first-child{{flex:1 1 320px;min-width:0}}
.market h2{{margin:4px 0 4px;font-family:'Instrument Serif',Georgia,serif;font-weight:400;font-size:34px;line-height:1.05}}
.market p{{margin:0;color:var(--muted);font-size:14px;max-width:60ch}}
.lage-upp h2{{color:var(--up)}}.lage-ned h2{{color:var(--down)}}.lage-sidled h2{{color:var(--oversald)}}
.market-nums{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px 22px;align-content:start}}
.market-spark{{width:100%;height:90px;display:block}}
.idx-line{{fill:none;stroke:var(--fg);stroke-width:2;vector-effect:non-scaling-stroke}}
.ma-line{{fill:none;stroke:var(--muted);stroke-width:1.5;stroke-dasharray:5 5;vector-effect:non-scaling-stroke}}
.legend{{font-size:12px!important}}
</style></head>
<body><div class="wrap">
<header>
  <span class="eyebrow">Stängningskurser {datum} · {len(rader)} av {len(BOLAG)} bolag · skannat {nu:%H:%M}</span>
  <h1>OMXS30-skanning</h1>
  <div class="chips">{''.join(chips)}</div>
  <div class="links"><a class="toplink" href="backtest.html">Så har strategin fungerat de senaste 3 åren →</a>
  <a class="toplink" href="optimering.html">Så kan strategin bli bättre →</a>
  <a class="toplink" href="analys.html">Djupanalys: stöd, motstånd och R/R →</a></div>
</header>
{_marknad_html(mk)}
{''.join(sektioner)}
<details><summary>Övriga bolag ({len(ovriga)})</summary>
<div class="tablewrap"><table><tr><th>Bolag</th><th>Kurs</th><th>RSI</th><th>MACD-hist</th><th>R/R</th></tr>{ovr_rader}</table></div></details>
<footer>Histogram = avståndet mellan MACD-linjen och signallinjen. Det kan stiga medan MACD-linjen fortfarande faller – då faller kursen långsammare, och det är just den tidiga vändningen.<br>
Röd streckad linje i grafen = föreslagen stop (1 % under senaste bottnen). R/R räknas mot senaste toppen; grönt betyder minst {_tal(MIN_RR, 1)}.
Ett verktyg för att hitta kandidater, inte köpråd. Kolla alltid grafen och rapportdatum innan du köper.</footer>
</div>{GRAF_JS}</body></html>"""
    fil.write_text(html, encoding="utf-8")


def skicka_notis(titel: str, text: str, prio: int = 3):
    """Skickar en pushnotis via ntfy om NTFY_TOPIC är satt (i GitHub: ett secret)."""
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        print("Ingen NTFY_TOPIC – hoppar över notis.")
        Path("notis_status.txt").write_text("ingen kanal", encoding="utf-8")
        return
    data = {"topic": topic, "title": titel, "message": text, "priority": prio, "tags": ["chart_with_upwards_trend"]}
    sida = os.environ.get("SIDA_URL", "").strip()
    if sida:
        data["click"] = sida.lower()
    try:
        req = urlrequest.Request("https://ntfy.sh/", data=json.dumps(data).encode("utf-8"),
                                 headers={"Content-Type": "application/json", "User-Agent": "omx-skanner/1.0"},
                                 method="POST")
        svar = urlrequest.urlopen(req, timeout=15).read().decode("utf-8", "replace")
        print(f"Notis skickad: {svar[:200]}")
        Path("notis_status.txt").write_text("skickad", encoding="utf-8")
    except Exception as e:
        print(f"Kunde inte skicka notis: {e}")
        Path("notis_status.txt").write_text(f"fel: {e}", encoding="utf-8")


def notis_for_signaler(rader: list, mk=None):
    signaler = [r for r in rader if r["Status"] in (TIDIG, KORS)]
    pa_vag = [r for r in rader if r["Status"] == OVERSALD and r["MACD-hist"] == "stiger"]
    if not signaler and not pa_vag:
        if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
            skicka_notis("OMXS30: testkörning klar", "Inga signaler just nu. Notiserna fungerar.", prio=2)
        return  # tyst dag – ingen notis
    rader_txt = []
    for r in signaler:
        ikon = "🟢" if r["Status"] == TIDIG else "🔵"
        typ = "MACD vänder upp" if r["Status"] == TIDIG else "MACD-korsning"
        rr = f"R/R {_tal(r['R/R'], 1)}" if r["R/R"] is not None else ""
        d = r.get("_rapport_dagar")
        rapport = f" · ⚠ rapport om {d} d" if d is not None and d < MIN_DAGAR_RAPPORT else ""
        rader_txt.append(f"{ikon} {r['Bolag']}: {typ} · RSI {r['RSI']:.0f} · {rr}{rapport}")
    for r in pa_vag:
        rader_txt.append(f"🟡 {r['Bolag']}: översåld, histogrammet börjar vända · RSI {r['RSI']:.0f}")
    titel = f"OMXS30: {len(signaler)} signaler, {len(pa_vag)} på väg" if signaler else f"OMXS30: {len(pa_vag)} på väg att vända"
    if mk:
        rader_txt.append(f"Marknaden: {LAGE_TEXT[mk['lage']][0].lower()}")
    skicka_notis(titel, "\n".join(rader_txt), prio=4 if signaler else 3)


def main():
    print(f"Skannar {len(BOLAG)} bolag …")
    resultat = []
    for namn, ticker in BOLAG.items():
        try:
            r = analysera(namn, ticker)
            if r:
                resultat.append(r)
            else:
                print(f"  Ingen data för {namn} ({ticker}) – kolla tickern.")
        except Exception as e:  # ett bolag som strular ska inte stoppa resten
            print(f"  Fel för {namn}: {e}")

    if len(resultat) < MIN_BOLAG_DATA:
        skicka_notis("OMXS30-skanningen misslyckades",
                     f"Fick bara data för {len(resultat)} av {len(BOLAG)} bolag (Yahoo strulade). Sidan behåller gårdagens resultat.", prio=2)
        sys.exit(f"Fick bara data för {len(resultat)} bolag – sidan uppdateras inte.")

    tabell = pd.DataFrame(resultat)
    tabell["_prio"] = tabell["Status"].map(PRIO)
    tabell = tabell.sort_values(["_prio", "RSI"]).drop(columns="_prio")

    pd.set_option("display.width", 200)
    print()
    print(tabell.drop(columns=[c for c in tabell.columns if c.startswith("_")]).to_string(index=False))

    rader = tabell.to_dict("records")
    # Rapportdatum och Avanza-länk hämtas bara för bolagen som syns som kort
    for r in rader:
        if not r["Status"]:
            continue
        d = nasta_rapport(r["_namn_sok"], r["_ticker"])
        r["_rapport_datum"] = d.isoformat() if d else None
        r["_rapport_dagar"] = (d - date.today()).days if d else None
        r["_avanza"] = avanza_lank(r["_namn_sok"])
    i_molnet = os.environ.get("GITHUB_ACTIONS") == "true"
    if i_molnet:
        mapp = Path("site")
        mapp.mkdir(exist_ok=True)
        fil = mapp / "index.html"
    else:
        fil = Path(__file__).with_name("omx30_skanning.html")
    mk = marknad()
    spara_html(rader, fil, mk)
    try:
        Path("results").mkdir(exist_ok=True)
        (Path("results") / "skanning.json").write_text(json.dumps({
            "tid": datetime.now().isoformat(timespec="minutes"), "bolag_med_data": len(rader),
            "saknas": [n for n in BOLAG if n not in {r["Bolag"] for r in rader}],
            "marknad": mk["lage"] if mk else None,
            "rader": [{"Bolag": r["Bolag"], "Status": r["Status"], "Kurs": r["Kurs"], "RSI": r["RSI"],
                       "Datum": r.get("_datum")} for r in rader],
        }, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:
        print(f"Kunde inte spara skanning.json: {e}")
    print(f"\nSparat: {fil}")
    notis_for_signaler(rader, mk)
    if not i_molnet:
        try:
            webbrowser.open(fil.resolve().as_uri())
        except Exception:
            pass


if __name__ == "__main__":
    main()
