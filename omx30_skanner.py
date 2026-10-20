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
from datetime import datetime
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

# Kategorier, i den ordning de visas
TIDIG = "1. MACD vänder upp (tidigt)"
KORS = "2. MACD-korsning"
OVERSALD = "3. Översålda – bevaka"
PRIO = {TIDIG: 0, KORS: 1, OVERSALD: 2, "": 9}


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


def analysera(namn: str, ticker: str):
    df = yf.download(ticker, period="9mo", interval="1d", progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None
    close, low, high = kolumn(df, "Close"), kolumn(df, "Low"), kolumn(df, "High")
    if len(close) < 60:
        return None

    r = rsi(close)
    linje, signal, hist = macd(close)

    kurs = float(close.iloc[-1])
    rsi_nu, rsi_igar = float(r.iloc[-1]), float(r.iloc[-2])
    rsi_min5 = float(r.iloc[-5:].min())
    h0, h1, h2 = float(hist.iloc[-1]), float(hist.iloc[-2]), float(hist.iloc[-3])
    hist_vander = h0 < 0 and h0 > h1 > h2           # under noll men stiger två dagar i rad
    kors_upp = any(
        linje.iloc[-i] > signal.iloc[-i] and linje.iloc[-i - 1] <= signal.iloc[-i - 1]
        for i in range(1, KORS_DAGAR + 1)
    )

    botten = float(low.iloc[-DAGAR_BOTTEN:].min())
    topp = float(high.iloc[-DAGAR_TOPP:].max())
    stop = botten * (1 - STOP_MARGINAL)
    risk = kurs - stop
    rr = (topp - kurs) / risk if risk > 0 else float("nan")

    if kors_upp and rsi_min5 < RSI_LAG + 10:
        status = KORS
    elif hist_vander and rsi_min5 < RSI_LAG and rsi_nu >= rsi_igar:
        status = TIDIG
    elif rsi_nu < RSI_OVERSALD:
        status = OVERSALD
    else:
        status = ""

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
        "_spark": [round(float(x), 2) for x in close.iloc[-60:]],
        "_datum": close.index[-1].strftime("%Y-%m-%d"),
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
  {_spark_svg(r.get('_spark') or [], r['Stop (under botten)'])}
  <div class="stats">
    <div class="stat"><span class="lbl">RSI</span><span class="val">{_tal(r['RSI'], 1)}</span>
      <span class="bar"><i style="width:{max(0, min(100, rsi_v)):.0f}%"></i><b style="left:{RSI_OVERSALD}%"></b></span></div>
    <div class="stat"><span class="lbl">Histogram</span><span class="val {'up' if r['MACD-hist'] == 'stiger' else 'down'}">{pil}</span></div>
    <div class="stat"><span class="lbl">MACD-linje</span><span class="val {'up' if r.get('MACD-linje') == 'stiger' else 'down'}">{linje_pil}</span></div>
    <div class="stat"><span class="lbl">Stop</span><span class="val">{_tal(r['Stop (under botten)'])}</span></div>
    <div class="stat"><span class="lbl">Mål</span><span class="val">{_tal(r['Mål (senaste topp)'])}</span></div>
    <div class="stat"><span class="lbl">R/R</span><span class="val rr {'ok' if rr_ok else 'low'}">{_tal(rr)}</span></div>
  </div>
  <a class="chart" href="https://www.tradingview.com/chart/?symbol=OMXSTO:{escape(tv)}" target="_blank" rel="noopener">Öppna graf →</a>
</article>"""


def spara_html(rader: list, fil: Path):
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
.empty{{margin:0;color:var(--muted);font-size:14px;padding:14px 16px;border:1px dashed var(--line);border-radius:14px}}
details{{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:14px 18px}}
summary{{cursor:pointer;font-weight:600}}
.tablewrap{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;margin-top:12px;font-size:14px;font-variant-numeric:tabular-nums}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}}
th{{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}}
footer{{color:var(--muted);font-size:13px}}
</style></head>
<body><div class="wrap">
<header>
  <span class="eyebrow">Stängningskurser {datum} · skannat {nu:%H:%M}</span>
  <h1>OMXS30-skanning</h1>
  <div class="chips">{''.join(chips)}</div>
</header>
{''.join(sektioner)}
<details><summary>Övriga bolag ({len(ovriga)})</summary>
<div class="tablewrap"><table><tr><th>Bolag</th><th>Kurs</th><th>RSI</th><th>MACD-hist</th><th>R/R</th></tr>{ovr_rader}</table></div></details>
<footer>Histogram = avståndet mellan MACD-linjen och signallinjen. Det kan stiga medan MACD-linjen fortfarande faller – då faller kursen långsammare, och det är just den tidiga vändningen.<br>
Röd streckad linje i grafen = föreslagen stop (1 % under senaste bottnen). R/R räknas mot senaste toppen; grönt betyder minst {_tal(MIN_RR, 1)}.
Ett verktyg för att hitta kandidater, inte köpråd. Kolla alltid grafen och rapportdatum innan du köper.</footer>
</div></body></html>"""
    fil.write_text(html, encoding="utf-8")


def skicka_notis(titel: str, text: str, prio: int = 3):
    """Skickar en pushnotis via ntfy om NTFY_TOPIC är satt (i GitHub: ett secret)."""
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        return
    data = {"topic": topic, "title": titel, "message": text, "priority": prio, "tags": ["chart_with_upwards_trend"]}
    sida = os.environ.get("SIDA_URL", "").strip()
    if sida:
        data["click"] = sida
    try:
        req = urlrequest.Request("https://ntfy.sh/", data=json.dumps(data).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
        urlrequest.urlopen(req, timeout=15).read()
        print("Notis skickad.")
    except Exception as e:
        print(f"Kunde inte skicka notis: {e}")


def notis_for_signaler(rader: list):
    signaler = [r for r in rader if r["Status"] in (TIDIG, KORS)]
    pa_vag = [r for r in rader if r["Status"] == OVERSALD and r["MACD-hist"] == "stiger"]
    if not signaler and not pa_vag:
        return  # tyst dag – ingen notis
    rader_txt = []
    for r in signaler:
        ikon = "🟢" if r["Status"] == TIDIG else "🔵"
        typ = "MACD vänder upp" if r["Status"] == TIDIG else "MACD-korsning"
        rr = f"R/R {_tal(r['R/R'], 1)}" if r["R/R"] is not None else ""
        rader_txt.append(f"{ikon} {r['Bolag']}: {typ} · RSI {r['RSI']:.0f} · {rr}")
    for r in pa_vag:
        rader_txt.append(f"🟡 {r['Bolag']}: översåld, histogrammet börjar vända · RSI {r['RSI']:.0f}")
    titel = f"OMXS30: {len(signaler)} signaler, {len(pa_vag)} på väg" if signaler else f"OMXS30: {len(pa_vag)} på väg att vända"
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

    if not resultat:
        skicka_notis("OMXS30-skanningen misslyckades", "Fick ingen kursdata i dag. Försök igen senare eller kör den på datorn.", prio=2)
        sys.exit("Fick ingen data alls. Kolla internetanslutningen.")

    tabell = pd.DataFrame(resultat)
    tabell["_prio"] = tabell["Status"].map(PRIO)
    tabell = tabell.sort_values(["_prio", "RSI"]).drop(columns="_prio")

    pd.set_option("display.width", 200)
    print()
    print(tabell.drop(columns=[c for c in tabell.columns if c.startswith("_")]).to_string(index=False))

    rader = tabell.to_dict("records")
    i_molnet = os.environ.get("GITHUB_ACTIONS") == "true"
    if i_molnet:
        mapp = Path("site")
        mapp.mkdir(exist_ok=True)
        fil = mapp / "index.html"
    else:
        fil = Path(__file__).with_name("omx30_skanning.html")
    spara_html(rader, fil)
    print(f"\nSparat: {fil}")
    notis_for_signaler(rader)
    if not i_molnet:
        try:
            webbrowser.open(fil.resolve().as_uri())
        except Exception:
            pass


if __name__ == "__main__":
    main()
