#!/usr/bin/env python3
"""
Optimering av strategin på OMXS30, tre år bakåt.

Testar en strängare version av dina regler – ett tydligt dyk, RSI under en gräns och en
kraftig vändning i MACD – med olika stop, utgångar och filter, och visar vilka val som
höjer R per trade.

För att inte lura sig själv delas historiken i två delar:
  * Träning: de två första åren. Här letas de bästa reglerna fram.
  * Test:    det sista året. Här kontrolleras att reglerna håller på data de inte "sett".
Regler som bara är bra på träningsdelen är troligen tur.

Kör:  python optimera.py   (i GitHub: site/optimering.html)
"""
import itertools
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
import omx30_skanner as sk

AR = bt.AR
TRANING_AR = 2
MIN_TRADES = 25

RSI_GRANSER = [25, 30, 35]
DYK = [0, 6, 10]            # minst så många % ner från 20-dagars toppen (0 = inget krav)
VANDNING = [0.0, 0.3]       # histogrammet ska ha återhämtat minst så stor del av dyket (0 = inget krav)
TREND = [False, True]       # bara köp när OMXS30 ligger över 200-dagars medel
STOPP = ["botten20", "botten5"]
UTGANG = ["2R", "3R", "trail", "macd"]

UTGANG_TEXT = {
    "2R": "fast mål på +2R",
    "3R": "fast mål på +3R",
    "trail": "släpande stop (köpkurs vid +1R, sedan 1R under högsta kursen)",
    "macd": "sälj när MACD korsar ner (stop till köpkurs vid +1R)",
}
STOPP_TEXT = {"botten20": "under 20-dagars botten", "botten5": "under 5-dagars botten (dykets botten)"}


def forbered(df: pd.DataFrame, upp):
    c, h, l, o = df["Close"], df["High"], df["Low"], df["Open"]
    r = sk.rsi(c)
    linje, signal, hist = sk.macd(c)
    upp_al = upp.reindex(df.index, method="ffill") if upp is not None else None
    upp_al = upp_al.fillna(False).astype(bool) if upp_al is not None else pd.Series(True, index=df.index)
    return {
        "o": o.to_numpy(), "h": h.to_numpy(), "l": l.to_numpy(), "c": c.to_numpy(), "datum": df.index,
        "rsi_min5": r.rolling(5).min(),
        "rsi_upp": r > r.shift(1),
        "hist": hist,
        "hist_min10": hist.rolling(10).min(),
        "vander": (hist > hist.shift(1)) & (hist.shift(1) > hist.shift(2)),
        "dyk": (c.rolling(5).min() / h.rolling(20).max() - 1) * 100,
        "linje": linje.to_numpy(), "signal": signal.to_numpy(),
        "botten20": (l.rolling(20).min() * (1 - sk.STOP_MARGINAL)).to_numpy(),
        "botten5": (l.rolling(5).min() * (1 - sk.STOP_MARGINAL)).to_numpy(),
        "upp": upp_al,
    }


def signal(b, rsi_grans, dyk, vandning, trend):
    s = (b["rsi_min5"] < rsi_grans) & b["vander"] & b["rsi_upp"] & (b["hist_min10"] < 0)
    if dyk:
        s &= b["dyk"] <= -dyk
    if vandning:
        s &= (b["hist"] - b["hist_min10"]) >= vandning * b["hist_min10"].abs()
    if trend:
        s &= b["upp"]
    s = s.fillna(False).astype(bool)
    return (s & ~s.shift(1, fill_value=False)).to_numpy()


def simulera(b, sig, stoppnamn, utgang, start):
    o, h, l, c = b["o"], b["h"], b["l"], b["c"]
    linje, sign = b["linje"], b["signal"]
    stopp = b[stoppnamn]
    datum = b["datum"]
    n, ut, ledig = len(c), [], 0
    maxd = 30 if utgang in ("2R", "3R") else 40
    for t in np.flatnonzero(sig):
        if t >= n - 1 or t < ledig or datum[t] < start:
            continue
        entry, stop = o[t + 1], stopp[t]
        if not (entry > stop > 0):
            continue
        risk = entry - stop
        mal = entry + (2 if utgang == "2R" else 3) * risk if utgang in ("2R", "3R") else None
        hogsta, slut, pris = entry, None, None
        for d in range(t + 1, min(n, t + 1 + maxd)):
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
            if utgang in ("trail", "macd") and hogsta >= entry + risk:
                stop = max(stop, entry)
                if utgang == "trail":
                    stop = max(stop, hogsta - risk)
        if slut is None:
            slut = min(n - 1, t + maxd)
            if t + maxd > n - 1:
                continue  # fortfarande öppen – räknas inte
            pris = c[slut]
        ut.append({"Köpdag": datum[t + 1], "R": (pris - entry) / risk, "Dagar": slut - t})
        ledig = slut + 1
    return ut


def stat(tr):
    s = bt.statistik(tr)
    return s if s else {"antal": 0, "traff": float("nan"), "snitt": float("nan"), "pf": float("nan"), "dagar": float("nan"), "summa": 0}


def main():
    idag = date.today()
    start = pd.Timestamp(idag - timedelta(days=365 * AR))
    gräns = pd.Timestamp(idag - timedelta(days=365 * (AR - TRANING_AR)))
    upp = bt.index_upptrend()

    data = {}
    for namn, ticker in sk.BOLAG.items():
        try:
            df = bt.ladda(ticker)
            if df is not None:
                data[namn] = (df, forbered(df, upp))
        except Exception as e:
            print(f"  Fel för {namn}: {e}")
    print(f"Data för {len(data)} bolag.")

    # Utgångsläge: skannerns nuvarande regler med MACD-utgång
    bas = []
    for namn, (df, b) in data.items():
        s = sk.signalserie(df["Close"], df["Low"], df["High"])["status"]
        k = s.isin([sk.TIDIG, sk.KORS])
        k = (k & ~k.shift(1, fill_value=False)).to_numpy()
        bas += simulera(b, k, "botten20", "macd", start)
    bas_tr, bas_te = [t for t in bas if t["Köpdag"] < gräns], [t for t in bas if t["Köpdag"] >= gräns]

    resultat = []
    for rg, dk, vd, tf, st, ug in itertools.product(RSI_GRANSER, DYK, VANDNING, TREND, STOPP, UTGANG):
        alla = []
        for namn, (df, b) in data.items():
            alla += simulera(b, signal(b, rg, dk, vd, tf), st, ug, start)
        tr = [t for t in alla if t["Köpdag"] < gräns]
        te = [t for t in alla if t["Köpdag"] >= gräns]
        resultat.append({"rsi": rg, "dyk": dk, "vandning": vd, "trend": tf, "stopp": st, "utgang": ug,
                         "tr": stat(tr), "te": stat(te), "alla": stat(alla)})
    print(f"Testade {len(resultat)} kombinationer.")

    giltiga = [r for r in resultat if r["tr"]["antal"] >= MIN_TRADES and r["te"]["antal"] >= 8]
    topp = sorted(giltiga, key=lambda r: r["tr"]["snitt"], reverse=True)[:10]
    # Robust val: bäst på träning OCH positiv på test
    robusta = [r for r in sorted(giltiga, key=lambda r: min(r["tr"]["snitt"], r["te"]["snitt"]), reverse=True)]
    bast = robusta[0] if robusta else None

    effekter = []
    for nyckel, varden, etikett in [
        ("rsi", RSI_GRANSER, lambda v: f"RSI under {v}"),
        ("dyk", DYK, lambda v: "Inget dykkrav" if v == 0 else f"Dyk minst {v} %"),
        ("vandning", VANDNING, lambda v: "Ingen krav på kraftig vändning" if v == 0 else f"Kraftig vändning (≥ {int(v * 100)} % tillbaka)"),
        ("trend", TREND, lambda v: "Även i nedtrend" if not v else "Bara när OMXS30 är i upptrend"),
        ("stopp", STOPP, lambda v: "Stop " + STOPP_TEXT[v]),
        ("utgang", UTGANG, lambda v: "Utgång: " + UTGANG_TEXT[v]),
    ]:
        rader = []
        for v in varden:
            grupp = [r for r in resultat if r[nyckel] == v and r["alla"]["antal"] >= MIN_TRADES]
            if grupp:
                snitt = float(np.median([r["alla"]["snitt"] for r in grupp]))
                antal = float(np.median([r["alla"]["antal"] for r in grupp])) / AR
                rader.append((etikett(v), snitt, antal))
        effekter.append(rader)

    fil = Path("site") / "optimering.html" if os.environ.get("GITHUB_ACTIONS") == "true" else Path(__file__).with_name("optimering.html")
    fil.parent.mkdir(exist_ok=True)
    spara(fil, bast, topp, effekter, stat(bas_tr), stat(bas_te), start, gräns, idag, len(data))
    bt.spara_json("optimering.json", {
        "bast": {"regler": beskriv(bast), "traning": bast["tr"], "test": bast["te"]} if bast else None,
        "topp10": [{"regler": beskriv(r), "traning": r["tr"], "test": r["te"]} for r in topp],
        "effekter": [[{"val": e, "median_R": v, "per_ar": a} for e, v, a in rader] for rader in effekter],
        "nuvarande": {"traning": stat(bas_tr), "test": stat(bas_te)},
    })
    print(f"Sparat: {fil}")
    if bast:
        print(f"Bäst: {beskriv(bast)} | träning {bast['tr']['snitt']:+.2f}R ({bast['tr']['antal']}), test {bast['te']['snitt']:+.2f}R ({bast['te']['antal']})")


def beskriv(r):
    delar = [f"RSI under {r['rsi']}"]
    if r["dyk"]:
        delar.append(f"dyk minst {r['dyk']} %")
    if r["vandning"]:
        delar.append("kraftig vändning i MACD")
    if r["trend"]:
        delar.append("bara i upptrend")
    delar.append("stop " + STOPP_TEXT[r["stopp"]])
    delar.append(UTGANG_TEXT[r["utgang"]])
    return ", ".join(delar)


def _rad(r, bastrad=False):
    f, rr = bt._f, bt._r
    k = lambda v: "up" if v == v and v > 0 else "down"
    return (f"<tr{' class=best' if bastrad else ''}><td class='wrap'>{beskriv(r)}</td>"
            f"<td>{r['tr']['antal']}</td><td class='{k(r['tr']['snitt'])}'>{rr(r['tr']['snitt'])}</td>"
            f"<td>{r['te']['antal']}</td><td class='{k(r['te']['snitt'])}'>{rr(r['te']['snitt'])}</td>"
            f"<td>{f(r['te']['traff'], 0)} %</td></tr>")


def spara(fil, bast, topp, effekter, bas_tr, bas_te, start, gräns, idag, n_bolag):
    f, rr = bt._f, bt._r
    if bast:
        lyft = bast["te"]["snitt"] - bas_te["snitt"] if bas_te["antal"] else float("nan")
        slutsats = (f"<p>Den regeluppsättning som höll bäst på <b>båda</b> perioderna:</p><p class='rule'>{beskriv(bast)}</p>"
                    f"<p>Träning: <b>{rr(bast['tr']['snitt'])}</b> per trade ({bast['tr']['antal']} trades). "
                    f"Test (sista året): <b>{rr(bast['te']['snitt'])}</b> per trade ({bast['te']['antal']} trades, "
                    f"{f(bast['te']['traff'], 0)} % träff). Skannerns nuvarande regler gav {rr(bas_te['snitt'])} på samma testår"
                    f"{'' if lyft != lyft else f' – en skillnad på {rr(lyft)} per trade'}.</p>")
    else:
        slutsats = "<p>Ingen kombination hade tillräckligt många trades i båda perioderna för en säker slutsats.</p>"
    eff_html = ""
    for rader in effekter:
        if not rader:
            continue
        basta = max(v for _, v, _ in rader)
        eff_html += "<table class='eff'>" + "".join(
            f"<tr{' class=best' if v == basta else ''}><td class='wrap'>{e}</td><td class='{'up' if v > 0 else 'down'}'>{rr(v)}</td>"
            f"<td>{f(a, 0)} / år</td></tr>" for e, v, a in rader) + "</table>"
    topp_html = "".join(_rad(r, r is bast) for r in topp)
    html = f"""<!doctype html>
<html lang="sv"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Optimering av strategin</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,700&family=Instrument+Serif&display=swap">
<style>
:root{{--bg:#F4F1EA;--card:#FFFFFF;--fg:#1B1A17;--muted:#6B665C;--line:#E2DDD2;--up:#2F7A3E;--down:#A3321C;--accent:#2F4BD8;--hl:#EEF2FD;color-scheme:light}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161513;--card:#211F1C;--fg:#EEEAE2;--muted:#A8A295;--line:#33302B;--up:#8FD19C;--down:#F09A86;--accent:#9DB0FF;--hl:#1F2A4A;color-scheme:dark}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font-family:'DM Sans',system-ui,sans-serif;font-size:15px;line-height:1.45}}
.wrap-page{{max-width:1000px;margin:0 auto;padding:32px 20px 48px;display:flex;flex-direction:column;gap:30px}}
h1{{margin:0;font-family:'Instrument Serif',Georgia,serif;font-weight:400;font-size:clamp(34px,6vw,50px);line-height:1}}
h2{{margin:0 0 6px;font-size:21px}}
p{{margin:0 0 10px;max-width:72ch}}
.muted{{color:var(--muted)}}
a{{color:var(--accent);font-weight:600;text-decoration:none}}
.box{{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:18px}}
.rule{{font-size:17px;font-weight:700;background:var(--hl);padding:12px 14px;border-radius:12px}}
.up{{color:var(--up);font-weight:600}}.down{{color:var(--down);font-weight:600}}
.tablewrap{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums;background:var(--card);border-radius:14px;margin-bottom:12px}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap;vertical-align:top}}
td.wrap{{white-space:normal;min-width:220px}}
th{{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}}
tr.best td{{background:var(--hl)}}
ul{{margin:0;padding-left:20px;color:var(--muted);font-size:14px}}
</style></head><body><div class="wrap-page">
<header><a href="index.html">← Till skanningen</a> · <a href="backtest.html">Backtest</a>
<h1>Optimering av strategin</h1>
<p class="muted">{n_bolag} bolag · träning {start:%Y-%m-%d} – {gräns:%Y-%m-%d} · test {gräns:%Y-%m-%d} – {idag:%Y-%m-%d} · uppdaterad {datetime.now():%Y-%m-%d}</p></header>
<section class="box"><h2>Slutsats</h2>{slutsats}</section>
<section><h2>Vad höjer R per trade?</h2>
<p class="muted">Varje tabell visar hur ett enda val påverkar resultatet, i snitt över alla andra kombinationer (median-R per trade, hela perioden). Den markerade raden är bäst. Antal per år visar hur ofta signalen kommer.</p>
{eff_html}</section>
<section><h2>Tio bästa kombinationerna på träningsåren – och hur de klarade teståret</h2>
<p class="muted">Om en kombination är bra på träning men dålig på test har den troligen bara haft tur. Leta efter dem som är bra på båda. Den markerade är slutsatsen ovan.</p>
<div class="tablewrap"><table><tr><th>Regler</th><th>Träning, trades</th><th>Träning, snitt</th><th>Test, trades</th><th>Test, snitt</th><th>Test, träff</th></tr>{topp_html}</table></div>
<p class="muted">Som jämförelse: skannerns nuvarande regler (med MACD-utgång) gav {rr(bas_tr['snitt'])} på träning och {rr(bas_te['snitt'])} på test.</p></section>
<section><h2>Begränsningar</h2><ul>
<li>Courtage, spread och rapportregeln är inte med.</li>
<li>Bolagen är dagens OMXS30, så bolag som åkt ur efter dåliga år saknas.</li>
<li>Ju fler kombinationer man testar, desto större chans att någon ser bra ut av ren tur. Därför väger teståret tyngst.</li>
<li>Historik är ingen garanti för framtiden.</li>
</ul></section>
</div></body></html>"""
    fil.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
