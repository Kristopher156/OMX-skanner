#!/usr/bin/env python3
"""
Backtest av skannerns signaler – hur hade strategin gått de senaste åren?

Kör:  python backtest.py
Resultatet sparas som backtest.html (i GitHub: site/backtest.html).

Två sätt att sälja testas, med samma köpsignaler som skannern
(grönt = MACD vänder upp, blått = MACD-korsning):

  A. Mål eller stop: sälj vid senaste toppen (målet på kortet) eller vid stoppen.
  B. MACD-utgång: stoppen flyttas till köpkursen vid +1R, och du säljer när
     MACD korsar ner – ditt sätt att låta vinnarna löpa.

Köp sker på öppningskursen dagen efter signalen. Ett innehav åt gången per bolag.
"""
import os
from datetime import date, datetime, timedelta
from html import escape
from pathlib import Path

import pandas as pd
import yfinance as yf

import omx30_skanner as sk

AR = 3                 # så många år bakåt testas
MAX_DAGAR_A = 30       # tidsstopp för variant A (handelsdagar)
MAX_DAGAR_B = 40       # tidsstopp för variant B
KOPSIGNALER = (sk.TIDIG, sk.KORS)


def ladda(ticker: str):
    df = yf.download(ticker, period=f"{AR + 1}y", interval="1d", progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None
    ut = pd.DataFrame({k: sk.kolumn(df, k) for k in ["Open", "High", "Low", "Close"]}).dropna()
    ut.index = pd.to_datetime(ut.index)
    if ut.index.tz is not None:
        ut.index = ut.index.tz_localize(None)
    return ut if len(ut) > 300 else None


def index_upptrend():
    """True de dagar OMXS30 låg över sitt 200-dagars medelvärde."""
    df = ladda(sk.INDEX_TICKER)
    if df is None:
        return None
    c = df["Close"]
    return (c > c.rolling(200).mean()).rename("upp")


def simulera(df: pd.DataFrame, sig: pd.DataFrame, upp, namn: str, start: pd.Timestamp, variant: str):
    o, h, l, c = (df[k].to_numpy() for k in ["Open", "High", "Low", "Close"])
    status = sig["status"].to_numpy()
    stopp = sig["stop"].to_numpy()
    topp = sig["topp"].to_numpy()
    rr_sig = sig["rr"].to_numpy()
    linje, signal = sig["linje"].to_numpy(), sig["signal"].to_numpy()
    datum = df.index
    upp_al = upp.reindex(datum, method="ffill") if upp is not None else None
    n = len(df)
    trades, ledig_fran = [], 0

    for t in range(1, n - 1):
        if datum[t] < start or t < ledig_fran:
            continue
        ny = status[t] in KOPSIGNALER and status[t - 1] not in KOPSIGNALER
        if not ny:
            continue
        entry, stop0 = o[t + 1], stopp[t]
        if not (entry > stop0 > 0):
            continue
        risk = entry - stop0
        mal = topp[t]
        stop, slut, pris, orsak = stop0, None, None, None
        maxd = MAX_DAGAR_A if variant == "A" else MAX_DAGAR_B
        for d in range(t + 1, min(n, t + 1 + maxd)):
            if l[d] <= stop:                                  # stop först (försiktigt antagande)
                pris = o[d] if o[d] < stop else stop
                orsak = "stop" if stop < entry else "stop (köpkurs)"
                slut = d
                break
            if variant == "A" and mal > entry and h[d] >= mal:
                pris = o[d] if o[d] > mal else mal
                orsak, slut = "mål", d
                break
            if variant == "B":
                if d > t + 1 and linje[d] < signal[d] and linje[d - 1] >= signal[d - 1]:
                    pris, orsak, slut = c[d], "MACD ner", d
                    break
                if h[d] >= entry + risk:                       # +1R nådd: stop till köpkurs från nästa dag
                    stop = max(stop, entry)
        if slut is None:
            slut = min(n - 1, t + maxd)
            if slut >= n - 1 and t + maxd > n - 1:
                orsak = "öppen"
            else:
                orsak = "tid"
            pris = c[slut]
        trades.append({
            "Bolag": namn, "Signal": "Tidig vändning" if status[t] == sk.TIDIG else "MACD-korsning",
            "Köpdag": datum[t + 1].date(), "Köp": entry, "Säljdag": datum[slut].date(), "Sälj": pris,
            "R": (pris - entry) / risk, "Dagar": slut - t, "Orsak": orsak,
            "RR vid signal": rr_sig[t],
            "Marknad": ("upp" if bool(upp_al.iloc[t]) else "ned") if upp_al is not None and upp_al.iloc[t] == upp_al.iloc[t] else "okänd",
        })
        ledig_fran = slut + 1
    return trades


def statistik(tr):
    if not tr:
        return None
    r = pd.Series([t["R"] for t in tr])
    vinst, forlust = r[r > 0].sum(), -r[r < 0].sum()
    return {
        "antal": len(r), "traff": (r > 0).mean() * 100, "snitt": r.mean(), "median": r.median(),
        "summa": r.sum(), "pf": (vinst / forlust) if forlust > 0 else float("inf"),
        "dagar": sum(t["Dagar"] for t in tr) / len(tr),
    }


def _f(v, d=2):
    if v is None or v != v:
        return "–"
    if v == float("inf"):
        return "∞"
    return f"{v:,.{d}f}".replace(",", " ").replace(".", ",")


def _r(v):
    return ("+" if v >= 0 else "−") + _f(abs(v)) + "R"


def sektion(titel, forklaring, tr):
    s = statistik([t for t in tr if t["Orsak"] != "öppen"])
    if not s:
        return f"<section><h2>{titel}</h2><p class='muted'>Inga avslutade trades.</p></section>"
    klar = [t for t in tr if t["Orsak"] != "öppen"]
    grupper = [
        ("Alla signaler", klar),
        ("Bara när R/R ≥ 2 vid signal", [t for t in klar if t["RR vid signal"] == t["RR vid signal"] and t["RR vid signal"] >= sk.MIN_RR]),
        ("När marknaden var i upptrend", [t for t in klar if t["Marknad"] == "upp"]),
        ("När marknaden var i nedtrend", [t for t in klar if t["Marknad"] == "ned"]),
        ("Grön: tidig vändning", [t for t in klar if t["Signal"] == "Tidig vändning"]),
        ("Blå: MACD-korsning", [t for t in klar if t["Signal"] == "MACD-korsning"]),
    ]
    rader = ""
    for namn, g in grupper:
        st = statistik(g)
        if not st:
            rader += f"<tr><td>{namn}</td><td colspan='5' class='muted'>inga trades</td></tr>"
            continue
        klass = "up" if st["snitt"] > 0 else "down"
        rader += (f"<tr><td>{namn}</td><td>{st['antal']}</td><td>{_f(st['traff'], 0)} %</td>"
                  f"<td class='{klass}'>{_r(st['snitt'])}</td><td>{_f(st['pf'])}</td><td>{_f(st['dagar'], 0)}</td></tr>")
    klass = "up" if s["snitt"] > 0 else "down"
    tiles = f"""<div class="tiles">
  <div class="tile"><span class="lbl">Trades</span><span class="big">{s['antal']}</span></div>
  <div class="tile"><span class="lbl">Träffprocent</span><span class="big">{_f(s['traff'], 0)} %</span></div>
  <div class="tile"><span class="lbl">Snitt per trade</span><span class="big {klass}">{_r(s['snitt'])}</span></div>
  <div class="tile"><span class="lbl">Trades per år</span><span class="big">{_f(s['antal'] / AR, 0)}</span></div>
  <div class="tile"><span class="lbl">Profit factor</span><span class="big">{_f(s['pf'])}</span></div>
  <div class="tile"><span class="lbl">Snitt dagar</span><span class="big">{_f(s['dagar'], 0)}</span></div>
</div>
<p class="explain">{forklara(s)}</p>"""
    senaste = sorted(tr, key=lambda t: t["Köpdag"], reverse=True)[:12]
    lista = "".join(
        f"<tr><td>{t['Köpdag']}</td><td>{escape(t['Bolag'])}</td><td>{t['Signal']}</td><td>{_f(t['Köp'])}</td>"
        f"<td>{_f(t['Sälj'])}</td><td class='{'up' if t['R'] > 0 else 'down'}'>{_r(t['R'])}</td><td>{t['Orsak']}</td></tr>"
        for t in senaste)
    return f"""<section>
<h2>{titel}</h2><p class="muted">{forklaring}</p>
{tiles}
<div class="tablewrap"><table><tr><th>Urval</th><th>Trades</th><th>Träff</th><th>Snitt-R</th><th>Profit factor</th><th>Dagar</th></tr>{rader}</table></div>
<details><summary>Senaste 12 trades</summary><div class="tablewrap"><table>
<tr><th>Köpdag</th><th>Bolag</th><th>Signal</th><th>Köp</th><th>Sälj</th><th>R</th><th>Orsak</th></tr>{lista}</table></div></details>
</section>"""


def forklara(s):
    """Översätter R till något begripligt."""
    per_ar = s["antal"] / AR
    kanske = min(per_ar, 40)          # realistiskt antal trades per år för en person med ett litet konto
    return (f"<b>Snitt per trade är siffran att titta på.</b> Om du hade tagit alla {s['antal']} signaler hade det blivit "
            f"{_r(s['summa'])} sammanlagt ({_r(s['snitt'])} × {s['antal']}), men det hade aldrig gått i verkligheten – "
            f"det är {_f(per_ar, 0)} trades per år, många samtidigt. Säg att du tar ungefär {_f(kanske, 0)} trades per år med "
            f"2 % risk var: då ger {_r(s['snitt'])} per trade ungefär {_f(kanske * s['snitt'] * 2, 0)} % per år på tradingkontot, "
            f"grovt räknat och före courtage.")


def bedomning(tr_b):
    s = statistik([t for t in tr_b if t["Orsak"] != "öppen"])
    if not s:
        return "För få trades för att säga något."
    if s["antal"] < 30:
        return f"Bara {s['antal']} trades – för få för säkra slutsatser. Se det som en fingervisning."
    if s["snitt"] >= 0.25:
        return (f"Historiskt positivt: i snitt {_r(s['snitt'])} per trade över {s['antal']} trades. "
                "Det tyder på att signalerna har haft en verklig fördel – men historik är ingen garanti.")
    if s["snitt"] > 0:
        return (f"Svagt positivt: {_r(s['snitt'])} per trade. Det räcker knappt för att täcka courtage och misstag. "
                "Titta på vilka urval som går bäst nedan och skärp reglerna där.")
    return (f"Negativt: {_r(s['snitt'])} per trade. Reglerna som de är nu har inte fungerat historiskt. "
            "Titta på urvalen nedan – kanske fungerar de bara i upptrend eller med R/R ≥ 2.")


def spara(tr_a, tr_b, period, bh_index, fil: Path):
    nu = datetime.now()
    html = f"""<!doctype html>
<html lang="sv"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Backtest av strategin</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,700&family=Instrument+Serif&display=swap">
<style>
:root{{--bg:#F4F1EA;--card:#FFFFFF;--fg:#1B1A17;--muted:#6B665C;--line:#E2DDD2;--up:#2F7A3E;--down:#A3321C;--accent:#2F4BD8;color-scheme:light}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161513;--card:#211F1C;--fg:#EEEAE2;--muted:#A8A295;--line:#33302B;--up:#8FD19C;--down:#F09A86;--accent:#9DB0FF;color-scheme:dark}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font-family:'DM Sans',system-ui,sans-serif;font-size:15px;line-height:1.45}}
.wrap{{max-width:1000px;margin:0 auto;padding:32px 20px 48px;display:flex;flex-direction:column;gap:32px}}
h1{{margin:0;font-family:'Instrument Serif',Georgia,serif;font-weight:400;font-size:clamp(34px,6vw,50px);line-height:1}}
h2{{margin:0 0 4px;font-size:21px}}
.muted{{color:var(--muted);margin:0 0 12px;max-width:70ch}}
a{{color:var(--accent);font-weight:600;text-decoration:none}}
.verdict{{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:18px;font-size:16px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:10px;margin-bottom:14px}}
.tile{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:12px 14px;display:flex;flex-direction:column;gap:2px}}
.lbl{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}}
.big{{font-size:22px;font-weight:700;font-variant-numeric:tabular-nums}}
.up{{color:var(--up)}}.down{{color:var(--down)}}
.tablewrap{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums;background:var(--card);border-radius:14px}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap}}
th{{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}}
details{{margin-top:12px}}summary{{cursor:pointer;font-weight:600}}
.explain{{margin:0 0 14px;font-size:14px;max-width:75ch}}
ul{{margin:0;padding-left:20px;color:var(--muted);font-size:14px}}
</style></head><body><div class="wrap">
<header><a href="index.html">← Till skanningen</a> · <a href="optimering.html">Optimering</a>
<h1>Backtest av din strategi</h1>
<p class="muted">{period} · {len(sk.BOLAG)} bolag · uppdaterad {nu:%Y-%m-%d}. OMXS30-indexet gick {bh_index} under samma period.</p></header>
<div class="verdict"><b>Bedömning:</b> {bedomning(tr_b)}</div>
{sektion("B. MACD-utgång (låt vinnarna löpa)", "Stop under bottnen. Vid +1R flyttas stoppen till köpkursen. Sälj när MACD korsar ner, senast efter " + str(MAX_DAGAR_B) + " handelsdagar.", tr_b)}
{sektion("A. Mål eller stop", "Sälj vid senaste toppen (målet på kortet) eller vid stoppen, senast efter " + str(MAX_DAGAR_A) + " handelsdagar.", tr_a)}
<section><h2>Så ska du läsa det</h2><ul>
<li><b>R</b> = vinst eller förlust i förhållande till risken. −1R betyder att stoppen togs. Med 2 % risk per trade är 1R ungefär 2 % av kontot.</li>
<li><b>Profit factor</b> = summan av vinsterna delat med summan av förlusterna. Över 1,5 är bra, under 1 betyder förlust.</li>
<li><b>Upptrend/nedtrend</b> = om OMXS30 låg över eller under sitt 200-dagars medelvärde när signalen kom.</li>
</ul></section>
<section><h2>Begränsningar</h2><ul>
<li>Courtage och spread är inte medräknade – dra av lite från varje trade.</li>
<li>Rapportregeln är inte med, eftersom gamla rapportdatum inte finns i datan.</li>
<li>Bolagen är dagens OMXS30. Bolag som åkt ur indexet efter dåliga år saknas, vilket kan göra resultatet något för bra.</li>
<li>Om stop och mål nås samma dag antas stoppen ha tagits först.</li>
<li>Historik är ingen garanti för framtiden.</li>
</ul></section>
</div></body></html>"""
    fil.write_text(html, encoding="utf-8")


def main():
    start = pd.Timestamp(date.today() - timedelta(days=365 * AR))
    upp = index_upptrend()
    idx = ladda(sk.INDEX_TICKER)
    bh_index = "–"
    if idx is not None:
        c = idx["Close"][idx.index >= start]
        if len(c) > 1:
            v = (c.iloc[-1] / c.iloc[0] - 1) * 100
            bh_index = ("+" if v >= 0 else "−") + _f(abs(v), 0) + " %"
    tr_a, tr_b = [], []
    for namn, ticker in sk.BOLAG.items():
        try:
            df = ladda(ticker)
            if df is None:
                print(f"  Ingen data för {namn}")
                continue
            sig = sk.signalserie(df["Close"], df["Low"], df["High"])
            tr_a += simulera(df, sig, upp, namn, start, "A")
            tr_b += simulera(df, sig, upp, namn, start, "B")
        except Exception as e:
            print(f"  Fel för {namn}: {e}")
    period = f"{start:%Y-%m-%d} – {date.today():%Y-%m-%d}"
    for namn, tr in (("A", tr_a), ("B", tr_b)):
        s = statistik([t for t in tr if t["Orsak"] != "öppen"])
        if s:
            print(f"Variant {namn}: {s['antal']} trades, träff {s['traff']:.0f} %, snitt {s['snitt']:+.2f}R, PF {s['pf']:.2f}")
    if os.environ.get("GITHUB_ACTIONS") == "true":
        Path("site").mkdir(exist_ok=True)
        fil = Path("site") / "backtest.html"
    else:
        fil = Path(__file__).with_name("backtest.html")
    spara(tr_a, tr_b, period, bh_index, fil)
    print(f"Sparat: {fil}")


if __name__ == "__main__":
    main()
