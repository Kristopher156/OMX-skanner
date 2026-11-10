#!/usr/bin/env python3
"""
Exits, utstoppningar och ett dynamiskt system.

1. Blir traderna utstoppade innan de hinner stiga? (hur snabbt, och hur ofta vänder de upp efteråt)
2. Sju sätt att sälja, testade på varje metod: fast 2R, släpande stop, halva vid 2R, MACD vänder,
   under 20-dagars, säljcandle, stop till köpkurs + släpande.
3. Dynamiskt system: varje dag används bara de metoder som de senaste 3 åren har fungerat i
   just det marknadsläget (bara kunskap som fanns då). Plus en nödbroms: faller kontot 20 %
   från toppen pausas nya köp i 8 veckor.
Resultat: results/exits.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import metoder as M
import omx30_skanner as sk

TRAN_TILL = "2022-01-01"   # exit väljs på åren före, testas på åren efter
METODER = [m for m in M.METODER if m != "slump"]
RISK = 2.0


def stat(r):
    r = np.asarray(r, dtype=float)
    if not len(r):
        return {"n": 0}
    return {"n": int(len(r)), "E": round(float(r.mean()), 3), "traff": round(float((r > 0).mean() * 100), 1),
            "osakerhet": round(float(1.96 * r.std(ddof=1) / np.sqrt(len(r))), 3) if len(r) > 1 else None}


def portfolj(kandidater, max_oppna, max_per_vecka, broms_R=None, paus_dagar=56):
    """Väljer trades i datumordning: max X öppna, max Y nya per vecka, nödbroms vid nedgång."""
    valda, veckor = [], {}
    for x in kandidater:
        veckor.setdefault(pd.Timestamp(x["datum"]).strftime("%G-%V"), []).append(x)
    paus_till, utlost_vid_topp = None, None
    for v in sorted(veckor):
        nya = 0
        for x in sorted(veckor[v], key=lambda x: (-M.rang(x), x["datum"])):
            if paus_till and x["datum"] < paus_till:
                continue
            if max_per_vecka and nya >= max_per_vecka:
                break
            oppna = [o for o in valda if o["datum"] <= x["datum"] < o["ut_datum"]]
            if len(oppna) >= max_oppna or any(o["bolag"] == x["bolag"] for o in oppna):
                continue
            if broms_R:
                stangda = sorted([o["R"] for o in valda if o["ut_datum"] <= x["datum"]])
                kurva = np.cumsum([o["R"] for o in sorted([o for o in valda if o["ut_datum"] <= x["datum"]], key=lambda o: o["ut_datum"])])
                if len(kurva):
                    topp = max(0.0, float(kurva.max()))
                    if topp - kurva[-1] >= broms_R and utlost_vid_topp != topp:
                        utlost_vid_topp = topp
                        paus_till = str((pd.Timestamp(x["datum"]) + pd.Timedelta(days=paus_dagar)).date())
                        continue
            valda.append(x)
            nya += 1
    return valda


def arsresultat(valda):
    ar = {}
    for x in valda:
        ar.setdefault(x["ut_datum"][:4], 0.0)
        ar[x["ut_datum"][:4]] += x["R"] * RISK
    kurva = np.cumsum([x["R"] for x in sorted(valda, key=lambda x: x["ut_datum"])])
    dd = float((np.maximum.accumulate(np.r_[0, kurva])[1:] - kurva).max()) * RISK if len(kurva) else 0
    return {y: round(v, 1) for y, v in sorted(ar.items())}, round(dd, 1)


def sammanfatta(namn, valda, ar_tot, ar_test):
    per_ar, dd = arsresultat(valda)
    tr = [x for x in valda if x["datum"] < TRAN_TILL]
    te = [x for x in valda if x["datum"] >= TRAN_TILL]
    ar_tr = ar_tot - ar_test
    return {"namn": namn, "trades_per_ar": round(len(valda) / ar_tot, 1), **stat([x["R"] for x in valda]),
            "procent_per_ar": round(sum(x["R"] for x in valda) / ar_tot * RISK, 1),
            "procent_per_ar_tran": round(sum(x["R"] for x in tr) / ar_tr * RISK, 1) if ar_tr > 0 else None,
            "procent_per_ar_test": round(sum(x["R"] for x in te) / ar_test * RISK, 1),
            "per_ar": per_ar, "bast_ar": max(per_ar.values()) if per_ar else None,
            "samst_ar": min(per_ar.values()) if per_ar else None, "storsta_fall": dd}


def main():
    idx = M.ladda(sk.INDEX_TICKER)
    idx_close, idx_trend = idx["Close"], M.trend(idx["Close"])
    data = {}
    for namn, t in sk.BOLAG.items():
        df = M.ladda(t)
        if df is not None:
            data[namn] = M.forbered(df)
    print(f"{len(data)} bolag")
    alla = {}
    for ex in M.EXITS:
        lista = []
        for i, (namn, d) in enumerate(data.items()):
            for m in METODER:
                lista += M.simulera(namn, m, d, idx_close, idx_trend, fro=i, exit_typ=ex)
        alla[ex] = lista
        print(f"  exit {ex}: {len(lista)} trades")
    forsta = min(x["datum"] for x in alla[None])
    sista = max(x["ut_datum"] for x in alla[None])
    ar_tot = (pd.Timestamp(sista) - pd.Timestamp(forsta)).days / 365.25
    ar_test = (pd.Timestamp(sista) - pd.Timestamp(TRAN_TILL)).days / 365.25

    # ---- 1. utstoppningar (standard-exit)
    stopp = [x for x in alla[None] if x["orsak"] == "stop"]
    vande = 0
    for x in stopp:
        d = data[x["bolag"]]
        pos = d.index.get_loc(pd.Timestamp(x["ut_datum"]))
        risk = x["kop"] - x["stop"]
        fram = d["High"].iloc[pos + 1:pos + 21]
        if len(fram) and fram.max() >= x["kop"] + 2 * risk:
            vande += 1
    dagar = np.array([x["dagar"] for x in stopp])
    utstopp = {
        "andel_stoppade": round(len(stopp) / len(alla[None]) * 100, 1),
        "median_dagar_till_stop": float(np.median(dagar)) if len(dagar) else None,
        "andel_inom_5_dagar": round(float((dagar <= 5).mean() * 100), 1) if len(dagar) else None,
        "andel_var_plus_1R_forst": round(float(np.mean([x["mfe"] >= 1 for x in stopp]) * 100), 1),
        "andel_var_plus_halv_R_forst": round(float(np.mean([x["mfe"] >= 0.5 for x in stopp]) * 100), 1),
        "andel_nadde_2R_inom_20_dagar_efter_stop": round(vande / len(stopp) * 100, 1) if stopp else None,
        "per_metod": {m: round(sum(1 for x in alla[None] if x["metod"] == m and x["orsak"] == "stop") /
                               max(1, sum(1 for x in alla[None] if x["metod"] == m)) * 100, 1) for m in METODER},
    }

    # ---- 2. exit för exit, metod för metod
    exit_tabell = {}
    for m in METODER:
        exit_tabell[m] = {}
        for ex, lista in alla.items():
            l = [x for x in lista if x["metod"] == m]
            tr = [x["R"] for x in l if x["datum"] < TRAN_TILL]
            te = [x["R"] for x in l if x["datum"] >= TRAN_TILL]
            exit_tabell[m][str(ex)] = {"alla": stat([x["R"] for x in l]), "tran": stat(tr), "test": stat(te),
                                       "dagar": float(np.median([x["dagar"] for x in l])) if l else None,
                                       "per_lage": {lg: stat([x["R"] for x in l if x["index_lage"] == lg]) for lg in ["upp", "sidled", "ned"]}}
    # bästa exit per metod, vald bara på träningsåren
    bast_exit = {}
    for m in METODER:
        kand = [(ex, v["tran"].get("E", -9)) for ex, v in exit_tabell[m].items() if v["tran"].get("n", 0) >= 30]
        bast_exit[m] = max(kand, key=lambda x: x[1])[0] if kand else "None"

    # ---- 3. dynamiskt system
    def lista_for(exitval):
        ut = []
        for m in METODER:
            ex = None if exitval[m] == "None" else exitval[m]
            ut += [x for x in alla[ex] if x["metod"] == m]
        return sorted(ut, key=lambda x: x["datum"])

    std_exit = {m: "None" for m in METODER}

    def statisk(lista):
        return [x for x in lista if x["metod"] in M.LAGE_METODER.get(x["index_lage"], [])]

    def adaptiv(lista, min_n=15, min_E=0.1, ar_bak=3):
        """Väljer metod efter hur den gått i samma marknadsläge de senaste åren (bara avslutade trades)."""
        import bisect
        grupp = {}
        for y in sorted(lista, key=lambda y: y["ut_datum"]):
            grupp.setdefault((y["metod"], y["index_lage"]), []).append(y)
        idx = {k: ([y["ut_datum"] for y in v], np.r_[0, np.cumsum([y["R"] for y in v])]) for k, v in grupp.items()}
        ut = []
        for x in lista:
            k = (x["metod"], x["index_lage"])
            if k not in idx:
                continue
            datum, cs = idx[k]
            fran = str((pd.Timestamp(x["datum"]) - pd.Timedelta(days=365 * ar_bak)).date())
            a, b = bisect.bisect_left(datum, fran), bisect.bisect_left(datum, x["datum"])
            n = b - a
            if n >= min_n and (cs[b] - cs[a]) / n >= min_E:
                ut.append(x)
        return ut

    def kvalitet(lista):
        return [x for x in lista if (x["rr_mot"] is None or x["rr_mot"] >= 2) and (x["vol_x"] is None or x["vol_x"] >= 1)]

    system = []
    for exnamn, exval in (("standard-exit (2R)", std_exit), ("bästa exit per metod", bast_exit)):
        lista = lista_for(exval)
        ad = adaptiv(lista)
        for namn, kand in ((f"Fast schema läge→metod, {exnamn}", kvalitet(statisk(lista))),
                           (f"Dynamiskt (metoden som fungerat senast i läget), {exnamn}", kvalitet(ad)),
                           (f"Dynamiskt utan R/R-krav, {exnamn}", [x for x in ad if x["vol_x"] is None or x["vol_x"] >= 1])):
            for mo in (2, 4):
                system.append(sammanfatta(f"{namn} · max {mo} samtidigt", portfolj(kand, mo, 2), ar_tot, ar_test))
                system.append(sammanfatta(f"{namn} · max {mo} samtidigt · nödbroms −20 %",
                                          portfolj(kand, mo, 2, broms_R=10), ar_tot, ar_test))
    system.sort(key=lambda x: -(x["procent_per_ar_test"] or -99))

    res = {"period": f"{forsta} – {sista}", "tran_till": TRAN_TILL, "bolag": len(data), "exits": M.EXITS,
           "utstopp": utstopp, "exit_tabell": exit_tabell, "bast_exit": bast_exit, "system": system}
    Path("results").mkdir(exist_ok=True)
    (Path("results") / "exits.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("Utstopp:", utstopp)
    print("Bästa exit per metod (vald på träningsår):", bast_exit)
    for x in system[:8]:
        print(f"  {x['namn'][:95]:95} {x['trades_per_ar']}/år E={x.get('E')} %/år {x['procent_per_ar']} test {x['procent_per_ar_test']} bäst {x['bast_ar']} sämst {x['samst_ar']} fall {x['storsta_fall']}")


if __name__ == "__main__":
    main()
