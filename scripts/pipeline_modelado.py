"""Pipeline de modelado: partición, outliers, filtros univariado / bivariado / multivariado y datasets finales.

Etapas (se pueden correr por separado; cada una lee lo que dejó la anterior):

  python scripts/pipeline_modelado.py outliers    # partición train/test + reasignación de outliers por tasa de default
  python scripts/pipeline_modelado.py binning     # trameado qcut (5 y 10) y OptBinning sobre train, para todas las variables
  python scripts/pipeline_modelado.py seleccion   # filtro univariado, bivariado (IV / Gini) y multivariado (Spearman)
  python scripts/pipeline_modelado.py datasets    # dataset logístico (WoE) y dataset ML (SMOTE solo en train)
  python scripts/pipeline_modelado.py todo        # las cuatro, en orden

Principio de la guía («Errores frecuentes»): todo lo que usa el TARGET —reasignación de outliers, bines, WoE, IV, Gini,
selección y SMOTE— se aprende SOLO con train y se aplica después a test.

Entradas:  artifacts/tablon_imputado.parquet (notebook, Paso 3: faltantes y centinelas)
Salidas:   artifacts/tablon_tratado.parquet, artifacts/parametros_outliers.json, artifacts/modelado/*
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts"
OUT = ART / "modelado"
DATASETS = OUT / "datasets"
ID, TARGET = "SK_ID_CURR", "TARGET"
SEMILLA = 42
TEST_SIZE = 0.30

# ── Outliers ────────────────────────────────────────────────────────────────────────────────────────
# Umbral de detección: el mismo percentil decidido en la etapa anterior (evidencia en «Calidad»), recalculado en train.
# OWN_CAR_AGE: el umbral es de negocio (64–65 años es un bloque de codificación y 91 es imposible; ≤ 63 es plausible).
UMBRAL_OUTLIER = {"AMT_INCOME_TOTAL": ("percentil", 99.9), "CNT_CHILDREN": ("percentil", 99.9),
                  "CNT_FAM_MEMBERS": ("percentil", 99.9), "OBS_30_CNT_SOCIAL_CIRCLE": ("percentil", 99.9),
                  "OBS_60_CNT_SOCIAL_CIRCLE": ("percentil", 99.9), "AMT_REQ_CREDIT_BUREAU_QRT": ("percentil", 99.9),
                  "OWN_CAR_AGE": ("valor", 63)}
TOPES_ANTERIORES = {"AMT_INCOME_TOTAL": 900000.0, "CNT_CHILDREN": 4.0, "CNT_FAM_MEMBERS": 6.0, "OBS_30_CNT_SOCIAL_CIRCLE": 17.0,
                    "OBS_60_CNT_SOCIAL_CIRCLE": 16.0, "AMT_REQ_CREDIT_BUREAU_QRT": 4.0, "OWN_CAR_AGE": 30.0}
MAX_VALORES_DISCRETOS = 70     # hasta aquí cada valor es un tramo candidato; si no, qcut en 20 tramos
N_TRAMOS_CONTINUA = 20
TOLERANCIA_RD = 0.005          # el tramo destino no puede mover su tasa de default más de 0.5 p.p. al recibir a los atípicos

# ── Filtros ─────────────────────────────────────────────────────────────────────────────────────────
DOMINANTE_MAX = 0.99            # univariado: un valor concentra ≥ 99 % de los datos no nulos → varianza casi nula
NULOS_MAX = 0.50                # univariado: más de 50 % de nulos…
CARDINALIDAD_ALTA = 15          # univariado: más de 15 categorías → agrupar por tasa de default
IV_MIN = 0.10                   # bivariado logística: rangos «Medium», «Strong» y «≥ 0.5»
IV_SOSPECHOSO = 0.50
GINI_MIN = 0.08                 # bivariado ML (ver justificación en la app): equivale a IV = 0.02 (inicio de «Weak»)
GINI_SOSPECHOSO = 0.38          # equivale a IV = 0.50
RHO_MAX = 0.60                  # multivariado: |ρ de Spearman| > 0.6 → redundantes
OPTB = dict(max_n_prebins=20, min_prebin_size=0.05, max_n_bins=5)   # parámetros de la guía (Paso 6.2)


def guardar(nombre: str, tabla: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tabla.to_parquet(OUT / f"{nombre}.parquet", index=False)


def leer(nombre: str) -> pd.DataFrame:
    return pd.read_parquet(OUT / f"{nombre}.parquet")


def wilson(k: float, n: int, z: float = 1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


def tope(s: pd.Series, p: float) -> float:
    s = s.dropna()
    entera = bool((s % 1 == 0).all())
    return float(s.quantile(p / 100, interpolation="higher" if entera else "linear"))


def cargar_base():
    imp = pd.read_parquet(ART / "tablon_imputado.parquet")
    part = leer("particion").set_index(ID)["muestra"]
    return imp, imp[ID].map(part)


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# 1. PARTICIÓN Y OUTLIERS
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def tramos_candidatos(s: pd.Series):
    """Etiqueta de tramo para los valores NO atípicos: cada valor (discretas) o 20 cuantiles (continuas)."""
    entera = bool((s % 1 == 0).all())
    if entera and s.nunique() <= MAX_VALORES_DISCRETOS:
        return s.astype(float), True
    return pd.qcut(s, N_TRAMOS_CONTINUA, duplicates="drop"), False


def etapa_outliers() -> None:
    from sklearn.model_selection import train_test_split

    imp = pd.read_parquet(ART / "tablon_imputado.parquet")
    tr_ids, _ = train_test_split(imp[ID], test_size=TEST_SIZE, stratify=imp[TARGET], random_state=SEMILLA)
    muestra = np.where(imp[ID].isin(set(tr_ids)), "train", "test")
    guardar("particion", pd.DataFrame({ID: imp[ID], "muestra": muestra}))
    es_tr = pd.Series(muestra == "train", index=imp.index)
    tr = imp[es_tr]

    trat = imp.copy()
    resumen, tramos_out, params = [], [], {}
    for v, (regla, par) in UMBRAL_OUTLIER.items():
        s = tr[v].dropna()
        y = tr.loc[s.index, TARGET]
        umbral = tope(s, par) if regla == "percentil" else float(par)
        es_out = s > umbral
        n_out, k_out = int(es_out.sum()), int(y[es_out].sum())
        rd_out = k_out / n_out
        lo, hi = wilson(k_out, n_out)

        sv, yv = s[~es_out], y[~es_out]
        etq, discreta = tramos_candidatos(sv)
        g = pd.DataFrame({"x": sv, "y": yv, "t": etq}).groupby("t", observed=True).agg(
            n=("y", "size"), k=("y", "sum"), mediana=("x", "median"), desde=("x", "min"), hasta=("x", "max"))
        g["rd"] = g["k"] / g["n"]
        g["orden"] = np.arange(len(g))
        # RD del tramo si recibiera a los atípicos: mide cuánto se «distorsiona» un rango sano (la crítica al tope)
        g["rd_si_recibe"] = (g["k"] + k_out) / (g["n"] + n_out)
        g["desplazamiento"] = (g["rd_si_recibe"] - g["rd"]).abs()
        g["compatible"] = (g["rd"] >= lo) & (g["rd"] <= hi) & (g["n"] >= n_out)
        ok = g["compatible"] & (g["desplazamiento"] <= TOLERANCIA_RD)
        if ok.any():
            dest = g[ok].iloc[-1]                              # el más cercano al umbral: preserva el orden de la variable
            criterio = "RD dentro del IC 95 % de los atípicos y desplazamiento ≤ 0.5 p.p.; el tramo más cercano al umbral"
        elif g["compatible"].any():
            dest = g[g["compatible"]].sort_values("desplazamiento").iloc[0]
            criterio = "RD dentro del IC 95 %; el tramo que menos se desplaza"
        else:
            dest = g.sort_values("desplazamiento").iloc[0]
            criterio = "Ningún tramo dentro del IC 95 %: el que menos se desplaza"
        valor = float(dest.name) if discreta else float(dest["mediana"])
        if bool((s % 1 == 0).all()):
            valor = float(round(valor))

        # Comparación con el tope anterior (todo lo que superaba el tope se llevaba al tope)
        cap = TOPES_ANTERIORES[v]
        s_cap = s.clip(upper=cap)
        s_new = s.where(~es_out, valor)

        def tramo_de(x):
            """Posición (0..k-1) del tramo candidato al que cae cada valor."""
            if discreta:
                pos = {val: i for i, val in enumerate(g.index)}
                return x.astype(float).map(pos)
            bordes = [iv.right for iv in g.index[:-1]]
            return pd.Series(np.searchsorted(bordes, x.to_numpy(), side="left"), index=x.index)

        t_cap, t_new = tramo_de(s_cap), tramo_de(s_new)
        pos_dest = int(dest["orden"])
        pos_cap = int(t_cap[s > cap].iloc[0]) if (s > cap).any() else None
        for t_, r in g.iterrows():
            i = int(r["orden"])
            m_cap, m_new = t_cap == i, t_new == i
            tramos_out.append({
                "variable": v, "orden": i,
                "tramo": fmt(t_) if discreta else f"{fmt(r['desde'])}–{fmt(r['hasta'])}",
                "n": int(r["n"]), "rd": r["rd"], "compatible": bool(r["compatible"]),
                "n_reasignacion": int(m_new.sum()), "rd_reasignacion": float(y[m_new].mean()),
                "n_tope": int(m_cap.sum()), "rd_tope": float(y[m_cap].mean()) if m_cap.any() else np.nan,
                "es_destino": i == pos_dest, "es_destino_tope": i == pos_cap})
        tramo_cap = pos_cap
        d_cap = next(x for x in tramos_out[::-1] if x["variable"] == v and x["es_destino_tope"]) if tramo_cap is not None else None
        d_new = next(x for x in tramos_out[::-1] if x["variable"] == v and x["es_destino"])
        # test: cuántos atípicos y su RD (la regla se aplica con lo aprendido en train)
        te = imp[~es_tr]
        te_out = te[v] > umbral
        pv = stats.chi2_contingency([[k_out, n_out - k_out], [int(dest["k"]), int(dest["n"] - dest["k"])]])[1]
        resumen.append({
            "variable": v, "regla": "Percentil 99.9 (train)" if regla == "percentil" else "Negocio: > 63 años",
            "umbral": umbral, "n_atipicos_train": n_out, "n_atipicos_test": int(te_out.sum()),
            "rd_atipicos": rd_out, "ic_bajo": lo, "ic_alto": hi,
            "rd_atipicos_test": float(te.loc[te_out, TARGET].mean()) if te_out.any() else np.nan,
            "tramo_destino": d_new["tramo"], "n_destino": d_new["n"], "rd_destino_antes": d_new["rd"],
            "rd_destino_despues": d_new["rd_reasignacion"], "valor_asignado": valor, "p_valor": pv, "criterio": criterio,
            "tope_anterior": cap, "n_afectados_tope": int((s > cap).sum()),
            "tramo_tope": d_cap["tramo"] if d_cap else None, "n_tramo_tope": d_cap["n"] if d_cap else np.nan,
            "rd_tramo_tope_antes": d_cap["rd"] if d_cap else np.nan, "rd_tramo_tope_despues": d_cap["rd_tope"] if d_cap else np.nan,
        })
        params[v] = {"umbral": umbral, "valor_asignado": valor, "regla": regla}
        trat.loc[trat[v] > umbral, v] = valor

    trat.to_parquet(ART / "tablon_tratado.parquet", index=False, compression="zstd")
    json.dump({"metodo": "Reasignación por tasa de default: los valores > umbral se llevan al tramo cuya tasa de default "
                         "es estadísticamente similar (IC 95 % de Wilson) y que, al recibirlos, no mueve su tasa más de 0.5 p.p.; "
                         "entre los compatibles, el más cercano al umbral. Aprendido solo con train.",
               "aprendido_en": "train", "semilla_particion": SEMILLA, "test_size": TEST_SIZE,
               "umbrales": {v: p["umbral"] for v, p in params.items()},
               "valores_asignados": {v: p["valor_asignado"] for v, p in params.items()},
               "topes_anteriores": TOPES_ANTERIORES},
              open(ART / "parametros_outliers.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    guardar("outliers_reasignacion", pd.DataFrame(resumen))
    guardar("outliers_tramos", pd.DataFrame(tramos_out))
    print(pd.DataFrame(resumen)[["variable", "umbral", "n_atipicos_train", "rd_atipicos", "tramo_destino", "rd_destino_antes",
                                 "rd_destino_despues", "valor_asignado", "tramo_tope", "rd_tramo_tope_antes",
                                 "rd_tramo_tope_despues"]].to_string())



# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# 2. TRAMEADO (qcut de la guía y OptBinning) — aprendido en train
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def tipos_variables() -> pd.DataFrame:
    return pd.read_parquet(ART / "univariado" / "tipos.parquet")


def es_texto(s: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(s)


def a_objeto(s: pd.Series):
    return s.astype(object).where(s.notna(), np.nan).to_numpy() if es_texto(s) else s.to_numpy(dtype=float)


def tabla_tramos(etq_tr, y_tr, etq_te, y_te, orden=None) -> pd.DataFrame:
    """n, tasa de default, WoE e IV por tramo (train) + n y tasa en test con los mismos tramos."""
    a = pd.DataFrame({"t": etq_tr, "y": y_tr}).groupby("t", observed=True, sort=False)["y"].agg(n="size", k="sum")
    b = pd.DataFrame({"t": etq_te, "y": y_te}).groupby("t", observed=True, sort=False)["y"].agg(n_test="size", k_test="sum")
    t = a.join(b, how="left").fillna({"n_test": 0, "k_test": 0})
    if orden is not None:
        t = t.reindex([o for o in orden if o in t.index])
    K, G = t["k"].sum(), (t["n"] - t["k"]).sum()
    db = (t["k"] + 0.5) / (K + 0.5 * len(t))
    dg = (t["n"] - t["k"] + 0.5) / (G + 0.5 * len(t))
    t["pct"] = t["n"] / t["n"].sum()
    t["rd"] = t["k"] / t["n"]
    t["woe"] = np.log(dg / db)
    t["iv"] = (dg - db) * t["woe"]
    t["rd_test"] = np.where(t["n_test"] > 0, t["k_test"] / t["n_test"].where(t["n_test"] > 0, 1), np.nan)
    return t.reset_index().rename(columns={"t": "tramo"})


def gini_desde_tasa(etq, y, tasa_por_tramo: dict) -> float:
    from sklearn.metrics import roc_auc_score
    score = pd.Series(etq).map(tasa_por_tramo).astype(float)
    ok = score.notna().to_numpy()
    return float(2 * roc_auc_score(np.asarray(y)[ok], score[ok]) - 1) if ok.sum() and len(set(np.asarray(y)[ok])) > 1 else np.nan


def etiquetas_qcut(x_tr: pd.Series, x_te: pd.Series, q: int):
    """qcut de la guía sobre train; test se corta con los mismos bordes. El nulo es su propio tramo."""
    v = x_tr.dropna()
    _, bordes = pd.qcut(v, q, retbins=True, duplicates="drop")
    bordes = np.unique(bordes)
    if len(bordes) < 2:
        bordes = np.array([v.min(), v.max()])
    cortes = np.r_[-np.inf, bordes[1:-1], np.inf]
    nombres = []
    for i in range(len(cortes) - 1):
        lo, hi = bordes[i], bordes[i + 1]
        nombres.append(f"[{fmt_corte(lo)}, {fmt_corte(hi)}]" if i == 0 else f"({fmt_corte(lo)}, {fmt_corte(hi)}]")
    if len(set(nombres)) < len(nombres):
        nombres = [f"T{i + 1}: {n}" for i, n in enumerate(nombres)]
    def corta(x):
        e = pd.cut(x, cortes, labels=nombres, include_lowest=True).astype(object)
        return e.where(x.notna(), "Sin dato")
    return corta(x_tr), corta(x_te), nombres + ["Sin dato"]


def patron_tendencia(rds: list) -> str:
    """Forma de la tasa de default a lo largo de tramos ordenados (sin el tramo «Sin dato»)."""
    r = [x for x in rds if pd.notna(x)]
    if len(r) < 3:
        return "Dos niveles" if len(r) == 2 else "Un solo tramo"
    if max(r) - min(r) < 0.01:
        return "Sin patrón claro"
    d = np.sign(np.diff(r))
    d = d[d != 0]
    if (d > 0).all():
        return "Monótona creciente"
    if (d < 0).all():
        return "Monótona decreciente"
    rho = stats.spearmanr(range(len(r)), r)[0]
    i_max, i_min = int(np.argmax(r)), int(np.argmin(r))
    if 0 < i_max < len(r) - 1 and abs(rho) < 0.5:
        return "No lineal (∩)"
    if 0 < i_min < len(r) - 1 and abs(rho) < 0.5:
        return "No lineal (U)"
    return "Casi monótona creciente" if rho > 0 else "Casi monótona decreciente"


def etapa_binning() -> None:
    import joblib
    from optbinning import BinningProcess
    from sklearn.metrics import roc_auc_score

    trat = pd.read_parquet(ART / "tablon_tratado.parquet")
    part = leer("particion").set_index(ID)["muestra"]
    m = trat[ID].map(part)
    tr, te = trat[m == "train"].reset_index(drop=True), trat[m == "test"].reset_index(drop=True)
    tipos = tipos_variables()
    variables = tipos["variable"].tolist()
    cat = [v for v in variables if es_texto(trat[v])]

    # ── OptBinning (Paso 6.2 de la guía) ──
    bp = BinningProcess(variable_names=variables, categorical_variables=cat, n_jobs=2, **OPTB)
    X_tr = pd.DataFrame({v: a_objeto(tr[v]) for v in variables})
    X_te = pd.DataFrame({v: a_objeto(te[v]) for v in variables})
    bp.fit(X_tr, tr[TARGET].to_numpy())
    DATASETS.mkdir(parents=True, exist_ok=True)
    joblib.dump(bp, DATASETS / "binning_process.pkl")
    resumen = bp.summary().set_index("name")

    filas_q, filas_o, met = [], [], []
    for v in variables:
        tipo = tipos.set_index("variable").loc[v, "tipo"]
        ob = bp.get_binned_variable(v)
        bt = ob.binning_table.build()
        bt = bt[bt.index != "Totals"]
        etiquetas = []
        for b in bt["Bin"]:
            if isinstance(b, str):
                etiquetas.append("Sin dato" if b == "Missing" else ("Especial" if b == "Special" else b))
            else:
                cats = [str(c) for c in list(b)]
                etiquetas.append(", ".join(cats) if len(cats) <= 4 else f"{', '.join(cats[:3])} y {len(cats) - 3} más")
        idx_tr = ob.transform(X_tr[v].to_numpy(), metric="indices", metric_missing="empirical", metric_special="empirical")
        idx_te = ob.transform(X_te[v].to_numpy(), metric="indices", metric_missing="empirical", metric_special="empirical")
        mapa = dict(enumerate(etiquetas))
        e_tr, e_te = pd.Series(idx_tr).map(mapa), pd.Series(idx_te).map(mapa)
        to = tabla_tramos(e_tr, tr[TARGET], e_te, te[TARGET], orden=etiquetas)
        miembros = {}
        for lab, b in zip(etiquetas, bt["Bin"]):
            if not isinstance(b, str):
                miembros[lab] = " · ".join(str(c) for c in list(b))
        to["miembros"] = to["tramo"].map(miembros)
        to.insert(0, "variable", v)
        to["orden"] = range(len(to))
        filas_o.append(to)
        tasa_o = dict(zip(to["tramo"], to["rd"]))
        gini_o_tr = gini_desde_tasa(e_tr, tr[TARGET], tasa_o)
        gini_o_te = gini_desde_tasa(e_te, te[TARGET], tasa_o)
        tt = to[to["n_test"] > 0]
        Kt, Gt = tt["k_test"].sum(), (tt["n_test"] - tt["k_test"]).sum()
        dbt = (tt["k_test"] + 0.5) / (Kt + 0.5 * len(tt))
        dgt = (tt["n_test"] - tt["k_test"] + 0.5) / (Gt + 0.5 * len(tt))
        iv_o_te = float(((dgt - dbt) * np.log(dgt / dbt)).sum())

        # ── qcut de la guía (Paso 6.1): 5 y 10 tramos para numéricas; categorías para el resto ──
        reg = {"variable": v, "tipo": tipo}
        for q in (5, 10):
            if tipo == "numérica":
                q_tr, q_te, orden = etiquetas_qcut(tr[v], te[v], q)
            else:
                q_tr = tr[v].astype(object).where(tr[v].notna(), "Sin dato").map(lambda z: fmt(z) if isinstance(z, (int, float)) else str(z))
                q_te = te[v].astype(object).where(te[v].notna(), "Sin dato").map(lambda z: fmt(z) if isinstance(z, (int, float)) else str(z))
                vc = q_tr.value_counts()
                orden = sorted(vc.index, key=lambda z: (z == "Sin dato", -vc[z])) if es_texto(tr[v]) else \
                    sorted(vc.index, key=lambda z: (z == "Sin dato", float(z) if z != "Sin dato" else 0))
            tq = tabla_tramos(q_tr, tr[TARGET], q_te, te[TARGET], orden=orden)
            tq.insert(0, "variable", v)
            tq.insert(1, "q", q if tipo == "numérica" else 0)
            tq["orden"] = range(len(tq))
            if tipo != "numérica" and q == 10:
                continue
            filas_q.append(tq)
            reg[f"iv_q{q}"] = float(tq["iv"].sum())
            reg[f"gini_q{q}"] = gini_desde_tasa(q_tr, tr[TARGET], dict(zip(tq["tramo"], tq["rd"])))
            if q == 5:
                ordenada = tipo == "numérica" or not es_texto(tr[v])
                sin_nulo = tq[tq["tramo"] != "Sin dato"]
                reg["patron"] = patron_tendencia(sin_nulo["rd"].tolist()) if ordenada else "Categórica (sin orden)"
                reg["rho_tramos_train_test"] = float(stats.spearmanr(sin_nulo["rd"], sin_nulo["rd_test"])[0]) \
                    if len(sin_nulo) >= 3 else np.nan
        x = tr[v]
        nn = x.dropna()
        reg.update({
            "iv": float(resumen.loc[v, "iv"]), "gini": gini_o_tr, "iv_test": iv_o_te, "gini_test": gini_o_te,
            "n_tramos_optb": int(resumen.loc[v, "n_bins"]), "pct_nulos": float(x.isna().mean()),
            "pct_dominante": float(nn.value_counts(normalize=True).iloc[0]) if len(nn) else 1.0,
            "valor_dominante": fmt(nn.value_counts().index[0]) if len(nn) and not es_texto(x) else (str(nn.value_counts().index[0]) if len(nn) else ""),
            "n_categorias": int(nn.nunique()),
        })
        met.append(reg)
    guardar("binning_qcut", pd.concat(filas_q, ignore_index=True))
    guardar("binning_optb", pd.concat(filas_o, ignore_index=True))
    met = pd.DataFrame(met).merge(tipos[["variable", "familia", "descripcion"]], on="variable", how="left")
    guardar("metricas_binning", met)
    print(met.sort_values("iv", ascending=False)[["variable", "iv", "gini", "iv_test", "gini_test", "iv_q5", "gini_q5", "patron"]]
          .head(25).to_string())


def cramer_v(a: pd.Series, b: pd.Series) -> float:
    t = pd.crosstab(a.fillna("Sin dato"), b.fillna("Sin dato"))
    chi2 = stats.chi2_contingency(t, correction=False)[0]
    n = t.to_numpy().sum()
    k = min(t.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 else np.nan


def rango_iv(iv: float) -> str:
    return ("< 0.02 · Not good" if iv < 0.02 else "0.02–0.1 · Weak" if iv < 0.1 else "0.1–0.3 · Medium" if iv < 0.3
            else "0.3–0.5 · Strong" if iv < 0.5 else "≥ 0.5 · Revisar sobreajuste")


def rango_gini(g: float) -> str:
    return ("< 0.08 · Sin poder" if g < 0.08 else "0.08–0.18 · Débil" if g < 0.18 else "0.18–0.30 · Medio" if g < 0.30
            else "0.30–0.38 · Fuerte" if g < GINI_SOSPECHOSO else "≥ 0.38 · Revisar sobreajuste")


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# 3. SELECCIÓN: univariado → bivariado → multivariado
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def etapa_seleccion() -> None:
    import joblib

    met = leer("metricas_binning").set_index("variable")
    bp = joblib.load(DATASETS / "binning_process.pkl")
    trat = pd.read_parquet(ART / "tablon_tratado.parquet")
    part = leer("particion").set_index(ID)["muestra"]
    tr = trat[trat[ID].map(part) == "train"].reset_index(drop=True)

    # ── 3.1 Filtro univariado ──
    fu, grupos = [], []
    for v, r in met.iterrows():
        decision, motivo = "Pasa", ""
        if r["pct_dominante"] >= DOMINANTE_MAX:
            decision, motivo = "Elimina", f"Varianza casi nula: «{r['valor_dominante']}» concentra el {r['pct_dominante']:.2%} de los datos"
        elif r["pct_nulos"] > NULOS_MAX and r["gini"] < GINI_MIN:
            decision, motivo = "Elimina", f"{r['pct_nulos']:.0%} de nulos y Gini {r['gini']:.3f} < {GINI_MIN}: el poder discriminante no lo compensa"
        elif r["tipo"] == "categórica" and r["n_categorias"] > CARDINALIDAD_ALTA:
            ob = bp.get_binned_variable(v)
            bt = ob.binning_table.build()
            bt = bt[(bt.index != "Totals") & (bt["Count"] > 0)]
            reales = [b for b in bt["Bin"] if not isinstance(b, str)]
            if len(reales) >= 2:
                decision = "Agrupa"
                motivo = f"{int(r['n_categorias'])} categorías → {len(reales)} grupos por tasa de default (OptBinning, train)"
                vc = tr[v].value_counts()
                rd_cat = tr.groupby(v)[TARGET].mean()
                for i, (b, (_, fila)) in enumerate(zip(bt["Bin"], bt.iterrows())):
                    nombre = "Sin dato" if isinstance(b, str) and b == "Missing" else f"G{i + 1}"
                    miembros = [str(c) for c in list(b)] if not isinstance(b, str) else ["(nulo)"]
                    for c in miembros:
                        grupos.append({"variable": v, "grupo": nombre, "categoria": c, "n": int(vc.get(c, fila["Count"] if c == "(nulo)" else 0)),
                                       "rd": float(rd_cat.get(c, fila["Event rate"])), "rd_grupo": float(fila["Event rate"]),
                                       "n_grupo": int(fila["Count"])})
            else:
                decision, motivo = "Elimina", "Alta cardinalidad sin grupos con tasa de default distinta"
        elif r["pct_nulos"] > NULOS_MAX:
            motivo = f"{r['pct_nulos']:.0%} de nulos, pero Gini {r['gini']:.3f} ≥ {GINI_MIN}: se conserva (el nulo es informativo)"
        fu.append({"variable": v, "tipo": r["tipo"], "familia": r["familia"], "pct_nulos": r["pct_nulos"],
                   "pct_dominante": r["pct_dominante"], "valor_dominante": r["valor_dominante"], "n_categorias": r["n_categorias"],
                   "iv": r["iv"], "gini": r["gini"], "decision": decision, "motivo": motivo})
    fu = pd.DataFrame(fu)
    guardar("filtro_univariado", fu)
    guardar("grupos_cardinalidad", pd.DataFrame(grupos))
    pasan_u = fu.loc[fu["decision"] != "Elimina", "variable"].tolist()

    # ── 3.2 Filtro bivariado (IV para logística, Gini para ML) ──
    bi = met.loc[pasan_u].reset_index()
    bi["rango_iv"] = bi["iv"].map(rango_iv)
    bi["rango_gini"] = bi["gini"].map(rango_gini)
    bi["pasa_logistica"] = bi["iv"] >= IV_MIN
    bi["pasa_ml"] = bi["gini"] >= GINI_MIN
    bi["sospechosa"] = (bi["iv"] >= IV_SOSPECHOSO) | (bi["gini"] >= GINI_SOSPECHOSO)
    # Estabilidad: con los tramos de OptBinning (los que miden IV y Gini), el orden de la tasa de default en test debe
    # parecerse al de train (ρ ≥ 0.5) y el Gini de test no puede caer a menos de la mitad.
    bo = leer("binning_optb")
    bo = bo[(bo["n"] > 0) & (bo["n_test"] > 0)]
    rho_o = bo.groupby("variable").apply(lambda d: stats.spearmanr(d["rd"], d["rd_test"])[0] if len(d) >= 3 else np.nan)
    bi["rho_tramos_train_test"] = bi["variable"].map(rho_o)
    bi["estable"] = ((bi["gini_test"] >= 0.5 * bi["gini"]) & (bi["rho_tramos_train_test"].fillna(1) >= 0.5))
    bi.loc[~bi["estable"], ["pasa_logistica", "pasa_ml"]] = False
    guardar("bivariado_resumen", bi)
    # Boxplot por clase (guía, Paso 5.2.1): cuantiles de cada numérica según TARGET, en train
    caja = []
    for v in bi.loc[bi["tipo"] == "numérica", "variable"]:
        for clase, g in tr.groupby(TARGET)[v]:
            x = g.dropna()
            q = x.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
            caja.append({"variable": v, "clase": "1 · default" if clase == 1 else "0 · paga", "p5": q[0.05], "p25": q[0.25],
                         "p50": q[0.5], "p75": q[0.75], "p95": q[0.95], "media": x.mean(), "n": len(x)})
    guardar("box_clase", pd.DataFrame(caja))

    # ── 3.3 Multivariado: Spearman entre numéricas (+ dicotómicas 0/1); se conserva la de mayor IV / Gini ──
    def codificar(v):
        s = tr[v]
        if es_texto(s):
            clases = sorted(s.dropna().unique(), key=str)
            return (s == clases[-1]).astype(float).where(s.notna())
        return s.astype(float)

    pares_all, sel_all, mats = [], [], []
    for dataset, col_pasa, metrica in [("logistica", "pasa_logistica", "iv"), ("ml", "pasa_ml", "gini")]:
        cand = bi[bi[col_pasa]].sort_values(metrica, ascending=False)
        # Numéricas, dicotómicas (0/1) y categóricas ordinales codificadas como número entran a Spearman (guía, Paso 6.5);
        # las categóricas nominales no, y entre ellas se usa V de Cramér con el mismo umbral (extensión a la guía).
        nominal = [v for v in cand["variable"] if cand.set_index("variable").loc[v, "tipo"] == "categórica" and es_texto(tr[v])]
        num = [v for v in cand["variable"] if v not in nominal]
        X = pd.DataFrame({v: codificar(v) for v in num})
        corr = X.corr(method="spearman", min_periods=1000) if len(num) > 1 else pd.DataFrame(1.0, index=num, columns=num)
        val = cand.set_index("variable")[metrica]
        elegidas, motivo, medida = [], {}, {}
        for v in num:
            choque = [s_ for s_ in elegidas if abs(corr.loc[v, s_]) > RHO_MAX]
            if choque:
                motivo[v], medida[v] = choque[0], float(corr.loc[v, choque[0]])
            else:
                elegidas.append(v)
        nom_ok = []
        for v in nominal:
            choque = [s_ for s_ in nom_ok if cramer_v(tr[v], tr[s_]) > RHO_MAX]
            if choque:
                motivo[v], medida[v] = choque[0], cramer_v(tr[v], tr[choque[0]])
            else:
                nom_ok.append(v)
        finales = set(elegidas) | set(nom_ok)
        for grupo, medir in [(num, lambda a_, b_: corr.loc[a_, b_]), (nominal, lambda a_, b_: cramer_v(tr[a_], tr[b_]))]:
            for i, a_ in enumerate(grupo):
                for b_ in grupo[i + 1:]:
                    rho = medir(a_, b_)
                    if pd.notna(rho) and abs(rho) > RHO_MAX:
                        hi_, lo_ = (a_, b_) if val[a_] >= val[b_] else (b_, a_)
                        if hi_ in finales and lo_ not in finales:
                            res = f"Conserva {hi_} · excluye {lo_}"
                        elif hi_ not in finales and lo_ not in finales:
                            res = "Ambas excluidas por otra variable más fuerte"
                        elif hi_ not in finales:
                            res = f"Conserva {lo_} · {hi_} ya había salido por otra variable"
                        else:
                            res = "Ambas se conservan"
                        pares_all.append({"dataset": dataset, "medida": "Spearman" if grupo is num else "V de Cramér",
                                          "variable_1": a_, "variable_2": b_, "rho": float(rho),
                                          "metrica_1": float(val[a_]), "metrica_2": float(val[b_]), "resultado": res})
        for v in cand["variable"]:
            sel_all.append({"dataset": dataset, "variable": v, "tipo": cand.set_index("variable").loc[v, "tipo"],
                            "metrica": float(val[v]), "seleccionada": v in finales,
                            "redundante_con": motivo.get(v, ""), "medida_redundancia": medida.get(v, np.nan),
                            "familia": cand.set_index("variable").loc[v, "familia"]})
        cl = corr.stack().reset_index()
        cl.columns = ["v1", "v2", "rho"]
        cl.insert(0, "dataset", dataset)
        mats.append(cl)
    guardar("multivariado_pares", pd.DataFrame(pares_all))
    guardar("multivariado_seleccion", pd.DataFrame(sel_all))
    guardar("multivariado_corr", pd.concat(mats, ignore_index=True))
    s_ = pd.DataFrame(sel_all)
    print(fu["decision"].value_counts().to_dict(), "| logística:", int(bi["pasa_logistica"].sum()), "→",
          int(s_.query("dataset=='logistica' and seleccionada").shape[0]), "| ML:", int(bi["pasa_ml"].sum()), "→",
          int(s_.query("dataset=='ml' and seleccionada").shape[0]))


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# 4. DATASETS FINALES
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
FEATURES_PROPUESTAS = {
    "RATIO_CREDITO_INGRESO": ("AMT_CREDIT / AMT_INCOME_TOTAL", lambda d: d["AMT_CREDIT"] / d["AMT_INCOME_TOTAL"],
                              "Apalancamiento: cuántos ingresos anuales representa la deuda solicitada."),
    "RATIO_CUOTA_INGRESO": ("AMT_ANNUITY / AMT_INCOME_TOTAL", lambda d: d["AMT_ANNUITY"] / d["AMT_INCOME_TOTAL"],
                            "Carga financiera (debt service ratio): parte del ingreso que se va en la cuota."),
    "RATIO_CREDITO_BIEN": ("AMT_CREDIT / AMT_GOODS_PRICE", lambda d: d["AMT_CREDIT"] / d["AMT_GOODS_PRICE"],
                           "Financiamiento sobre el precio del bien (≈ LTV): > 1 incluye seguros o comisiones."),
    "PLAZO_IMPLICITO": ("AMT_CREDIT / AMT_ANNUITY", lambda d: d["AMT_CREDIT"] / d["AMT_ANNUITY"],
                        "Número aproximado de cuotas: plazos largos suelen asociarse a más riesgo."),
    "RATIO_EMPLEO_EDAD": ("DAYS_EMPLOYED / DAYS_BIRTH", lambda d: d["DAYS_EMPLOYED"] / d["DAYS_BIRTH"],
                          "Proporción de la vida en el empleo actual: estabilidad laboral relativa a la edad."),
    "INGRESO_POR_MIEMBRO": ("AMT_INCOME_TOTAL / CNT_FAM_MEMBERS", lambda d: d["AMT_INCOME_TOTAL"] / d["CNT_FAM_MEMBERS"],
                            "Ingreso disponible por integrante del hogar."),
    "EXT_SOURCE_PROMEDIO": ("media(EXT_SOURCE_1, 2, 3)", lambda d: d[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1),
                            "Consenso de los tres scores externos."),
}


def etapa_datasets() -> None:
    import joblib
    from imblearn.over_sampling import SMOTENC
    from optbinning import OptimalBinning
    from sklearn.metrics import roc_auc_score

    bp = joblib.load(DATASETS / "binning_process.pkl")
    trat = pd.read_parquet(ART / "tablon_tratado.parquet")
    part = leer("particion").set_index(ID)["muestra"]
    m = trat[ID].map(part)
    tr, te = trat[m == "train"].reset_index(drop=True), trat[m == "test"].reset_index(drop=True)
    sel = leer("multivariado_seleccion")
    tipos = tipos_variables().set_index("variable")
    grupos = leer("grupos_cardinalidad")
    fichas, resumen = [], []

    # ── 4.1 Logística: WoE de OptBinning ajustado en train (guía, Paso 8) ──
    v_log = sel.query("dataset == 'logistica' and seleccionada").sort_values("metrica", ascending=False)["variable"].tolist()
    def woe(df):
        out = pd.DataFrame({ID: df[ID], TARGET: df[TARGET]})
        for v in v_log:
            ob = bp.get_binned_variable(v)
            out[f"WOE_{v}"] = ob.transform(a_objeto(df[v]), metric="woe", metric_missing="empirical", metric_special="empirical")
        return out
    log_tr, log_te = woe(tr), woe(te)
    log_tr.to_parquet(DATASETS / "logistica_train.parquet", index=False)
    log_te.to_parquet(DATASETS / "logistica_test.parquet", index=False)
    for v in v_log:
        fichas.append({"dataset": "logistica", "columna": f"WOE_{v}", "variable": v, "tipo": tipos.loc[v, "tipo"],
                       "transformacion": "WoE (OptBinning, train)", "metrica": float(sel.query("dataset=='logistica' and variable==@v")["metrica"].iloc[0]),
                       "familia": tipos.loc[v, "familia"], "pct_nulos_train": float(tr[v].isna().mean())})
    W = log_tr[[f"WOE_{v}" for v in v_log]].to_numpy()
    vif = []
    for j in range(W.shape[1]):
        otros = np.delete(W, j, axis=1)
        Xo = np.c_[np.ones(len(W)), otros]
        beta = np.linalg.lstsq(Xo, W[:, j], rcond=None)[0]
        r2 = 1 - ((W[:, j] - Xo @ beta) ** 2).sum() / ((W[:, j] - W[:, j].mean()) ** 2).sum()
        vif.append({"columna": f"WOE_{v_log[j]}", "vif": 1 / (1 - r2) if r2 < 1 else np.inf})
    guardar("logistica_vif", pd.DataFrame(vif))
    guardar("logistica_corr_woe", log_tr[[f"WOE_{v}" for v in v_log]].corr().stack().reset_index()
            .set_axis(["v1", "v2", "rho"], axis=1))

    # ── 4.2 ML: variables originales; categóricas agrupadas; nulos → mediana + indicador; SMOTE-NC solo en train ──
    v_ml = sel.query("dataset == 'ml' and seleccionada").sort_values("metrica", ascending=False)["variable"].tolist()
    mapa_grupo = {v: dict(zip(g["categoria"], g["grupo"])) for v, g in grupos.groupby("variable")}
    medianas, indicadores, categoricas = {}, [], []
    def preparar(df, ajustar=False):
        out = pd.DataFrame(index=df.index)
        for v in v_ml:
            s_ = df[v]
            if v in mapa_grupo:
                out[v] = s_.map(mapa_grupo[v]).where(s_.notna(), "Sin dato").fillna("Sin dato").astype(str)
                if ajustar:
                    categoricas.append(v)
            elif es_texto(s_) or tipos.loc[v, "tipo"] == "categórica":
                out[v] = s_.astype(object).where(s_.notna(), "Sin dato").map(lambda z: fmt(z) if isinstance(z, (int, float)) else str(z))
                if ajustar:
                    categoricas.append(v)
            else:
                if ajustar:
                    medianas[v] = float(tr[v].median())
                    if tr[v].isna().mean() > 0.01:
                        indicadores.append(v)
                if v in indicadores:
                    out[f"{v}__nulo"] = s_.isna().astype(int)
                out[v] = s_.fillna(medianas[v]).astype(float)
                if tipos.loc[v, "tipo"] == "dicotómica" and ajustar:
                    categoricas.append(v)
        return out
    X_tr = preparar(tr, ajustar=True)
    X_te = preparar(te)
    binarias = [c for c in X_tr.columns if c.endswith("__nulo")]
    cat_cols = list(dict.fromkeys(categoricas + binarias))
    pd.concat([tr[[ID, TARGET]], X_tr], axis=1).to_parquet(DATASETS / "ml_train.parquet", index=False)
    pd.concat([te[[ID, TARGET]], X_te], axis=1).to_parquet(DATASETS / "ml_test.parquet", index=False)
    idx_cat = [X_tr.columns.get_loc(c) for c in cat_cols]
    sm = SMOTENC(categorical_features=idx_cat, sampling_strategy=1.0, k_neighbors=5, random_state=SEMILLA)
    Xs, ys = sm.fit_resample(X_tr, tr[TARGET])
    Xs = pd.DataFrame(Xs, columns=X_tr.columns)
    sint = np.r_[np.zeros(len(X_tr), dtype=bool), np.ones(len(Xs) - len(X_tr), dtype=bool)]
    for c in X_tr.columns:
        if c not in cat_cols:
            Xs[c] = Xs[c].astype(float)
        elif c in binarias or (c in categoricas and pd.api.types.is_numeric_dtype(X_tr[c])):
            Xs[c] = Xs[c].astype(float)
    # Coherencia de las sintéticas: SMOTE-NC interpola cada continua por separado y vota las categóricas, así que puede
    # crear filas imposibles (indicador de nulo = 1 con un valor ≠ mediana, o 2.37 créditos). Se corrigen:
    for v in indicadores:
        Xs.loc[Xs[f"{v}__nulo"] == 1, v] = medianas[v]
    enteras = [c for c in X_tr.columns if c not in cat_cols and bool((X_tr[c] % 1 == 0).all())]
    Xs[enteras] = Xs[enteras].round()
    guardar("smote_correcciones", pd.DataFrame({"tipo": ["Indicador de nulo = 1 → valor = mediana", "Conteos redondeados a entero"],
                                                "variables": [", ".join(indicadores), ", ".join(enteras)]}))
    ml_s = pd.concat([pd.DataFrame({ID: np.r_[tr[ID].to_numpy(dtype=float), np.full(sint.sum(), np.nan)],
                                    TARGET: np.asarray(ys), "es_sintetica": sint}), Xs.reset_index(drop=True)], axis=1)
    ml_s.to_parquet(DATASETS / "ml_train_smote.parquet", index=False, compression="zstd")
    for c in X_tr.columns:
        base = c.replace("__nulo", "")
        fichas.append({"dataset": "ml", "columna": c, "variable": base, "tipo": "dicotómica" if c in binarias else tipos.loc[base, "tipo"],
                       "transformacion": ("Indicador de nulo (creado)" if c in binarias else
                                          "Agrupada por tasa de default" if base in mapa_grupo else
                                          f"Original · nulo → mediana ({fmt(medianas[base])})" if base in indicadores else
                                          "Original · categórica" if base in categoricas and es_texto(tr[base]) else "Original"),
                       "metrica": float(sel.query("dataset=='ml' and variable==@base")["metrica"].iloc[0]),
                       "familia": tipos.loc[base, "familia"], "pct_nulos_train": float(tr[base].isna().mean())})
    guardar("dataset_features", pd.DataFrame(fichas))

    for nombre, df_, y_ in [("Logística · train", log_tr, log_tr[TARGET]), ("Logística · test", log_te, log_te[TARGET]),
                            ("ML · train (original)", X_tr, tr[TARGET]), ("ML · train (SMOTE)", Xs, pd.Series(ys)),
                            ("ML · test", X_te, te[TARGET])]:
        n_feat = df_.shape[1] - (2 if nombre.startswith("Logística") else 0)
        resumen.append({"dataset": nombre, "filas": len(df_), "columnas": n_feat, "n_default": int(y_.sum()),
                        "n_paga": int(len(y_) - y_.sum()), "tasa_default": float(y_.mean()),
                        "sinteticas": int(sint.sum()) if "SMOTE" in nombre else 0})
    guardar("dataset_resumen", pd.DataFrame(resumen))

    # Calidad de las observaciones sintéticas: ¿se parecen a los defaults reales?
    real = Xs[(np.asarray(ys) == 1) & ~sint]
    fake = Xs[sint]
    comp, hist = [], []
    for c in X_tr.columns:
        if c in cat_cols:
            continue
        d_ks = stats.ks_2samp(real[c], fake[c]).statistic
        comp.append({"variable": c, "media_real": real[c].mean(), "media_sintetica": fake[c].mean(),
                     "std_real": real[c].std(), "std_sintetica": fake[c].std(), "D_KS": d_ks})
        lo, hi = np.percentile(real[c], [0.5, 99.5])
        if lo == hi:
            continue
        bins = np.linspace(lo, hi, 31)
        for serie, datos in [("Default real", real[c]), ("Default sintético", fake[c]), ("No default", Xs.loc[np.asarray(ys) == 0, c])]:
            h, _ = np.histogram(datos.clip(lo, hi), bins=bins, density=True)
            hist += [{"variable": c, "serie": serie, "desde": a, "hasta": b, "densidad": float(hh)} for a, b, hh in zip(bins[:-1], bins[1:], h)]
    guardar("smote_comparacion", pd.DataFrame(comp))
    guardar("smote_hist", pd.DataFrame(hist))
    cat_comp = []
    for c in cat_cols:
        for serie, datos in [("Default real", real[c]), ("Default sintético", fake[c])]:
            vc = datos.astype(str).value_counts(normalize=True)
            cat_comp += [{"variable": c, "serie": serie, "categoria": k, "pct": float(p)} for k, p in vc.items()]
    guardar("smote_categoricas", pd.DataFrame(cat_comp))
    guardar("muestra_logistica", log_tr.head(25))
    muestra_ml = pd.concat([ml_s[~ml_s["es_sintetica"]].head(12), ml_s[ml_s["es_sintetica"]].head(13)])
    guardar("muestra_ml", muestra_ml.astype({c: str for c in muestra_ml.columns if muestra_ml[c].dtype == object}))

    # ── 4.3 Espacio de feature engineering (propuestas; NO entran a los datasets) ──
    fe = []
    finales_ml = [v for v in v_ml if not es_texto(tr[v])]
    for nombre, (formula, f, lectura) in FEATURES_PROPUESTAS.items():
        x_tr = f(tr).replace([np.inf, -np.inf], np.nan)
        x_te = f(te).replace([np.inf, -np.inf], np.nan)
        ob = OptimalBinning(name=nombre, dtype="numerical", **OPTB)
        ob.fit(x_tr.to_numpy(), tr[TARGET].to_numpy())
        iv = float(ob.binning_table.build().loc["Totals", "IV"])
        w_tr = ob.transform(x_tr.to_numpy(), metric="woe", metric_missing="empirical")
        w_te = ob.transform(x_te.to_numpy(), metric="woe", metric_missing="empirical")
        rx = x_tr.rank()
        rho = {v: abs(rx.corr(tr[v].rank())) for v in finales_ml}     # Spearman = Pearson de los rangos (nulos por pares)
        top = max(rho, key=rho.get)
        bt = ob.binning_table.build()
        bt = bt[(bt.index != "Totals") & (bt["Count"] > 0)]
        fe.append({"feature": nombre, "formula": formula, "lectura": lectura, "iv": iv,
                   "gini": float(2 * roc_auc_score(tr[TARGET], -w_tr) - 1), "gini_test": float(2 * roc_auc_score(te[TARGET], -w_te) - 1),
                   "max_rho_seleccionadas": rho[top], "variable_mas_correlacionada": top,
                   "pasaria_logistica": iv >= IV_MIN, "pasaria_ml": float(2 * roc_auc_score(tr[TARGET], -w_tr) - 1) >= GINI_MIN,
                   "tramos": " | ".join(f"{b}: {r:.1%}" for b, r in zip(bt["Bin"].astype(str), bt["Event rate"]))})
    guardar("feature_engineering", pd.DataFrame(fe))
    print(pd.DataFrame(resumen).to_string())
    print(pd.DataFrame(fe)[["feature", "iv", "gini", "max_rho_seleccionadas", "variable_mas_correlacionada"]].to_string())


def fmt_corte(v) -> str:
    v = float(v)
    if abs(v) >= 1e6:
        return f"{v / 1e6:,.2f} M"
    if abs(v) >= 100 or v.is_integer():
        return f"{v:,.0f}"
    return f"{v:.3g}"


def fmt(v) -> str:
    v = float(v)
    if abs(v) >= 1e6:
        return f"{v / 1e6:,.2f} M"
    if abs(v) >= 1000:
        return f"{v / 1e3:,.0f} mil"
    return f"{v:g}"


if __name__ == "__main__":
    etapa = sys.argv[1] if len(sys.argv) > 1 else "todo"
    pasos = {"outliers": [etapa_outliers], "binning": [etapa_binning], "seleccion": [etapa_seleccion],
             "datasets": [etapa_datasets], "todo": [etapa_outliers, etapa_binning, etapa_seleccion, etapa_datasets]}
    for f in pasos[etapa]:
        f()
