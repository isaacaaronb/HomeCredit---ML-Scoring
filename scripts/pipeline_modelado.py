"""Pipeline de modelado: partición, outliers, filtros univariado / bivariado / multivariado y dataset final.

Etapas (se pueden correr por separado; cada una lee lo que dejó la anterior):

  python scripts/pipeline_modelado.py outliers    # partición 80/20 + valores imposibles, eliminación y capeo p0.1–p99.9
  python scripts/pipeline_modelado.py binning     # trameado qcut (5 y 10) y OptBinning sobre train, para todas las variables
  python scripts/pipeline_modelado.py seleccion   # filtro univariado, bivariado (IV ≥ 0.05) y multivariado (Spearman, menor IV sale)
  python scripts/pipeline_modelado.py datasets    # dataset de entrenamiento único: versión original y versión SMOTE
  python scripts/pipeline_modelado.py todo        # las cuatro, en orden

Principio de la guía («Errores frecuentes»): todo lo que se aprende de los datos —percentiles de capeo, bines, WoE, IV,
selección y SMOTE— se calcula SOLO con train y se aplica después a test.

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
TEST_SIZE = 0.20                # partición 80/20 estratificada (indicación de la profesora)

# ── Outliers (indicación de la profesora: simple; eliminar lo imposible y escaso, capear el resto) ──────────────
CAR_RELLENO = (64, 65)          # OWN_CAR_AGE: bloque de 64–65 años = código de relleno → nulo («antigüedad desconocida»)
FACTOR_EXTREMO = 3.0            # «super extremo»: más de 3 veces el percentil 99.9 (o 3 veces el 0.1 si es negativo)
MAX_FILAS_ELIMINAR = 20         # …y tan escaso (≤ 20 créditos en toda la base) que se elimina la fila; si son más, se capea
PCT_CAPEO = (0.1, 99.9)         # capeo de las dos colas en los percentiles 0.1 y 99.9 de train (comparado con 1–99)

# ── Filtros ─────────────────────────────────────────────────────────────────────────────────────────
DOMINANTE_MAX = 0.99            # univariado: un valor concentra ≥ 99 % de los datos no nulos → varianza casi nula
NULOS_MAX = 0.50                # univariado: más de 50 % de nulos…
GINI_NULOS = 0.08               # …y Gini < 0.08: el poder discriminante no compensa los nulos
CARDINALIDAD_ALTA = 15          # univariado: más de 15 categorías → agrupar por tasa de default
IV_MIN = 0.05                   # bivariado: criterio ÚNICO para todos los modelos (IV del WoE de OptBinning)
IV_SOSPECHOSO = 0.50            # se informa: posible sobreajuste / fuga
GINI_SOSPECHOSO = 0.38          # (informativo) equivale a IV = 0.50
RHO_MAX = 0.60                  # multivariado: |ρ de Spearman| > 0.6 → redundantes; sale la de menor IV
OPTB = dict(max_n_prebins=20, min_prebin_size=0.05, max_n_bins=5)   # parámetros de la guía (Paso 6.2)


def guardar(nombre: str, tabla: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tabla.to_parquet(OUT / f"{nombre}.parquet", index=False)


def leer(nombre: str) -> pd.DataFrame:
    return pd.read_parquet(OUT / f"{nombre}.parquet")


# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# 1. PARTICIÓN Y OUTLIERS
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def limite(s: pd.Series, p: float) -> float:
    """Percentil p (0–100) de una serie; en variables enteras se toma un valor observado hacia afuera de la cola."""
    s = s.dropna()
    if s.empty:
        return np.nan
    entera = bool((s % 1 == 0).all())
    modo = "linear" if not entera else ("higher" if p >= 50 else "lower")
    return float(s.quantile(p / 100, interpolation=modo))


def n_tukey(x: pd.Series) -> int:
    x = x.dropna()
    q1, q3 = x.quantile([0.25, 0.75])
    i = q3 - q1
    return int(((x < q1 - 1.5 * i) | (x > q3 + 1.5 * i)).sum())


def iv_optb(x: pd.Series, y: pd.Series) -> float:
    from optbinning import OptimalBinning
    ob = OptimalBinning(name="x", dtype="numerical", **OPTB)
    ob.fit(x.to_numpy(dtype=float), y.to_numpy())
    return float(ob.binning_table.build().loc["Totals", "IV"])


def etapa_outliers() -> None:
    """Tres pasos, en este orden:
    A. Valores sin sentido de negocio → nulo (bloque 64–65 de OWN_CAR_AGE).
    B. Valores «super extremos» (> 3 × p99.9 de train) y escasos (≤ 20 créditos): se elimina la fila (train y test).
    C. Capeo de las dos colas de cada numérica en los percentiles 0.1 y 99.9 de train (se aplica igual a test).
    """
    from sklearn.model_selection import train_test_split

    imp = pd.read_parquet(ART / "tablon_imputado.parquet")
    nums = tipos_variables().query("tipo == 'numérica'")["variable"].tolist()
    tr_ids, _ = train_test_split(imp[ID], test_size=TEST_SIZE, stratify=imp[TARGET], random_state=SEMILLA)
    es_tr = imp[ID].isin(set(tr_ids))
    trat = imp.copy()

    # ── A. Código de relleno → nulo ──
    relleno = trat["OWN_CAR_AGE"].isin(CAR_RELLENO)
    resumen_a = {"variable": "OWN_CAR_AGE", "regla": f"Valores {CAR_RELLENO[0]}–{CAR_RELLENO[1]} años → nulo",
                 "n_total": int(relleno.sum()), "n_train": int((relleno & es_tr).sum()),
                 "rd_train": float(trat.loc[relleno & es_tr, TARGET].mean()),
                 "rd_resto_con_auto": float(trat.loc[~relleno & es_tr & trat["OWN_CAR_AGE"].notna(), TARGET].mean())}
    trat.loc[relleno, "OWN_CAR_AGE"] = np.nan

    # ── B. Super extremos y escasos → eliminar la fila ──
    tr0 = trat[es_tr]
    elim, borrar = [], pd.Series(False, index=trat.index)
    for v in nums:
        s = tr0[v]
        p_hi, p_lo = limite(s, 99.9), limite(s, 0.1)
        m_hi = trat[v] > FACTOR_EXTREMO * p_hi if p_hi > 0 else pd.Series(False, index=trat.index)
        m_lo = trat[v] < FACTOR_EXTREMO * p_lo if p_lo < 0 else pd.Series(False, index=trat.index)
        for lado, m, p_ref in [("superior", m_hi, p_hi), ("inferior", m_lo, p_lo)]:
            n = int(m.sum())
            if n == 0:
                continue
            decision = "Elimina la fila" if n <= MAX_FILAS_ELIMINAR else "Se capea (demasiados casos para eliminar)"
            vals = trat.loc[m, v].sort_values(ascending=(lado == "inferior"))
            elim.append({"variable": v, "lado": lado, "p999": p_ref, "umbral": FACTOR_EXTREMO * p_ref, "n": n,
                         "n_train": int((m & es_tr).sum()), "valores": ", ".join(fmt(x) for x in vals.head(6)) + (" …" if n > 6 else ""),
                         "rd": float(trat.loc[m, TARGET].mean()), "decision": decision})
            if n <= MAX_FILAS_ELIMINAR:
                borrar |= m
    elim = pd.DataFrame(elim)
    filas_borradas = trat.loc[borrar, [ID, TARGET]].assign(muestra=np.where(es_tr[borrar], "train", "test"))
    guardar("particion", pd.DataFrame({ID: imp[ID], "muestra": np.where(es_tr, "train", "test"), "eliminada": borrar.to_numpy()}))
    trat, es_tr = trat[~borrar].copy(), es_tr[~borrar]

    # ── C. Capeo p0.1–p99.9 (train) + comparación con 1–99 ──
    tr = trat[es_tr]
    y_tr = tr[TARGET]
    cap, params = [], {}
    for v in nums:
        s = tr[v]
        if s.dropna().empty:
            continue
        lo, hi = limite(s, PCT_CAPEO[0]), limite(s, PCT_CAPEO[1])
        lo99, hi99 = limite(s, 1), limite(s, 99)
        x_all = trat[v]
        n_lo, n_hi = int((x_all < lo).sum()), int((x_all > hi).sum())
        sc, s99 = s.clip(lo, hi), s.clip(lo99, hi99)
        cap.append({"variable": v, "lim_inf": lo, "lim_sup": hi, "n_inf": n_lo, "n_sup": n_hi, "n_modificados": n_lo + n_hi,
                    "n_mod_train": int(((s < lo) | (s > hi)).sum()), "n_mod_p99_train": int(((s < lo99) | (s > hi99)).sum()),
                    "min_antes": float(x_all.min()), "max_antes": float(x_all.max()),
                    "media_antes": float(x_all.mean()), "std_antes": float(x_all.std()), "asim_antes": float(x_all.skew()),
                    "tukey_antes": n_tukey(s), "tukey_p999": n_tukey(sc), "tukey_p99": n_tukey(s99),
                    "iv_sin_capeo": iv_optb(s, y_tr), "iv_p999": iv_optb(sc, y_tr), "iv_p99": iv_optb(s99, y_tr)})
        params[v] = {"inf": lo, "sup": hi}
        trat[v] = x_all.clip(lo, hi)
        cap[-1].update({"min_despues": float(trat[v].min()), "max_despues": float(trat[v].max()), "media_despues": float(trat[v].mean()),
                        "std_despues": float(trat[v].std()), "asim_despues": float(trat[v].skew())})
    cap = pd.DataFrame(cap)

    trat.to_parquet(ART / "tablon_tratado.parquet", index=False, compression="zstd")
    json.dump({"metodo": "A) OWN_CAR_AGE 64–65 → nulo (código de relleno). B) Se eliminan las filas con valores > 3 × p99.9 "
                         "(o < 3 × p0.1 si es negativo) cuando son ≤ 20 créditos en la base. C) Capeo de las dos colas de cada "
                         "numérica en los percentiles 0.1 y 99.9 de train.",
               "aprendido_en": "train", "semilla_particion": SEMILLA, "test_size": TEST_SIZE,
               "car_relleno": list(CAR_RELLENO), "factor_extremo": FACTOR_EXTREMO, "max_filas_eliminar": MAX_FILAS_ELIMINAR,
               "percentiles_capeo": list(PCT_CAPEO), "filas_eliminadas": filas_borradas[ID].astype(int).tolist(),
               "limites": params},
              open(ART / "parametros_outliers.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    guardar("outliers_relleno", pd.DataFrame([resumen_a]))
    guardar("outliers_eliminacion", elim)
    guardar("outliers_filas_eliminadas", filas_borradas)
    guardar("outliers_capeo", cap)
    print(f"A) OWN_CAR_AGE → nulo: {resumen_a['n_total']:,} | B) filas eliminadas: {int(borrar.sum())} "
          f"(train {int((filas_borradas['muestra'] == 'train').sum())}) | C) valores capeados: {int(cap['n_modificados'].sum()):,}")
    print(elim.to_string())
    print("IV: máx |p99.9 − sin| =", float((cap["iv_p999"] - cap["iv_sin_capeo"]).abs().max()),
          "| máx |p99 − sin| =", float((cap["iv_p99"] - cap["iv_sin_capeo"]).abs().max()))



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
        elif r["pct_nulos"] > NULOS_MAX and r["gini"] < GINI_NULOS:
            decision, motivo = "Elimina", f"{r['pct_nulos']:.0%} de nulos y Gini {r['gini']:.3f} < {GINI_NULOS}: el poder discriminante no lo compensa"
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
            motivo = f"{r['pct_nulos']:.0%} de nulos, pero Gini {r['gini']:.3f} ≥ {GINI_NULOS}: se conserva (el nulo es informativo)"
        fu.append({"variable": v, "tipo": r["tipo"], "familia": r["familia"], "pct_nulos": r["pct_nulos"],
                   "pct_dominante": r["pct_dominante"], "valor_dominante": r["valor_dominante"], "n_categorias": r["n_categorias"],
                   "iv": r["iv"], "gini": r["gini"], "decision": decision, "motivo": motivo})
    fu = pd.DataFrame(fu)
    guardar("filtro_univariado", fu)
    guardar("grupos_cardinalidad", pd.DataFrame(grupos))
    pasan_u = fu.loc[fu["decision"] != "Elimina", "variable"].tolist()

    # ── 3.2 Filtro bivariado: criterio ÚNICO para todos los modelos → IV ≥ 0.05 (Gini se reporta, no decide) ──
    bi = met.loc[pasan_u].reset_index()
    bi["rango_iv"] = bi["iv"].map(rango_iv)
    bi["rango_gini"] = bi["gini"].map(rango_gini)
    bi["pasa"] = bi["iv"] >= IV_MIN
    bi["sospechosa"] = bi["iv"] >= IV_SOSPECHOSO
    # Estabilidad (informativa): con los tramos de OptBinning, el orden de la tasa de default en test debería parecerse
    # al de train (ρ ≥ 0.5) y el Gini de test no debería caer a menos de la mitad.
    bo = leer("binning_optb")
    bo = bo[(bo["n"] > 0) & (bo["n_test"] > 0)]
    rho_o = bo.groupby("variable").apply(lambda d: stats.spearmanr(d["rd"], d["rd_test"])[0] if len(d) >= 3 else np.nan)
    bi["rho_tramos_train_test"] = bi["variable"].map(rho_o)
    bi["estable"] = ((bi["gini_test"] >= 0.5 * bi["gini"]) & (bi["rho_tramos_train_test"].fillna(1) >= 0.5))
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

    # ── 3.3 Multivariado: Spearman entre numéricas (+ dicotómicas 0/1); de cada par redundante sale la de MENOR IV ──
    def codificar(v):
        s = tr[v]
        if es_texto(s):
            clases = sorted(s.dropna().unique(), key=str)
            return (s == clases[-1]).astype(float).where(s.notna())
        return s.astype(float)

    cand = bi[bi["pasa"]].sort_values("iv", ascending=False)
    # Numéricas, dicotómicas (0/1) y categóricas ordinales codificadas como número entran a Spearman (guía, Paso 6.5);
    # las categóricas nominales no, y entre ellas se usa V de Cramér con el mismo umbral (extensión a la guía).
    nominal = [v for v in cand["variable"] if cand.set_index("variable").loc[v, "tipo"] == "categórica" and es_texto(tr[v])]
    num = [v for v in cand["variable"] if v not in nominal]
    X = pd.DataFrame({v: codificar(v) for v in num})
    corr = X.corr(method="spearman", min_periods=1000) if len(num) > 1 else pd.DataFrame(1.0, index=num, columns=num)
    iv_, gini_ = cand.set_index("variable")["iv"], cand.set_index("variable")["gini"]
    elegidas, motivo, medida = [], {}, {}
    for v in num:                                   # orden de mayor a menor IV: entra si no es redundante con las que ya entraron
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
    pares = []
    for grupo, medir in [(num, lambda a_, b_: corr.loc[a_, b_]), (nominal, lambda a_, b_: cramer_v(tr[a_], tr[b_]))]:
        for i, a_ in enumerate(grupo):
            for b_ in grupo[i + 1:]:
                rho = medir(a_, b_)
                if pd.notna(rho) and abs(rho) > RHO_MAX:
                    hi_, lo_ = (a_, b_) if iv_[a_] >= iv_[b_] else (b_, a_)
                    if hi_ in finales and lo_ not in finales:
                        res = f"Conserva {hi_} · excluye {lo_}"
                    elif hi_ not in finales and lo_ not in finales:
                        res = "Ambas excluidas por otra variable de mayor IV"
                    elif hi_ not in finales:
                        res = f"Conserva {lo_} · {hi_} ya había salido por otra variable"
                    else:
                        res = "Ambas se conservan"
                    # ¿El criterio anterior (mayor Gini) habría elegido distinto en este par?
                    g_hi = a_ if gini_[a_] >= gini_[b_] else b_
                    pares.append({"medida": "Spearman" if grupo is num else "V de Cramér", "variable_1": a_, "variable_2": b_,
                                  "rho": float(rho), "iv_1": float(iv_[a_]), "iv_2": float(iv_[b_]),
                                  "gini_1": float(gini_[a_]), "gini_2": float(gini_[b_]),
                                  "gini_elegiria_otra": g_hi != hi_, "resultado": res})
    sel = [{"variable": v, "tipo": cand.set_index("variable").loc[v, "tipo"], "iv": float(iv_[v]), "gini": float(gini_[v]),
            "seleccionada": v in finales, "redundante_con": motivo.get(v, ""), "medida_redundancia": medida.get(v, np.nan),
            "familia": cand.set_index("variable").loc[v, "familia"]} for v in cand["variable"]]
    cl = corr.stack().reset_index()
    cl.columns = ["v1", "v2", "rho"]
    guardar("multivariado_pares", pd.DataFrame(pares))
    guardar("multivariado_seleccion", pd.DataFrame(sel))
    guardar("multivariado_corr", cl)
    print(fu["decision"].value_counts().to_dict(), "| bivariado (IV ≥ 0.05):", int(bi["pasa"].sum()), "→ multivariado:", len(finales))


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
    """Dataset de entrenamiento ÚNICO (las variables que salen del multivariado) en dos versiones que solo difieren en el
    balance del TARGET: original (8 % de default) y rebalanceada con SMOTE-NC. Test no se toca."""
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
    v_fin = sel.query("seleccionada").sort_values("iv", ascending=False)["variable"].tolist()
    iv_de = sel.set_index("variable")["iv"]

    # ── 4.1 Columnas del dataset: categóricas agrupadas; nulos → mediana de train + indicador (SMOTE no acepta nulos) ──
    mapa_grupo = {v: dict(zip(g["categoria"], g["grupo"])) for v, g in grupos.groupby("variable")}
    medianas, indicadores, categoricas = {}, [], []
    def preparar(df, ajustar=False):
        out = pd.DataFrame(index=df.index)
        for v in v_fin:
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
    pd.concat([tr[[ID, TARGET]], X_tr], axis=1).to_parquet(DATASETS / "train_original.parquet", index=False)
    pd.concat([te[[ID, TARGET]], X_te], axis=1).to_parquet(DATASETS / "test.parquet", index=False)

    # ── 4.2 Versión rebalanceada: SMOTE-NC solo sobre train ──
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
                                                "variables": [", ".join(indicadores) or "—", ", ".join(enteras) or "—"]}))
    ml_s = pd.concat([pd.DataFrame({ID: np.r_[tr[ID].to_numpy(dtype=float), np.full(sint.sum(), np.nan)],
                                    TARGET: np.asarray(ys), "es_sintetica": sint}), Xs.reset_index(drop=True)], axis=1)
    ml_s.to_parquet(DATASETS / "train_smote.parquet", index=False, compression="zstd")

    fichas = []
    for c in X_tr.columns:
        base = c.replace("__nulo", "")
        fichas.append({"columna": c, "variable": base, "tipo": "dicotómica" if c in binarias else tipos.loc[base, "tipo"],
                       "transformacion": ("Indicador de nulo (creado)" if c in binarias else
                                          "Agrupada por tasa de default" if base in mapa_grupo else
                                          f"Original · nulo → mediana ({fmt(medianas[base])})" if base in indicadores else
                                          "Original · categórica" if base in categoricas and es_texto(tr[base]) else "Original"),
                       "iv": float(iv_de[base]), "gini": float(sel.set_index("variable").loc[base, "gini"]),
                       "familia": tipos.loc[base, "familia"], "pct_nulos_train": float(tr[base].isna().mean())})
    guardar("dataset_features", pd.DataFrame(fichas))

    # ── 4.3 Vista WoE: cómo verá la regresión logística esas mismas variables (el WoE trata el nulo como tramo propio) ──
    def woe(df):
        out = pd.DataFrame({ID: df[ID], TARGET: df[TARGET]})
        for v in v_fin:
            ob = bp.get_binned_variable(v)
            out[f"WOE_{v}"] = ob.transform(a_objeto(df[v]), metric="woe", metric_missing="empirical", metric_special="empirical")
        return out
    log_tr, log_te = woe(tr), woe(te)
    log_tr.to_parquet(DATASETS / "woe_train.parquet", index=False)
    log_te.to_parquet(DATASETS / "woe_test.parquet", index=False)
    W = log_tr[[f"WOE_{v}" for v in v_fin]].to_numpy()
    vif = []
    for j in range(W.shape[1]):
        otros = np.delete(W, j, axis=1)
        Xo = np.c_[np.ones(len(W)), otros]
        beta = np.linalg.lstsq(Xo, W[:, j], rcond=None)[0]
        r2 = 1 - ((W[:, j] - Xo @ beta) ** 2).sum() / ((W[:, j] - W[:, j].mean()) ** 2).sum()
        vif.append({"columna": f"WOE_{v_fin[j]}", "vif": 1 / (1 - r2) if r2 < 1 else np.inf})
    guardar("woe_vif", pd.DataFrame(vif))
    guardar("woe_corr", log_tr[[f"WOE_{v}" for v in v_fin]].corr().stack().reset_index().set_axis(["v1", "v2", "rho"], axis=1))
    guardar("muestra_woe", log_tr.head(25))

    resumen = []
    for nombre, df_, y_ in [("Train original", X_tr, tr[TARGET]), ("Train SMOTE", Xs, pd.Series(ys)), ("Test", X_te, te[TARGET])]:
        resumen.append({"dataset": nombre, "filas": len(df_), "columnas": df_.shape[1], "n_default": int(y_.sum()),
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
    muestra = pd.concat([ml_s[~ml_s["es_sintetica"]].head(12), ml_s[ml_s["es_sintetica"]].head(13)])
    guardar("muestra_dataset", muestra.astype({c: str for c in muestra.columns if muestra[c].dtype == object}))

    # ── 4.4 Espacio de feature engineering (propuestas; NO entran al dataset) ──
    fe = []
    finales_num = [v for v in v_fin if not es_texto(tr[v])]
    for nombre, (formula, f, lectura) in FEATURES_PROPUESTAS.items():
        x_tr = f(tr).replace([np.inf, -np.inf], np.nan)
        x_te = f(te).replace([np.inf, -np.inf], np.nan)
        ob = OptimalBinning(name=nombre, dtype="numerical", **OPTB)
        ob.fit(x_tr.to_numpy(), tr[TARGET].to_numpy())
        iv = float(ob.binning_table.build().loc["Totals", "IV"])
        w_tr = ob.transform(x_tr.to_numpy(), metric="woe", metric_missing="empirical")
        w_te = ob.transform(x_te.to_numpy(), metric="woe", metric_missing="empirical")
        rx = x_tr.rank()
        rho = {v: abs(rx.corr(tr[v].rank())) for v in finales_num}     # Spearman = Pearson de los rangos (nulos por pares)
        top = max(rho, key=rho.get)
        bt = ob.binning_table.build()
        bt = bt[(bt.index != "Totals") & (bt["Count"] > 0)]
        fe.append({"feature": nombre, "formula": formula, "lectura": lectura, "iv": iv,
                   "gini": float(2 * roc_auc_score(tr[TARGET], -w_tr) - 1), "gini_test": float(2 * roc_auc_score(te[TARGET], -w_te) - 1),
                   "max_rho_seleccionadas": rho[top], "variable_mas_correlacionada": top, "pasaria": iv >= IV_MIN,
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
