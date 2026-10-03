"""Precalcula todo lo que muestra la página «EDA univariado» (Paso 4 de la guía).

Dos pasadas, como pide la guía:
  - ex-ante : tablón oficial tal como llega        (artifacts/tablon_general.parquet)
  - ex-post : tablón tras el preprocesamiento      (artifacts/tablon_tratado.parquet)

La metodología replica la del notebook `notebooks/01_eda_univariado.ipynb` (Paso 4):
  - clasificación de variables con la regla del notebook (≤ 5 valores numéricos → categórica), fijada con el
    tablón ex-ante para que el tratamiento no cambie el tipo de una variable;
  - métricas numéricas, centinela, conteos (enteros ≥ 0 con ≤ 150 valores);
  - ajuste de distribuciones por máxima verosimilitud sobre 20,000 observaciones (semilla 42), elección por AIC y
    D de Kolmogorov–Smirnov; continuas: normal, logística, skew-normal, lognormal, gamma, Weibull, exponencial, beta;
    conteos: Poisson y binomial negativa;
  - vista del histograma p0.5–p99.5 (las métricas usan todos los datos); vista log1p si min ≥ 0, asimetría > 2 y no
    es conteo;
  - categóricas: entropía normalizada, HHI, categorías raras (< 1 %); dicotómicas: casi constantes (< 1 %).

Salida: artifacts/univariado/*.parquet + resumen.json        Uso: python scripts/build_univariado.py
"""
import json
import re
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts"
OUT = ART / "univariado"
SEMILLA = 42
N_MUESTRA_AJUSTE = 20_000
UMBRAL_CAT_NUM = 5
FORZAR_TIPO = {"AMT_REQ_CREDIT_BUREAU_HOUR": "numérica"}
UMBRAL_CENTINELA_PCT, K_IQR_CENTINELA, MAX_UNICOS_CONTEO = 0.01, 10, 150
UMBRAL_MINORITARIA, UMBRAL_CATEGORIA_RARA, UMBRAL_CARDINALIDAD = 0.01, 0.01, 15
N_SCATTER = 4_000
ID, TARGET = "SK_ID_CURR", "TARGET"

FAMILIAS_CONTINUAS = {
    "normal": stats.norm, "logística": stats.logistic, "skew-normal": stats.skewnorm,
    "lognormal": stats.lognorm, "gamma": stats.gamma, "weibull": stats.weibull_min,
    "exponencial": stats.expon, "beta": stats.beta,
}
FAMILIAS_CON_COTA = {"lognormal", "gamma", "weibull", "exponencial"}
FAMILIAS_CONTEO = {"poisson": stats.poisson, "binomial negativa": stats.nbinom}
DIST_TODAS = {**FAMILIAS_CONTINUAS, **FAMILIAS_CONTEO}


def clasificar(serie: pd.Series) -> str:
    n = serie.nunique(dropna=True)
    if n <= 1:
        return "constante"
    if n == 2:
        return "dicotómica"
    if not pd.api.types.is_numeric_dtype(serie):
        return "categórica"
    return "categórica" if n <= UMBRAL_CAT_NUM else "numérica"


# ── Numéricas ────────────────────────────────────────────────────────────────────────────────────────
def metricas_numericas(serie: pd.Series) -> dict:
    x = serie.dropna()
    q = x.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    q1, med, q3 = q[0.25], q[0.50], q[0.75]
    iqr = q3 - q1
    media, std = x.mean(), x.std()
    vc = x.value_counts()
    valor_moda, pct_moda = vc.index[0], vc.iloc[0] / len(x)
    if pct_moda >= UMBRAL_CENTINELA_PCT and iqr > 0 and abs(valor_moda - med) > K_IQR_CENTINELA * iqr:
        centinela, pct_centinela = valor_moda, pct_moda
    else:
        centinela, pct_centinela = np.nan, 0.0
    es_entero = bool(np.all(np.mod(x.to_numpy(), 1) == 0))
    return {
        "n_validos": len(x), "fill_rate": len(x) / len(serie), "pct_missing": 1 - len(x) / len(serie),
        "media": media, "std": std, "cv": std / abs(media) if media != 0 else np.nan,
        "min": x.min(), "p1": q[0.01], "p5": q[0.05], "p10": q[0.10], "p25": q1, "p50": med, "p75": q3,
        "p90": q[0.90], "p95": q[0.95], "p99": q[0.99], "max": x.max(), "rango": x.max() - x.min(), "iqr": iqr,
        "asimetria": x.skew(), "curtosis": x.kurt(), "pct_ceros": (x == 0).mean(), "pct_negativos": (x < 0).mean(),
        "pct_outliers_iqr": ((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).mean(),
        "n_outliers_iqr": int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum()),
        "pct_outliers_extremos": ((x < q1 - 3.0 * iqr) | (x > q3 + 3.0 * iqr)).mean(),
        "n_unicos": x.nunique(), "valor_moda": valor_moda, "pct_moda": pct_moda,
        "centinela": centinela, "pct_centinela": pct_centinela,
        "es_conteo": es_entero and x.min() >= 0 and x.nunique() <= MAX_UNICOS_CONTEO,
    }


def ajustar_continua(x):
    xmin, xmax = x.min(), x.max()
    loc0 = xmin - 1e-3 * (xmax - xmin)
    filas, params = [], {}
    for nombre, dist in FAMILIAS_CONTINUAS.items():
        try:
            xx = x
            if nombre in FAMILIAS_CON_COTA:
                p = dist.fit(x, floc=loc0)
            elif nombre == "beta":
                if xmin < 0 or xmax > 1:
                    continue
                xx = np.clip(x, 1e-6, 1 - 1e-6)
                p = dist.fit(xx, floc=0, fscale=1)
            else:
                p = dist.fit(x)
            logl = float(np.sum(dist.logpdf(xx, *p)))
            if not np.isfinite(logl):
                continue
            d_ks = float(stats.kstest(xx, dist.cdf, args=p).statistic)
            filas.append({"distribucion": nombre, "k": len(p), "logL": logl, "AIC": 2 * len(p) - 2 * logl, "D_KS": d_ks})
            params[nombre] = [float(v) for v in p]
        except Exception:
            continue
    return filas, params


def ajustar_conteo(x):
    x = x.astype(int)
    n, m, v = len(x), x.mean(), x.var(ddof=1)
    soporte = np.arange(0, x.max() + 1)
    ecdf = np.searchsorted(np.sort(x), soporte, side="right") / n
    filas, params = [], {}
    candidatas = {"poisson": (m,)}
    if v > m > 0:
        r = m ** 2 / (v - m)
        candidatas["binomial negativa"] = (r, r / (r + m))
    for nombre, p in candidatas.items():
        dist = FAMILIAS_CONTEO[nombre]
        logl = float(np.sum(dist.logpmf(x, *p)))
        if not np.isfinite(logl):
            continue
        d_ks = float(np.max(np.abs(ecdf - dist.cdf(soporte, *p))))
        filas.append({"distribucion": nombre, "k": len(p), "logL": logl, "AIC": 2 * len(p) - 2 * logl, "D_KS": d_ks})
        params[nombre] = [float(v) for v in p]
    return filas, params


def analizar_numerica(tarea):
    """Todo lo gráfico de una variable numérica en una pasada (se ejecuta en paralelo)."""
    pasada, col, x_all, x_fit, m = tarea
    sent = m["centinela"]
    x = x_all if np.isnan(sent) else x_all[x_all != sent]
    res = {"ajuste": [], "hist": [], "curvas": [], "qq": [], "box": None, "box_pts": [], "log": None, "log_hist": [], "log_curva": []}
    if len(x) == 0 or x.min() == x.max():
        return pasada, col, res
    es_conteo = bool(m["es_conteo"])
    filas, params = ajustar_conteo(x_fit) if es_conteo else ajustar_continua(x_fit)
    tabla = pd.DataFrame(filas).sort_values("AIC").reset_index(drop=True)
    tabla["dAIC"] = tabla["AIC"] - tabla["AIC"].min()
    tabla["elegida"] = tabla.index == 0
    res["ajuste"] = [{"pasada": pasada, "variable": col, **r} for r in tabla.to_dict("records")]
    mejor = tabla.loc[0, "distribucion"]

    # Histograma en la vista p0.5–p99.5; densidad relativa a TODOS los datos (compatible con la pdf ajustada)
    lo, hi = np.percentile(x, [0.5, 99.5])
    if lo == hi:
        lo, hi = x.min(), x.max()
    vis = x[(x >= lo) & (x <= hi)]
    if es_conteo:
        edges = np.arange(np.floor(lo) - 0.5, np.ceil(hi) + 1.5, 1.0)
    else:
        n_bins = int(np.clip(len(np.histogram_bin_edges(vis, bins="fd")) - 1, 25, 70))
        edges = np.linspace(lo, hi, n_bins + 1)
    cnt, edges = np.histogram(vis, bins=edges)
    ancho = np.diff(edges)
    res["hist"] = [{"pasada": pasada, "variable": col, "desde": float(a), "hasta": float(b), "n": int(c),
                    "densidad": float(c / (len(x) * w))} for a, b, c, w in zip(edges[:-1], edges[1:], cnt, ancho)]

    # Curvas teóricas: la mejor y la normal (referencia)
    for nombre in dict.fromkeys([mejor, "normal"]):
        if es_conteo:
            if nombre == "normal":
                grid = np.linspace(edges[0], edges[-1], 200)
                y = stats.norm.pdf(grid, x.mean(), x.std())
            else:
                grid = np.arange(np.ceil(lo), np.floor(hi) + 1)
                y = FAMILIAS_CONTEO[nombre].pmf(grid, *params[nombre])
        else:
            if nombre not in params:
                continue
            grid = np.linspace(lo, hi, 200)
            y = DIST_TODAS[nombre].pdf(np.clip(grid, 1e-6, 1 - 1e-6) if nombre == "beta" else grid, *params[nombre])
        rol = "mejor" if nombre == mejor else "normal"
        res["curvas"] += [{"pasada": pasada, "variable": col, "distribucion": nombre, "rol": rol, "x": float(g), "y": float(v)}
                          for g, v in zip(grid, y) if np.isfinite(v)]

    # Q-Q contra la mejor distribución
    probs = np.linspace(0.005, 0.995, 199)
    obs = np.quantile(x, probs)
    dist = DIST_TODAS[mejor]
    teo = dist.ppf(probs, *params[mejor])
    res["qq"] = [{"pasada": pasada, "variable": col, "p": float(p), "teorico": float(t), "observado": float(o)}
                 for p, t, o in zip(probs, teo, obs) if np.isfinite(t)]

    # Boxplot. Convención acordada con la profesora: los bigotes llegan a los percentiles del capeo (0.1 y 99.9).
    # Ex-post son los límites de capeo aprendidos en train, así que tras el tratamiento no queda ningún punto fuera;
    # ex-ante son los percentiles 0.1 y 99.9 de la base original y los puntos son justo lo que se capeó o eliminó.
    # La regla de Tukey (1.5·IQR) se sigue reportando como dato.
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    iqr = q3 - q1
    t_lo = float(x[x >= q1 - 1.5 * iqr].min())
    t_hi = float(x[x <= q3 + 1.5 * iqr].max())
    if m.get("lim_inf") is not None and np.isfinite(m["lim_inf"]):
        wl, wh = float(m["lim_inf"]), float(m["lim_sup"])
    else:
        entera = bool(np.all(x % 1 == 0))
        wl = float(np.quantile(x, 0.001, method="lower" if entera else "linear"))
        wh = float(np.quantile(x, 0.999, method="higher" if entera else "linear"))
    res["box"] = {"pasada": pasada, "variable": col, "q1": q1, "mediana": med, "q3": q3, "bigote_inf": wl, "bigote_sup": wh,
                  "tukey_inf": t_lo, "tukey_sup": t_hi, "n_fuera_tukey": int(((x < t_lo) | (x > t_hi)).sum()),
                  "media": float(x.mean()), "vista_min": float(min(lo, wl)), "vista_max": float(max(hi, wh)),
                  "n_fuera": int(((x < wl) | (x > wh)).sum()), "n_fuera_vista": int(((x < lo) | (x > hi)).sum())}
    fuera = x[(x < wl) | (x > wh)]
    if len(fuera):
        rg = np.random.default_rng(SEMILLA)
        fuera = rg.choice(fuera, size=min(len(fuera), 500), replace=False)
        jit = rg.uniform(-0.3, 0.3, len(fuera))
        res["box_pts"] = [{"pasada": pasada, "variable": col, "valor": float(v), "jitter": float(j)} for v, j in zip(fuera, jit)]

    # Vista log1p (continuas no negativas de cola derecha pesada)
    if m["min"] >= 0 and m["asimetria"] > 2 and not es_conteo:
        z = np.log1p(x[x > -1])
        mu, sd = z.mean(), z.std()
        rg = np.random.default_rng(SEMILLA)
        d_ks = stats.kstest(rg.choice(z, min(len(z), N_MUESTRA_AJUSTE), replace=False), "norm", args=(mu, sd)).statistic
        lo_z, hi_z = np.percentile(z, [0.5, 99.5])
        vz = z[(z >= lo_z) & (z <= hi_z)]
        nb = int(np.clip(len(np.histogram_bin_edges(vz, bins="fd")) - 1, 25, 70))
        cz, ez = np.histogram(vz, bins=np.linspace(lo_z, hi_z, nb + 1))
        wz = np.diff(ez)
        res["log"] = {"pasada": pasada, "variable": col, "asim_log": float(pd.Series(z).skew()), "curt_log": float(pd.Series(z).kurt()),
                      "D_KS_normal_log": float(d_ks), "media_log": float(mu), "std_log": float(sd)}
        res["log_hist"] = [{"pasada": pasada, "variable": col, "desde": float(a), "hasta": float(b), "densidad": float(c / (len(z) * w))}
                           for a, b, c, w in zip(ez[:-1], ez[1:], cz, wz)]
        g = np.linspace(lo_z, hi_z, 200)
        res["log_curva"] = [{"pasada": pasada, "variable": col, "x": float(a), "y": float(b)} for a, b in zip(g, stats.norm.pdf(g, mu, sd))]
    return pasada, col, res


def top_valores(serie: pd.Series, k: int, dropna: bool) -> list:
    vc = serie.value_counts(dropna=dropna)
    n = len(serie) if not dropna else serie.notna().sum()
    return [{"rango": i + 1, "valor": "Nulo" if pd.isna(v) else v, "n": int(c), "pct": c / n} for i, (v, c) in enumerate(vc.head(k).items())]


def fmt_valor(v) -> str:
    if isinstance(v, str):
        return v
    v = float(v)
    return f"{v:,.0f}" if v.is_integer() else f"{v:,.4g}"


# ── Categóricas y dicotómicas ──────────────────────────────────────────────────────────────────────
def metricas_categorica(serie: pd.Series) -> dict:
    x = serie.dropna()
    p = x.value_counts(normalize=True)
    k = len(p)
    h = float(-(p * np.log2(p)).sum())
    return {
        "n_validos": len(x), "fill_rate": len(x) / len(serie), "n_categorias": k, "moda": fmt_valor(p.index[0]),
        "pct_moda": p.iloc[0], "pct_top5": float(p.head(5).sum()), "pct_top10": float(p.head(10).sum()),
        "entropia_norm": h / np.log2(k) if k > 1 else 0.0, "hhi": float((p ** 2).sum()),
        "n_cat_raras": int((p < UMBRAL_CATEGORIA_RARA).sum()), "pct_obs_en_raras": float(p[p < UMBRAL_CATEGORIA_RARA].sum()),
        "cardinalidad_alta": k > UMBRAL_CARDINALIDAD,
        "n_codificados": int(serie.astype(str).isin(["XNA", "Unknown"]).sum()),
    }


def metricas_dicotomica(serie: pd.Series) -> dict:
    x = serie.dropna()
    vc = x.value_counts()
    n_may, n_min = vc.iloc[0], vc.iloc[1]
    n_otros = int(vc.iloc[2:].sum()) if len(vc) > 2 else 0       # p. ej. CODE_GENDER = 'XNA' en la pasada ex-ante
    p_min = n_min / len(serie)
    pm = n_min / (n_may + n_min)
    return {
        "n_validos": len(x), "fill_rate": len(x) / len(serie), "clase_mayoritaria": str(vc.index[0]),
        "clase_minoritaria": str(vc.index[1]), "n_mayoritaria": int(n_may), "n_minoritaria": int(n_min),
        "pct_mayoritaria": n_may / len(serie), "pct_minoritaria": p_min, "pct_otros": n_otros / len(serie),
        "pct_nulo": 1 - len(x) / len(serie), "pct_minoritaria_validos": pm, "razon_desbalance": n_may / n_min,
        "entropia_binaria": float(-(pm * np.log2(pm) + (1 - pm) * np.log2(1 - pm))), "casi_constante": pm < UMBRAL_MINORITARIA,
    }


LIMITES: dict = {}
# Indicadores creados en el preprocesamiento: van a la familia de la fuente cuyo faltante marcan
FAMILIA_FLAGS = {"flag_sin_buro": "05 · Buró de crédito", "flag_sin_previas": "06 · Historial en Home Credit",
                 "flag_sin_empleo": "02 · Perfil del solicitante", "flag_sin_info_vivienda": "03 · Vivienda y entorno"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bases = {"ex-ante": pd.read_parquet(ART / "tablon_general.parquet"), "ex-post": pd.read_parquet(ART / "tablon_tratado.parquet")}
    dicc = pd.read_csv(REPO / "data_dictionary" / "diccionario_tablon.csv").set_index("variable")
    ante, post = bases["ex-ante"], bases["ex-post"]
    global LIMITES
    LIMITES = json.load(open(ART / "parametros_outliers.json", encoding="utf-8")).get("limites", {})
    N = len(post)

    # ── Tipos: regla del notebook sobre la base ex-ante (estable frente al tratamiento) ──
    filas_tipo = []
    for c in post.columns:
        if c in (ID, TARGET):
            continue
        base_c = ante[c] if c in ante else post[c]
        tipo = FORZAR_TIPO.get(c, clasificar(base_c))
        nota = ""
        if c == "CODE_GENDER":
            tipo, nota = "dicotómica", "Ex-ante tenía 3 valores (F, M y 4 'XNA'); 'XNA' era un faltante codificado."
        elif c not in ante:
            nota = "Creada en el preprocesamiento."
        elif clasificar(post[c]) != clasificar(ante[c]) and c not in FORZAR_TIPO:
            nota = f"Con la regla (≤ {UMBRAL_CAT_NUM} valores) sería «{clasificar(post[c])}» tras el capeo; se mantiene «{tipo}»."
        familia = dicc.loc[c, "familia"] if c in dicc.index else FAMILIA_FLAGS.get(c, "02 · Perfil del solicitante")
        bloque = dicc.loc[c, "bloque"] if c in dicc.index else "Indicadores del preprocesamiento"
        filas_tipo.append({"variable": c, "tipo": tipo, "familia": familia, "bloque": bloque,
                           "tipo_dato": dicc.loc[c, "tipo_dato"] if c in dicc.index else "Dicotómica",
                           "descripcion": dicc.loc[c, "descripcion"] if c in dicc.index else "", "nota_tipo": nota,
                           "orden": int(dicc.loc[c, "orden"]) if c in dicc.index else 1000 + len(filas_tipo)})
    tipos = pd.DataFrame(filas_tipo)
    tipos.to_parquet(OUT / "tipos.parquet", index=False)
    NUM = tipos.query("tipo == 'numérica'")["variable"].tolist()
    CAT = tipos.query("tipo == 'categórica'")["variable"].tolist()
    DIC = tipos.query("tipo == 'dicotómica'")["variable"].tolist()

    # ── Numéricas: métricas, top-5 y tareas de ajuste ──
    met, top5, tareas = [], [], []
    for pasada, df in bases.items():
        rng = np.random.default_rng(SEMILLA)                 # misma secuencia que el notebook
        for c in NUM:
            m = metricas_numericas(df[c])
            met.append({"pasada": pasada, "variable": c, **m})
            top5 += [{"pasada": pasada, "variable": c, **r, "valor": fmt_valor(r["valor"])} for r in top_valores(df[c], 5, dropna=True)]
            x = df[c].dropna().to_numpy(dtype=float)
            if not np.isnan(m["centinela"]):
                x = x[x != m["centinela"]]
            x_fit = rng.choice(x, size=min(len(x), N_MUESTRA_AJUSTE), replace=False)
            lim = LIMITES.get(c) if pasada == "ex-post" else None
            m_t = dict(m, lim_inf=lim["inf"] if lim else None, lim_sup=lim["sup"] if lim else None)
            tareas.append((pasada, c, x if len(x) else np.array([0.0]), x_fit, m_t))
    met = pd.DataFrame(met)

    print(f"Ajustando {len(tareas)} distribuciones…", flush=True)
    acum = {k: [] for k in ["ajuste", "hist", "curvas", "qq", "box_pts", "log_hist", "log_curva"]}
    box, logs = [], []
    with ProcessPoolExecutor(max_workers=2) as ex:
        for i, (pasada, col, res) in enumerate(ex.map(analizar_numerica, tareas, chunksize=2), 1):
            for k in acum:
                acum[k] += res[k]
            if res["box"]:
                box.append(res["box"])
            if res["log"]:
                logs.append(res["log"])
            if i % 20 == 0:
                print(f"  {i}/{len(tareas)}", flush=True)
    for k, v in acum.items():
        pd.DataFrame(v).to_parquet(OUT / f"num_{k}.parquet", index=False)
    pd.DataFrame(box).to_parquet(OUT / "num_box.parquet", index=False)
    pd.DataFrame(logs).to_parquet(OUT / "num_log.parquet", index=False)
    pd.DataFrame(top5).to_parquet(OUT / "num_top5.parquet", index=False)

    aj = pd.DataFrame(acum["ajuste"])
    mejor = aj[aj["elegida"]].set_index(["pasada", "variable"])[["distribucion", "D_KS"]]
    d_norm = aj[aj["distribucion"] == "normal"].set_index(["pasada", "variable"])["D_KS"]
    met = met.set_index(["pasada", "variable"])
    met["mejor_distribucion"] = mejor["distribucion"]
    met["D_KS_mejor"] = mejor["D_KS"]
    met["D_KS_normal"] = d_norm
    met = met.reset_index()
    met.to_parquet(OUT / "num_metricas.parquet", index=False)

    # ── Categóricas ──
    catm, catf = [], []
    for pasada, df in bases.items():
        for c in CAT:
            catm.append({"pasada": pasada, "variable": c, **metricas_categorica(df[c])})
            s = df[c]
            vc = s.value_counts(dropna=False)
            ordinal = pd.api.types.is_numeric_dtype(s)
            for v, n in vc.items():
                catf.append({"pasada": pasada, "variable": c, "categoria": "Nulo" if pd.isna(v) else (fmt_valor(v) if ordinal else str(v)),
                             "orden_num": float(v) if ordinal and not pd.isna(v) else np.nan, "n": int(n), "pct": n / N,
                             "es_nulo": bool(pd.isna(v)), "codificado": str(v) in ("XNA", "Unknown")})
    pd.DataFrame(catm).to_parquet(OUT / "cat_metricas.parquet", index=False)
    pd.DataFrame(catf).to_parquet(OUT / "cat_frecuencias.parquet", index=False)

    # ── Dicotómicas y objetivo ──
    dic = []
    for pasada, df in bases.items():
        for c in DIC:
            if c in df:
                dic.append({"pasada": pasada, "variable": c, **metricas_dicotomica(df[c])})
    pd.DataFrame(dic).to_parquet(OUT / "dic_metricas.parquet", index=False)
    vc_t = post[TARGET].value_counts().sort_index()

    # ── Muestra para dispersión numérica–numérica ──
    idx = np.random.default_rng(SEMILLA).choice(N, size=N_SCATTER, replace=False)
    muestra = pd.concat([ante.iloc[idx][NUM].assign(pasada="ex-ante"), post.iloc[idx][NUM].assign(pasada="ex-post")])
    muestra.to_parquet(OUT / "muestra_dispersion.parquet", index=False)

    # ── Vista previa de redundancia en vivienda (_AVG/_MODE/_MEDI): se confirma en el multivariado ──
    red = []
    bases_viv = sorted({re.sub(r"_(AVG|MODE|MEDI)$", "", c) for c in NUM if re.search(r"_(AVG|MODE|MEDI)$", c)})
    for b in bases_viv:
        trio = [f"{b}_{s}" for s in ("AVG", "MEDI", "MODE") if f"{b}_{s}" in post]
        if len(trio) < 2:
            continue
        sub = post[trio].dropna()
        rho = sub.rank().corr()
        red.append({"base": b, "n_variables": len(trio), "rho_avg_medi": rho.iloc[0, 1] if len(trio) > 1 else np.nan,
                    "rho_avg_mode": rho.iloc[0, 2] if len(trio) > 2 else np.nan,
                    "pct_avg_igual_medi": float((sub.iloc[:, 0] == sub.iloc[:, 1]).mean()), "n_completos": len(sub)})
    pd.DataFrame(red).to_parquet(OUT / "redundancia_vivienda.parquet", index=False)

    # ── Dicotómicas redundantes: pares idénticos o complementarios en ≥ 99.9 % de las filas (ex-post) ──
    bin01 = {}
    for c in DIC:
        s = post[c]
        if s.isna().any():
            continue
        clases = sorted(s.unique(), key=str)
        v = (s == clases[1]).to_numpy()
        if min(v.mean(), 1 - v.mean()) >= UMBRAL_MINORITARIA:     # las casi constantes coincidirían entre sí por azar
            bin01[c] = v
    red_d, nombres = [], list(bin01)
    for i, a in enumerate(nombres):
        for b in nombres[i + 1:]:
            igual = float((bin01[a] == bin01[b]).mean())
            acuerdo, relacion = (igual, "idéntica") if igual >= 0.5 else (1 - igual, "complementaria")
            if acuerdo >= 0.999:
                red_d.append({"variable_a": a, "variable_b": b, "relacion": relacion, "acuerdo": acuerdo,
                              "n_discrepancias": int(round((1 - acuerdo) * N))})
    pd.DataFrame(red_d).to_parquet(OUT / "redundancia_dicotomicas.parquet", index=False)

    # ── Verificación ex-ante vs ex-post (guía, Paso 4, precisión b) ──
    ver = []
    for pasada, df in bases.items():
        mm = met[met.pasada == pasada]
        cm = pd.DataFrame(catm).query("pasada == @pasada")
        cols = [c for c in df.columns if c not in (ID, TARGET)]
        ver.append({
            "pasada": pasada, "columnas": len(cols), "pct_celdas_nulas": float(df[cols].isna().mean().mean()),
            "variables_con_nulos": int(df[cols].isna().any().sum()),
            "num_asim_mayor_2": int((mm["asimetria"].abs() > 2).sum()), "asim_max": float(mm["asimetria"].abs().max()),
            "num_outliers_mayor_5": int((mm["pct_outliers_iqr"] > 0.05).sum()),
            "num_con_centinela": int(mm["centinela"].notna().sum()),
            "cat_codificados": int(cm["n_codificados"].sum()), "cat_cardinalidad_max": int(cm["n_categorias"].max()),
        })
    pd.DataFrame(ver).to_parquet(OUT / "verificacion.parquet", index=False)

    resumen = {"filas": N, "n_numericas": len(NUM), "n_categoricas": len(CAT), "n_dicotomicas": len(DIC),
               "target_0": int(vc_t.loc[0]), "target_1": int(vc_t.loc[1]), "tasa_default": float(post[TARGET].mean()),
               "n_muestra_ajuste": N_MUESTRA_AJUSTE, "semilla": SEMILLA}
    json.dump(resumen, open(OUT / "resumen.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"Listo: {len(list(OUT.glob('*.parquet')))} tablas en {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
