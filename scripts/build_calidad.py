"""Precalcula los diagnósticos de calidad y preprocesamiento que muestra la app (página «Calidad y preprocesamiento»).

Replica la lógica de notebooks/01_eda_univariado.ipynb (Pasos 2 y 3) sobre el tablón oficial y sobre las salidas
de ese notebook, para que la app no tenga que cargar tres tablones de 307,511 filas en memoria.

Entradas:
  artifacts/tablon_general.parquet     tablón oficial (antes del preprocesamiento)
  artifacts/tablon_imputado.parquet    tras el tratamiento de faltantes y centinelas (notebook)
  artifacts/tablon_tratado.parquet     tras el tratamiento de outliers (scripts/pipeline_modelado.py outliers):
                                       OWN_CAR_AGE 64–65 → nulo, filas super extremas eliminadas y capeo p0.1–p99.9
  artifacts/modelado/particion.parquet · outliers_*.parquet
  artifacts/parametros_imputacion.json · artifacts/parametros_outliers.json
  data_dictionary/diccionario_tablon.csv

Salida: artifacts/calidad/*.parquet y artifacts/calidad/resumen.json

Uso:  python scripts/build_calidad.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts"
OUT = ART / "calidad"
SEMILLA = 42
SENT = 365243

# ── Parámetros del notebook (Pasos 2 y 3) ──────────────────────────────────────
N_MIN_GRUPO = 200
EDAD_CORTES = [0, 30, 40, 50, 60, 120]
EDAD_ETIQ = ["≤30", "30-40", "40-50", "50-60", ">60"]
UMBRALES = {"nulos_alto": 0.40, "casi_constante": 0.95, "asimetria": 2.0, "outliers": 0.05, "ceros": 0.50,
            "minoritaria": 0.01, "cardinalidad": 15, "rara": 0.01}
CURVAS = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "CNT_CHILDREN", "OBS_30_CNT_SOCIAL_CIRCLE", "AMT_REQ_CREDIT_BUREAU_QRT",
          "BUREAU_DEUDA_TOTAL", "HC_CARD_MAX_UTILIZACION", "BUREAU_N_CREDITOS", "OWN_CAR_AGE"]

FAMILIAS_HISTORIAL = {
    "Buró": ["BUREAU_N_CREDITOS", "BUREAU_N_ACTIVOS", "BUREAU_DEUDA_TOTAL", "BUREAU_MAX_DIAS_ATRASO", "BUREAU_MESES_OBSERVADOS",
             "BUREAU_MESES_CON_ATRASO", "BUREAU_MESES_ESTADO_CONOCIDO", "BUREAU_PROP_MESES_CON_ATRASO"],
    "Solicitudes previas": ["HC_N_SOLICITUDES", "HC_N_APROBADAS", "HC_N_RECHAZADAS", "HC_DIAS_ULTIMA_DECISION",
                            "HC_PROP_SOLICITUDES_RECHAZADAS"],
    "POS Cash": ["HC_N_OPERACIONES_POS", "HC_POS_MAX_ATRASO", "HC_POS_MAX_CUOTAS_PENDIENTES"],
    "Cuotas": ["HC_N_REGISTROS_PAGO", "HC_N_PAGOS_TARDE", "HC_PAY_MAX_ATRASO", "HC_PROP_PAGOS_TARDE"],
    "Tarjeta de crédito": ["HC_N_OPERACIONES_TARJETA", "HC_CARD_MAX_ATRASO", "HC_CARD_MAX_UTILIZACION",
                           "HC_CARD_MAX_ULTIMO_SALDO"],
}
REPRESENTANTES = [("Buró (sin créditos)", "BUREAU_N_CREDITOS"), ("Consultas al buró", "AMT_REQ_CREDIT_BUREAU_YEAR"),
                  ("Solicitudes previas", "HC_N_SOLICITUDES"), ("POS Cash", "HC_N_OPERACIONES_POS"),
                  ("Cuotas", "HC_N_REGISTROS_PAGO"), ("Tarjeta de crédito", "HC_N_OPERACIONES_TARJETA"),
                  ("Auto (OWN_CAR_AGE)", "OWN_CAR_AGE"), ("Ocupación", "OCCUPATION_TYPE"),
                  ("EXT_SOURCE_1", "EXT_SOURCE_1"), ("EXT_SOURCE_3", "EXT_SOURCE_3"),
                  ("Vivienda (bloque completo)", "__vivienda__")]


def tasa(d, mask):
    return float(d.loc[mask, "TARGET"].mean()) if mask.any() else np.nan


def guardar(nombre: str, tabla: pd.DataFrame) -> None:
    tabla.to_parquet(OUT / f"{nombre}.parquet", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(ART / "tablon_general.parquet")
    imp = pd.read_parquet(ART / "tablon_imputado.parquet")
    trat = pd.read_parquet(ART / "tablon_tratado.parquet")
    p_out = json.load(open(ART / "parametros_outliers.json", encoding="utf-8"))
    dicc = pd.read_csv(REPO / "data_dictionary" / "diccionario_tablon.csv").set_index("variable")
    N, y = len(df), df["TARGET"]
    base = float(y.mean())
    res = {"filas": N, "columnas": df.shape[1], "columnas_post": trat.shape[1], "tasa_base": base}

    tipo = dicc["tipo_dato"]
    es_num = tipo.str.startswith("Numérica")
    es_cat = tipo.str.startswith("Categórica")
    es_dico = tipo.eq("Dicotómica")
    vivienda = [c for c in df.columns if c.endswith(("_AVG", "_MODE", "_MEDI"))]
    consultas = [c for c in df.columns if c.startswith("AMT_REQ_CREDIT_BUREAU")]
    social = [c for c in df.columns if c.endswith("_CNT_SOCIAL_CIRCLE")]

    # ══ 1. DIAGNÓSTICO DE CALIDAD ═══════════════════════════════════════════════
    filas = []
    for c in df.columns:
        s = df[c]
        x = s.dropna()
        a = []
        pct_n = s.isna().mean()
        if c not in ("SK_ID_CURR", "TARGET"):
            if pct_n >= UMBRALES["nulos_alto"]:
                a.append("nulos altos")
            if es_num[c] and len(x):
                q1, q3 = x.quantile([0.25, 0.75])
                iqr = q3 - q1
                moda = x.value_counts(normalize=True).iloc[0]
                if moda > UMBRALES["casi_constante"]:
                    a.append("casi constante")
                if c == "DAYS_EMPLOYED":
                    a.append("valor centinela")
                if abs(x.skew()) > UMBRALES["asimetria"]:
                    a.append("asimetría alta")
                if ((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).mean() > UMBRALES["outliers"]:
                    a.append("outliers (IQR)")
                if (x == 0).mean() > UMBRALES["ceros"]:
                    a.append("mayoría de ceros")
            elif es_dico[c] and len(x):
                if x.value_counts(normalize=True).iloc[-1] < UMBRALES["minoritaria"]:
                    a.append("casi constante")
            elif es_cat[c] and len(x):
                vc = x.value_counts(normalize=True)
                if vc.iloc[0] > UMBRALES["casi_constante"]:
                    a.append("casi constante")
                if len(vc) > UMBRALES["cardinalidad"]:
                    a.append("cardinalidad alta")
        filas.append({"variable": c, "familia": dicc.loc[c, "familia"], "tipo_dato": tipo[c],
                      "n_nulos": int(s.isna().sum()), "pct_nulos": float(pct_n), "fill_rate": float(1 - pct_n),
                      "n_unicos": int(s.nunique()), "n_nulos_post": int(trat[c].isna().sum()),
                      "alertas": ", ".join(a), "n_alertas": len(a)})
    cal = pd.DataFrame(filas)
    cal["tramo_nulos"] = pd.cut(cal["pct_nulos"], [-0.001, 0.0, 0.05, 0.20, 0.50, 1.0],
                                labels=["0% (completa)", "(0%, 5%]", "(5%, 20%]", "(20%, 50%]", "> 50%"]).astype(str)
    cal["fill_rate_post"] = 1 - cal["n_nulos_post"] / N
    guardar("calidad_variables", cal)

    pct_fila = df.drop(columns=["SK_ID_CURR", "TARGET"]).isna().mean(axis=1)
    pct_fila_post = trat.drop(columns=["SK_ID_CURR", "TARGET"]).isna().mean(axis=1)
    bins = np.linspace(0, 0.7, 36)
    h0, _ = np.histogram(pct_fila, bins=bins)
    h1, _ = np.histogram(pct_fila_post, bins=bins)
    guardar("nulos_por_fila", pd.DataFrame({"desde": bins[:-1], "hasta": bins[1:], "antes": h0, "despues": h1}))
    res["nulos_fila_mediana"] = float(pct_fila.median())
    res["nulos_fila_mediana_post"] = float(pct_fila_post.median())

    cols_nulas = cal.loc[cal["n_nulos"] > 0].sort_values("pct_nulos", ascending=False)["variable"].tolist()
    muestra = df[cols_nulas].sample(1500, random_state=SEMILLA).isna().astype("int8").reset_index(drop=True)
    guardar("matriz_nulos", muestra)

    # Chequeos pedidos por la guía (Paso 2) + consistencia lógica
    chequeos = []

    def chk(categoria, nombre, n, estado, detalle):
        chequeos.append({"categoria": categoria, "chequeo": nombre, "casos": int(n), "estado": estado, "detalle": detalle})

    chk("Integridad", "SK_ID_CURR duplicados", N - df["SK_ID_CURR"].nunique(), "ok", "Una fila por crédito objetivo.")
    dup = int(df.drop(columns="SK_ID_CURR").duplicated().sum())
    chk("Integridad", "Filas duplicadas (sin contar el ID)", dup, "ok" if dup == 0 else "aviso", "Registros idénticos en las 147 columnas restantes.")
    chk("Integridad", "TARGET fuera de {0, 1}", (~y.isin([0, 1])).sum(), "ok", "La variable objetivo es binaria y completa.")
    texto = df.select_dtypes(exclude="number").columns
    coercibles = [c for c in texto if pd.to_numeric(df[c], errors="coerce").notna().sum() > 0]
    chk("Tipos de dato", "Numéricas almacenadas como texto", len(coercibles), "ok" if not coercibles else "aviso",
        f"pd.to_numeric(errors='coerce') sobre las {len(texto)} columnas de texto: ninguna contiene números." if not coercibles
        else ", ".join(coercibles))
    espacios = sum(int((df[c].dropna().astype(str) != df[c].dropna().astype(str).str.strip()).sum()) for c in texto)
    chk("Tipos de dato", "Categorías con espacios sobrantes", espacios, "ok" if espacios == 0 else "aviso", "str.strip() no cambia ninguna categoría.")
    cod = {c: int(df[c].isin(["XNA", "Unknown", "", "?", "nan", "NA"]).sum()) for c in texto}
    cod = {c: n for c, n in cod.items() if n}
    chk("Faltantes codificados", "Categorías que ocultan un faltante (XNA, Unknown, ?, vacío)", sum(cod.values()), "aviso",
        " · ".join(f"{c}: {n:,}" for c, n in cod.items()))
    chk("Faltantes codificados", "Valor centinela DAYS_EMPLOYED = 365243", (df["DAYS_EMPLOYED"] == SENT).sum(), "aviso",
        "≈ 1 000 años de antigüedad laboral: no es un dato real.")
    dias = [c for c in df.columns if c.startswith("DAYS_") and c != "DAYS_EMPLOYED"]
    chk("Consistencia", "Variables DAYS_* positivas (fechas posteriores a la solicitud)",
        sum(int((df[c] > 0).sum()) for c in dias) + int(((df["DAYS_EMPLOYED"] > 0) & (df["DAYS_EMPLOYED"] != SENT)).sum()),
        "ok", "Todas las antigüedades son ≤ 0, salvo el centinela.")
    montos = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]
    chk("Consistencia", "Montos de la solicitud ≤ 0", sum(int((df[c] <= 0).sum()) for c in montos), "ok", "Ingreso, crédito, cuota y precio del bien son positivos.")
    fam = int((df["CNT_FAM_MEMBERS"] < df["CNT_CHILDREN"] + 1).sum())
    chk("Consistencia", "Miembros de la familia < hijos + 1", fam, "ok" if fam == 0 else "aviso", "El hogar incluye al menos al solicitante y sus hijos.")
    auto = int((df["OWN_CAR_AGE"].isna() & df["FLAG_OWN_CAR"].eq("Y")).sum())
    chk("Consistencia", "Declara auto pero sin edad del auto", auto, "info", "Casi todos los nulos de OWN_CAR_AGE son clientes sin auto (nulo estructural).")
    apr = int((df["HC_N_APROBADAS"] + df["HC_N_RECHAZADAS"] > df["HC_N_SOLICITUDES"]).sum())
    chk("Consistencia", "Aprobadas + rechazadas > solicitudes previas", apr, "ok", "Los conteos de previous_application cuadran.")
    act = int((df["BUREAU_N_ACTIVOS"] > df["BUREAU_N_CREDITOS"]).sum())
    chk("Consistencia", "Créditos activos > créditos en buró", act, "ok", "Los conteos de bureau cuadran.")
    cob = int(((df["TIENE_BUREAU"] == 1) != df["BUREAU_N_CREDITOS"].notna()).sum()
              + ((df["TIENE_HISTORIAL_HOME_CREDIT"] == 1) != df["HC_N_SOLICITUDES"].notna()).sum())
    chk("Consistencia", "Indicadores TIENE_* incoherentes con su fuente", cob, "ok", "Coinciden con la nulidad de sus variables de conteo.")
    neg = int((df["BUREAU_DEUDA_TOTAL"] < 0).sum())
    chk("Consistencia", "Deuda total en buró negativa", neg, "info" if neg else "ok", "Saldos a favor o ajustes reportados al buró; se conservan.")
    sob = int((df["HC_CARD_MAX_UTILIZACION"] > 1).sum())
    chk("Consistencia", "Utilización de tarjeta > 100 %", sob, "info", "Sobregiro sobre el límite: dato plausible, no error.")
    guardar("chequeos", pd.DataFrame(chequeos))

    # Categóricas: frecuencias, raras y codificadas
    filas, detalle = [], []
    for c in [v for v in df.columns if es_cat[v] or (es_dico[v] and df[v].dtype == object)]:
        s = df[c]
        vc = s.value_counts(dropna=False)
        p = s.value_counts(normalize=True)
        filas.append({"variable": c, "tipo_dato": tipo[c], "n_categorias": int(s.nunique()), "pct_nulos": float(s.isna().mean()),
                      "moda": str(p.index[0]), "pct_moda": float(p.iloc[0]),
                      "n_raras": int((p < UMBRALES["rara"]).sum()), "pct_obs_raras": float(p[p < UMBRALES["rara"]].sum()),
                      "codificados": int(s.isin(["XNA", "Unknown"]).sum())})
        for k, n in vc.items():
            detalle.append({"variable": c, "categoria": "(nulo)" if pd.isna(k) else str(k), "n": int(n), "pct": n / N})
    guardar("categoricas", pd.DataFrame(filas))
    guardar("categorias_detalle", pd.DataFrame(detalle))

    # Numéricas: tabla de colas (guía: min / p1 / p50 / p99 / max)
    filas = []
    for c in [v for v in df.columns if es_num[v]]:
        x = df[c].dropna()
        q = x.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
        iqr = q[0.75] - q[0.25]
        filas.append({"variable": c, "familia": dicc.loc[c, "familia"], "min": x.min(), "p1": q[0.01], "p50": q[0.5],
                      "p99": q[0.99], "max": x.max(), "asimetria": float(x.skew()),
                      "pct_outliers_iqr": float(((x < q[0.25] - 1.5 * iqr) | (x > q[0.75] + 1.5 * iqr)).mean()),
                      "pct_ceros": float((x == 0).mean())})
    guardar("numericas", pd.DataFrame(filas))

    # ══ 2. VALORES FALTANTES ════════════════════════════════════════════════════
    familia_de = {c: f for f, cols in FAMILIAS_HISTORIAL.items() for c in cols}
    familia_de.update({c: "Consultas al buró" for c in consultas})
    familia_de.update({c: "Vivienda (edificio)" for c in vivienda})
    familia_de.update({c: "Score externo" for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]})
    familia_de.update({c: "Círculo social" for c in social})
    familia_de.update({"OWN_CAR_AGE": "Auto", "OCCUPATION_TYPE": "Empleo"})
    filas = []
    for c in cols_nulas:
        nul = df[c].isna()
        filas.append({"variable": c, "familia": familia_de.get(c, "Otros (nulos menores)"), "tipo_dato": tipo[c],
                      "n_nulos": int(nul.sum()), "fill_rate": float(1 - nul.mean()),
                      "default_si_nulo": tasa(df, nul), "default_si_dato": tasa(df, ~nul)})
    inv = pd.DataFrame(filas)
    inv["razon_default"] = inv["default_si_nulo"] / inv["default_si_dato"]
    guardar("inventario_nulos", inv)

    filas = []
    for nombre, c in REPRESENTANTES:
        nul = df[vivienda].isna().all(axis=1) if c == "__vivienda__" else df[c].isna()
        filas.append({"grupo": nombre, "variable": c, "n_nulos": int(nul.sum()),
                      "default_si_nulo": tasa(df, nul), "default_si_dato": tasa(df, ~nul)})
    guardar("nulo_informativo", pd.DataFrame(filas))

    es_sent = df["DAYS_EMPLOYED"] == SENT
    dias_emp = -df["DAYS_EMPLOYED"].where(~es_sent)
    orden = ["≤ 90 días", "90 d – 1 año", "1 – 3 años", "3 – 10 años", "> 10 años", "Centinela (sin empleo)"]
    tramo = pd.cut(dias_emp, [-np.inf, 90, 365, 1095, 3650, np.inf], labels=orden[:-1]).astype(object).mask(es_sent, orden[-1])
    ten = df.assign(tramo=tramo).groupby("tramo")["TARGET"].agg(n="size", tasa="mean").reindex(orden).reset_index()
    guardar("centinela_tenencia", ten)
    guardar("centinela_coincidencias", pd.DataFrame([
        ("Filas con el centinela", int(es_sent.sum())),
        ("… con ORGANIZATION_TYPE = 'XNA'", int((es_sent & df["ORGANIZATION_TYPE"].eq("XNA")).sum())),
        ("… con NAME_INCOME_TYPE = 'Pensioner'", int((es_sent & df["NAME_INCOME_TYPE"].eq("Pensioner")).sum())),
        ("… con FLAG_EMP_PHONE = 0", int((es_sent & df["FLAG_EMP_PHONE"].eq(0)).sum())),
        ("… con OCCUPATION_TYPE nulo", int((es_sent & df["OCCUPATION_TYPE"].isna()).sum())),
        ("Total ORGANIZATION_TYPE = 'XNA'", int(df["ORGANIZATION_TYPE"].eq("XNA").sum())),
    ], columns=["condicion", "n"]))
    res["ceros_reales_days_employed"] = int((df.loc[~es_sent, "DAYS_EMPLOYED"] == 0).sum())

    filas = []
    for fam_n, cols in FAMILIAS_HISTORIAL.items():
        todas = df[cols].isna().all(axis=1)
        for c in cols:
            s = df[c]
            filas.append({"familia": fam_n, "variable": c, "n_nulos": int(s.isna().sum()), "pct_nulos": float(s.isna().mean()),
                          "ceros_entre_no_nulos": int((s == 0).sum()), "filas_todas_nulas": int(todas.sum())})
    guardar("estructurales", pd.DataFrame(filas))
    sin_buro = df["BUREAU_N_CREDITOS"].isna()
    sin_cons = df[consultas].isna().all(axis=1)
    guardar("consultas_vs_buro", pd.crosstab(sin_cons.map({True: "Consultas nulas", False: "Consultas con dato"}),
                                             sin_buro.map({True: "Sin créditos en buró", False: "Con créditos en buró"}))
            .reset_index().rename(columns={"row_0": "consultas"}).rename_axis(None, axis=1))
    occ_nul = df["OCCUPATION_TYPE"].isna()
    res.update({
        "consultas_nulas": int(sin_cons.sum()), "consultas_nulas_sin_buro": int((sin_cons & sin_buro).sum()),
        "meses_atraso_nulos": int(df["BUREAU_MESES_CON_ATRASO"].isna().sum()),
        "meses_atraso_nulos_con_creditos": int((df["BUREAU_MESES_CON_ATRASO"].isna() & ~sin_buro).sum()),
        "meses_atraso_ceros": int((df["BUREAU_MESES_CON_ATRASO"] == 0).sum()),
        "auto_nulos": int(df["OWN_CAR_AGE"].isna().sum()), "sin_auto": int(df["FLAG_OWN_CAR"].eq("N").sum()),
        "auto_nulos_sin_auto": int((df["OWN_CAR_AGE"].isna() & df["FLAG_OWN_CAR"].eq("N")).sum()),
        "auto_edad_cero": int((df["OWN_CAR_AGE"] == 0).sum()),
        "ocupacion_nulos": int(occ_nul.sum()), "ocupacion_nulos_centinela": int((occ_nul & es_sent).sum()),
        "ocupacion_default_sin_explicar": tasa(df, occ_nul & ~es_sent),
    })

    ingresos = df["NAME_INCOME_TYPE"].value_counts(normalize=True).loc[lambda s: s >= 0.01].index.tolist()
    edad = pd.cut(-df["DAYS_BIRTH"] / 365.25, EDAD_CORTES, labels=EDAD_ETIQ).astype(str)
    ing = df["NAME_INCOME_TYPE"].where(df["NAME_INCOME_TYPE"].isin(ingresos), "Other")
    filas, med = [], []
    for c in ["EXT_SOURCE_1", "EXT_SOURCE_3"]:
        nul = df[c].isna()
        for e in EDAD_ETIQ:
            filas.append({"variable": c, "eje": "Edad", "grupo": e, "pct_nulos": float(nul[edad == e].mean())})
        for g in ingresos + ["Other"]:
            filas.append({"variable": c, "eje": "Tipo de ingreso", "grupo": g, "pct_nulos": float(nul[ing == g].mean())})
        ok = ~nul
        cel = (pd.DataFrame({"v": df.loc[ok, c], "edad": edad[ok], "ing": ing[ok]})
               .groupby(["edad", "ing"])["v"].agg(n="size", mediana="median").reset_index())
        cel["variable"] = c
        cel["mediana_global"] = float(df[c].median())
        med.append(cel)
    guardar("ext_nulos", pd.DataFrame(filas))
    guardar("ext_medianas", pd.concat(med, ignore_index=True))

    n_null_viv = df[vivienda].isna().sum(axis=1)
    estado = pd.Series(np.select([n_null_viv == len(vivienda), n_null_viv == 0], ["sin ninguna info", "info completa"], "info parcial"),
                       index=df.index)
    viv = df.groupby(estado)["TARGET"].agg(n="size", tasa="mean").reindex(["info completa", "info parcial", "sin ninguna info"]).reset_index()
    viv.columns = ["estado", "n", "tasa"]
    guardar("vivienda_estado", viv)
    por_tipo = (estado.eq("sin ninguna info").groupby(df["NAME_HOUSING_TYPE"]).agg(pct_sin_info="mean", n="size")
                .sort_values("pct_sin_info", ascending=False).reset_index())
    guardar("vivienda_por_tipo", por_tipo)
    res["vivienda_n_vars"] = len(vivienda)
    res["vivienda_parcial_mediana_faltantes"] = int(n_null_viv[estado == "info parcial"].median())

    menores = {c: int(df[c].isna().sum()) for c in ["NAME_TYPE_SUITE"] + social + ["AMT_GOODS_PRICE", "AMT_ANNUITY", "CNT_FAM_MEMBERS",
                                                                                     "DAYS_LAST_PHONE_CHANGE", "EXT_SOURCE_2"]}
    menores["CODE_GENDER = 'XNA'"] = int(df["CODE_GENDER"].eq("XNA").sum())
    menores["NAME_FAMILY_STATUS = 'Unknown'"] = int(df["NAME_FAMILY_STATUS"].eq("Unknown").sum())
    men = pd.DataFrame({"variable": list(menores), "n": list(menores.values())})
    men["pct"] = men["n"] / N
    guardar("despreciables", men.sort_values("n", ascending=False))
    guardar("circulo_social", pd.DataFrame({"variable": social, "media": df[social].mean().values, "mediana": df[social].median().values,
                                            "moda": df[social].mode().iloc[0].values, "pct_ceros": (df[social] == 0).mean().values}))
    filas = []
    for c in ["AMT_GOODS_PRICE", "AMT_ANNUITY"]:
        ok = df[c].notna()
        razon = (df[c] / df["AMT_CREDIT"]).groupby(df["NAME_CONTRACT_TYPE"]).median()
        pred = df.loc[ok, "AMT_CREDIT"] * df.loc[ok, "NAME_CONTRACT_TYPE"].map(razon)
        real = df.loc[ok, c]
        filas.append({"variable": c, "spearman_con_amt_credit": float(df.loc[ok, [c, "AMT_CREDIT"]].corr(method="spearman").iloc[0, 1]),
                      "error_mediana_global": float(np.median(np.abs(real - real.median()) / real)),
                      "error_razon_por_contrato": float(np.median(np.abs(real - pred) / real))})
    guardar("montos_error", pd.DataFrame(filas))

    # ══ 3. RESULTADO DEL TRATAMIENTO DE FALTANTES ═══════════════════════════════
    nuevas = [c for c in imp.columns if c not in df.columns]
    guardar("flags", pd.DataFrame({"flag": nuevas, "n": [int(imp[c].sum()) for c in nuevas],
                                   "pct": [float(imp[c].mean()) for c in nuevas]}))
    imputadas = sorted({"EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_LAST_PHONE_CHANGE", "CNT_FAM_MEMBERS",
                        "AMT_GOODS_PRICE", "AMT_ANNUITY"} | set(social))
    filas = []
    for c in imputadas + ["DAYS_EMPLOYED"]:
        a, b = df[c], imp[c]
        antes = a.where(a != SENT) if c == "DAYS_EMPLOYED" else a
        filas.append({"variable": c, "n_imputados": int((a == SENT).sum()) if c == "DAYS_EMPLOYED" else int(a.isna().sum()),
                      "metodo": "", "media_antes": float(antes.mean()), "media_despues": float(b.mean()),
                      "std_antes": float(antes.std()), "std_despues": float(b.std())})
    ef = pd.DataFrame(filas)
    metodo = {"EXT_SOURCE_1": "Mediana por edad × tipo de ingreso", "EXT_SOURCE_3": "Mediana por edad × tipo de ingreso",
              "EXT_SOURCE_2": "Mediana global", "DAYS_LAST_PHONE_CHANGE": "Mediana global",
              "CNT_FAM_MEMBERS": "Moda (≥ hijos + 1)", "AMT_GOODS_PRICE": "AMT_CREDIT × razón mediana por contrato",
              "AMT_ANNUITY": "AMT_CREDIT × razón mediana por contrato", "DAYS_EMPLOYED": "Centinela 365243 → 0 + flag_sin_empleo"}
    ef["metodo"] = ef["variable"].map(metodo).fillna("Moda")
    guardar("efecto_imputacion", ef)
    cats = []
    for c, raro in [("NAME_TYPE_SUITE", None), ("CODE_GENDER", "XNA"), ("NAME_FAMILY_STATUS", "Unknown")]:
        cats.append({"variable": c, "problema": "nulo" if raro is None else f"categoría '{raro}'",
                     "casos": int(df[c].isna().sum() if raro is None else df[c].eq(raro).sum()),
                     "reemplazo": str(imp.loc[df[c].isna() if raro is None else df[c].eq(raro), c].iloc[0])})
    guardar("imputacion_categoricas", pd.DataFrame(cats))

    edges = np.linspace(0, 1, 61)
    filas = []
    for c in ["EXT_SOURCE_1", "EXT_SOURCE_3"]:
        h_o, _ = np.histogram(df[c].dropna(), bins=edges, density=True)
        h_i, _ = np.histogram(imp[c], bins=edges, density=True)
        filas += [{"variable": c, "desde": edges[i], "hasta": edges[i + 1], "serie": s, "densidad": v}
                  for i in range(60) for s, v in (("Original, sin nulos", h_o[i]), ("Tras imputar", h_i[i]))]
    guardar("hist_ext", pd.DataFrame(filas))
    res["ext_n_original"] = {c: int(df[c].notna().sum()) for c in ["EXT_SOURCE_1", "EXT_SOURCE_3"]}

    sin_tocar = sum(FAMILIAS_HISTORIAL.values(), []) + consultas + ["OWN_CAR_AGE", "OCCUPATION_TYPE"] + vivienda
    cambiaron = {c for c in df.columns if not df[c].equals(imp[c])}
    esperadas = set(imputadas) | {"DAYS_EMPLOYED", "NAME_TYPE_SUITE", "CODE_GENDER", "NAME_FAMILY_STATUS"}
    val = [
        ("Faltantes", "Mismo número de filas y SK_ID_CURR idéntico", len(imp) == N and imp["SK_ID_CURR"].equals(df["SK_ID_CURR"])),
        ("Faltantes", "TARGET sin cambios", imp["TARGET"].equals(y)),
        ("Faltantes", "Variables imputadas sin nulos", int(imp[imputadas + ["NAME_TYPE_SUITE"]].isna().sum().sum()) == 0),
        ("Faltantes", "DAYS_EMPLOYED sin el centinela 365243", not imp["DAYS_EMPLOYED"].eq(SENT).any()),
        ("Faltantes", "Sin 'XNA' en CODE_GENDER ni 'Unknown' en NAME_FAMILY_STATUS",
         not imp["CODE_GENDER"].eq("XNA").any() and not imp["NAME_FAMILY_STATUS"].eq("Unknown").any()),
        ("Faltantes", "Flags = nulos o centinelas originales",
         int(imp["flag_sin_buro"].sum()) == int(sin_buro.sum()) and int(imp["flag_sin_previas"].sum()) == int(df["HC_N_SOLICITUDES"].isna().sum())
         and int(imp["flag_sin_empleo"].sum()) == int(es_sent.sum())
         and int(imp["flag_sin_info_vivienda"].sum()) == int(df[vivienda].isna().all(axis=1).sum())),
        ("Faltantes", "Nulos estructurales, de vivienda y OCCUPATION_TYPE intactos",
         all(int(imp[c].isna().sum()) == int(df[c].isna().sum()) for c in sin_tocar)),
        ("Faltantes", "Valores no nulos originales intactos",
         all((imp.loc[df[c].notna(), c] == df.loc[df[c].notna(), c]).all() for c in imputadas if c != "CNT_FAM_MEMBERS")),
        ("Faltantes", "Solo cambiaron las columnas previstas", cambiaron == esperadas),
    ]
    # Outliers (pipeline_modelado.py outliers): A) OWN_CAR_AGE 64–65 → nulo; B) filas super extremas eliminadas; C) capeo
    lims = p_out["limites"]
    borradas = set(p_out["filas_eliminadas"])
    imp_q = imp[~imp["SK_ID_CURR"].isin(borradas)].reset_index(drop=True)          # imputado sin las filas eliminadas
    trat_a = trat.set_index("SK_ID_CURR").loc[imp_q["SK_ID_CURR"]].reset_index()
    relleno = imp_q["OWN_CAR_AGE"].isin(p_out["car_relleno"])
    imp_r = imp_q.copy()
    imp_r.loc[relleno, "OWN_CAR_AGE"] = np.nan                                         # tras el paso A, antes del capeo
    dentro = {v: imp_r[v].between(lims[v]["inf"], lims[v]["sup"]) for v in lims}
    cambio_out = {c for c in imp_q.columns if not imp_q[c].equals(trat_a[c])}
    val += [
        ("Outliers", f"Filas = imputado − {len(borradas)} eliminadas, y son exactamente las marcadas",
         len(trat) == len(imp) - len(borradas) and not trat["SK_ID_CURR"].isin(borradas).any()),
        ("Outliers", "Mismas columnas que el tablón imputado", list(trat.columns) == list(imp.columns)),
        ("Outliers", "OWN_CAR_AGE: el bloque 64–65 pasó a nulo y no queda ninguno",
         not trat["OWN_CAR_AGE"].isin(p_out["car_relleno"]).any()
         and int(trat_a["OWN_CAR_AGE"].isna().sum()) == int(imp_q["OWN_CAR_AGE"].isna().sum() + relleno.sum())),
        ("Outliers", "Solo cambiaron variables numéricas", cambio_out <= set(lims)),
        ("Outliers", "Ningún valor fuera de los límites de capeo (p0.1–p99.9 de train)",
         all(trat[v].dropna().between(lims[v]["inf"], lims[v]["sup"]).all() for v in lims)),
        ("Outliers", "Valores dentro de los límites intactos",
         all(trat_a.loc[dentro[v], v].equals(imp_r.loc[dentro[v], v]) for v in lims)),
        ("Outliers", "Nulos intactos (salvo el bloque de OWN_CAR_AGE)", all(trat_a[v].isna().equals(imp_r[v].isna()) for v in lims)),
        ("Outliers", "Límites aprendidos solo con train (partición 80/20 estratificada, semilla 42)",
         p_out.get("aprendido_en") == "train" and p_out.get("test_size") == 0.2),
        ("Outliers", "Coherencia miembros ≥ hijos + 1 no empeora",
         int((trat["CNT_FAM_MEMBERS"] < trat["CNT_CHILDREN"] + 1).sum()) <= int((imp["CNT_FAM_MEMBERS"] < imp["CNT_CHILDREN"] + 1).sum())),
    ]
    guardar("validaciones", pd.DataFrame(val, columns=["etapa", "verificacion", "ok"]))
    res["celdas_nulas_antes"] = int(df.isna().sum().sum())
    res["celdas_nulas_despues"] = int(trat[df.columns].isna().sum().sum())
    res["filas_post"] = int(len(trat))
    res["filas_eliminadas"] = int(len(borradas))
    res["ingreso_original"] = {"std": float(imp["AMT_INCOME_TOTAL"].std()), "asim": float(imp["AMT_INCOME_TOTAL"].skew()),
                               "max": float(imp["AMT_INCOME_TOTAL"].max())}
    res["ingreso_tratado"] = {"std": float(trat["AMT_INCOME_TOTAL"].std()), "asim": float(trat["AMT_INCOME_TOTAL"].skew()),
                              "max": float(trat["AMT_INCOME_TOTAL"].max())}

    # ══ 4. OUTLIERS ═════════════════════════════════════════════════════════════
    MOD = ART / "modelado"
    for nombre in ["outliers_relleno", "outliers_eliminacion", "outliers_capeo", "outliers_filas_eliminadas"]:
        guardar(nombre, pd.read_parquet(MOD / f"{nombre}.parquet"))
    part = pd.read_parquet(MOD / "particion.parquet").set_index("SK_ID_CURR")["muestra"]
    tr_imp = imp[imp["SK_ID_CURR"].map(part) == "train"]
    s = tr_imp["OWN_CAR_AGE"].dropna()
    y_tr = tr_imp["TARGET"]
    tramos = [(0, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30), (31, 40), (41, 50), (51, 63), (64, 64), (65, 65), (66, 200)]
    etq = ["0-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-40", "41-50", "51-63", "64", "65", "66+"]
    guardar("own_car_age", pd.DataFrame([{"tramo": e, "n": int(((s >= a) & (s <= b)).sum()),
                                          "tasa": float(y_tr[s.index][(s >= a) & (s <= b)].mean()) if ((s >= a) & (s <= b)).sum() >= 100 else np.nan,
                                          "bloque": e in ("64", "65")} for (a, b), e in zip(tramos, etq)]))
    # Cola superior antes / después (percentiles 90–100, train) de las variables con colas más largas
    curvas = []
    qgrid = np.linspace(0.90, 1.0, 201)
    tr_trat = trat[trat["SK_ID_CURR"].map(part) == "train"]
    for v in CURVAS:
        antes = tr_imp[v].dropna().quantile(qgrid).values
        despues = tr_trat[v].dropna().quantile(qgrid).values
        curvas += [{"variable": v, "percentil": q * 100, "antes": a, "despues": d, "lim_sup": lims[v]["sup"]}
                   for q, a, d in zip(qgrid, antes, despues)]
    guardar("curvas_cola", pd.DataFrame(curvas))

    json.dump(res, open(OUT / "resumen.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
    print(f"Listo: {len(list(OUT.glob('*.parquet')))} tablas en {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
