"""Dataset final: lo que entra al entrenamiento (logística con WoE y ML con SMOTE) y espacio de feature engineering."""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

REPO_DIR = Path(__file__).resolve().parents[1]
MOD = REPO_DIR / "artifacts" / "modelado"
AZUL, NARANJA, VERDE, GRIS, TINTA, BARRA = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85", "#2b2b29", "#c9c8c3"


@st.cache_data(show_spinner=False)
def t(nombre: str) -> pd.DataFrame:
    return pd.read_parquet(MOD / f"{nombre}.parquet")


def tabla_texto(df: pd.DataFrame, formatos: dict | None = None) -> None:
    st.table(df.reset_index(drop=True).style.hide(axis="index").format(formatos or {}, na_rep="—"))


def fam_corta(f) -> str:
    return f.split("·", 1)[-1].strip() if isinstance(f, str) else ""


res = t("dataset_resumen")
feat = t("dataset_features")
bo = t("binning_optb")
R = res.set_index("dataset")

st.title("Dataset final")
st.markdown(
    "Los dos conjuntos con los que se entrenarán los modelos (Pasos 8–9 de la guía). Comparten la partición **70 / 30 "
    "estratificada** (semilla 42) y todo lo que usa el `TARGET` se aprendió en train. Se diferencian en **qué variables** llevan "
    "(IV vs. Gini) y en **cómo** las representan:\n\n"
    "- **Logística**: cada variable se reemplaza por su **WoE** (OptBinning), que linealiza la relación con el log-odds.\n"
    "- **Machine learning** (árbol, Random Forest, XGBoost): variables en su **escala original** y train **rebalanceado con SMOTE**.")
n_var_ml = int((feat["dataset"].eq("ml") & ~feat["columna"].str.endswith("__nulo")).sum())
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Variables · logística", int(R.loc["Logística · train", "columnas"]), border=True)
k2.metric("Columnas · ML", int(R.loc["ML · train (SMOTE)", "columnas"]), border=True,
          help=f"{n_var_ml} variables + indicadores de nulo.")
k3.metric("Train", f"{int(R.loc['Logística · train', 'filas']):,}", border=True, help="70 % del tablón.")
k4.metric("Test (sin tocar)", f"{int(R.loc['Logística · test', 'filas']):,}", border=True, help="30 %: nunca se rebalancea.")
k5.metric("Train ML tras SMOTE", f"{int(R.loc['ML · train (SMOTE)', 'filas']):,}", f"+{int(R.loc['ML · train (SMOTE)', 'sinteticas']):,} sintéticas",
          delta_color="off", border=True)

tab_log, tab_ml, tab_sm, tab_fe = st.tabs([":material/functions: Dataset logístico", ":material/forest: Dataset ML",
                                           ":material/balance: Rebalanceo con SMOTE", ":material/construction: Feature engineering"])

# ══════════════════════════════════════════════════════════════════════════════
with tab_log:
    fl = feat.query("dataset == 'logistica'")
    c1, c2 = st.columns([2, 3], gap="large")
    with c1:
        st.markdown(f"**{len(fl)} variables en WoE** · IV ≥ 0.1, sin redundancias (|ρ| ≤ 0.6)")
        tabla_texto(fl.assign(familia=fl["familia"].map(fam_corta))[["columna", "variable", "familia", "metrica"]]
                    .rename(columns={"columna": "Columna", "variable": "Variable original", "familia": "Familia", "metrica": "IV"}), {"IV": "{:.3f}"})
        tabla_texto(res[res["dataset"].str.startswith("Logística")][["dataset", "filas", "n_default", "tasa_default"]]
                    .rename(columns={"dataset": "Muestra", "filas": "Filas", "n_default": "Defaults", "tasa_default": "Tasa de default"}),
                    {"Filas": "{:,}", "Defaults": "{:,}", "Tasa de default": "{:.2%}"})
        st.caption("Sin rebalanceo: la regresión logística estima bien probabilidades con 8 % de eventos y 215 mil filas; "
                   "rebalancear solo desplazaría el intercepto.")
    with c2:
        vw = st.segmented_control("Del valor original al WoE", fl["variable"].tolist(), default=fl["variable"].iloc[0], key="woe_var")
        d = bo.query("variable == @vw and n > 0").copy()
        d["signo"] = np.where(d["woe"] >= 0, "Menos riesgo que el promedio", "Más riesgo que el promedio")
        st.altair_chart((alt.Chart(d).mark_bar(size=40, cornerRadiusEnd=3).encode(
            x=alt.X("tramo:N", sort=d["tramo"].tolist(), title=f"Tramo de {vw} (OptBinning, train)", axis=alt.Axis(labelAngle=0, labelLimit=140)),
            y=alt.Y("woe:Q", title="WoE = ln(% buenos / % malos)"),
            color=alt.Color("signo:N", scale=alt.Scale(domain=["Menos riesgo que el promedio", "Más riesgo que el promedio"], range=[AZUL, NARANJA]),
                            legend=alt.Legend(orient="top", title=None, labelLimit=260)),
            tooltip=[alt.Tooltip("tramo:N"), alt.Tooltip("woe:Q", format=".3f"), alt.Tooltip("rd:Q", format=".2%", title="Tasa de default"),
                     alt.Tooltip("n:Q", format=",", title="Créditos")])
            + alt.Chart(d[d["woe"] >= 0]).mark_text(dy=-8, fontSize=11, color=TINTA).encode(
                x=alt.X("tramo:N", sort=d["tramo"].tolist()), y="woe:Q", text=alt.Text("woe:Q", format=".2f"))
            + alt.Chart(d[d["woe"] < 0]).mark_text(dy=10, fontSize=11, color=TINTA).encode(
                x=alt.X("tramo:N", sort=d["tramo"].tolist()), y="woe:Q", text=alt.Text("woe:Q", format=".2f")))
            .properties(title={"text": f"Valor que recibe la logística en WoE_{vw}", "subtitle": "Variable original → tramo → WoE → coeficiente"}),
            width="stretch", height=320)
    st.markdown("**Primeras filas de train** (lo que verá `LogisticRegression`)")
    st.dataframe(t("muestra_logistica"), hide_index=True, width="stretch", height=260,
                 column_config={c: st.column_config.NumberColumn(c, format="%.4f") for c in t("muestra_logistica").columns if c.startswith("WOE_")})
    st.caption("Con WoE = ln(% buenos / % malos), un WoE alto significa menos riesgo: los coeficientes de la logística saldrán "
               "**negativos** (más WoE, menos probabilidad de default). Es la convención de Siddiqi y de OptBinning.")

# ══════════════════════════════════════════════════════════════════════════════
with tab_ml:
    fm = feat.query("dataset == 'ml'").copy()
    st.markdown(f"**{fm['variable'].nunique()} variables** (Gini ≥ 0.08, sin redundancias) → **{len(fm)} columnas** al agregar los "
                "indicadores de nulo.")
    fm["clase"] = np.where(fm["columna"].str.endswith("__nulo"), "Indicador de nulo",
                           np.where(fm["transformacion"].str.startswith("Agrupada"), "Agrupada por default",
                                    np.where(fm["transformacion"].str.contains("mediana"), "Original + mediana", "Original")))
    m1, m2 = st.columns([3, 2], gap="large")
    m1.dataframe(fm.assign(familia=fm["familia"].map(fam_corta))[["columna", "transformacion", "metrica", "pct_nulos_train", "familia", "tipo"]],
                 hide_index=True, width="stretch", height=520,
                 column_config={"columna": st.column_config.TextColumn("Columna", pinned=True, width=240),
                                "tipo": st.column_config.TextColumn("Tipo", width=90), "familia": st.column_config.TextColumn("Familia", width=170),
                                "transformacion": st.column_config.TextColumn("Tratamiento", width=210),
                                "metrica": st.column_config.NumberColumn("Gini", format="%.3f"),
                                "pct_nulos_train": st.column_config.ProgressColumn("% nulos (train)", format="percent", min_value=0, max_value=1)})
    with m2:
        cnt = fm.groupby("clase").size().reset_index(name="n")
        orden_cl = ["Original", "Original + mediana", "Indicador de nulo", "Agrupada por default"]
        base_cl = alt.Chart(cnt, title=alt.TitleParams("Columnas por tratamiento", anchor="start")).encode(
            y=alt.Y("clase:N", sort=orden_cl, title=None, axis=alt.Axis(labelLimit=200, labelFontSize=11)),
            x=alt.X("n:Q", title="Columnas", axis=alt.Axis(tickMinStep=1)))
        st.altair_chart(base_cl.mark_bar(height={"band": 0.6}, cornerRadiusEnd=4).encode(
            color=alt.Color("clase:N", scale=alt.Scale(domain=orden_cl, range=[AZUL, "#7fb0ea", GRIS, VERDE]), legend=None),
            tooltip=[alt.Tooltip("clase:N", title="Tratamiento"), alt.Tooltip("n:Q", title="Columnas")])
            + base_cl.mark_text(align="left", dx=4, color=TINTA, fontSize=12).encode(text="n:Q"),
            width="stretch", height=200)
        st.markdown(
            "- **¿Por qué imputar si XGBoost acepta nulos?** Porque **SMOTE no**: necesita valores para interpolar. Se imputa "
            "con la mediana de train y se agrega un **indicador** para no perder la señal del nulo.\n"
            "- `OCCUPATION_TYPE` y `ORGANIZATION_TYPE` entran **agrupadas** por tasa de default (G1 = menor riesgo); las demás "
            "categóricas, con sus categorías originales. El *encoding* final (one-hot u ordinal) lo decide cada modelo.")
    st.markdown("**Muestra del train rebalanceado** (12 reales y 13 sintéticas; `SK_ID_CURR` vacío en las sintéticas)")
    mm = t("muestra_ml")
    st.dataframe(mm, hide_index=True, width="stretch", height=300,
                 column_config={"es_sintetica": st.column_config.CheckboxColumn("Sintética"), "SK_ID_CURR": st.column_config.NumberColumn(format="%d")})
    st.download_button("Descargar la lista de columnas del dataset ML (CSV)", fm[["columna", "variable", "tipo", "transformacion", "metrica"]]
                       .to_csv(index=False).encode("utf-8"), "features_ml.csv", "text/csv", icon=":material/download:")

# ══════════════════════════════════════════════════════════════════════════════
with tab_sm:
    st.markdown(
        "Con 8 % de defaults, un modelo de árboles puede aprender a decir «paga» casi siempre. **SMOTE** crea defaults sintéticos "
        "interpolando entre un default real y uno de sus 5 vecinos más cercanos (también defaults). Como el dataset tiene "
        "categóricas, se usa **SMOTE-NC**: interpola las continuas y, en las categóricas, toma el valor más frecuente entre los vecinos.")
    b = res[res["dataset"].isin(["ML · train (original)", "ML · train (SMOTE)", "ML · test"])].copy()
    lb = b.melt(id_vars="dataset", value_vars=["n_paga", "n_default"], var_name="clase", value_name="n")
    lb["clase"] = lb["clase"].map({"n_paga": "0 · paga", "n_default": "1 · default"})
    lb["pct"] = lb["n"] / lb.groupby("dataset")["n"].transform("sum")
    s1, s2 = st.columns([3, 2], gap="large")
    s1.altair_chart(alt.Chart(lb).mark_bar(size=60).encode(
        x=alt.X("dataset:N", sort=["ML · train (original)", "ML · train (SMOTE)", "ML · test"], title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("n:Q", title="Créditos", stack="zero"),
        color=alt.Color("clase:N", scale=alt.Scale(domain=["0 · paga", "1 · default"], range=[AZUL, NARANJA]), legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("dataset:N"), alt.Tooltip("clase:N"), alt.Tooltip("n:Q", format=","), alt.Tooltip("pct:Q", format=".1%")])
        .properties(title="Balance de clases antes y después"), width="stretch", height=320)
    with s2:
        st.markdown("**Configuración**")
        tabla_texto(pd.DataFrame({"Parámetro": ["Técnica", "Proporción objetivo", "Vecinos (k)", "Dónde se aplica", "Semilla"],
                                  "Valor": ["SMOTE-NC (imbalanced-learn)", "1 : 1 (sampling_strategy = 1.0)", "5", "Solo train", "42"]}))
        st.markdown("**Correcciones de coherencia** aplicadas a las sintéticas")
        tabla_texto(t("smote_correcciones").rename(columns={"tipo": "Corrección", "variables": "Variables"}))

    st.markdown("**¿Se parecen los defaults sintéticos a los reales?**")
    comp = t("smote_comparacion")
    h1, h2 = st.columns([1, 1], gap="large")
    h1.dataframe(comp, hide_index=True, width="stretch", height=420,
                 column_config={"variable": st.column_config.TextColumn("Variable", width=190),
                                **{c: st.column_config.NumberColumn(c.replace("_", " ").replace("sintetica", "sint.").capitalize(), format="%.4g", width=62)
                                   for c in ["media_real", "media_sintetica", "std_real", "std_sintetica"]},
                                "D_KS": st.column_config.ProgressColumn("D KS", format="%.3f", min_value=0, max_value=0.3)})
    with h2:
        lista_vh = comp.sort_values("D_KS", ascending=False)["variable"].tolist()
        vh = st.selectbox("Distribución (ordenadas por D de KS, de mayor a menor diferencia)", lista_vh, key="var_sm",
                          index=lista_vh.index("EXT_SOURCE_3") if "EXT_SOURCE_3" in lista_vh else 0)
        hh = t("smote_hist").query("variable == @vh")
        st.altair_chart(alt.Chart(hh).mark_line(interpolate="step-after", strokeWidth=2).encode(
            x=alt.X("desde:Q", title=vh), y=alt.Y("densidad:Q", title="Densidad"),
            color=alt.Color("serie:N", scale=alt.Scale(domain=["Default real", "Default sintético", "No default"], range=[NARANJA, AZUL, BARRA]),
                            legend=alt.Legend(orient="top", title=None)),
            strokeDash=alt.StrokeDash("serie:N", scale=alt.Scale(domain=["Default real", "Default sintético", "No default"],
                                                                 range=[[1, 0], [5, 3], [1, 0]]), legend=None),
            tooltip=[alt.Tooltip("serie:N"), alt.Tooltip("desde:Q", format=",.4g"), alt.Tooltip("densidad:Q", format=".3g")]),
            width="stretch", height=330)
    st.markdown(
        "- **Medias casi idénticas, dispersión menor.** Interpolar entre vecinos «rellena» el interior de la nube de defaults: la "
        "desviación estándar sintética es hasta ~20 % menor y las colas quedan sub-representadas.\n"
        "- **Las categóricas se concentran en la moda**: con voto de mayoría, el grupo más frecuente gana peso (en `OCCUPATION_TYPE`, "
        "G4 pasa de 29 % de los defaults reales a 37 % de los sintéticos).")
    sc = t("smote_categoricas")
    lista_sc = sorted(sc["variable"].unique())
    vc_ = st.selectbox("Variable categórica", lista_sc, index=lista_sc.index("OCCUPATION_TYPE") if "OCCUPATION_TYPE" in lista_sc else 0, key="cat_sm")
    dsc = sc.query("variable == @vc_")
    st.altair_chart(alt.Chart(dsc).mark_bar(height={"band": 0.8}).encode(
        y=alt.Y("categoria:N", title=None, sort="-x", axis=alt.Axis(labelOverlap=False, labelLimit=260)), yOffset="serie:N", x=alt.X("pct:Q", title="% de los defaults", axis=alt.Axis(format="%")),
        color=alt.Color("serie:N", scale=alt.Scale(domain=["Default real", "Default sintético"], range=[NARANJA, AZUL]), legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("categoria:N"), alt.Tooltip("serie:N"), alt.Tooltip("pct:Q", format=".1%")]),
        width="stretch", height=max(160, 46 * dsc["categoria"].nunique()))
    st.warning(
        "**Tres reglas al usar este dataset.** (1) SMOTE se aplicó **solo a train**: test conserva el 8 % real y es la única "
        "medida honesta del desempeño. (2) Un modelo entrenado con 50 % de defaults producirá **probabilidades infladas**; para "
        "usarlas como PD hay que recalibrarlas (p. ej. corrección por prevalencia: odds × 0.0807/0.9193 ÷ (0.5/0.5), o "
        "calibración isotónica en una muestra sin rebalancear). (3) El umbral de 50 % de la guía deja de ser neutral: conviene "
        "comparar modelos con AUC / Gini / KS, que no dependen del umbral.", icon=":material/rule:")

# ══════════════════════════════════════════════════════════════════════════════
with tab_fe:
    st.markdown(
        "Espacio reservado para **feature engineering**, a revisar con la profesora. Son variables **propuestas** a partir de "
        "relaciones de negocio; se midió su IV y Gini en train (mismos parámetros de OptBinning) y su redundancia con las "
        "variables ya seleccionadas, **pero no entran a los datasets** hasta que se apruebe su inclusión.")
    fe = t("feature_engineering")
    fe["veredicto"] = np.select(
        [fe["iv"] >= 0.5, fe["max_rho_seleccionadas"] > 0.6, fe["pasaria_ml"]],
        ["IV ≥ 0.5: revisar sobreajuste", "Redundante con una seleccionada", "Candidata para ML"], default="Sin poder suficiente")
    st.dataframe(fe[["feature", "formula", "lectura", "iv", "gini", "gini_test", "max_rho_seleccionadas", "variable_mas_correlacionada", "veredicto"]],
                 hide_index=True, width="stretch", height=300,
                 column_config={"feature": st.column_config.TextColumn("Feature", pinned=True, width=200),
                                "formula": st.column_config.TextColumn("Fórmula", width=230), "lectura": st.column_config.TextColumn("Idea de negocio", width=330),
                                "iv": st.column_config.NumberColumn("IV", format="%.3f"), "gini": st.column_config.NumberColumn("Gini", format="%.3f"),
                                "gini_test": st.column_config.NumberColumn("Gini test", format="%.3f"),
                                "max_rho_seleccionadas": st.column_config.NumberColumn("|ρ| máx.", format="%.2f", help="Con las variables del dataset ML."),
                                "variable_mas_correlacionada": st.column_config.TextColumn("Más parecida a", width=200),
                                "veredicto": st.column_config.TextColumn("Lectura preliminar", width=230)})
    f1, f2 = st.columns([3, 2], gap="large")
    f1.altair_chart((alt.Chart(fe).mark_circle(size=160, opacity=0.85, stroke="white").encode(
        x=alt.X("max_rho_seleccionadas:Q", title="|ρ| máximo con las variables ya seleccionadas", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("gini:Q", title="Gini univariado (train)"),
        color=alt.Color("veredicto:N", scale=alt.Scale(domain=["Candidata para ML", "Redundante con una seleccionada", "IV ≥ 0.5: revisar sobreajuste", "Sin poder suficiente"],
                                                       range=[VERDE, GRIS, NARANJA, BARRA]), legend=alt.Legend(orient="top", title=None, columns=2)),
        tooltip=[alt.Tooltip("feature:N"), alt.Tooltip("gini:Q", format=".3f"), alt.Tooltip("max_rho_seleccionadas:Q", format=".2f")])
        + alt.Chart(fe).mark_text(dx=10, align="left", fontSize=10, color=TINTA).encode(x="max_rho_seleccionadas:Q", y="gini:Q", text="feature:N")
        + alt.Chart(pd.DataFrame({"x": [0.6]})).mark_rule(strokeDash=[4, 3], color=GRIS).encode(x="x:Q")
        + alt.Chart(pd.DataFrame({"y": [0.08]})).mark_rule(strokeDash=[4, 3], color=GRIS).encode(y="y:Q"))
        .properties(title="Poder vs. redundancia de las propuestas"), width="stretch", height=380)
    with f2:
        st.markdown(
            "**Para discutir**\n\n"
            "- `EXT_SOURCE_PROMEDIO` tiene **IV 0.61**: cae en el rango «revisar sobreajuste». No hay fuga (los scores existen al "
            "momento de la solicitud), pero combina tres variables ya seleccionadas: ¿reemplazarlas o sumarla?\n"
            "- `RATIO_CREDITO_BIEN` (≈ LTV) aporta señal (Gini 0.14) y **casi no se parece** a nada seleccionado: es la candidata "
            "más limpia para el dataset ML.\n"
            "- `RATIO_EMPLEO_EDAD` es casi la misma variable que `DAYS_EMPLOYED` (ρ = 0.98).\n"
            "- Los ratios de capacidad de pago (`RATIO_CUOTA_INGRESO`, `RATIO_CREDITO_INGRESO`) tienen **IV < 0.02**: la "
            "intuición de negocio no siempre se confirma en los datos.")
    st.caption("Estas cifras se calcularon solo con train. Si alguna propuesta se aprueba, debe pasar por los mismos filtros "
               "(univariado, bivariado y multivariado) antes de entrar al dataset final.")

st.divider()
st.caption("Fuente: `scripts/pipeline_modelado.py datasets`. Los archivos completos (`artifacts/modelado/datasets/*.parquet`) se "
           "generan localmente con ese comando y no se publican en el repositorio.")
