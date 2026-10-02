"""Calidad y preprocesamiento: diagnóstico del tablón oficial, decisiones sobre faltantes y outliers, y resultado."""
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

REPO_DIR = Path(__file__).resolve().parents[1]
CAL = REPO_DIR / "artifacts" / "calidad"

AZUL, NARANJA, VERDE, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"
COLOR_TIPO = alt.Scale(domain=["Numérica continua", "Numérica discreta", "Dicotómica", "Categórica nominal", "Categórica ordinal"],
                       range=[AZUL, "#7f5fc8", VERDE, NARANJA, "#c98a2b"])
PCT = alt.Axis(format="%")


@st.cache_data(show_spinner=False)
def _leer(ruta: str, version: float) -> pd.DataFrame:
    return pd.read_parquet(ruta)


def t(nombre: str) -> pd.DataFrame:
    """Lee un artefacto; la fecha de modificación entra en la llave del caché para que un redeploy no sirva datos viejos."""
    ruta = CAL / f"{nombre}.parquet"
    return _leer(str(ruta), ruta.stat().st_mtime)


@st.cache_data(show_spinner=False)
def _resumen(ruta: str, version: float) -> dict:
    return json.load(open(ruta, encoding="utf-8"))


def resumen() -> dict:
    ruta = CAL / "resumen.json"
    return _resumen(str(ruta), ruta.stat().st_mtime)


def regla_base(base: float, eje: str = "x"):
    datos = pd.DataFrame({"v": [base]})
    enc = {eje: alt.X("v:Q") if eje == "x" else alt.Y("v:Q")}
    return alt.Chart(datos).mark_rule(color=NARANJA, strokeDash=[5, 4], strokeWidth=1.5).encode(**enc)


def nota(texto: str, icono: str = ":material/insights:") -> None:
    st.info(texto, icon=icono)


def tabla_texto(df: pd.DataFrame, formatos: dict | None = None) -> None:
    """Tabla estática que ajusta el texto largo en varias líneas (st.dataframe lo corta)."""
    st.table(df.style.hide(axis="index").format(formatos or {}, na_rep="—"))


TIPO_CORTO = {"Numérica continua": "Continua", "Numérica discreta": "Discreta", "Dicotómica": "Dicotómica",
              "Categórica nominal": "Nominal", "Categórica ordinal": "Ordinal", "Identificador": "ID / objetivo",
              "Objetivo (dicotómica)": "ID / objetivo"}


R = resumen()
BASE = R["tasa_base"]
N = R["filas"]
cal = t("calidad_variables")

# ── Encabezado ─────────────────────────────────────────────────────────────────
st.title("Calidad y preprocesamiento")
st.markdown(
    "Antes de modelar hay que responder dos preguntas: **¿se puede confiar en el tablón tal como llega?** y **¿qué se "
    "corrigió, por qué y con qué efecto?** Esta página sigue los Pasos 2 (*Análisis de calidad*) y 3 (*Preprocesamiento*) "
    "de la guía: primero el diagnóstico, después las decisiones sobre **valores faltantes** y **outliers**, y al final la "
    "comparación **antes / después** con sus validaciones."
)
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Variables con nulos", f"{(cal.n_nulos > 0).sum()}", border=True, help=f"De {len(cal)} variables del tablón oficial.")
k2.metric("Celdas nulas", f"{R['celdas_nulas_antes'] / (N * R['columnas']):.1%}", border=True,
          help=f"{R['celdas_nulas_antes']:,} de {N * R['columnas']:,} celdas del tablón oficial.")
k3.metric("Variables imputadas", "15", border=True, help="12 numéricas + 3 categóricas; más el centinela de DAYS_EMPLOYED.")
k4.metric("Outliers reasignados", f"{len(t('plan_outliers'))} variables", border=True, help="Reasignación al tramo con tasa de default similar (aprendida en train).")
val = t("validaciones")
k5.metric("Validaciones OK", f"{val.ok.sum()}/{len(val)}", border=True, help="Chequeos automáticos del preprocesamiento.")

tab_diag, tab_na, tab_out, tab_res = st.tabs([":material/fact_check: Diagnóstico de calidad",
                                              ":material/format_color_reset: Valores faltantes",
                                              ":material/stacked_line_chart: Outliers",
                                              ":material/compare_arrows: Resultado del preprocesamiento"])

# ══════════════════════════════════════════════════════════════════════════════
# 1. DIAGNÓSTICO
# ══════════════════════════════════════════════════════════════════════════════
with tab_diag:
    st.subheader("Chequeos de integridad y consistencia")
    st.caption("Los que pide la guía (tipos de dato, faltantes reales y codificados, duplicados, valores inesperados) y reglas "
               "de negocio adicionales para descartar registros imposibles.")
    chq = t("chequeos").copy()
    chq["estado"] = chq["estado"].map({"ok": "✔ Correcto", "aviso": "▲ Requiere tratamiento", "info": "ℹ Informativo"})
    tabla_texto(chq[["estado", "categoria", "chequeo", "casos", "detalle"]].rename(columns={
        "estado": "Estado", "categoria": "Tipo", "chequeo": "Chequeo", "casos": "Casos", "detalle": "Detalle"}), {"Casos": "{:,}"})
    nota("La base es **íntegra** (sin duplicados, sin tipos mal asignados, sin fechas o montos imposibles). Los problemas reales "
         "son de **faltantes**: nulos por ausencia de historial, un **centinela** en `DAYS_EMPLOYED` y categorías que ocultan un "
         "faltante (`XNA`, `Unknown`). `ORGANIZATION_TYPE = 'XNA'` coincide exactamente con el centinela: no es un faltante "
         "sino la categoría *sin empleador*.")

    st.subheader("¿Cuánta información falta?")
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        familias = sorted(cal["familia"].unique())
        sel = st.multiselect("Filtrar por familia", familias, placeholder="Todas las familias", key="fam_fill")
        d = cal[(cal.n_nulos > 0) & (cal.familia.isin(sel) if sel else pd.Series(True, index=cal.index))].sort_values("fill_rate")
        barras = alt.Chart(d).mark_bar(cornerRadiusEnd=3, height={"band": 0.75}).encode(
            y=alt.Y("variable:N", sort=d["variable"].tolist(), title=None,
                    axis=alt.Axis(labelFontSize=10, labelLimit=260, labelOverlap=False)),
            x=alt.X("fill_rate:Q", title="Fill rate (% de créditos con dato)", scale=alt.Scale(domain=[0, 1]), axis=PCT),
            color=alt.Color("tipo_dato:N", scale=COLOR_TIPO, title=None, legend=alt.Legend(orient="top", columns=3)),
            tooltip=[alt.Tooltip("variable:N", title="Variable"), alt.Tooltip("familia:N", title="Familia"),
                     alt.Tooltip("fill_rate:Q", title="Fill rate", format=".1%"), alt.Tooltip("n_nulos:Q", title="Nulos", format=",")])
        regla = alt.Chart(pd.DataFrame({"v": [0.5]})).mark_rule(color=GRIS, strokeDash=[4, 4]).encode(x="v:Q")
        st.altair_chart((barras + regla).properties(height=max(180, 17 * len(d)), title="Fill rate de las variables con nulos"),
                        width="stretch")
    with c2:
        sin_id = cal[~cal["tipo_dato"].isin(["Identificador", "Objetivo (dicotómica)"])]
        tramos = pd.crosstab(sin_id["tramo_nulos"], sin_id["tipo_dato"].map(TIPO_CORTO), margins=True, margins_name="Total")
        tramos = tramos[[c for c in ["Continua", "Discreta", "Dicotómica", "Nominal", "Ordinal", "Total"] if c in tramos]]
        tramos.index.name, tramos.columns.name = "% de nulos", None
        orden = ["0% (completa)", "(0%, 5%]", "(5%, 20%]", "(20%, 50%]", "> 50%", "Total"]
        st.markdown("**Variables por tramo de nulos** (sin ID ni TARGET)")
        corto = {"0% (completa)": "0%", "(0%, 5%]": "≤5%", "(5%, 20%]": "5–20%", "(20%, 50%]": "20–50%",
                 "> 50%": "50%+", "Total": "Total"}
        tabla_texto(tramos.reindex(orden).rename(index=corto).T.rename_axis("Tipo").reset_index())
        nf = t("nulos_por_fila")
        hist = alt.Chart(nf).mark_bar(color=AZUL, binSpacing=1).encode(
            x=alt.X("desde:Q", bin="binned", title="% de variables nulas en el crédito", axis=PCT),
            x2="hasta:Q", y=alt.Y("antes:Q", title="Créditos"),
            tooltip=[alt.Tooltip("desde:Q", format=".0%", title="Desde"), alt.Tooltip("hasta:Q", format=".0%", title="Hasta"),
                     alt.Tooltip("antes:Q", format=",", title="Créditos")])
        med = alt.Chart(pd.DataFrame({"v": [R["nulos_fila_mediana"]]})).mark_rule(color="#333", strokeDash=[4, 3]).encode(x="v:Q")
        st.altair_chart((hist + med).properties(height=260, title=f"Nulos por crédito (mediana {R['nulos_fila_mediana']:.1%})"),
                        width="stretch")
        nota("Los faltantes se concentran en **vivienda** (47 variables con 30–53 % de fill rate), los **scores externos 1 y 3** y el "
             "**historial** (buró y tarjetas). Casi ningún crédito está completo: la mediana es de ~31 % de variables nulas y solo ~2 % de los créditos tiene menos del 2 %.")

    with st.expander("Patrón de nulos en una muestra de 1,500 créditos", icon=":material/grid_on:"):
        m = t("matriz_nulos")
        m = m.iloc[::5].reset_index(drop=True)  # 300 filas: suficiente para ver el patrón por bloques
        largo = m.reset_index().melt(id_vars="index", var_name="variable", value_name="nulo")
        mapa = alt.Chart(largo).mark_rect().encode(
            x=alt.X("variable:N", sort=list(m.columns), title=None, axis=alt.Axis(labelAngle=-90, labelFontSize=8)),
            y=alt.Y("index:O", title="Créditos (muestra)", axis=None),
            color=alt.Color("nulo:N", scale=alt.Scale(domain=[0, 1], range=["#eaf1fb", "#1f4e8c"]),
                            legend=alt.Legend(title=None, labelExpr="datum.value == 1 ? 'Nulo' : 'Con dato'", orient="top")),
            tooltip=[alt.Tooltip("variable:N"), alt.Tooltip("nulo:N", title="Nulo (1 = sí)")])
        st.altair_chart(mapa.properties(height=320), width="stretch")
        st.caption("Columnas ordenadas por % de nulos. Los bloques verticales revelan nulos que ocurren juntos: vivienda, "
                   "historial de buró y tarjetas. Es la huella de un faltante **estructural**, no aleatorio.")

    st.subheader("Numéricas: rangos y colas")
    num = t("numericas")
    f1, f2 = st.columns([2, 1], vertical_alignment="bottom")
    fam_num = f1.multiselect("Familia", sorted(num["familia"].unique()), placeholder="Todas", key="fam_num")
    solo_alerta = f2.toggle("Solo con > 5 % de outliers (IQR)", key="solo_out")
    mask = num["familia"].isin(fam_num) if fam_num else pd.Series(True, index=num.index)
    if solo_alerta:
        mask &= num["pct_outliers_iqr"] > 0.05
    dn = num[mask]
    st.dataframe(dn.drop(columns="familia"), hide_index=True, width="stretch", height=380,
                 column_config={"variable": st.column_config.TextColumn("Variable", pinned=True, width=230),
                                **{c: st.column_config.NumberColumn(c, format="compact") for c in ["min", "p1", "p50", "p99", "max"]},
                                "asimetria": st.column_config.NumberColumn("Asimetría", format="%.2f"),
                                "pct_outliers_iqr": st.column_config.ProgressColumn("% outliers (1.5·IQR)", format="percent",
                                                                                    min_value=0, max_value=0.3),
                                "pct_ceros": st.column_config.ProgressColumn("% ceros", format="percent", min_value=0, max_value=1)})
    st.caption("Tabla de colas pedida por la guía (mín, p1, p50, p99, máx). Un outlier estadístico no es un error: la decisión de "
               "tratarlo se toma en la pestaña **Outliers** con evidencia de riesgo.")

    st.subheader("Categóricas: cardinalidad, categorías raras y valores inesperados")
    cat = t("categoricas")
    cc1, cc2 = st.columns([3, 2], gap="large")
    with cc1:
        st.dataframe(cat.assign(tipo_dato=cat["tipo_dato"].map(TIPO_CORTO)), hide_index=True, width="stretch",
                     height=38 + 35 * len(cat),
                     column_config={"variable": st.column_config.TextColumn("Variable", width=200, pinned=True),
                                    "tipo_dato": st.column_config.TextColumn("Tipo", width=75),
                                    "n_categorias": st.column_config.NumberColumn("Categorías", width=90),
                                    "pct_nulos": st.column_config.NumberColumn("% nulos", format="percent", width=80),
                                    "moda": st.column_config.TextColumn("Moda", width=180),
                                    "pct_moda": st.column_config.NumberColumn("% moda", format="percent", width=80),
                                    "n_raras": st.column_config.NumberColumn("Raras (< 1 %)", width=95),
                                    "pct_obs_raras": st.column_config.NumberColumn("% obs. en raras", format="percent", width=100),
                                    "codificados": st.column_config.NumberColumn("XNA / Unknown", width=100)})
    with cc2:
        var_c = st.selectbox("Frecuencias de", cat["variable"], index=int(cat.index[cat.variable == "ORGANIZATION_TYPE"][0]),
                             key="cat_freq")
        det = t("categorias_detalle").query("variable == @var_c").sort_values("n", ascending=False).head(20).copy()
        det["tipo"] = det.apply(lambda r: "Nulo / codificado" if r.categoria in ("(nulo)", "XNA", "Unknown")
                                else ("Rara (< 1 %)" if r.pct < 0.01 else "Categoría"), axis=1)
        barras_c = alt.Chart(det).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y("categoria:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200, labelOverlap=False)),
            x=alt.X("pct:Q", title="% de créditos", axis=PCT),
            color=alt.Color("tipo:N", scale=alt.Scale(domain=["Categoría", "Rara (< 1 %)", "Nulo / codificado"],
                                                        range=[AZUL, "#9db8d9", GRIS]), legend=alt.Legend(orient="top", title=None, columns=2)),
            tooltip=[alt.Tooltip("categoria:N", title="Categoría"), alt.Tooltip("n:Q", format=",", title="Créditos"),
                     alt.Tooltip("pct:Q", format=".2%", title="%")])
        st.altair_chart(barras_c.properties(height=max(170, 26 * len(det)), title=f"{var_c} (top 20)"), width="stretch")
    nota("`ORGANIZATION_TYPE` (58 categorías, 41 raras) y `OCCUPATION_TYPE` (18) tienen **cardinalidad alta**: conviene agruparlas "
         "en el bivariado según su tasa de default, no por frecuencia. `CODE_GENDER = 'XNA'` (4) y `NAME_FAMILY_STATUS = 'Unknown'` (2) "
         "son faltantes disfrazados de categoría.")

    st.subheader("Alertas por variable")
    alertas = cal[cal.n_alertas > 0].copy()
    largo = alertas.assign(alerta=alertas["alertas"].str.split(", ")).explode("alerta", ignore_index=True)
    matriz = pd.crosstab(largo["alerta"], largo["tipo_dato"].map(TIPO_CORTO), margins=True, margins_name="Total")
    matriz.index.name, matriz.columns.name = "Alerta", None
    a1, a2 = st.columns([2, 3], gap="large")
    a1.dataframe(matriz, width="stretch")
    a1.caption(f"{(cal.n_alertas == 0).sum() - 2} variables sin ninguna alerta (sin contar ID y TARGET). Las alertas son para "
               "revisar, no exclusiones automáticas.")
    a2.dataframe(alertas.sort_values(["n_alertas", "fill_rate"], ascending=[False, True])[["variable", "tipo_dato", "fill_rate", "alertas"]],
                 hide_index=True, width="stretch", height=360,
                 column_config={"variable": st.column_config.TextColumn("Variable", width=230),
                                "tipo_dato": st.column_config.TextColumn("Tipo", width=130),
                                "fill_rate": st.column_config.ProgressColumn("Fill rate", format="percent", min_value=0, max_value=1, width=110),
                                "alertas": st.column_config.TextColumn("Alertas", width=380)})

# ══════════════════════════════════════════════════════════════════════════════
# 2. VALORES FALTANTES
# ══════════════════════════════════════════════════════════════════════════════
with tab_na:
    st.markdown("Cada faltante se clasifica según **por qué falta**, porque de eso depende el tratamiento correcto:")
    t1, t2, t3, t4 = st.columns(4, gap="medium")
    with t1, st.container(border=True):
        st.markdown(":blue-badge[Estructural]  \nEl dato **no puede existir**: sin créditos en el buró no hay deuda en el buró.  \n"
                    "**Tratamiento:** no imputar; marcar la ausencia con un indicador.")
    with t2, st.container(border=True):
        st.markdown(":orange-badge[Falta de información]  \nEl dato existe pero **no se registró**, y la ausencia depende del perfil.  \n"
                    "**Tratamiento:** imputación condicionada (mediana por grupo) o indicador de bloque.")
    with t3, st.container(border=True):
        st.markdown(":green-badge[Despreciable]  \nMenos de **0.5 %** de los créditos; su impacto es marginal.  \n"
                    "**Tratamiento:** moda, mediana o una regla de negocio simple.")
    with t4, st.container(border=True):
        st.markdown(":violet-badge[Centinela / codificado]  \nUn valor que **simula** un dato (`365243`, `XNA`, `Unknown`).  \n"
                    "**Tratamiento:** reemplazar y conservar la información en un flag.")

    st.subheader("¿El nulo es informativo?")
    ni = t("nulo_informativo")
    largo = ni.melt(id_vars=["grupo", "n_nulos"], value_vars=["default_si_dato", "default_si_nulo"], var_name="serie", value_name="tasa")
    largo["serie"] = largo["serie"].map({"default_si_dato": "Con dato", "default_si_nulo": "Nulo (sin dato)"})
    orden_g = ni["grupo"].tolist()
    barras_ni = alt.Chart(largo).mark_bar(cornerRadiusEnd=3, height={"band": 0.8}).encode(
        y=alt.Y("grupo:N", sort=orden_g, title=None, axis=alt.Axis(labelLimit=220)),
        yOffset=alt.YOffset("serie:N", sort=["Con dato", "Nulo (sin dato)"]),
        x=alt.X("tasa:Q", title="Tasa de default", axis=PCT),
        color=alt.Color("serie:N", scale=alt.Scale(domain=["Con dato", "Nulo (sin dato)"], range=[GRIS, AZUL]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("grupo:N", title="Grupo"), alt.Tooltip("serie:N", title="Serie"),
                 alt.Tooltip("tasa:Q", format=".2%", title="Default"), alt.Tooltip("n_nulos:Q", format=",", title="Nulos")])
    n1, n2 = st.columns([3, 2], gap="large")
    n1.altair_chart((barras_ni + regla_base(BASE)).properties(height=470), width="stretch")
    with n2:
        st.markdown(f"La línea naranja es la tasa base (**{BASE:.1%}**).")
        st.markdown(
            "- **Sin buró** y **sin consultas** → más default (~10 %): no tener historial externo es una **señal de riesgo**.\n"
            "- **Sin solicitudes previas / POS / cuotas** → menos default (~6 %): los clientes nuevos en Home Credit incumplen menos.\n"
            "- **Sin `EXT_SOURCE_3`** y **sin vivienda** → más default.\n"
            "- **Sin ocupación** → menos default: el grupo está dominado por pensionistas.\n\n"
            "Conclusión: **el nulo contiene señal**. Imputarlo a ciegas la destruiría; por eso se conserva el `NaN` en los "
            "estructurales y se agregan indicadores de ausencia."
        )

    with st.expander("Inventario completo de nulos (91 variables)", icon=":material/table_view:"):
        inv = t("inventario_nulos")
        fam_inv = st.multiselect("Familia", sorted(inv["familia"].unique()), placeholder="Todas", key="fam_inv")
        st.dataframe(inv[inv.familia.isin(fam_inv)] if fam_inv else inv, hide_index=True, width="stretch", height=420,
                     column_config={"variable": st.column_config.TextColumn("Variable", pinned=True, width=230),
                                    "familia": st.column_config.TextColumn("Familia", width=170),
                                    "tipo_dato": st.column_config.TextColumn("Tipo", width=140),
                                    "n_nulos": st.column_config.NumberColumn("Nulos", format="localized"),
                                    "fill_rate": st.column_config.ProgressColumn("Fill rate", format="percent", min_value=0, max_value=1),
                                    "default_si_nulo": st.column_config.NumberColumn("Default si nulo", format="percent"),
                                    "default_si_dato": st.column_config.NumberColumn("Default con dato", format="percent"),
                                    "razon_default": st.column_config.NumberColumn("Razón", format="%.2f",
                                                                                   help="Default si nulo / default con dato.")})

    st.subheader("Evidencia por tipo de faltante")
    sub = st.segmented_control("Caso", ["Centinela DAYS_EMPLOYED", "Nulos estructurales", "EXT_SOURCE 1 y 3", "Bloque de vivienda",
                                        "Despreciables"], default="Centinela DAYS_EMPLOYED", key="caso_na")

    if sub == "Centinela DAYS_EMPLOYED":
        ten = t("centinela_tenencia")
        ten["grupo"] = ["Antigüedad"] * 5 + ["Centinela"]
        e1, e2 = st.columns([3, 2], gap="large")
        barras_t = alt.Chart(ten).mark_bar(cornerRadiusEnd=4, size=46).encode(
            x=alt.X("tramo:N", sort=ten["tramo"].tolist(), title=None, axis=alt.Axis(labelAngle=0, labelLimit=160, labelOverlap=False)),
            y=alt.Y("tasa:Q", title="Tasa de default", axis=PCT),
            color=alt.Color("grupo:N", scale=alt.Scale(domain=["Antigüedad", "Centinela"], range=[AZUL, NARANJA]), legend=None),
            tooltip=[alt.Tooltip("tramo:N", title="Tramo"), alt.Tooltip("n:Q", format=",", title="Créditos"),
                     alt.Tooltip("tasa:Q", format=".2%", title="Default")])
        etiquetas = alt.Chart(ten).mark_text(dy=-8, fontSize=11, color="#52514e").encode(
            x=alt.X("tramo:N", sort=ten["tramo"].tolist()), y="tasa:Q", text=alt.Text("tasa:Q", format=".1%"))
        e1.altair_chart((barras_t + etiquetas + regla_base(BASE, "y")).properties(
            height=330, title="Default por antigüedad laboral: el centinela se parece a los más antiguos"), width="stretch")
        e1.caption("Créditos por tramo en el tooltip. Tramos de antigüedad con DAYS_EMPLOYED ≤ 0.")
        with e2:
            st.dataframe(t("centinela_coincidencias"), hide_index=True, width="stretch",
                         column_config={"condicion": st.column_config.TextColumn("Condición", width=280),
                                        "n": st.column_config.NumberColumn("Créditos", format="localized")})
            st.markdown(f"`DAYS_EMPLOYED = 365243` (≈ 1,000 años) marca a **55,374** créditos: todos con `ORGANIZATION_TYPE = 'XNA'` "
                        f"y `FLAG_EMP_PHONE = 0`, y casi todos pensionistas. Su default (5.4 %) se parece al de los más antiguos, no "
                        f"al de un empleo recién iniciado. Solo hay **{R['ceros_reales_days_employed']}** ceros reales.")
            st.success("**Decisión:** reemplazar por `0` (sin días de empleo) y crear `flag_sin_empleo` para conservar la señal.",
                       icon=":material/check_circle:")

    elif sub == "Nulos estructurales":
        est = t("estructurales")
        e1 = st.container()
        e1.dataframe(est, hide_index=True, width="stretch", height=38 + 35 * len(est),
                     column_config={"familia": st.column_config.TextColumn("Familia", width=140),
                                    "variable": st.column_config.TextColumn("Variable", width=240),
                                    "n_nulos": st.column_config.NumberColumn("Nulos", format="localized"),
                                    "pct_nulos": st.column_config.ProgressColumn("% nulos", format="percent", min_value=0, max_value=1),
                                    "ceros_entre_no_nulos": st.column_config.NumberColumn("Ceros reales", format="localized"),
                                    "filas_todas_nulas": st.column_config.NumberColumn("Sin registros en la familia", format="localized")})
        e2, e3 = st.columns([2, 3], gap="large")
        with e2:
            st.markdown("**Consultas al buró vs. créditos en el buró**")
            tabla_texto(t("consultas_vs_buro").rename(columns={"consultas": ""}),
                        {"Con créditos en buró": "{:,}", "Sin créditos en buró": "{:,}"})
            st.success("**Decisión:** no imputar. Conservar el `NaN` y marcar la ausencia de historial con `flag_sin_buro` y "
                       "`flag_sin_previas`.", icon=":material/check_circle:")
        with e3:
            st.markdown(
                f"- Las **{R['consultas_nulas']:,}** consultas nulas corresponden todas a créditos **sin buró**: el nulo significa "
                f"*no figura en el buró*, no *cero consultas*.\n"
                f"- `BUREAU_MESES_CON_ATRASO`: {R['meses_atraso_nulos']:,} nulos, de los cuales **{R['meses_atraso_nulos_con_creditos']:,}** "
                f"tienen créditos pero **sin historial mensual**; hay {R['meses_atraso_ceros']:,} ceros reales. Imputar 0 mezclaría "
                f"*sin detalle* con *sin atraso*.\n"
                f"- `OWN_CAR_AGE`: {R['auto_nulos']:,} nulos, **{R['auto_nulos_sin_auto']:,}** son clientes sin auto; ya lo informa "
                f"`FLAG_OWN_CAR` y existen {R['auto_edad_cero']:,} autos de edad 0 reales.\n"
                f"- `OCCUPATION_TYPE`: {R['ocupacion_nulos']:,} nulos; **{R['ocupacion_nulos_centinela']:,}** coinciden con el centinela "
                f"de empleo. Quedan {R['ocupacion_nulos'] - R['ocupacion_nulos_centinela']:,} sin explicación (default "
                f"{R['ocupacion_default_sin_explicar']:.1%})."
            )

    elif sub == "EXT_SOURCE 1 y 3":
        ext = t("ext_nulos")
        graf = []
        for var in ["EXT_SOURCE_1", "EXT_SOURCE_3"]:
            for eje, color in [("Edad", AZUL), ("Tipo de ingreso", VERDE)]:
                d = ext.query("variable == @var and eje == @eje")
                orden_x = d["grupo"].tolist() if eje == "Edad" else d.sort_values("pct_nulos", ascending=False)["grupo"].tolist()
                graf.append(alt.Chart(d).mark_bar(color=color, cornerRadiusEnd=3, size=34).encode(
                    x=alt.X("grupo:N", sort=orden_x, title=None,
                            axis=alt.Axis(labelAngle=0, labelLimit=150, labelOverlap=False, labelExpr="split(datum.label, ' ')")),
                    y=alt.Y("pct_nulos:Q", title="% nulos", axis=PCT),
                    tooltip=[alt.Tooltip("grupo:N", title=eje), alt.Tooltip("pct_nulos:Q", format=".1%", title="% nulos")])
                    .properties(title=f"{var}: % de nulos por {eje.lower()}", height=200, width=330))
        st.altair_chart(alt.vconcat(alt.hconcat(graf[0], graf[1]), alt.hconcat(graf[2], graf[3])).resolve_scale(y="independent"),
                        width="content")
        med = t("ext_medianas")
        var_m = st.segmented_control("Medianas por celda", ["EXT_SOURCE_1", "EXT_SOURCE_3"], default="EXT_SOURCE_1", key="ext_med")
        dm = med.query("variable == @var_m")
        dm_ok = dm[dm.n >= 200]
        base_m = alt.Chart(dm).encode(
            x=alt.X("ing:N", title="Tipo de ingreso",
                    axis=alt.Axis(labelAngle=0, labelOverlap=False, labelLimit=130, labelExpr="split(datum.label, ' ')")),
            y=alt.Y("edad:N", sort=["≤30", "30-40", "40-50", "50-60", ">60"], title="Edad"))
        mapa = base_m.mark_rect(stroke="white", strokeWidth=1.5).encode(
            color=alt.condition("datum.n >= 200", alt.Color("mediana:Q", scale=alt.Scale(scheme="blues"), title="Mediana"),
                                alt.value("#e4e4e0")),
            tooltip=[alt.Tooltip("edad:N"), alt.Tooltip("ing:N", title="Ingreso"), alt.Tooltip("mediana:Q", format=".3f"),
                     alt.Tooltip("n:Q", format=",", title="Obs. con dato")])
        texto = base_m.mark_text(fontSize=11).encode(
            text=alt.Text("mediana:Q", format=".2f"),
            color=alt.condition("datum.n >= 200 && datum.mediana > 0.55", alt.value("white"), alt.value("#222")),
            opacity=alt.condition("datum.n >= 200", alt.value(1), alt.value(0.45)))
        g1, g2 = st.columns([3, 2], gap="large")
        g1.altair_chart((mapa + texto).properties(title=f"{var_m}: mediana por edad × tipo de ingreso"), width="stretch", height=340)
        with g2:
            st.markdown(f"La mediana global de **{var_m}** es **{dm['mediana_global'].iloc[0]:.3f}**, pero por celda va de "
                        f"**{dm_ok['mediana'].min():.2f}** a **{dm_ok['mediana'].max():.2f}** (celdas con ≥ 200 casos; las grises usan el respaldo). El nulo depende de la edad y del tipo de "
                        "ingreso, y el score también: una mediana global sesgaría a los jóvenes hacia arriba y a los mayores "
                        "hacia abajo.")
            st.success("**Decisión:** imputar `EXT_SOURCE_1` y `EXT_SOURCE_3` con la **mediana por edad × tipo de ingreso** "
                       f"(celdas con ≥ 200 observaciones; si no, mediana por tipo de ingreso y luego global). "
                       "`EXT_SOURCE_2` (0.2 % de nulos) con la mediana global.", icon=":material/check_circle:")

    elif sub == "Bloque de vivienda":
        viv = t("vivienda_estado")
        viv["color"] = viv["estado"]
        e1, e2 = st.columns([3, 2], gap="large")
        b = alt.Chart(viv).mark_bar(cornerRadiusEnd=4, size=70).encode(
            x=alt.X("estado:N", sort=viv["estado"].tolist(), title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("tasa:Q", title="Tasa de default", axis=PCT),
            color=alt.Color("color:N", scale=alt.Scale(domain=viv["estado"].tolist(), range=[AZUL, "#7fb0ea", NARANJA]), legend=None),
            tooltip=[alt.Tooltip("estado:N"), alt.Tooltip("n:Q", format=","), alt.Tooltip("tasa:Q", format=".2%", title="Default")])
        lab = alt.Chart(viv).mark_text(dy=-8, fontSize=11, color="#52514e").encode(
            x=alt.X("estado:N", sort=viv["estado"].tolist()), y="tasa:Q", text=alt.Text("tasa:Q", format=".1%"))
        e1.altair_chart((b + lab + regla_base(BASE, "y")).properties(height=320, title="Vivienda: el estado de la información también discrimina"),
                        width="stretch")
        with e2:
            st.dataframe(t("vivienda_por_tipo"), hide_index=True, width="stretch",
                         column_config={"NAME_HOUSING_TYPE": st.column_config.TextColumn("Tipo de vivienda"),
                                        "pct_sin_info": st.column_config.ProgressColumn("% sin info de vivienda", format="percent",
                                                                                        min_value=0, max_value=1),
                                        "n": st.column_config.NumberColumn("Créditos", format="localized")})
            st.markdown(f"Las **{R['vivienda_n_vars']}** variables del edificio faltan en bloque: el 47 % de los créditos no tiene "
                        f"ninguna y quienes tienen información parcial carecen, en mediana, de {R['vivienda_parcial_mediana_faltantes']}. "
                        "Imputar 47 variables inventaría casi la mitad de la base; en cambio, **no tener información de vivienda** "
                        "sube el default a 9.3 %.")
            st.success("**Decisión:** no imputar valores; crear `flag_sin_info_vivienda` (1 si faltan las 47).",
                       icon=":material/check_circle:")

    else:
        e1, e2 = st.columns([2, 3], gap="large")
        with e1:
            st.markdown("**Faltantes por debajo de 0.5 %**")
            tabla_texto(t("despreciables").rename(columns={"variable": "Variable / caso", "n": "Casos", "pct": "% del total"}),
                        {"Casos": "{:,}", "% del total": "{:.3%}"})
        with e2:
            st.markdown("**Círculo social: ¿media, mediana o moda?**")
            st.dataframe(t("circulo_social"), hide_index=True, width="stretch",
                         column_config={"media": st.column_config.NumberColumn(format="%.3f"),
                                        "pct_ceros": st.column_config.ProgressColumn("% ceros", format="percent", min_value=0, max_value=1)})
            st.markdown("**Montos: error relativo mediano al imputar (menor es mejor)**")
            st.dataframe(t("montos_error"), hide_index=True, width="stretch",
                         column_config={"spearman_con_amt_credit": st.column_config.NumberColumn("ρ Spearman con AMT_CREDIT", format="%.3f"),
                                        "error_mediana_global": st.column_config.NumberColumn("Mediana global", format="percent"),
                                        "error_razon_por_contrato": st.column_config.NumberColumn("Razón por contrato", format="percent")})
            st.success("**Decisión:** conteos del círculo social y `CNT_FAM_MEMBERS` con la **moda** (0 y 2; miembros ≥ hijos + 1); "
                       "`AMT_GOODS_PRICE` y `AMT_ANNUITY` con `AMT_CREDIT × razón mediana por tipo de contrato` (el error cae de 50 % a "
                       "6 % en el precio del bien); `DAYS_LAST_PHONE_CHANGE` con la mediana; `NAME_TYPE_SUITE`, `CODE_GENDER = 'XNA'` y "
                       "`NAME_FAMILY_STATUS = 'Unknown'` con la moda.", icon=":material/check_circle:")

    st.subheader("Plan de tratamiento de faltantes")
    plan = pd.DataFrame([
        ("Buró: BUREAU_* (8) y AMT_REQ_CREDIT_BUREAU_* (6)", "Estructural", "Sin imputar · flag_sin_buro", "Aplicado"),
        ("Solicitudes previas: HC_N_SOLICITUDES, HC_N_APROBADAS, HC_N_RECHAZADAS, HC_DIAS_ULTIMA_DECISION, HC_PROP_SOLICITUDES_RECHAZADAS",
         "Estructural", "Sin imputar · flag_sin_previas", "Aplicado"),
        ("POS Cash: HC_N_OPERACIONES_POS, HC_POS_MAX_ATRASO, HC_POS_MAX_CUOTAS_PENDIENTES", "Estructural", "Sin imputar · sin flag", "Aplicado"),
        ("Cuotas: HC_N_REGISTROS_PAGO, HC_N_PAGOS_TARDE, HC_PAY_MAX_ATRASO, HC_PROP_PAGOS_TARDE", "Estructural",
         "Sin imputar · sin flag", "Aplicado"),
        ("Tarjeta: HC_N_OPERACIONES_TARJETA, HC_CARD_MAX_ATRASO, HC_CARD_MAX_UTILIZACION, HC_CARD_MAX_ULTIMO_SALDO", "Estructural",
         "Sin imputar · sin flag", "Aplicado"),
        ("OWN_CAR_AGE", "Estructural", "Sin imputar (ya lo informa FLAG_OWN_CAR)", "Aplicado"),
        ("DAYS_EMPLOYED = 365243", "Centinela", "Reemplazo por 0 · flag_sin_empleo", "Aplicado"),
        ("EXT_SOURCE_1, EXT_SOURCE_3", "Falta de información", "Mediana por edad × tipo de ingreso", "Aplicado"),
        ("EXT_SOURCE_2", "Falta de información", "Mediana global", "Aplicado"),
        ("Vivienda (47 variables)", "Falta de información (bloque)", "Sin imputar · flag_sin_info_vivienda", "Aplicado"),
        ("OCCUPATION_TYPE", "Mixto (centinela + sin explicación)", "Sin imputar", "Pendiente"),
        ("OBS/DEF_*_CNT_SOCIAL_CIRCLE, CNT_FAM_MEMBERS", "Despreciable", "Moda (CNT_FAM_MEMBERS ≥ hijos + 1)", "Aplicado"),
        ("AMT_GOODS_PRICE, AMT_ANNUITY", "Despreciable", "AMT_CREDIT × razón mediana por tipo de contrato", "Aplicado"),
        ("DAYS_LAST_PHONE_CHANGE", "Despreciable", "Mediana", "Aplicado"),
        ("NAME_TYPE_SUITE (nulo), CODE_GENDER 'XNA', NAME_FAMILY_STATUS 'Unknown'", "Despreciable / codificado", "Moda", "Aplicado"),
        ("ORGANIZATION_TYPE = 'XNA'", "Categoría real (sin empleador)", "Se conserva como categoría", "Aplicado"),
    ], columns=["Variables", "Tipo de faltante", "Tratamiento", "Estado"])
    plan["Estado"] = plan["Estado"].map({"Aplicado": "✔ Aplicado", "Pendiente": "◷ Pendiente de decisión"})
    tabla_texto(plan)
    st.caption("`OCCUPATION_TYPE` sigue sin decisión. Nota: `flag_sin_buro` y `flag_sin_previas` son el complemento exacto de "
               "`TIENE_BUREAU` y `TIENE_HISTORIAL_HOME_CREDIT`.")

# ══════════════════════════════════════════════════════════════════════════════
# 3. OUTLIERS
# ══════════════════════════════════════════════════════════════════════════════
with tab_out:
    st.markdown(
        "Un valor extremo no se corrige por ser extremo: el criterio es de **riesgo**. La corrección se hace en dos pasos y "
        "**solo con train** (partición 70/30 estratificada, semilla 42), porque usa el `TARGET`:\n\n"
        "1. **Detección.** Dónde están las colas y si su tasa de default se parece a la del tramo previo.\n"
        "2. **Reasignación por tasa de default.** El valor atípico **no se lleva al tope**, sino al **tramo cuya tasa de default "
        "es estadísticamente similar** a la de los atípicos."
    )
    with st.container(border=True):
        st.markdown(
            ":orange-badge[Corrección tras el feedback del asistente de docencia] Antes se topeaba (*winsorización*): todo lo "
            "que superaba el percentil se llevaba al tope. Eso **infla la tasa de default del tramo que recibe los atípicos**. "
            "El ejemplo es `OWN_CAR_AGE`: llevar los autos de 64–65 y 91 años (imposibles) a 30 años subía el default de los "
            "autos de 30 años de **6.0 % a 8.8 %**, una distorsión que no existe en los datos. Ahora cada grupo de atípicos va a un "
            "tramo con su misma tasa: la variable conserva la información que aporta.")

    st.subheader("Paso 1 · Detección: dónde están las colas")
    incluir = st.toggle("Incluir las variables de historial con colas largas (no se tratan)", key="cand",
                        help="BUREAU_* y HC_*: su cola tiene una tasa de default distinta a la del tramo previo; tratarla borraría señal.")
    estados = ["Tratada", "Candidata (sin tratar)"] if incluir else ["Tratada"]
    colas = t("colas").query("estado in @estados")
    st.dataframe(colas, hide_index=True, width="stretch", height=38 + 35 * len(colas),
                 column_config={"variable": st.column_config.TextColumn("Variable", width=240),
                                "estado": st.column_config.TextColumn("Estado", width=170),
                                **{c: st.column_config.NumberColumn(c, format="compact") for c in ["mediana", "p95", "p99", "p99.9", "max"]},
                                "max_sobre_p999": st.column_config.NumberColumn("Máx / p99.9", format="%.1f",
                                                                                help="Cuántas veces supera el máximo al percentil 99.9.")})
    dc = t("default_por_cola").query("estado in @estados").copy()
    dc["fiable"] = dc["n"].ge(300).map({True: "n ≥ 300", False: "n < 300 (poco fiable)"})
    dc["etiqueta"] = dc.apply(lambda r: "sin casos" if r.n == 0 else ("n < 20" if pd.isna(r.tasa) else f"{r.tasa:.1%}"), axis=1)
    dc["y_etq"] = dc["tasa"].fillna(0)
    techo = float(dc["tasa"].max()) * 1.18
    paneles = []
    for v in colas["variable"]:
        d = dc.query("variable == @v")
        barra = alt.Chart(d).mark_bar(cornerRadiusEnd=3, size=30).encode(
            x=alt.X("tramo:N", sort=["≤ p95", "p95–99", "p99–99.9", "> p99.9"], title=None, axis=alt.Axis(labelAngle=0, labelFontSize=9)),
            y=alt.Y("tasa:Q", title=None, axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, techo])),
            color=alt.Color("fiable:N", scale=alt.Scale(domain=["n ≥ 300", "n < 300 (poco fiable)"], range=[AZUL, GRIS]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("tramo:N"), alt.Tooltip("n:Q", format=","), alt.Tooltip("tasa:Q", format=".2%", title="Default")])
        texto = alt.Chart(d).mark_text(dy=-7, fontSize=10, color="#52514e").encode(
            x=alt.X("tramo:N", sort=["≤ p95", "p95–99", "p99–99.9", "> p99.9"]), y=alt.Y("y_etq:Q"), text="etiqueta:N")
        paneles.append((barra + texto + regla_base(BASE, "y")).properties(title=v, width=195, height=170))
    filas_p = [alt.hconcat(*paneles[i:i + 4]) for i in range(0, len(paneles), 4)]
    st.altair_chart(alt.vconcat(*filas_p).configure_title(fontSize=11, anchor="start"), width="content")
    st.caption(f"Tasa de default por tramo de cola en train. Línea naranja: tasa base {BASE:.1%}. En las variables de atraso "
               "(si se activa el interruptor) la cola tiene **más** default que el resto: es señal de riesgo, no error, y no se toca.")

    with st.expander("Evidencia estadística por percentil (χ², Δ logL, Δ AUC)", icon=":material/table_view:"):
        ev = t("evidencia_topes").query("estado in @estados").copy()
        ev["resultado"] = ev["p_valor"].apply(lambda p: "n/d" if pd.isna(p) else ("Homogénea (p ≥ 0.05)" if p >= 0.05 else "Difiere (p < 0.05)"))
        ev["texto"] = ev.apply(lambda r: f"n/d · n={r.n_afectados:,}" if pd.isna(r.p_valor) else
                               (f"p<0.001 · n={r.n_afectados:,}" if r.p_valor < 0.001 else f"p={r.p_valor:.3f} · n={r.n_afectados:,}"), axis=1)
        ev["corte"] = [f"cola sobre p{float(x):g}" for x in ev["percentil"]]
        base_h = alt.Chart(ev).encode(x=alt.X("corte:N", title=None, axis=alt.Axis(orient="top", labelAngle=0)),
                                      y=alt.Y("variable:N", sort=colas["variable"].tolist(), title=None, axis=alt.Axis(labelLimit=240)))
        calor = base_h.mark_rect(stroke="white", strokeWidth=2).encode(
            color=alt.Color("resultado:N", scale=alt.Scale(domain=["Homogénea (p ≥ 0.05)", "Difiere (p < 0.05)", "n/d"],
                                                           range=["#b5e4d2", "#f6c8b3", "#dcdcd8"]), legend=alt.Legend(orient="bottom", title=None)),
            tooltip=[alt.Tooltip("variable:N"), alt.Tooltip("corte:N", title="Corte"), alt.Tooltip("tope:Q", format=",.4g", title="Valor del percentil"),
                     alt.Tooltip("p_valor:Q", format=".4f", title="p-valor χ²"), alt.Tooltip("n_afectados:Q", format=",", title="Créditos en la cola")])
        st.altair_chart((calor + base_h.mark_text(fontSize=11, color="#222").encode(text="texto:N")).properties(height=34 * len(colas) + 40),
                        width="stretch")
        st.caption("Compara la tasa de default de la cola (por encima del percentil) con la del tramo inmediatamente anterior. "
                   "Es la evidencia que fijó el **umbral de detección** (p99.9) de seis variables.")

    st.subheader("Paso 2 · Reasignación por tasa de default")
    st.markdown(
        "Para cada variable, los valores **por encima del umbral** forman un grupo; se calcula su tasa de default en train y su "
        "intervalo de confianza al 95 % (Wilson). El destino es un tramo de valores normales (cada valor en las discretas, 20 "
        "cuantiles en `AMT_INCOME_TOTAL`) que cumpla tres condiciones:\n\n"
        "- su tasa de default está **dentro del intervalo** de los atípicos (no se distinguen estadísticamente);\n"
        "- al recibir a los atípicos su tasa **no se mueve más de 0.5 p.p.** (la crítica del asistente, convertida en regla);\n"
        "- entre los que cumplen, el **más cercano al umbral**, para respetar el orden de la variable."
    )
    pl = t("plan_outliers")
    dist = pd.concat([
        pl.assign(metodo="Tope anterior", antes=pl["rd_tramo_tope_antes"], despues=pl["rd_tramo_tope_despues"]),
        pl.assign(metodo="Reasignación", antes=pl["rd_destino_antes"], despues=pl["rd_destino_despues"]),
    ])
    dist["desplazamiento"] = (dist["despues"] - dist["antes"]) * 100
    dist["tramo"] = np.where(dist["metodo"] == "Tope anterior", dist["tramo_tope"], dist["tramo_destino"])
    d1, d2 = st.columns([3, 2], gap="large")
    d1.altair_chart((alt.Chart(dist).mark_bar(height={"band": 0.8}).encode(
        y=alt.Y("variable:N", sort=pl["variable"].tolist(), title=None, axis=alt.Axis(labelLimit=240)),
        yOffset=alt.YOffset("metodo:N", sort=["Tope anterior", "Reasignación"]),
        x=alt.X("desplazamiento:Q", title="Cambio en la tasa de default del tramo que recibe los atípicos (p.p.)"),
        color=alt.Color("metodo:N", scale=alt.Scale(domain=["Tope anterior", "Reasignación"], range=[NARANJA, AZUL]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("variable:N"), alt.Tooltip("metodo:N", title="Método"), alt.Tooltip("tramo:N", title="Tramo que recibe"),
                 alt.Tooltip("antes:Q", format=".2%", title="Default antes"), alt.Tooltip("despues:Q", format=".2%", title="Default después"),
                 alt.Tooltip("desplazamiento:Q", format="+.2f", title="Cambio (p.p.)")])
        + alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#2b2b29").encode(x="x:Q"))
        .properties(title="¿Cuánto se distorsiona el tramo que recibe a los atípicos?"), width="stretch", height=340)
    with d2:
        st.markdown(
            "- **`OWN_CAR_AGE`**: el tope llevaba 3,600 autos a «30 años» y subía su default **+2.8 p.p.** La reasignación los "
            "lleva a **16 años** (8.65 % de default frente a 8.61 % de los atípicos): **−0.02 p.p.**\n"
            "- **`OBS_30` / `OBS_60_CNT_SOCIAL_CIRCLE`**: el tope inflaba los tramos 17 y 16 (+2.9 y +0.4 p.p.); ahora van a 12.\n"
            "- En **ingresos, hijos, miembros de la familia y consultas** el tramo del tope ya tenía la misma tasa que los atípicos: "
            "la reasignación llega al mismo tramo y confirma que ahí el tope no distorsionaba.")
    tabla_texto(pl.assign(ic=lambda d: d.apply(lambda r: f"{r.ic_bajo:.1%} – {r.ic_alto:.1%}", axis=1))
                [["variable", "regla", "umbral", "n_atipicos_train", "rd_atipicos", "ic", "tramo_destino", "rd_destino_antes",
                  "rd_destino_despues", "valor_asignado", "p_valor"]]
                .rename(columns={"variable": "Variable", "regla": "Umbral de detección", "umbral": "Umbral", "n_atipicos_train": "Atípicos (train)",
                                 "rd_atipicos": "Default atípicos", "ic": "IC 95 %", "tramo_destino": "Tramo destino",
                                 "rd_destino_antes": "Default destino", "rd_destino_despues": "Default destino (después)",
                                 "valor_asignado": "Valor asignado", "p_valor": "p-valor χ²"}),
                {"Umbral": "{:,.0f}", "Atípicos (train)": "{:,}", "Default atípicos": "{:.2%}", "Default destino": "{:.2%}",
                 "Default destino (después)": "{:.2%}", "Valor asignado": "{:,.0f}", "p-valor χ²": "{:.3f}"})
    st.caption("p-valor χ² entre los atípicos y su tramo destino: valores altos confirman que no se distinguen. En test se aplica "
               "la misma regla con los umbrales y destinos aprendidos en train.")

    st.markdown("**Tasa de default por tramo: original, con el tope anterior y con la reasignación**")
    tr_o = t("outliers_tramos")
    var_o = st.selectbox("Variable", pl["variable"].tolist(), index=pl["variable"].tolist().index("OWN_CAR_AGE"), key="var_out")
    d = tr_o.query("variable == @var_o").copy()
    d = d[(d["n"] >= 100) | d["es_destino"] | d["es_destino_tope"]]
    fila = pl.set_index("variable").loc[var_o]
    largo = pd.concat([d.assign(serie="Original", tasa=d["rd"]), d.assign(serie="Con el tope anterior", tasa=d["rd_tope"]),
                       d.assign(serie="Con la reasignación", tasa=d["rd_reasignacion"])])
    orden_t = d["tramo"].tolist()
    xenc = alt.X("tramo:N", sort=orden_t, title=f"{var_o} (tramos con ≥ 100 casos)", axis=alt.Axis(labelAngle=-45 if len(d) > 15 else 0))
    banda = alt.Chart(pd.DataFrame({"lo": [fila.ic_bajo], "hi": [fila.ic_alto]})).mark_rect(color=VERDE, opacity=0.12).encode(
        y=alt.Y("lo:Q"), y2="hi:Q")
    lineas = alt.Chart(largo).mark_line(point=alt.OverlayMarkDef(size=36, filled=True), strokeWidth=2).encode(
        x=xenc, y=alt.Y("tasa:Q", title="Tasa de default", axis=PCT),
        color=alt.Color("serie:N", scale=alt.Scale(domain=["Original", "Con el tope anterior", "Con la reasignación"],
                                                   range=[GRIS, NARANJA, AZUL]), legend=alt.Legend(orient="top", title=None)),
        strokeDash=alt.StrokeDash("serie:N", scale=alt.Scale(domain=["Original", "Con el tope anterior", "Con la reasignación"],
                                                             range=[[2, 2], [6, 3], [1, 0]]), legend=None),
        tooltip=[alt.Tooltip("tramo:N"), alt.Tooltip("serie:N"), alt.Tooltip("tasa:Q", format=".2%", title="Default"),
                 alt.Tooltip("n:Q", format=",", title="Créditos (original)")])
    marcas = alt.Chart(d[d["es_destino"] | d["es_destino_tope"]].assign(
        rol=lambda x: np.where(x["es_destino"], "Destino de la reasignación", "Destino del tope"),
        y_marca=lambda x: np.where(x["es_destino"], x["rd_reasignacion"], x["rd_tope"]))).mark_text(dy=-16, fontSize=12, fontWeight="bold").encode(
        x=xenc, y=alt.Y("y_marca:Q"), text=alt.value("▼"),
        color=alt.Color("rol:N", scale=alt.Scale(domain=["Destino de la reasignación", "Destino del tope"], range=[AZUL, NARANJA]), legend=None))
    st.altair_chart(alt.layer(banda, lineas, marcas).resolve_scale(color="independent").properties(title={"text": f"{var_o}: tasa de default por tramo (train)",
                    "subtitle": f"Banda verde: IC 95 % de los atípicos ({fila.rd_atipicos:.2%}) · ▼ azul: destino de la reasignación · ▼ naranja: destino del tope"}),
                    width="stretch", height=360)
    st.altair_chart(alt.Chart(largo[largo["serie"] != "Original"]).mark_bar(opacity=0.85).encode(
        x=xenc, xOffset=alt.XOffset("serie:N"), y=alt.Y("n_eff:Q", title="Créditos"),
        color=alt.Color("serie:N", scale=alt.Scale(domain=["Con el tope anterior", "Con la reasignación"], range=[NARANJA, AZUL]), legend=None),
        tooltip=[alt.Tooltip("tramo:N"), alt.Tooltip("serie:N"), alt.Tooltip("n_eff:Q", format=",", title="Créditos")])
        .transform_calculate(n_eff="datum.serie == 'Con el tope anterior' ? datum.n_tope : datum.n_reasignacion"),
        width="stretch", height=170)

    st.subheader("Caso especial: la antigüedad del auto")
    oc = t("own_car_age").copy()
    oc["grupo"] = oc["bloque"].map({True: "Bloque 64–65 años", False: "Resto"})
    o1, o2 = st.columns(2, gap="large")
    orden_oc = oc["tramo"].tolist()
    col_oc = alt.Color("grupo:N", scale=alt.Scale(domain=["Resto", "Bloque 64–65 años"], range=[AZUL, NARANJA]),
                       legend=alt.Legend(orient="top", title=None))
    o1.altair_chart(alt.Chart(oc).mark_bar(size=24).encode(
        x=alt.X("tramo:N", sort=orden_oc, title="Edad del auto (años)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("n:Q", scale=alt.Scale(type="log", domainMin=1), title="Créditos (escala log)"), y2=alt.datum(1), color=col_oc,
        tooltip=[alt.Tooltip("tramo:N"), alt.Tooltip("n:Q", format=",")]).properties(height=280, title="Frecuencia por edad del auto (train)"),
        width="stretch")
    o2.altair_chart((alt.Chart(oc.dropna(subset=["tasa"])).mark_line(point=alt.OverlayMarkDef(size=60, filled=True), color="#333").encode(
        x=alt.X("tramo:N", sort=orden_oc, title="Edad del auto (años)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("tasa:Q", title="Tasa de default", axis=PCT, scale=alt.Scale(zero=True)),
        tooltip=[alt.Tooltip("tramo:N"), alt.Tooltip("tasa:Q", format=".2%", title="Default")]) + regla_base(BASE, "y"))
        .properties(height=280, title="Default por edad del auto (tramos con ≥ 100 casos)"), width="stretch")
    nota("Hay un **bloque anómalo de autos de 64–65 años** (≈ 2,300 créditos en train, más dos de 91 años) tras un tramo casi vacío "
         "entre 51 y 63: parece una **codificación de relleno**. Su default (8.6 %) se parece al de los autos de 13 a 16 años, no "
         "al de los de 30. Por eso el umbral aquí es de **negocio** (> 63 años) y el destino, **16 años**: el más cercano al "
         "umbral con la misma tasa. Los autos de 31 a 63 años, raros pero posibles, ya no se tocan.", ":material/directions_car:")

# ══════════════════════════════════════════════════════════════════════════════
# 4. RESULTADO
# ══════════════════════════════════════════════════════════════════════════════
with tab_res:
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Columnas", f"{R['columnas_post']}", f"+{R['columnas_post'] - R['columnas']} flags", border=True)
    r2.metric("Celdas nulas", f"{R['celdas_nulas_despues'] / 1e6:.2f} M", f"{R['celdas_nulas_despues'] - R['celdas_nulas_antes']:,}",
              delta_color="inverse", border=True, help="Las 148 columnas originales. Los nulos estructurales se conservan a propósito.")
    r3.metric("Nulos por crédito", f"{R['nulos_fila_mediana_post']:.1%}",
              f"{(R['nulos_fila_mediana_post'] - R['nulos_fila_mediana']) * 100:+.1f} p.p.", delta_color="inverse", border=True, help="Mediana del % de variables nulas por crédito.")
    r4.metric("Valores reasignados", f"{int(t('efecto_outliers')['n_modificados'].sum()):,}", border=True,
              help="Suma de valores atípicos llevados a su tramo destino en las 7 variables (train + test).")

    st.subheader("Validaciones del preprocesamiento")
    v = t("validaciones").copy()
    v["resultado"] = v["ok"].map({True: "✔ Supera", False: "✖ Falla"})
    st.dataframe(v[["etapa", "verificacion", "resultado"]], hide_index=True, width="stretch", height=38 + 35 * len(v),
                 column_config={"etapa": st.column_config.TextColumn("Etapa", width=120),
                                "verificacion": st.column_config.TextColumn("Verificación", width=620),
                                "resultado": st.column_config.TextColumn("Resultado", width=140)})

    st.subheader("Faltantes: antes y después")
    f1, f2 = st.columns([3, 2], gap="large")
    with f1:
        ef = t("efecto_imputacion")
        st.dataframe(ef, hide_index=True, width="stretch", height=38 + 35 * len(ef),
                     column_config={"variable": st.column_config.TextColumn("Variable", width=220),
                                    "n_imputados": st.column_config.NumberColumn("Imputados", format="localized"),
                                    "metodo": st.column_config.TextColumn("Método", width=260),
                                    **{c: st.column_config.NumberColumn(c.replace("_", " ").capitalize(), format="%.4g")
                                       for c in ["media_antes", "media_despues", "std_antes", "std_despues"]}})
        st.caption("Imputar concentra valores y **reduce la dispersión** (EXT_SOURCE_1: σ 0.211 → 0.184). En `DAYS_EMPLOYED` el "
                   "'antes' excluye el centinela.")
    with f2:
        fl = t("flags")
        st.altair_chart(alt.Chart(fl).mark_bar(color=VERDE, cornerRadiusEnd=3, height={"band": 0.6}).encode(
            y=alt.Y("flag:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False, labelLimit=220)),
            x=alt.X("pct:Q", title="% de créditos marcados", axis=PCT),
            tooltip=[alt.Tooltip("flag:N"), alt.Tooltip("n:Q", format=","), alt.Tooltip("pct:Q", format=".1%")])
            .properties(height=230, title="Indicadores creados"), width="stretch")
        st.markdown("**Categóricas imputadas con la moda**")
        tabla_texto(t("imputacion_categoricas").rename(columns={"variable": "Variable", "problema": "Problema", "casos": "Casos",
                                                               "reemplazo": "Reemplazo"}), {"Casos": "{:,}"})

    he = t("hist_ext")
    paneles = []
    for var in ["EXT_SOURCE_1", "EXT_SOURCE_3"]:
        d = he.query("variable == @var")
        paneles.append(alt.Chart(d).mark_bar(opacity=0.55, binSpacing=0).encode(
            x=alt.X("desde:Q", bin="binned", title="Valor del score"), x2="hasta:Q",
            y=alt.Y("densidad:Q", stack=None, title="Densidad"),
            color=alt.Color("serie:N", scale=alt.Scale(domain=["Original, sin nulos", "Tras imputar"], range=[AZUL, NARANJA]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("serie:N"), alt.Tooltip("desde:Q", format=".2f"), alt.Tooltip("densidad:Q", format=".2f")])
            .properties(title=f"{var}: los valores imputados forman picos", height=260, width=440))
    st.altair_chart(alt.hconcat(*paneles), width="content")
    st.caption("Cada pico es la mediana de una celda edad × ingreso. Es el costo conocido de imputar con medianas; en el bivariado "
               "conviene revisar si estos picos distorsionan los tramos (qcut) de estas variables.")

    nf = t("nulos_por_fila").melt(id_vars=["desde", "hasta"], var_name="serie", value_name="creditos")
    nf["serie"] = nf["serie"].map({"antes": "Antes", "despues": "Después"})
    st.altair_chart(alt.Chart(nf).mark_bar(opacity=0.6, binSpacing=0).encode(
        x=alt.X("desde:Q", bin="binned", title="% de variables nulas en el crédito", axis=PCT), x2="hasta:Q",
        y=alt.Y("creditos:Q", stack=None, title="Créditos"),
        color=alt.Color("serie:N", scale=alt.Scale(domain=["Antes", "Después"], range=[GRIS, AZUL]), legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("serie:N"), alt.Tooltip("desde:Q", format=".0%"), alt.Tooltip("creditos:Q", format=",")])
        .properties(height=280, title="Nulos por crédito antes y después"), width="stretch")

    st.subheader("Outliers: cola superior antes y después de la reasignación")
    cv = t("curvas_cola").query("estado == 'Tratada'").melt(id_vars=["variable", "percentil"], value_vars=["antes", "despues"],
                                                             var_name="serie", value_name="valor")
    cv["serie"] = cv["serie"].map({"antes": "Antes", "despues": "Después"})
    topes = t("efecto_outliers").set_index("variable")
    paneles = []
    for var in topes.index:
        d = cv.query("variable == @var")
        tp = topes.loc[var]
        paneles.append(alt.Chart(d).mark_line(strokeWidth=2.2).encode(
            x=alt.X("percentil:Q", title="Percentil", scale=alt.Scale(domain=[90, 100])),
            y=alt.Y("valor:Q", title=None, scale=alt.Scale(type="symlog", zero=False, nice=False,
                                                            domain=[d["valor"].min() * 0.95, d["valor"].max() * 1.05]),
                    axis=alt.Axis(values=sorted({v for v in [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4,
                                                             1e5, 2e5, 5e5, 1e6, 1e7, 1e8]
                                                 if d["valor"].min() * 0.95 <= v <= d["valor"].max() * 1.05} | {float(tp.umbral)}),
                                  labelExpr="datum.value >= 1e6 ? format(datum.value / 1e6, ',') + ' M' : "
                                            "datum.value >= 1e3 ? format(datum.value / 1e3, ',') + ' mil' : format(datum.value, ',')")),
            color=alt.Color("serie:N", scale=alt.Scale(domain=["Antes", "Después"], range=[AZUL, NARANJA]),
                            legend=alt.Legend(orient="top", title=None)),
            strokeDash=alt.StrokeDash("serie:N", scale=alt.Scale(domain=["Antes", "Después"], range=[[1, 0], [6, 4]]), legend=None),
            tooltip=[alt.Tooltip("serie:N"), alt.Tooltip("percentil:Q", format=".2f"), alt.Tooltip("valor:Q", format=",.4g")])
            .properties(title=f"{var} · > {tp.umbral:,.0f} → {tp.valor_asignado:,.0f}", width=265, height=210))
    filas_p = [alt.hconcat(*paneles[i:i + 3]) for i in range(0, len(paneles), 3)]
    st.altair_chart(alt.vconcat(*filas_p).configure_title(fontSize=11, anchor="start"), width="content")
    st.caption("Percentiles 90 a 100 en escala symlog (train). La línea discontinua (después) se separa de la continua solo en la cola "
               "reasignada: el resto de la distribución queda intacto. Título: umbral → valor asignado.")
    et = t("efecto_outliers")
    st.dataframe(et, hide_index=True, width="stretch", height=38 + 35 * len(et),
                 column_config={"variable": st.column_config.TextColumn("Variable", width=230),
                                "umbral": st.column_config.NumberColumn("Umbral", format="localized"),
                                "valor_asignado": st.column_config.NumberColumn("Valor asignado", format="localized"),
                                "n_modificados": st.column_config.NumberColumn("Modificados", format="localized"),
                                **{c: st.column_config.NumberColumn(c.replace("_", " ").capitalize(), format="compact")
                                   for c in ["max_antes", "max_despues", "media_antes", "media_despues", "std_antes", "std_despues"]},
                                "asim_antes": st.column_config.NumberColumn("Asimetría antes", format="%.2f"),
                                "asim_despues": st.column_config.NumberColumn("Asimetría después", format="%.2f")})
    ef_o = t("efecto_outliers").set_index("variable").loc["AMT_INCOME_TOTAL"]
    nota(f"El caso más claro es `AMT_INCOME_TOTAL`: con **{int(ef_o.n_modificados)}** valores reasignados (0.09 %), la desviación estándar "
         f"cae de {ef_o.std_antes / 1e3:,.0f} mil a {ef_o.std_despues / 1e3:,.0f} mil y la asimetría de {ef_o.asim_antes:.1f} a "
         f"{ef_o.asim_despues:.1f}, sin tocar al 99.9 % de los clientes.")

    st.subheader("Qué pasa después")
    st.markdown(
        "- **Redundancias** (`flag_sin_buro` vs. `TIENE_BUREAU`, versiones _AVG/_MODE/_MEDI): se resuelven en el **multivariado**.\n"
        "- **`OCCUPATION_TYPE`**: sus nulos quedan como categoría propia «Sin dato» al agrupar la variable por tasa de default "
        "(filtro univariado).\n"
        "- Los modelos de árboles son robustos a outliers y nulos: este tratamiento es deliberadamente **simple** y se concentra en "
        "no distorsionar la información, no en «limpiar» la distribución."
    )

st.divider()
st.caption("Fuente: `notebooks/01_eda_univariado.ipynb` (Pasos 2 y 3: faltantes) y `scripts/pipeline_modelado.py outliers` "
           "(reasignación de outliers sobre train). Tablas precalculadas con `scripts/build_calidad.py`.")
