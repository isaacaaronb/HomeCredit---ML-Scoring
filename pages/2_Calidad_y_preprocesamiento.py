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
BARRA_G = "#c9c8c3"
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
k3.metric("Variables imputadas", f"{R['n_imputadas']}", border=True,
          help="Faltantes despreciables (montos, círculo social, miembros, teléfono) y 3 categóricas; más el centinela de DAYS_EMPLOYED. Los scores externos no se imputan.")
k4.metric("Filas eliminadas", f"{R['filas_eliminadas']}", border=True,
          help="Valores super extremos (> 3 × p99.9) y escasos (≤ 20 créditos). El resto de las colas se capea en p0.1–p99.9.")
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
    st.caption("Tabla de colas pedida por la guía (mín, p1, p50, p99, máx), sobre el tablón original. Un outlier estadístico no es "
               "un error: el tratamiento (valores sin sentido, eliminación de lo super extremo y capeo) está en la pestaña **Outliers**.")

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
        st.markdown("**¿Qué habría hecho la imputación?** Se reconstruye la mediana por edad × tipo de ingreso (la regla que se usaba "
                    "antes) para ver a dónde llevaría a los nulos.")
        var_m = st.segmented_control("Score", ["EXT_SOURCE_1", "EXT_SOURCE_3"], default="EXT_SOURCE_1", key="ext_med") or "EXT_SOURCE_1"
        en = R["ext_nulos"][var_m]
        de = t("ext_destino").query("variable == @var_m").copy()
        de["etq"] = de.apply(lambda r: f"D{int(r.decil)}\n{r.desde:.2f}–{r.hasta:.2f}", axis=1)
        x_d = alt.X("tramo:N", sort=de["tramo"].tolist(), title="Decil del score (datos observados)", axis=alt.Axis(labelAngle=0))
        barras_d = alt.Chart(de).mark_bar(color=BARRA_G, width={"band": 0.75}).encode(
            x=x_d, y=alt.Y("pct_nulos_imputados:Q", title="% de los nulos que caerían ahí", axis=PCT, scale=alt.Scale(domain=[0, 0.5])),
            tooltip=[alt.Tooltip("tramo:N", title="Decil"), alt.Tooltip("desde:Q", format=".3f"), alt.Tooltip("hasta:Q", format=".3f"),
                     alt.Tooltip("pct_nulos_imputados:Q", format=".1%", title="% de nulos imputados"),
                     alt.Tooltip("rd_decil:Q", format=".2%", title="Default del decil")])
        linea_d = alt.Chart(de).mark_line(color=AZUL, strokeWidth=2.5, point=alt.OverlayMarkDef(color=AZUL, size=60, filled=True)).encode(
            x=x_d, y=alt.Y("rd_decil:Q", title="Tasa de default del decil", axis=alt.Axis(format="%", titleColor=AZUL),
                           scale=alt.Scale(domain=[0, max(de["rd_decil"].max(), en["rd_nulos"]) * 1.15])))
        regla_n = alt.Chart(pd.DataFrame({"y": [en["rd_nulos"]]})).mark_rule(color=NARANJA, strokeDash=[6, 4], strokeWidth=2).encode(
            y=alt.Y("y:Q", scale=alt.Scale(domain=[0, max(de["rd_decil"].max(), en["rd_nulos"]) * 1.15])))
        g1, g2 = st.columns([3, 2], gap="large")
        g1.altair_chart(alt.layer(barras_d, alt.layer(linea_d, regla_n)).resolve_scale(y="independent").properties(
            title={"text": f"{var_m}: a qué decil llevaría la imputación a los nulos",
                   "subtitle": f"Barras: % de los nulos · línea azul: default del decil · punteada naranja: default de los nulos ({en['rd_nulos']:.1%})"}),
            width="stretch", height=330)
        with g2:
            st.markdown(
                f"Los **{en['n_nulos']:,}** nulos de `{var_m}` ({en['pct_nulos']:.0%} de la base) tienen **{en['rd_nulos']:.1%}** de default, "
                f"más que quienes tienen dato ({en['rd_con_dato']:.1%}). La imputación los repartía en deciles cuyo default promedio es "
                f"**{en['rd_destino_ponderada']:.1%}**: deciles **sanos**, que al recibirlos verían inflada su tasa de default. Además, "
                "la mediana por celda concentra miles de créditos en unos pocos valores y crea **picos** (gráfico de abajo).")
            st.success("**Decisión (03-10-2026):** **no imputar** los tres scores externos. El nulo se conserva y el *binning* "
                       "(OptBinning) lo trata como un **tramo propio**, con su propio WoE. En la versión del dataset que va a SMOTE "
                       "(que no acepta nulos) se imputa la mediana de train **junto con un indicador de nulo**, que preserva la señal.",
                       icon=":material/check_circle:")
            st.caption(f"`EXT_SOURCE_2` tiene solo {R['ext2_nulos']['n']:,} nulos (default {R['ext2_nulos']['rd']:.1%}, casi el promedio): "
                       "el efecto era mínimo, pero se aplica la misma regla a los tres scores por coherencia.")
        he = t("hist_ext").query("variable == @var_m")
        st.markdown(f"**{var_m}: la imputación por mediana crearía picos** · misma escala en ambos paneles")
        ymax = float(he["densidad"].max()) * 1.05
        for serie, color in (("Original, sin nulos", AZUL), ("Si se imputara", NARANJA)):
            hs = he[he["serie"] == serie]
            st.altair_chart(alt.Chart(hs).mark_bar(binSpacing=0, color=color).encode(
                x=alt.X("desde:Q", bin="binned", title="Valor del score" if serie == "Si se imputara" else None,
                        scale=alt.Scale(domain=[0, 1])), x2="hasta:Q",
                y=alt.Y("densidad:Q", title="Densidad", scale=alt.Scale(domain=[0, ymax])),
                tooltip=[alt.Tooltip("desde:Q", format=".2f", title="Desde"), alt.Tooltip("hasta:Q", format=".2f", title="Hasta"),
                         alt.Tooltip("densidad:Q", format=".2f", title="Densidad")])
                .properties(title=serie, height=200), width="stretch")

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
        ("EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3", "Falta de información", "Sin imputar: el nulo es un tramo propio en el WoE", "Aplicado (03-10)"),
        ("Vivienda (47 variables)", "Falta de información (bloque)", "Sin imputar · flag_sin_info_vivienda", "Aplicado"),
        ("OCCUPATION_TYPE", "Mixto (centinela + sin explicación)", "Sin imputar", "Pendiente"),
        ("OBS/DEF_*_CNT_SOCIAL_CIRCLE, CNT_FAM_MEMBERS", "Despreciable", "Moda (CNT_FAM_MEMBERS ≥ hijos + 1)", "Aplicado"),
        ("AMT_GOODS_PRICE, AMT_ANNUITY", "Despreciable", "AMT_CREDIT × razón mediana por tipo de contrato", "Aplicado"),
        ("DAYS_LAST_PHONE_CHANGE", "Despreciable", "Mediana", "Aplicado"),
        ("NAME_TYPE_SUITE (nulo), CODE_GENDER 'XNA', NAME_FAMILY_STATUS 'Unknown'", "Despreciable / codificado", "Moda", "Aplicado"),
        ("ORGANIZATION_TYPE = 'XNA'", "Categoría real (sin empleador)", "Se conserva como categoría", "Aplicado"),
    ], columns=["Variables", "Tipo de faltante", "Tratamiento", "Estado"])
    plan["Estado"] = plan["Estado"].map({"Aplicado": "✔ Aplicado", "Aplicado (03-10)": "✔ Aplicado (cambio del 03-10)",
                                         "Pendiente": "◷ Pendiente de decisión"})
    tabla_texto(plan)
    st.caption("`OCCUPATION_TYPE` sigue sin decisión. Nota: `flag_sin_buro` y `flag_sin_previas` son el complemento exacto de "
               "`TIENE_BUREAU` y `TIENE_HISTORIAL_HOME_CREDIT`.")

# ══════════════════════════════════════════════════════════════════════════════
# 3. OUTLIERS
# ══════════════════════════════════════════════════════════════════════════════
with tab_out:
    rel = t("outliers_relleno").iloc[0]
    eli = t("outliers_eliminacion")
    cap = t("outliers_capeo")
    st.markdown(
        "El tratamiento es deliberadamente **simple**: los modelos que vienen (regresión logística sobre WoE, árboles, Random "
        "Forest, XGBoost) son robustos a los valores extremos, así que el objetivo es quitar lo que **no tiene sentido** y acotar "
        "las colas sin borrar información. Son tres pasos, en este orden, y los límites se aprenden **solo con train** "
        "(partición 80/20 estratificada, semilla 42):\n\n"
        "1. **Valores sin sentido de negocio → nulo.** El bloque de autos de 64–65 años.\n"
        "2. **Valores super extremos y escasos → se elimina la fila.** Más de 3 veces el percentil 99.9 y, como mucho, 20 créditos en la base.\n"
        "3. **Capeo de las dos colas** de cada numérica en los percentiles **0.1 y 99.9** de train."
    )
    with st.container(border=True):
        st.markdown(
            ":orange-badge[Cambio tras la conversación con la profesora] Antes cada grupo de atípicos se **reasignaba** al tramo con "
            "tasa de default similar. La observación fue que el boxplot del univariado seguía mostrando puntos aunque recibe el "
            "tablón tratado. La indicación: no complicarse, **eliminar** los máximos que son muy pocas observaciones, **capear** lo "
            "demás y cuidar los valores que carecen de sentido. Más abajo está la respuesta a por qué el capeo, por sí solo, no "
            "hacía desaparecer esos puntos.")
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("OWN_CAR_AGE → nulo", f"{int(rel.n_total):,}", border=True, help="Créditos con auto de 64 o 65 años (código de relleno).")
    o2.metric("Filas eliminadas", f"{int(R['filas_eliminadas'])}", f"{R['filas_eliminadas'] / N:.3%} de la base", delta_color="off",
              border=True, help="Valores > 3 × p99.9 en una variable, cuando son ≤ 20 créditos.")
    o3.metric("Valores capeados", f"{int(cap['n_modificados'].sum()):,}", border=True, help="Suma en las 93 numéricas (train + test).")
    o4.metric("Variables con capeo efectivo", f"{int((cap['n_modificados'] > 0).sum())} de {len(cap)}", border=True,
              help="En las acotadas (vivienda en [0, 1], EXT_SOURCE…) el p99.9 coincide con el máximo y el capeo no cambia nada.")

    st.subheader("Paso 1 · Valores sin sentido: el bloque de 64–65 años en OWN_CAR_AGE")
    oc = t("own_car_age").copy()
    oc["grupo"] = oc["bloque"].map({True: "Bloque 64–65 años", False: "Resto"})
    a1, a2 = st.columns(2, gap="large")
    orden_oc = oc["tramo"].tolist()
    col_oc = alt.Color("grupo:N", scale=alt.Scale(domain=["Resto", "Bloque 64–65 años"], range=[AZUL, NARANJA]),
                       legend=alt.Legend(orient="top", title=None))
    a1.altair_chart(alt.Chart(oc).mark_bar(size=24).encode(
        x=alt.X("tramo:N", sort=orden_oc, title="Edad del auto (años)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("n:Q", scale=alt.Scale(type="log", domainMin=1), title="Créditos (escala log)"), y2=alt.datum(1), color=col_oc,
        tooltip=[alt.Tooltip("tramo:N"), alt.Tooltip("n:Q", format=",")]).properties(height=280, title="Frecuencia por edad del auto (train, antes del tratamiento)"),
        width="stretch")
    a2.altair_chart((alt.Chart(oc.dropna(subset=["tasa"])).mark_line(point=alt.OverlayMarkDef(size=60, filled=True), color="#333").encode(
        x=alt.X("tramo:N", sort=orden_oc, title="Edad del auto (años)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("tasa:Q", title="Tasa de default", axis=PCT, scale=alt.Scale(zero=True)),
        tooltip=[alt.Tooltip("tramo:N"), alt.Tooltip("tasa:Q", format=".2%", title="Default")]) + regla_base(BASE, "y"))
        .properties(height=280, title="Default por edad del auto (tramos con ≥ 100 casos)"), width="stretch")
    nota(f"Tras un tramo casi vacío entre 51 y 63 años aparece un **bloque de {int(rel.n_total):,} créditos con autos de 64–65 años**. "
         "Es un patrón de **código de relleno**, no de autos reales. Son el 1.1 % de la base, así que no son «pocas observaciones» "
         "para eliminarlas, y capear no los toca (el p99 ya es 64). Se tratan como **antigüedad desconocida (nulo)**: se conservan las "
         f"filas y `FLAG_OWN_CAR` sigue indicando que tienen auto. Su default ({rel.rd_train:.1%}) no es muy distinto del resto de "
         f"dueños de auto ({rel.rd_resto_con_auto:.1%}).", ":material/directions_car:")

    st.subheader("Paso 2 · Valores super extremos y escasos: se elimina la fila")
    st.markdown(
        "Regla: un valor es **super extremo** si supera **3 veces el percentil 99.9 de train** (o queda por debajo de 3 veces el "
        "percentil 0.1, si la variable es negativa). Si en toda la base son **20 créditos o menos**, carecen de sentido estadístico "
        "y se elimina la fila; si son más, el valor no es una rareza aislada y pasa al capeo del paso 3.")
    def corto(v: float) -> str:
        return f"{v / 1e6:,.1f} M" if abs(v) >= 1e6 else f"{v:,.0f}" if abs(v) >= 100 or float(v).is_integer() else f"{v:.3g}"
    tabla_texto(eli.assign(umbral_txt=eli.apply(lambda r: f"{'más de' if r.lado == 'superior' else 'menos de'} {corto(r.umbral)}", axis=1))
                [["variable", "umbral_txt", "n", "valores", "rd", "decision"]]
                .rename(columns={"variable": "Variable", "umbral_txt": "Umbral (3 × p99.9)", "n": "Créditos", "valores": "Valores",
                                 "rd": "Default", "decision": "Decisión"}), {"Créditos": "{:,}", "Default": "{:.1%}"})
    fe_ = t("outliers_filas_eliminadas")
    st.caption(f"En total se eliminan **{len(fe_)} filas** ({int((fe_['muestra'] == 'train').sum())} de train y "
               f"{int((fe_['muestra'] == 'test').sum())} de test; un mismo crédito puede ser extremo en varias variables). Se quitan "
               "también de test porque un ingreso de 117 millones o 19 hijos no es un caso válido para evaluar el modelo.")

    st.subheader("Paso 3 · Capeo en los percentiles 0.1 y 99.9 de train")
    st.markdown(
        "Cada numérica se acota por las dos colas: lo que queda por debajo del percentil 0.1 de train toma ese valor, y lo que "
        "supera el percentil 99.9, también. Los límites se aplican igual a test.")
    cm = cap[cap["n_modificados"] > 0].sort_values("n_modificados", ascending=False)
    st.dataframe(cm[["variable", "lim_inf", "lim_sup", "n_inf", "n_sup", "max_antes", "max_despues", "asim_antes", "asim_despues"]],
                 hide_index=True, width="stretch", height=360,
                 column_config={"variable": st.column_config.TextColumn("Variable", pinned=True, width=230),
                                "lim_inf": st.column_config.NumberColumn("Límite inferior (p0.1)", format="compact"),
                                "lim_sup": st.column_config.NumberColumn("Límite superior (p99.9)", format="compact"),
                                "n_inf": st.column_config.NumberColumn("Capeados abajo", format="localized"),
                                "n_sup": st.column_config.NumberColumn("Capeados arriba", format="localized"),
                                "max_antes": st.column_config.NumberColumn("Máx. antes", format="compact"),
                                "max_despues": st.column_config.NumberColumn("Máx. después", format="compact"),
                                "asim_antes": st.column_config.NumberColumn("Asimetría antes", format="%.2f"),
                                "asim_despues": st.column_config.NumberColumn("Asimetría después", format="%.2f")})
    st.caption(f"{len(cm)} de {len(cap)} numéricas cambian. «Antes» es el tablón tras los pasos 1 y 2; las cifras cubren train y test.")

    st.markdown("#### ¿p99 o p99.9? ¿Cambia mucho?")
    c1, c2 = st.columns(2, gap="large")
    comp_iv = pd.concat([cap.assign(capeo="p0.1–p99.9 (elegido)", iv_cap=cap["iv_p999"]),
                         cap.assign(capeo="p1–p99", iv_cap=cap["iv_p99"])])
    lim_iv = float(cap["iv_sin_capeo"].max()) * 1.05
    c1.altair_chart((alt.Chart(comp_iv).mark_point(filled=True, size=55, opacity=0.75).encode(
        x=alt.X("iv_sin_capeo:Q", title="IV sin capeo", scale=alt.Scale(domain=[0, lim_iv])),
        y=alt.Y("iv_cap:Q", title="IV con capeo", scale=alt.Scale(domain=[0, lim_iv])),
        color=alt.Color("capeo:N", scale=alt.Scale(domain=["p0.1–p99.9 (elegido)", "p1–p99"], range=[AZUL, NARANJA]),
                        legend=alt.Legend(orient="top", title=None)),
        shape=alt.Shape("capeo:N", scale=alt.Scale(domain=["p0.1–p99.9 (elegido)", "p1–p99"], range=["circle", "cross"]), legend=None),
        tooltip=[alt.Tooltip("variable:N"), alt.Tooltip("capeo:N"), alt.Tooltip("iv_sin_capeo:Q", format=".4f", title="IV sin capeo"),
                 alt.Tooltip("iv_cap:Q", format=".4f", title="IV con capeo")])
        + alt.Chart(pd.DataFrame({"x": [0, lim_iv], "y": [0, lim_iv]})).mark_line(color=GRIS, strokeDash=[4, 3]).encode(x="x:Q", y="y:Q"))
        .properties(title={"text": "IV de las 93 numéricas: sin capeo vs con capeo", "subtitle": "Todos los puntos caen sobre la diagonal"}),
        width="stretch", height=420)
    top_mod = cap.nlargest(15, "n_mod_p99_train")
    mod = pd.concat([top_mod.assign(capeo="p0.1–p99.9 (elegido)", n=top_mod["n_mod_train"]),
                     top_mod.assign(capeo="p1–p99", n=top_mod["n_mod_p99_train"])])
    c2.altair_chart(alt.Chart(mod).mark_bar(height={"band": 0.8}).encode(
        y=alt.Y("variable:N", sort=top_mod["variable"].tolist(), title=None, axis=alt.Axis(labelLimit=230, labelFontSize=10, labelOverlap=False)),
        yOffset=alt.YOffset("capeo:N", sort=["p0.1–p99.9 (elegido)", "p1–p99"]),
        x=alt.X("n:Q", title="Valores modificados en train"),
        color=alt.Color("capeo:N", scale=alt.Scale(domain=["p0.1–p99.9 (elegido)", "p1–p99"], range=[AZUL, NARANJA]), legend=None),
        tooltip=[alt.Tooltip("variable:N"), alt.Tooltip("capeo:N"), alt.Tooltip("n:Q", format=",", title="Modificados")])
        .properties(title={"text": "Cuántos datos se alteran", "subtitle": "Las 15 variables más afectadas"}), width="stretch", height=420)
    d_iv = float(max((cap["iv_p999"] - cap["iv_sin_capeo"]).abs().max(), (cap["iv_p99"] - cap["iv_sin_capeo"]).abs().max()))
    nota(f"**El IV no cambia en ninguna de las 93 variables** (diferencia máxima: {d_iv:.4f}), capeando al p99 o al p99.9. La razón: "
         "OptBinning arma tramos de al menos 5 % de los datos, así que la cola completa ya cae dentro del último tramo y su WoE es "
         f"el mismo. Lo que sí cambia es cuánto se altera la data: con el p99 se tocan **{int(cap['n_mod_p99_train'].sum()):,}** valores "
         f"en train; con el p99.9, **{int(cap['n_mod_train'].sum()):,}**. Sin ganancia en poder predictivo, se elige la intervención "
         "mínima: **p99.9**.", ":material/balance:")

    st.markdown("#### ¿Por qué el boxplot seguía mostrando puntos?")
    tk = cap[cap["tukey_antes"] > 0].nlargest(12, "tukey_antes")
    tkl = pd.concat([tk.assign(serie="Sin capeo", n=tk["tukey_antes"]), tk.assign(serie="Capeo p0.1–p99.9", n=tk["tukey_p999"]),
                     tk.assign(serie="Capeo p1–p99", n=tk["tukey_p99"])])
    b1, b2 = st.columns([3, 2], gap="large")
    b1.altair_chart(alt.Chart(tkl).mark_bar(height={"band": 0.8}).encode(
        y=alt.Y("variable:N", sort=tk["variable"].tolist(), title=None, axis=alt.Axis(labelLimit=230, labelFontSize=10, labelOverlap=False)),
        yOffset=alt.YOffset("serie:N", sort=["Sin capeo", "Capeo p0.1–p99.9", "Capeo p1–p99"]),
        x=alt.X("n:Q", title="Puntos fuera de los bigotes de Tukey (1.5·IQR), train"),
        color=alt.Color("serie:N", scale=alt.Scale(domain=["Sin capeo", "Capeo p0.1–p99.9", "Capeo p1–p99"], range=[GRIS, AZUL, NARANJA]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("variable:N"), alt.Tooltip("serie:N"), alt.Tooltip("n:Q", format=",", title="Puntos")])
        .properties(title="Capear no mueve los bigotes de Tukey"), width="stretch", height=420)
    b2.markdown(
        "El boxplot clásico dibuja los bigotes a **1.5 veces el rango intercuartil** (Q3 + 1.5·IQR) y marca como punto todo lo que "
        "queda afuera. En una variable asimétrica ese bigote está **muy por debajo del p99**: en `AMT_INCOME_TOTAL` llega a unos 337 "
        "mil, mientras que el p99.9 es 900 mil. Por eso el capeo **no cambia el número de puntos**: los valores capeados se apilan en "
        "el límite, pero siguen fuera del bigote.\n\n"
        "**Convención adoptada en el univariado:** los bigotes llegan a los **percentiles del capeo** (0.1 y 99.9). En la pasada "
        "*ex-post* ya no queda ningún punto fuera; en la *ex-ante* los puntos son justo los valores que se capearon o eliminaron. "
        "El conteo de Tukey se sigue mostrando como dato: describe la forma de la distribución, no errores.")

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
    r4.metric("Filas", f"{R['filas_post']:,}", f"−{R['filas_eliminadas']} eliminadas", delta_color="off", border=True,
              help="Créditos que quedan tras eliminar los valores super extremos (train + test).")

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
        st.caption("Solo se imputan faltantes despreciables (< 0.5 %), por eso medias y dispersiones casi no cambian. En "
                   "`DAYS_EMPLOYED` el 'antes' excluye el centinela. Los scores externos conservan sus nulos.")
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

    nf = t("nulos_por_fila").melt(id_vars=["desde", "hasta"], var_name="serie", value_name="creditos")
    nf["serie"] = nf["serie"].map({"antes": "Antes", "despues": "Después"})
    st.altair_chart(alt.Chart(nf).mark_bar(opacity=0.6, binSpacing=0).encode(
        x=alt.X("desde:Q", bin="binned", title="% de variables nulas en el crédito", axis=PCT), x2="hasta:Q",
        y=alt.Y("creditos:Q", stack=None, title="Créditos"),
        color=alt.Color("serie:N", scale=alt.Scale(domain=["Antes", "Después"], range=[GRIS, AZUL]), legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("serie:N"), alt.Tooltip("desde:Q", format=".0%"), alt.Tooltip("creditos:Q", format=",")])
        .properties(height=280, title="Nulos por crédito antes y después"), width="stretch")

    st.subheader("Outliers: cola superior antes y después del capeo")
    cv = t("curvas_cola").melt(id_vars=["variable", "percentil", "lim_sup"], value_vars=["antes", "despues"],
                               var_name="serie", value_name="valor")
    cv["serie"] = cv["serie"].map({"antes": "Antes", "despues": "Después"})
    paneles = []
    for var in cv["variable"].unique():
        d = cv.query("variable == @var")
        lim = float(d["lim_sup"].iloc[0])
        v_min, v_max = float(d["valor"].min()), float(d["valor"].max())
        paneles.append(alt.Chart(d).mark_line(strokeWidth=2.2).encode(
            x=alt.X("percentil:Q", title="Percentil", scale=alt.Scale(domain=[90, 100])),
            y=alt.Y("valor:Q", title=None, scale=alt.Scale(type="symlog", zero=False, nice=False, domain=[v_min * 0.95, v_max * 1.05]),
                    axis=alt.Axis(values=sorted({v for v in [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4,
                                                             1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8]
                                                 if v_min * 0.95 <= v <= v_max * 1.05} | {float(f"{lim:.3g}")}),
                                  labelExpr="datum.value >= 1e6 ? format(datum.value / 1e6, ',') + ' M' : "
                                            "datum.value >= 1e3 ? format(datum.value / 1e3, ',') + ' mil' : format(datum.value, ',')")),
            color=alt.Color("serie:N", scale=alt.Scale(domain=["Antes", "Después"], range=[AZUL, NARANJA]),
                            legend=alt.Legend(orient="top", title=None)),
            strokeDash=alt.StrokeDash("serie:N", scale=alt.Scale(domain=["Antes", "Después"], range=[[1, 0], [6, 4]]), legend=None),
            tooltip=[alt.Tooltip("serie:N"), alt.Tooltip("percentil:Q", format=".2f"), alt.Tooltip("valor:Q", format=",.4g")])
            .properties(title=f"{var} · capeo en {lim / 1e6:,.2f} M" if lim >= 1e6 else f"{var} · capeo en {lim:,.4g}", width=265, height=200))
    filas_p = [alt.hconcat(*paneles[i:i + 3]) for i in range(0, len(paneles), 3)]
    st.altair_chart(alt.vconcat(*filas_p).configure_title(fontSize=11, anchor="start"), width="content")
    st.caption("Percentiles 90 a 100 en escala symlog (train). «Antes» es el tablón imputado; la línea discontinua (después) se "
               "separa solo en la última milésima: el resto de la distribución queda intacto. En `OWN_CAR_AGE` la curva cambia más "
               "porque el bloque de 64–65 años pasó a nulo.")
    ci = cap.set_index("variable").loc["AMT_INCOME_TOTAL"]
    io, it_ = R["ingreso_original"], R["ingreso_tratado"]
    nota(f"El caso más claro es `AMT_INCOME_TOTAL`: entre el tablón imputado y el tratado, la desviación estándar baja de "
         f"{io['std'] / 1e3:,.0f} mil a {it_['std'] / 1e3:,.0f} mil y la asimetría de {io['asim']:.0f} a {it_['asim']:.1f}. Casi todo el "
         f"efecto viene de **eliminar 18 ingresos absurdos** (117 M, 18 M…); el capeo posterior en {ci.lim_sup / 1e3:,.0f} mil toca a "
         f"{int(ci.n_modificados):,} créditos ({ci.n_modificados / R['filas_post']:.2%}).")

    st.subheader("Qué pasa después")
    st.markdown(
        "- **Redundancias** (`flag_sin_buro` vs. `TIENE_BUREAU`, versiones _AVG/_MODE/_MEDI): se resuelven en el **multivariado**.\n"
        "- **`OCCUPATION_TYPE`**: sus nulos quedan como categoría propia «Sin dato» al agrupar la variable por tasa de default "
        "(filtro univariado).\n"
        "- Los modelos de árboles son robustos a outliers y nulos: este tratamiento es deliberadamente **simple**. Quita lo que no "
        "tiene sentido y acota las colas sin cambiar el poder predictivo (el IV no se mueve)."
    )

st.divider()
st.caption("Fuente: `notebooks/01_eda_univariado.ipynb` (Pasos 2 y 3: faltantes) y `scripts/pipeline_modelado.py outliers` "
           "(valores sin sentido, eliminación y capeo, con límites aprendidos en train). Tablas precalculadas con `scripts/build_calidad.py`.")
