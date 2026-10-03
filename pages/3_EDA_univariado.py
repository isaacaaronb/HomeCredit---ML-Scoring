"""EDA univariado (Paso 4 de la guía): distribución, forma, colas y outliers de cada variable, por tipo."""
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

REPO_DIR = Path(__file__).resolve().parents[1]
UNI = REPO_DIR / "artifacts" / "univariado"

AZUL, NARANJA, VERDE, GRIS, TINTA = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85", "#2b2b29"
COLOR_TIPO = alt.Scale(domain=["numérica", "categórica", "dicotómica"], range=[AZUL, NARANJA, VERDE])
PCT = alt.Axis(format="%")
PASADAS = {"Ex-post · tras el preprocesamiento": "ex-post", "Ex-ante · tablón original": "ex-ante"}


@st.cache_data(show_spinner=False)
def _leer(ruta: str, version: float) -> pd.DataFrame:
    return pd.read_parquet(ruta)


def t(nombre: str) -> pd.DataFrame:
    """Lee un artefacto; la fecha de modificación entra en la llave del caché para que un redeploy no sirva datos viejos."""
    ruta = UNI / f"{nombre}.parquet"
    return _leer(str(ruta), ruta.stat().st_mtime)


@st.cache_data(show_spinner=False)
def _resumen(ruta: str, version: float) -> dict:
    return json.load(open(ruta, encoding="utf-8"))


def resumen() -> dict:
    ruta = UNI / "resumen.json"
    return _resumen(str(ruta), ruta.stat().st_mtime)


def nota(texto: str, icono: str = ":material/insights:") -> None:
    st.info(texto, icon=icono)


def tabla_texto(df: pd.DataFrame, formatos: dict | None = None) -> None:
    st.table(df.reset_index(drop=True).style.hide(axis="index").format(formatos or {}, na_rep="—"))


def fnum(v: float) -> str:
    """Número legible para textos y tooltips (miles con coma, decimales con sentido)."""
    if pd.isna(v):
        return "—"
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:,.2f} M"
    if a >= 1000 or float(v).is_integer():
        return f"{v:,.0f}"
    return f"{v:,.3g}" if a >= 0.01 else f"{v:.2e}"


def fam_corta(f: str) -> str:
    return f.split("·", 1)[-1].strip() if isinstance(f, str) else ""


# Lectura de negocio: qué se espera de cada variable (sirve para decir si la forma observada es "consistente")
ESPERADO = {
    "AMT_INCOME_TOTAL": "Ingresos: se espera sesgo a la derecha (muchos ingresos medios, pocos muy altos), típico de una lognormal. "
                        "Los ingresos absurdos (117 M, 18 M…: 18 créditos sobre 2.7 M) se eliminaron y el resto se capeó en 900 mil (p99.9 de train); la cola que queda es real.",
    "AMT_CREDIT": "Monto del crédito: sesgo a la derecha y picos en montos redondos (450 mil, 675 mil…), propios de productos "
                  "estandarizados. Consistente con lo esperado.",
    "AMT_ANNUITY": "Cuota: sigue al monto del crédito, con sesgo a la derecha. Los picos son cuotas de productos estándar.",
    "AMT_GOODS_PRICE": "Precio del bien: casi idéntico al monto del crédito (ρ de Spearman ≈ 0.98 en calidad); anticipa redundancia.",
    "REGION_POPULATION_RELATIVE": "Densidad poblacional normalizada de la región: solo 81 valores distintos (uno por región) y una "
                                  "región muy densa en la cola. Más que continua, se comporta como un código regional.",
    "DAYS_BIRTH": "Edad en días negativos (20 a 69 años). Distribución casi plana: no hay un grupo de edad dominante. Es la única "
                  "variable cercana a la simetría.",
    "DAYS_EMPLOYED": "Antigüedad laboral: la mayoría tiene pocos años en su empleo actual (sesgo hacia 0). Ex-post, el 18 % vale "
                     "exactamente 0 por el reemplazo del centinela (pensionistas y sin empleador).",
    "DAYS_REGISTRATION": "Antigüedad del registro del domicilio: cola larga hacia registros antiguos. Esperable.",
    "DAYS_ID_PUBLISH": "Antigüedad del documento de identidad: forma irregular porque la renovación sigue reglas por edad.",
    "OWN_CAR_AGE": "Edad del auto (solo quienes tienen auto): autos mayormente nuevos o de pocos años. El bloque anómalo de 64–65 "
                   "años (código de relleno) pasó a nulo y la cola se capeó en 42 años (p99.9 de train); ya no hay picos artificiales.",
    "CNT_CHILDREN": "Hijos: conteo con la mayoría en 0; la binomial negativa captura la sobredispersión frente a Poisson.",
    "CNT_FAM_MEMBERS": "Miembros de la familia: conteo con moda en 2 (pareja). Muy concentrado; poca dispersión.",
    "HOUR_APPR_PROCESS_START": "Hora de la solicitud: forma de campana alrededor del mediodía, horario comercial. Es operativa, no "
                               "un atributo del cliente: se espera poca relación con el riesgo.",
    "EXT_SOURCE_1": "Score externo 1: acotado en [0, 1] y sesgado a la izquierda (más clientes con score alto). Ex-post muestra picos "
                    "por la imputación con medianas por grupo.",
    "EXT_SOURCE_2": "Score externo 2: sesgado a la izquierda, casi sin nulos; forma consistente con un score de riesgo.",
    "EXT_SOURCE_3": "Score externo 3: sesgado a la izquierda; ex-post con picos de imputación.",
    "OBS_30_CNT_SOCIAL_CIRCLE": "Conteo del círculo social con atraso de 30 días: más de la mitad en 0 y cola larga.",
    "DEF_30_CNT_SOCIAL_CIRCLE": "Incumplimientos en el círculo social: casi todos 0 (89 %). Señal escasa pero potencialmente fuerte.",
    "DAYS_LAST_PHONE_CHANGE": "Días desde el último cambio de teléfono: el 12 % lo cambió el mismo día de la solicitud (valor 0).",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "Consultas al buró en el último año: conteo con sobredispersión; más consultas → más búsqueda de crédito.",
    "BUREAU_DEUDA_TOTAL": "Deuda total en buró: cola extremadamente larga (asimetría ~39) y un 27 % en 0. Es la variable más asimétrica "
                          "tras el tratamiento; no se trató para no borrar señal.",
    "BUREAU_MAX_DIAS_ATRASO": "Peor atraso en buró: 98.7 % en 0. Casi binaria (tuvo atraso o no) con una cola de miles de días.",
    "HC_POS_MAX_ATRASO": "Peor atraso en POS: 81 % en 0 y cola larga. Masa en cero: ninguna familia continua ajusta bien.",
    "HC_CARD_MAX_UTILIZACION": "Utilización máxima de tarjeta: bimodal (31 % en 0 y una masa alrededor de 100 %). Dos poblaciones: "
                               "tarjeta sin uso y tarjeta al tope.",
    "HC_N_REGISTROS_PAGO": "Registros de cuotas pagadas: sesgo a la derecha, bien descrito por una lognormal.",
    "HC_DIAS_ULTIMA_DECISION": "Días desde la última decisión en Home Credit: la mayoría es reciente; cola larga hacia atrás.",
}


R = resumen()
tipos = t("tipos")
N = R["filas"]

# ── Encabezado ─────────────────────────────────────────────────────────────────
st.title("EDA univariado")
st.markdown(
    "Antes de relacionar variables con el `TARGET` hay que entender **cada variable por separado**: su forma, sus colas, sus "
    "outliers y sus valores dominantes. Esta página sigue el Paso 4 de la guía (*Análisis Exploratorio — EDA univariado*): en las "
    "**numéricas** se revisan distribución, asimetría, rango y outliers; en las **categóricas**, cardinalidad, dominancia y "
    "categorías de baja frecuencia. Las **dicotómicas** se analizan aparte por su balance de clases."
)
c_p, c_i = st.columns([2, 3], vertical_alignment="center")
etiqueta = c_p.segmented_control("Pasada", list(PASADAS), default=list(PASADAS)[0], key="pasada",
                                 help="La guía propone dos pasadas: ex-ante (el dato tal como viene) y ex-post (después del "
                                      "preprocesamiento, para verificar que las transformaciones funcionaron).")
P = PASADAS[etiqueta or list(PASADAS)[0]]
c_i.caption("**Ex-post** (por defecto) es la base con la que se modelará. **Ex-ante** muestra el dato de origen, antes de imputar y "
            "capear. La clasificación de tipos se fija con el tablón original (regla del notebook: numérica con ≤ 5 valores → "
            "categórica) para que el tratamiento no cambie el tipo de una variable.")

num = t("num_metricas").query("pasada == @P").merge(tipos[["variable", "familia", "orden", "descripcion"]], on="variable")
_post = t("num_metricas").query("pasada == 'ex-post'")
VAR_ASIM_POST = _post.loc[_post["asimetria"].abs().idxmax(), "variable"]
cat = t("cat_metricas").query("pasada == @P").merge(tipos[["variable", "familia", "orden"]], on="variable")
dic = t("dic_metricas").query("pasada == @P").merge(tipos[["variable", "familia", "orden"]], on="variable")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Numéricas", R["n_numericas"], border=True)
k2.metric("Categóricas", R["n_categoricas"], border=True)
k3.metric("Dicotómicas", len(dic), border=True, help="Incluye los 4 flags creados en el preprocesamiento (solo ex-post).")
k4.metric("Tasa de default", f"{R['tasa_default']:.2%}", border=True, help=f"{R['target_1']:,} de {N:,} créditos.")
k5.metric("Casi constantes", int(dic["casi_constante"].sum()), border=True, help="Dicotómicas con clase minoritaria < 1 %.")

tab_pan, tab_num, tab_cat, tab_dic, tab_hal, tab_fil = st.tabs([":material/dashboard: Panorama", ":material/show_chart: Numéricas",
                                                                ":material/category: Categóricas", ":material/toggle_on: Dicotómicas",
                                                                ":material/task_alt: Hallazgos", ":material/filter_alt: Filtro univariado"])

# ══════════════════════════════════════════════════════════════════════════════
# PANORAMA
# ══════════════════════════════════════════════════════════════════════════════
with tab_pan:
    c1, c2 = st.columns([2, 3], gap="large")
    with c1:
        st.subheader("Variable objetivo")
        tg = pd.DataFrame({"clase": ["0 · paga", "1 · default"], "n": [R["target_0"], R["target_1"]]})
        tg["pct"] = tg["n"] / N
        tg["etq"] = tg.apply(lambda r: f"{r.n:,} ({r.pct:.2%})", axis=1)
        b = alt.Chart(tg).encode(x=alt.X("clase:N", title=None, axis=alt.Axis(labelAngle=0)),
                                 y=alt.Y("n:Q", title="Créditos", scale=alt.Scale(domain=[0, R["target_0"] * 1.15])))
        st.altair_chart(b.mark_bar(size=80, cornerRadiusEnd=4).encode(
            color=alt.Color("clase:N", scale=alt.Scale(range=[AZUL, NARANJA]), legend=None),
            tooltip=[alt.Tooltip("clase:N", title="Clase"), alt.Tooltip("n:Q", format=","), alt.Tooltip("pct:Q", format=".2%")])
            + b.mark_text(dy=-9, fontSize=12, color=TINTA).encode(text="etq:N"), width="stretch", height=300)
        razon = R["target_0"] / R["target_1"]
        st.markdown(f"Por cada crédito en default hay **{razon:.1f}** que pagan. El desbalance es moderado pero importa: un modelo "
                    f"que prediga siempre «paga» acierta el **{R['target_0'] / N:.1%}** y no sirve. Por eso se evaluará con AUC, "
                    "Gini y KS, no con *accuracy*, y la partición train/test debe ser **estratificada**.")
    with c2:
        st.subheader("¿Dónde vive cada tipo de variable?")
        comp = tipos.assign(fam=tipos["familia"].map(fam_corta)).groupby(["familia", "fam", "tipo"]).size().reset_index(name="n")
        orden_f = comp.sort_values("familia")["fam"].unique().tolist()
        st.altair_chart(alt.Chart(comp).mark_bar(height={"band": 0.72}).encode(
            y=alt.Y("fam:N", sort=orden_f, title=None, axis=alt.Axis(labelLimit=220, labelOverlap=False)),
            x=alt.X("n:Q", title="Variables", stack=True),
            color=alt.Color("tipo:N", scale=COLOR_TIPO, title=None, legend=alt.Legend(orient="top")),
            order=alt.Order("tipo:N"),
            tooltip=[alt.Tooltip("fam:N", title="Familia"), alt.Tooltip("tipo:N", title="Tipo"), alt.Tooltip("n:Q", title="Variables")]),
            width="stretch", height=420)
        st.caption("Vivienda concentra 47 numéricas (en tres versiones _AVG/_MODE/_MEDI); documentación y contacto son casi todo "
                   "dicotómicas.")

    st.subheader("¿Qué forma tienen las numéricas?")
    f1, f2 = st.columns([2, 3], gap="large")
    with f1:
        dist_n = num.groupby("mejor_distribucion").size().reset_index(name="n").sort_values("n", ascending=False)
        st.altair_chart(alt.Chart(dist_n).mark_bar(cornerRadiusEnd=3, height={"band": 0.7}, color=AZUL).encode(
            y=alt.Y("mejor_distribucion:N", sort="-x", title=None), x=alt.X("n:Q", title="Variables"),
            tooltip=[alt.Tooltip("mejor_distribucion:N", title="Distribución"), alt.Tooltip("n:Q", title="Variables")])
            .properties(title="Mejor distribución teórica (menor AIC)"), width="stretch", height=300)
    with f2:
        n_normal = int((num["mejor_distribucion"] == "normal").sum())
        mejora = (num["D_KS_normal"] - num["D_KS_mejor"]).median()
        st.markdown(
            f"- **Ninguna** variable se describe mejor con una normal ({n_normal} de {len(num)}): el perfil típico es **sesgado a la "
            "derecha** (montos, conteos, historial) o **acotado y sesgado a la izquierda** (scores externos).\n"
            f"- En las continuas, la mejor familia reduce el desajuste frente a la normal en **{mejora:.3f}** de D de "
            "Kolmogorov–Smirnov (mediana). Con 300 mil filas un *p-value* siempre rechazaría; por eso se reporta **D**, que mide el "
            "tamaño del desajuste (0 = ajuste perfecto).\n"
            f"- Los **conteos** (hijos, consultas, solicitudes previas) se ajustan con **binomial negativa**: su varianza supera a su "
            "media (sobredispersión), algo que Poisson no admite.\n"
            "- Las variables con **masa en cero** (peores atrasos, deuda, utilización) no las ajusta bien ninguna familia continua "
            "(D > 0.25): son mezclas de «nunca tuvo el evento» y «magnitud del evento»."
        )

    a1, a2 = st.columns(2, gap="large")
    top_as = num.assign(abs_as=num["asimetria"].abs()).nlargest(30, "abs_as")
    a1.altair_chart(alt.Chart(top_as).mark_bar(color=AZUL, height={"band": 0.72}).encode(
        y=alt.Y("variable:N", sort=top_as["variable"].tolist(), title=None, axis=alt.Axis(labelLimit=240, labelOverlap=False, labelFontSize=10)),
        x=alt.X("asimetria:Q", title="Asimetría (escala symlog)", scale=alt.Scale(type="symlog"),
                axis=alt.Axis(values=[-20, -10, -5, -2, 0, 2, 5, 10, 20, 40])),
        tooltip=[alt.Tooltip("variable:N"), alt.Tooltip("asimetria:Q", format=".2f", title="Asimetría"),
                 alt.Tooltip("curtosis:Q", format=".1f", title="Curtosis (exceso)")])
        .properties(title="Asimetría: las 30 variables más asimétricas"), width="stretch", height=640)
    top_out = num.nlargest(30, "pct_outliers_iqr")
    a2.altair_chart(alt.Chart(top_out).mark_bar(color=NARANJA, height={"band": 0.72}).encode(
        y=alt.Y("variable:N", sort=top_out["variable"].tolist(), title=None, axis=alt.Axis(labelLimit=240, labelOverlap=False, labelFontSize=10)),
        x=alt.X("pct_outliers_iqr:Q", title="% de observaciones fuera de los bigotes", axis=PCT),
        tooltip=[alt.Tooltip("variable:N"), alt.Tooltip("pct_outliers_iqr:Q", format=".2%", title="% outliers (1.5·IQR)"),
                 alt.Tooltip("pct_ceros:Q", format=".1%", title="% ceros")])
        .properties(title="% de outliers (regla 1.5·IQR): top 30"), width="stretch", height=640)
    nota("Un «outlier» por la regla 1.5·IQR no es un error. En variables con muchos ceros el IQR es 0 y **cualquier valor positivo** "
         "queda fuera de los bigotes de Tukey: por eso los atrasos y las consultas aparecen arriba, y el capeo no cambia este conteo. "
         "Los valores sin sentido se trataron en *Calidad y preprocesamiento*; lo que queda es forma de la distribución. Por eso el "
         "boxplot de cada ficha dibuja los bigotes en los **percentiles del capeo (0.1 y 99.9)**.")

    st.subheader("Verificación ex-ante vs ex-post")
    st.caption("Guía, Paso 4, precisión (b): ¿se redujo el % de missing?, ¿se controlaron los outliers?, ¿se redujo la cardinalidad?")
    v = t("verificacion").set_index("pasada")
    filas_v = [
        ("% de celdas nulas", "pct_celdas_nulas", "{:.1%}"), ("Variables con nulos", "variables_con_nulos", "{:,.0f}"),
        ("Numéricas con |asimetría| > 2", "num_asim_mayor_2", "{:,.0f}"), ("Asimetría máxima (|valor|)", "asim_max", "{:,.1f}"),
        ("Numéricas con > 5 % de outliers (IQR)", "num_outliers_mayor_5", "{:,.0f}"),
        ("Numéricas con valor centinela", "num_con_centinela", "{:,.0f}"),
        ("Celdas con categorías codificadas (XNA / Unknown)", "cat_codificados", "{:,.0f}"),
        ("Cardinalidad máxima (categóricas)", "cat_cardinalidad_max", "{:,.0f}"),
    ]
    tv = pd.DataFrame([{"Indicador": lab, "Ex-ante": fmt.format(v.loc["ex-ante", col]), "Ex-post": fmt.format(v.loc["ex-post", col])}
                       for lab, col, fmt in filas_v])
    v1, v2 = st.columns([3, 2], gap="large")
    with v1:
        tabla_texto(tv)
    with v2:
        st.markdown(
            "- **Missing**: bajó de 25.1 % a 23.9 % de celdas; el resto son nulos **estructurales** que se conservaron a propósito.\n"
            f"- **Outliers**: la asimetría máxima cae de **{v.loc['ex-ante', 'asim_max']:.1f}** (`AMT_INCOME_TOTAL`) a **{v.loc['ex-post', 'asim_max']:.1f}** "
            f"(`{VAR_ASIM_POST}`, atrasos con masa en cero). Tras el capeo no queda ningún punto fuera de los bigotes p0.1–p99.9; el conteo "
            "de variables asimétricas casi no cambia, porque su asimetría es forma, no error.\n"
            "- **Centinela**: `DAYS_EMPLOYED = 365243` ya no existe.\n"
            "- **Cardinalidad**: no se redujo (58 en `ORGANIZATION_TYPE`); los ≈ 55 mil `XNA` que quedan son la categoría real "
            "*sin empleador*. Agrupar categorías es tarea del bivariado, con su tasa de default."
        )

# ══════════════════════════════════════════════════════════════════════════════
# NUMÉRICAS
# ══════════════════════════════════════════════════════════════════════════════
with tab_num:
    st.subheader("Tabla descriptiva")
    t1, t2 = st.columns([3, 2], vertical_alignment="bottom")
    vista = t1.segmented_control("Vista", ["Estadísticos base", "Cuantiles", "Forma, outliers y ajuste", "Top-5 valores"],
                                 default="Estadísticos base", key="vista_num")
    fam_sel = t2.multiselect("Familia", sorted(num["familia"].unique()), placeholder="Todas", key="fam_num_uni",
                             format_func=fam_corta)
    dn = num[num["familia"].isin(fam_sel)] if fam_sel else num
    dn = dn.sort_values("orden")
    col_var = {"variable": st.column_config.TextColumn("Variable", pinned=True, width=230)}
    g = {c: st.column_config.NumberColumn(c, format="compact") for c in
         ["media", "std", "min", "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max"]}
    if vista == "Cuantiles":
        st.dataframe(dn[["variable", "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]], hide_index=True, width="stretch",
                     height=460, column_config={**col_var, **g})
        st.caption("Percentiles 1 a 99 (`quantile()`): la distancia entre p99 y p95, o entre p1 y p5, revela colas.")
    elif vista == "Forma, outliers y ajuste":
        st.dataframe(dn[["variable", "asimetria", "curtosis", "pct_ceros", "pct_outliers_iqr", "pct_outliers_extremos",
                         "mejor_distribucion", "D_KS_mejor", "D_KS_normal", "es_conteo"]], hide_index=True, width="stretch", height=460,
                     column_config={**col_var, "asimetria": st.column_config.NumberColumn("Asimetría", format="%.2f"),
                                    "curtosis": st.column_config.NumberColumn("Curtosis (exceso)", format="%.1f"),
                                    "pct_ceros": st.column_config.ProgressColumn("% ceros", format="percent", min_value=0, max_value=1),
                                    "pct_outliers_iqr": st.column_config.NumberColumn("% outliers 1.5·IQR", format="percent"),
                                    "pct_outliers_extremos": st.column_config.NumberColumn("% outliers 3·IQR", format="percent"),
                                    "mejor_distribucion": st.column_config.TextColumn("Mejor ajuste (AIC)"),
                                    "D_KS_mejor": st.column_config.NumberColumn("D KS (mejor)", format="%.3f"),
                                    "D_KS_normal": st.column_config.NumberColumn("D KS (normal)", format="%.3f"),
                                    "es_conteo": st.column_config.CheckboxColumn("Conteo")})
        st.caption("Curtosis en exceso (normal = 0). En los conteos el D se calcula contra la distribución discreta, por eso no hay "
                   "comparación con la normal.")
    elif vista == "Top-5 valores":
        tp = t("num_top5").query("pasada == @P and variable in @dn.variable")
        tp["txt"] = tp.apply(lambda r: f"{r.valor}  ({r.pct:.1%})", axis=1)
        ancho = tp.pivot(index="variable", columns="rango", values="txt").reindex(dn["variable"]).reset_index()
        ancho.columns = ["variable"] + [f"#{c}" for c in ancho.columns[1:]]
        st.dataframe(ancho, hide_index=True, width="stretch", height=460, column_config=col_var)
        st.caption("`value_counts().head(5)` sobre los valores no nulos: detecta valores dominantes (0 u otro) y picos de imputación.")
    else:
        st.dataframe(dn[["variable", "n_validos", "pct_missing", "media", "std", "min", "p25", "p50", "p75", "max", "cv", "n_unicos"]],
                     hide_index=True, width="stretch", height=460,
                     column_config={**col_var, **g, "n_validos": st.column_config.NumberColumn("count", format="localized"),
                                    "pct_missing": st.column_config.ProgressColumn("% missing", format="percent", min_value=0, max_value=1),
                                    "cv": st.column_config.NumberColumn("CV", format="%.2f", help="Desviación estándar / |media|."),
                                    "n_unicos": st.column_config.NumberColumn("Valores únicos", format="localized")})
        st.caption("`describe()` + % missing (`isna()`). Ex-post, el missing que queda es estructural (sin historial) o de vivienda.")

    st.divider()
    st.subheader("Ficha por variable")
    opciones = num.sort_values("orden")["variable"].tolist()
    var = st.selectbox("Variable numérica", opciones, index=opciones.index("AMT_ANNUITY"), key="var_num",
                       format_func=lambda v: f"{v}  ·  {fam_corta(num.set_index('variable').loc[v, 'familia'])}")
    m = num.set_index("variable").loc[var]
    if isinstance(m.get("descripcion"), str) and m["descripcion"]:
        st.caption(m["descripcion"])
    mm = st.columns(4) + st.columns(4)
    for col_, (lab, val) in zip(mm, [("n válidos", f"{int(m.n_validos):,}"), ("% missing", f"{m.pct_missing:.1%}"),
                                     ("Media", fnum(m.media)), ("Mediana", fnum(m.p50)), ("Desv. est.", fnum(m["std"])),
                                     ("Asimetría", f"{m.asimetria:.2f}"), ("Curtosis", f"{m.curtosis:.2f}"),
                                     ("% outliers", f"{m.pct_outliers_iqr:.1%}")]):
        col_.metric(lab, val)

    hist = t("num_hist").query("pasada == @P and variable == @var")
    curvas = t("num_curvas").query("pasada == @P and variable == @var")
    aj = t("num_ajuste").query("pasada == @P and variable == @var").sort_values("AIC")
    box = t("num_box").query("pasada == @P and variable == @var")
    pts = t("num_box_pts").query("pasada == @P and variable == @var")
    qq = t("num_qq").query("pasada == @P and variable == @var")
    es_conteo = bool(m["es_conteo"])
    mejor = aj.iloc[0]["distribucion"] if len(aj) else "—"

    if hist.empty:
        st.warning("La variable no tiene variabilidad en esta pasada; no se grafica.")
    else:
        lo, hi = float(hist["desde"].min()), float(hist["hasta"].max())
        xs = alt.Scale(domain=[lo, hi], nice=False)
        etq_mejor = f"Mejor ajuste: {mejor}"
        lineas = pd.DataFrame({"serie": ["Media", "Mediana"], "x": [m.media, m.p50]})
        lineas = lineas[(lineas.x >= lo) & (lineas.x <= hi)]
        cv_ = curvas.assign(serie=np.where(curvas["rol"] == "mejor", etq_mejor, "Normal (referencia)"))
        dom = ["Media", "Mediana", etq_mejor, "Normal (referencia)"]
        col_s = alt.Scale(domain=dom, range=[TINTA, TINTA, NARANJA, GRIS])
        dash_s = alt.Scale(domain=dom, range=[[1, 0], [2, 3], [1, 0], [6, 4]])
        barras = alt.Chart(hist).mark_bar(color=AZUL, opacity=0.5, binSpacing=1).encode(
            x=alt.X("desde:Q", bin="binned", scale=xs, title=var,
                    axis=alt.Axis(tickMinStep=1, format="d") if es_conteo else alt.Axis()), x2="hasta:Q",
            y=alt.Y("densidad:Q", title="Probabilidad" if es_conteo else "Densidad"),
            tooltip=[alt.Tooltip("desde:Q", format=",.4g", title="Desde"), alt.Tooltip("hasta:Q", format=",.4g", title="Hasta"),
                     alt.Tooltip("n:Q", format=",", title="Créditos")])
        if es_conteo:
            teo = alt.Chart(cv_[cv_.rol == "mejor"]).mark_line(point=alt.OverlayMarkDef(size=45, filled=True), strokeWidth=2).encode(
                x=alt.X("x:Q", scale=xs), y="y:Q", color=alt.Color("serie:N", scale=col_s, title=None),
                strokeDash=alt.StrokeDash("serie:N", scale=dash_s, legend=None))
            teo = teo + alt.Chart(cv_[cv_.rol == "normal"]).mark_line(strokeWidth=1.8).encode(
                x=alt.X("x:Q", scale=xs), y="y:Q", color=alt.Color("serie:N", scale=col_s, title=None),
                strokeDash=alt.StrokeDash("serie:N", scale=dash_s, legend=None))
        else:
            teo = alt.Chart(cv_).mark_line(strokeWidth=2.2).encode(
                x=alt.X("x:Q", scale=xs), y="y:Q", color=alt.Color("serie:N", scale=col_s, title=None),
                strokeDash=alt.StrokeDash("serie:N", scale=dash_s, legend=None), detail="serie:N",
                tooltip=[alt.Tooltip("serie:N"), alt.Tooltip("x:Q", format=",.4g"), alt.Tooltip("y:Q", format=".3g", title="Densidad")])
        reglas = alt.Chart(lineas).mark_rule(strokeWidth=1.8).encode(
            x=alt.X("x:Q", scale=xs), color=alt.Color("serie:N", scale=col_s, title=None, legend=alt.Legend(orient="top", labelLimit=320)),
            strokeDash=alt.StrokeDash("serie:N", scale=dash_s, legend=None),
            tooltip=[alt.Tooltip("serie:N"), alt.Tooltip("x:Q", format=",.4g", title="Valor")])
        h1, h2 = st.columns([5, 2], gap="medium")
        d_mejor = aj.iloc[0]["D_KS"]
        d_norm = aj.loc[aj.distribucion == "normal", "D_KS"]
        sub = f"{mejor} · D = {d_mejor:.3f}" + (f" · normal D = {d_norm.iloc[0]:.3f}" if len(d_norm) else "")
        h1.altair_chart((barras + teo + reglas).properties(title={"text": f"{var}: histograma", "subtitle": sub}),
                        width="stretch", height=360)
        qmin, qmax = float(min(qq.teorico.min(), qq.observado.min())), float(max(qq.teorico.max(), qq.observado.max()))
        diag = pd.DataFrame({"x": [qmin, qmax], "y": [qmin, qmax]})
        h2.altair_chart((alt.Chart(qq).mark_circle(size=18, color=AZUL, opacity=0.8).encode(
            x=alt.X("teorico:Q", title=f"Cuantiles teóricos ({mejor})", scale=alt.Scale(zero=False)),
            y=alt.Y("observado:Q", title="Cuantiles observados", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("p:Q", format=".1%", title="Probabilidad"), alt.Tooltip("teorico:Q", format=",.4g"),
                     alt.Tooltip("observado:Q", format=",.4g")])
            + alt.Chart(diag).mark_line(color=NARANJA, strokeWidth=1.6).encode(x="x:Q", y="y:Q"))
            .properties(title=f"Q-Q vs {mejor}"), width="stretch", height=360)
        b0 = box.iloc[0]
        st.caption(f"Vista: percentil 0.5 a 99.5 (deja fuera {int(b0.n_fuera_vista):,} créditos, "
                   f"{b0.n_fuera_vista / m.n_validos:.1%}); las métricas usan todos los datos. La densidad se calcula sobre el total, "
                   "así que es comparable con la curva teórica.")

        # Boxplot, en gráfico aparte
        caja = pd.DataFrame([{"y": 0, **b0.to_dict()}])
        blo, bhi = float(b0.vista_min), float(b0.vista_max)
        pad = (bhi - blo) * 0.02 or 1.0
        blo, bhi = blo - pad, bhi + pad
        yb = alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[-1, 1]))
        xb = alt.X("valor:Q", title=var, scale=alt.Scale(domain=[blo, bhi], nice=False, clamp=True))
        cap = caja.assign(bi=caja.bigote_inf, bs=caja.bigote_sup)
        capa_bigote = alt.Chart(cap).mark_rule(color=AZUL, strokeWidth=1.5).encode(
            x=alt.X("bi:Q", scale=alt.Scale(domain=[blo, bhi], nice=False), title=var), x2="bs:Q", y=yb)
        capa_topes = alt.Chart(pd.concat([cap.assign(x=cap.bi), cap.assign(x=cap.bs)])).mark_tick(color=AZUL, thickness=2, size=26).encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[blo, bhi], nice=False)), y=yb)
        capa_caja = alt.Chart(cap).mark_bar(color=AZUL, opacity=0.35, size=60, stroke=AZUL, strokeWidth=1).encode(
            x=alt.X("q1:Q", scale=alt.Scale(domain=[blo, bhi], nice=False)), x2="q3:Q", y=yb,
            tooltip=[alt.Tooltip("q1:Q", format=",.4g", title="Q1"), alt.Tooltip("mediana:Q", format=",.4g", title="Mediana"),
                     alt.Tooltip("q3:Q", format=",.4g", title="Q3"), alt.Tooltip("bigote_inf:Q", format=",.4g", title="Bigote inferior"),
                     alt.Tooltip("bigote_sup:Q", format=",.4g", title="Bigote superior")])
        capa_med = alt.Chart(cap).mark_tick(color=TINTA, thickness=3, size=60).encode(
            x=alt.X("mediana:Q", scale=alt.Scale(domain=[blo, bhi], nice=False)), y=yb)
        capa_media = alt.Chart(cap).mark_point(shape="diamond", filled=True, size=110, color=TINTA).encode(
            x=alt.X("media:Q", scale=alt.Scale(domain=[blo, bhi], nice=False)), y=yb,
            tooltip=[alt.Tooltip("media:Q", format=",.4g", title="Media")])
        capas = capa_bigote + capa_topes + capa_caja + capa_med + capa_media
        if len(pts):
            capas = alt.Chart(pts).mark_circle(size=12, color=AZUL, opacity=0.35).encode(
                x=xb, y=alt.Y("jitter:Q", axis=None, scale=alt.Scale(domain=[-1, 1])),
                tooltip=[alt.Tooltip("valor:Q", format=",.4g", title="Valor")]) + capas
        bigotes = "límites del capeo (p0.1 y p99.9 de train)" if P == "ex-post" else "percentiles 0.1 y 99.9"
        st.altair_chart(capas.properties(title={"text": f"{var}: boxplot", "subtitle":
                        f"Bigotes: {bigotes} · fuera: {int(b0.n_fuera):,} créditos · ◆ media · | mediana"}),
                        width="stretch", height=190)
        st.caption(f"Convención acordada con la profesora: los bigotes llegan a los percentiles del capeo. "
                   + ("En la pasada ex-post no queda ningún punto fuera. " if P == "ex-post" else
                      "En la pasada ex-ante, los puntos (muestra de hasta 500; los que exceden el eje se dibujan en el borde) son los "
                      "valores que se capearon o eliminaron. ")
                   + f"Con la regla de Tukey (1.5·IQR) quedarían {int(b0.n_fuera_tukey):,} créditos afuera "
                   f"({b0.n_fuera_tukey / m.n_validos:.1%}): describe la forma de la distribución, no errores.")

        z1, z2 = st.columns([3, 2], gap="large")
        with z1:
            st.markdown("**Ajuste de distribuciones** (máxima verosimilitud sobre 20,000 observaciones)")
            tabla_texto(aj.assign(elegida=aj["elegida"].map({True: "✔", False: ""}))
                        [["elegida", "distribucion", "k", "AIC", "dAIC", "D_KS"]]
                        .rename(columns={"elegida": "", "distribucion": "Distribución", "k": "Parámetros", "dAIC": "ΔAIC", "D_KS": "D KS"}),
                        {"AIC": "{:,.0f}", "ΔAIC": "{:,.0f}", "D KS": "{:.3f}"})
        with z2:
            st.markdown("**Top-5 valores más frecuentes**")
            tp5 = t("num_top5").query("pasada == @P and variable == @var")
            tabla_texto(tp5[["rango", "valor", "n", "pct"]].rename(columns={"rango": "Nº", "valor": "Valor", "n": "Créditos", "pct": "%"}),
                        {"Créditos": "{:,}", "%": "{:.1%}"})

        # Lectura: automática + esperada de negocio
        forma = ("aproximadamente simétrica" if abs(m.asimetria) < 0.5 else
                 ("moderadamente sesgada " if abs(m.asimetria) < 1 else "fuertemente sesgada ") +
                 ("a la derecha (cola hacia valores altos)" if m.asimetria > 0 else "a la izquierda (cola hacia valores bajos)"))
        colas = ("colas pesadas (leptocúrtica)" if m.curtosis > 3 else "colas ligeras / forma aplanada (platicúrtica)" if m.curtosis < -0.5
                 else "colas cercanas a las de una normal")
        lect = [f"Distribución **{forma}**, con {colas}; asimetría {m.asimetria:.2f} y curtosis en exceso {m.curtosis:.2f}."]
        lect.append(f"La describe mejor una **{mejor}** (D = {d_mejor:.3f}" + (f" frente a {d_norm.iloc[0]:.3f} de la normal)." if len(d_norm) else ")."))
        if d_mejor > 0.2:
            lect.append(f"**Ninguna familia ajusta bien** (D > 0.2): la forma no es simple (masa en un valor, varias modas). La lectura "
                        "por asimetría y curtosis resume mal esta variable; mira el histograma.")
        if m.pct_ceros > 0.3:
            lect.append(f"El 0 concentra el **{m.pct_ceros:.0%}** de los valores (valor dominante del conteo)." if es_conteo else
                        f"**{m.pct_ceros:.0%} de ceros**: la variable mezcla «sin evento» con «magnitud del evento».")
        if m.pct_outliers_iqr > 0.05:
            lect.append(f"{m.pct_outliers_iqr:.1%} de observaciones fuera de 1.5·IQR" +
                        (" (con IQR estrecho por la concentración en un valor)." if m.iqr == 0 or m.pct_moda > 0.5 else "."))
        if m.pct_missing > 0:
            lect.append(f"{m.pct_missing:.1%} de missing" + (" (estructural o de vivienda; se conserva)." if P == "ex-post" else "."))
        if pd.notna(m.centinela):
            lect.append(f"Valor centinela **{fnum(m.centinela)}** en el {m.pct_centinela:.1%} (se excluye de los gráficos).")
        st.markdown("**Lectura**\n\n" + "\n".join(f"- {x}" for x in lect))
        if var in ESPERADO:
            st.success(f"**¿Es lo esperado?** {ESPERADO[var]}", icon=":material/fact_check:")

        # Raw vs log
        lg = t("num_log").query("pasada == @P and variable == @var")
        if len(lg):
            st.markdown("**Escala original vs log(1 + x)** · la guía pide comparar ambas cuando la variable es muy asimétrica")
            lh = t("num_log_hist").query("pasada == @P and variable == @var")
            lc = t("num_log_curva").query("pasada == @P and variable == @var")
            l0 = lg.iloc[0]
            r1, r2 = st.columns(2, gap="large")
            r1.altair_chart(alt.Chart(hist).mark_bar(color=AZUL, opacity=0.6, binSpacing=1).encode(
                x=alt.X("desde:Q", bin="binned", title=var, scale=xs), x2="hasta:Q", y=alt.Y("densidad:Q", title="Densidad"))
                .properties(title={"text": "Escala original", "subtitle": f"asimetría {m.asimetria:.2f}"}), width="stretch", height=250)
            r2.altair_chart((alt.Chart(lh).mark_bar(color=VERDE, opacity=0.6, binSpacing=1).encode(
                x=alt.X("desde:Q", bin="binned", title=f"log(1 + {var})"), x2="hasta:Q", y=alt.Y("densidad:Q", title="Densidad"))
                + alt.Chart(lc).mark_line(color=GRIS, strokeDash=[6, 4], strokeWidth=2).encode(x="x:Q", y="y:Q"))
                .properties(title={"text": "Escala log(1 + x)", "subtitle": f"asimetría {l0.asim_log:.2f} · D vs normal {l0.D_KS_normal_log:.3f}"}),
                width="stretch", height=250)
            if m["max"] <= 1.5:
                st.warning("La variable está normalizada en [0, 1]: en ese rango log(1 + x) ≈ x, así que la transformación **casi no "
                           "cambia la forma**. Aquí el log no es la herramienta; si hace falta, mejor discretizar en el bivariado.",
                           icon=":material/warning:")
            elif abs(l0.asim_log) < 1:
                st.success(f"El log reduce la asimetría de {m.asimetria:.2f} a {l0.asim_log:.2f}: candidata a transformarse si el modelo "
                           "es sensible a la escala (regresión logística sin WoE).", icon=":material/check_circle:")
            else:
                st.info(f"El log reduce la asimetría a {l0.asim_log:.2f}, pero no la elimina (la masa en cero sigue ahí).",
                        icon=":material/info:")

    with st.expander("Relación entre dos numéricas (dispersión)", icon=":material/scatter_plot:"):
        st.caption("La guía incluye el scatterplot para detectar no linealidades y clusters. Muestra aleatoria de 4,000 créditos.")
        mu = t("muestra_dispersion").query("pasada == @P")
        s1, s2, s3 = st.columns([2, 2, 1], vertical_alignment="bottom")
        vx = s1.selectbox("Eje X", opciones, index=opciones.index("AMT_CREDIT"), key="sx")
        vy = s2.selectbox("Eje Y", opciones, index=opciones.index("AMT_ANNUITY"), key="sy")
        lg_ = s3.toggle("Escala symlog", key="slog")
        esc = alt.Scale(type="symlog") if lg_ else alt.Scale(zero=False)
        dd = mu[[vx, vy]].dropna() if vx != vy else mu[[vx]].dropna().assign(**{f"{vy} ": mu[vy]})
        if vx != vy:
            rho = dd[vx].rank().corr(dd[vy].rank())
            st.altair_chart(alt.Chart(dd).mark_circle(size=14, opacity=0.3, color=AZUL).encode(
                x=alt.X(f"{vx}:Q", scale=esc), y=alt.Y(f"{vy}:Q", scale=esc),
                tooltip=[alt.Tooltip(f"{vx}:Q", format=",.4g"), alt.Tooltip(f"{vy}:Q", format=",.4g")])
                .properties(title={"text": f"{vy} vs {vx}", "subtitle": f"n = {len(dd):,} · ρ de Spearman = {rho:.2f}"}),
                width="stretch", height=380)
            st.caption("La correlación entre explicativas (y la eliminación de redundantes) se decide en el multivariado (Pasos 6.5–6.6).")
        else:
            st.info("Elige dos variables distintas.")

# ══════════════════════════════════════════════════════════════════════════════
# CATEGÓRICAS
# ══════════════════════════════════════════════════════════════════════════════
with tab_cat:
    st.subheader("Cardinalidad y dominancia")
    cc = cat.sort_values("n_categorias", ascending=False)
    st.dataframe(cc[["variable", "n_categorias", "moda", "pct_moda", "pct_top5", "pct_top10", "entropia_norm", "hhi", "n_cat_raras",
                     "pct_obs_en_raras", "fill_rate"]], hide_index=True, width="stretch", height=38 + 35 * len(cc),
                 column_config={"variable": st.column_config.TextColumn("Variable", pinned=True, width=220),
                                "n_categorias": st.column_config.NumberColumn("Categorías", help="nunique()"),
                                "moda": st.column_config.TextColumn("Moda", width=170),
                                "pct_moda": st.column_config.NumberColumn("% top-1", format="percent"),
                                "pct_top5": st.column_config.NumberColumn("% top-5", format="percent"),
                                "pct_top10": st.column_config.NumberColumn("% top-10", format="percent"),
                                "entropia_norm": st.column_config.ProgressColumn("Entropía norm.", format="%.2f", min_value=0, max_value=1,
                                                                                 help="0 = una sola categoría; 1 = todas igual de frecuentes."),
                                "hhi": st.column_config.NumberColumn("HHI", format="%.3f", help="Σ p²: 1 = una categoría concentra todo."),
                                "n_cat_raras": st.column_config.NumberColumn("Raras (< 1 %)"),
                                "pct_obs_en_raras": st.column_config.NumberColumn("% obs. en raras", format="percent"),
                                "fill_rate": st.column_config.ProgressColumn("Fill rate", format="percent", min_value=0, max_value=1)})
    st.caption("Porcentajes sobre los valores no nulos. `HC_N_OPERACIONES_TARJETA` es un conteo que la regla (≤ 5 valores) clasifica "
               "como categórica.")

    st.subheader("Frecuencias por variable")
    ops_c = cc["variable"].tolist()
    vc_ = st.selectbox("Variable categórica", ops_c, index=ops_c.index("ORGANIZATION_TYPE"), key="var_cat")
    fr = t("cat_frecuencias").query("pasada == @P and variable == @vc_").copy()
    mc = cat.set_index("variable").loc[vc_]
    ordinal = fr["orden_num"].notna().any()
    validas = fr[~fr["es_nulo"]]
    validas = validas.sort_values("orden_num") if ordinal else validas.sort_values("n", ascending=False)
    top = validas.head(10) if not ordinal else validas
    resto = validas.iloc[len(top):]
    filas_g = top.assign(clase=np.where(top["codificado"], "Código XNA / Unknown", "Categoría"))
    if len(resto):
        filas_g = pd.concat([filas_g, pd.DataFrame([{"categoria": f"Otras ({len(resto)} categorías)", "n": resto["n"].sum(),
                                                     "pct": resto["pct"].sum(), "clase": "Otras"}])])
    nulos = fr[fr["es_nulo"]]
    if len(nulos):
        filas_g = pd.concat([filas_g, nulos.assign(clase="Nulo")])
    filas_g["etq"] = filas_g.apply(lambda r: f"{int(r.n):,} ({r.pct:.1%})", axis=1)
    orden_g = filas_g["categoria"].tolist()
    q1_, q2_ = st.columns([3, 2], gap="large")
    bc = alt.Chart(filas_g).encode(y=alt.Y("categoria:N", sort=orden_g, title=None, axis=alt.Axis(labelLimit=240, labelOverlap=False)))
    q1_.altair_chart((bc.mark_bar(height={"band": 0.72}, cornerRadiusEnd=3).encode(
        x=alt.X("n:Q", title="Créditos", scale=alt.Scale(domain=[0, filas_g["n"].max() * 1.28])),
        color=alt.Color("clase:N", scale=alt.Scale(domain=["Categoría", "Código XNA / Unknown", "Otras", "Nulo"],
                                                   range=[AZUL, "#7fb0ea", "#b9b8b3", GRIS]), legend=alt.Legend(orient="top", title=None, labelLimit=260)),
        tooltip=[alt.Tooltip("categoria:N", title="Categoría"), alt.Tooltip("n:Q", format=","), alt.Tooltip("pct:Q", format=".2%")])
        + bc.mark_text(align="left", dx=4, fontSize=11, color=TINTA).encode(x="n:Q", text="etq:N"))
        .properties(title={"text": f"{vc_}: countplot" + ("" if ordinal else " top-10"),
                           "subtitle": f"{int(mc.n_categorias)} categorías · fill rate {mc.fill_rate:.1%} · moda {mc.pct_moda:.1%} · "
                                       f"entropía norm. {mc.entropia_norm:.2f} · HHI {mc.hhi:.3f}"}),
        width="stretch", height=max(260, 42 * len(filas_g) + 130))
    with q2_:
        st.markdown("**Top-10 con % acumulado**")
        tt = validas.sort_values("n", ascending=False).head(10).copy()
        tt["pct_valido"] = tt["n"] / validas["n"].sum()
        tt["acum"] = tt["pct_valido"].cumsum()
        tabla_texto(tt[["categoria", "n", "pct_valido", "acum"]].rename(columns={"categoria": "Categoría", "n": "Créditos",
                                                                                 "pct_valido": "%", "acum": "% acumulado"}),
                    {"Créditos": "{:,}", "%": "{:.1%}", "% acumulado": "{:.1%}"})
        lect_c = []
        if mc.hhi > 0.6:
            lect_c.append(f"**Dominancia fuerte**: la moda concentra el {mc.pct_moda:.0%}; poca variabilidad para discriminar.")
        elif mc.entropia_norm > 0.9:
            lect_c.append("Categorías **casi equiprobables** (entropía alta): la variable reparte bien, pero no dice nada de riesgo "
                          "hasta verla contra el TARGET.")
        if mc.cardinalidad_alta:
            lect_c.append(f"**Cardinalidad alta** ({int(mc.n_categorias)}): one-hot generaría muchas columnas; conviene **agrupar** "
                          "por tasa de default en el bivariado.")
        if mc.n_cat_raras > 0:
            lect_c.append(f"{int(mc.n_cat_raras)} categorías raras (< 1 %) con el {mc.pct_obs_en_raras:.1%} de las observaciones: "
                          "candidatas a agruparse en «Otras», salvo que tengan un default muy distinto.")
        if vc_ == "ORGANIZATION_TYPE":
            lect_c.append("`XNA` (18 %) **no es un faltante**: es la categoría *sin empleador* y coincide fila a fila con el centinela "
                          "de `DAYS_EMPLOYED` (ver *Calidad y preprocesamiento*).")
        if mc.fill_rate < 0.7:
            lect_c.append(f"Solo {mc.fill_rate:.0%} con dato: el nulo se tratará como categoría propia en el bivariado.")
        if lect_c:
            st.markdown("**Lectura**\n\n" + "\n".join(f"- {x}" for x in lect_c))

# ══════════════════════════════════════════════════════════════════════════════
# DICOTÓMICAS
# ══════════════════════════════════════════════════════════════════════════════
with tab_dic:
    st.subheader("Balance de clases")
    dd_ = dic.sort_values("pct_minoritaria_validos", ascending=False)
    orden_d = dd_["variable"].tolist()
    largo = pd.concat([
        dd_.assign(parte="Clase mayoritaria", valor=dd_["pct_mayoritaria"], k=0),
        dd_.assign(parte="Clase minoritaria", valor=dd_["pct_minoritaria"], k=1),
        dd_.assign(parte="Nulo u otro valor", valor=dd_["pct_nulo"] + dd_["pct_otros"], k=2),
    ])
    largo["detalle"] = largo.apply(lambda r: f"{r.parte}: {r.valor:.2%}", axis=1)
    etq_d = dd_.assign(etq=dd_.apply(lambda r: f"minoritaria «{r.clase_minoritaria}»: {r.pct_minoritaria_validos:.2%}", axis=1), x=1.0)
    ybal = alt.Y("variable:N", sort=orden_d, title=None, axis=alt.Axis(labelLimit=240, labelOverlap=False, labelFontSize=10))
    barras_d = alt.Chart(largo).mark_bar(height={"band": 0.7}).encode(
        y=ybal, x=alt.X("valor:Q", stack="zero", title="% del total de filas", scale=alt.Scale(domain=[0, 1.42]),
                axis=alt.Axis(format="%", values=[0, 0.2, 0.4, 0.6, 0.8, 1.0])),
        color=alt.Color("parte:N", scale=alt.Scale(domain=["Clase mayoritaria", "Clase minoritaria", "Nulo u otro valor"],
                                                   range=[AZUL, NARANJA, GRIS]), legend=alt.Legend(orient="top", title=None)),
        order=alt.Order("k:Q"), tooltip=[alt.Tooltip("variable:N"), alt.Tooltip("detalle:N", title="Parte")])
    textos_d = alt.Chart(etq_d).mark_text(align="left", dx=6, fontSize=10, color=TINTA).encode(
        y=ybal, x=alt.X("x:Q"), text="etq:N",
        opacity=alt.condition("datum.casi_constante", alt.value(0.55), alt.value(1)))
    regla = alt.Chart(pd.DataFrame({"x": [0.99]})).mark_rule(color=TINTA, strokeDash=[3, 3], opacity=0.5).encode(x="x:Q")
    st.altair_chart((barras_d + textos_d + regla).properties(title="Balance de clases en las variables dicotómicas (% del total de filas)"),
                    width="stretch", height=24 * len(dd_) + 80)
    st.caption("Ordenadas de más a menos balanceada. La línea punteada marca el 99 %: a su derecha la clase minoritaria tiene menos "
               "del 1 % (casi constante). En la pasada ex-ante, «otro valor» es `CODE_GENDER = 'XNA'`.")

    d1, d2 = st.columns([5, 6], gap="large")
    with d1:
        cc_ = dd_[dd_["casi_constante"]]
        st.markdown(f"**Casi constantes ({len(cc_)})** · clase minoritaria < 1 %")
        tabla_texto(cc_[["variable", "clase_minoritaria", "n_minoritaria", "pct_minoritaria_validos"]]
                    .rename(columns={"variable": "Variable", "clase_minoritaria": "Minoritaria", "n_minoritaria": "Créditos",
                                     "pct_minoritaria_validos": "%"}), {"Créditos": "{:,}", "%": "{:.3%}"})
    with d2:
        st.markdown("**Pares redundantes** · idénticos o complementarios en ≥ 99.9 % de las filas (ex-post)")
        rd = t("redundancia_dicotomicas")
        tabla_texto(rd[["variable_a", "variable_b", "relacion", "n_discrepancias"]]
                    .rename(columns={"variable_a": "Variable A", "variable_b": "Variable B", "relacion": "Relación",
                                     "n_discrepancias": "Distintas"}), {"Distintas": "{:,}"})
        st.markdown(
            "- `flag_sin_buro` y `flag_sin_previas` son el **complemento exacto** de `TIENE_BUREAU` y `TIENE_HISTORIAL_HOME_CREDIT`.\n"
            "- `flag_sin_empleo` es casi el complemento de `FLAG_EMP_PHONE`: el centinela coincide con «no dio teléfono del empleador».\n"
            "- Aportan la misma información dos veces: en el modelo basta una de cada par.")
    nota("Una dicotómica casi constante no puede separar clases: con 13 créditos (`FLAG_DOCUMENT_2`) o 1 (`FLAG_MOBIL`) en la clase "
         "minoritaria no hay base estadística para estimar su efecto. Por eso el **filtro univariado** elimina toda variable cuyo valor "
         "dominante concentra ≥ 99 % de los datos (ver la pestaña *Filtro univariado*).")

# ══════════════════════════════════════════════════════════════════════════════
# HALLAZGOS
# ══════════════════════════════════════════════════════════════════════════════
with tab_hal:
    st.subheader("Observaciones del univariado")
    st.markdown(
        "Se evaluó cada variable una vez finalizadas las pruebas de calidad y las correcciones (pasada ex-post). Las variables presentan:")
    o1, o2 = st.columns(2, gap="large")
    with o1, st.container(border=True):
        st.markdown(
            "**1 · Distribución: ninguna es normal**\n\n"
            "- Montos (`AMT_*`): sesgo a la derecha, bien descritos por **gamma** (D ≈ 0.02–0.08). Picos en montos redondos por "
            "productos estandarizados.\n"
            "- Scores externos: acotados en [0, 1] y **sesgados a la izquierda** (skew-normal).\n"
            "- `DAYS_BIRTH`: casi plana, la única cercana a la simetría.\n"
            "- Conteos: **binomial negativa** (sobredispersión) en hijos, consultas, solicitudes y círculo social.")
    with o2, st.container(border=True):
        st.markdown(
            "**2 · Asimetría y colas**\n\n"
            "- 61 numéricas con |asimetría| > 2, sobre todo **vivienda** (en [0, 1], con muchos ceros) e **historial** (buró, POS, "
            "tarjeta, cuotas).\n"
            "- Las de atraso tienen **masa en cero** (47–99 % de ceros) y colas de miles de días: son mezclas «sin evento / magnitud».\n"
            "- `HC_CARD_MAX_UTILIZACION` es **bimodal** (tarjeta sin uso vs. tarjeta al tope).")
    o3, o4 = st.columns(2, gap="large")
    with o3, st.container(border=True):
        st.markdown(
            "**3 · Outliers relevantes**\n\n"
            "- Tras eliminar lo absurdo y capear en p0.1–p99.9, lo que la regla de Tukey sigue marcando es **forma, no error**: "
            "variables de historial cuyo IQR es 0 por la concentración en cero.\n"
            f"- La más asimétrica ex-post es `{VAR_ASIM_POST}`: masa en cero y una cola real de días de atraso.\n"
            "- `DAYS_EMPLOYED` ex-post tiene un **pico artificial en 0** (18 %) por el reemplazo del centinela.")
    with o4, st.container(border=True):
        st.markdown(
            "**4 · Consistencia con lo esperado**\n\n"
            "- Montos, ingresos y antigüedades se comportan como en cualquier cartera de consumo: cola derecha y valores redondos.\n"
            "- El 8.07 % de default es coherente con el público de Home Credit: clientes con historial crediticio escaso o nulo.\n"
            "- Lo **no esperado** está en variables operativas (`HOUR_APPR_PROCESS_START`, `WEEKDAY_APPR_PROCESS_START`) y en "
            "documentos casi nunca entregados: poca razón de negocio para que expliquen el riesgo.")

    st.subheader("Preguntas para discutir antes del bivariado")
    st.markdown(
        "1. **`DAYS_EMPLOYED = 0` junta dos perfiles opuestos.** El reemplazo del centinela pone a los pensionistas (default ≈ 5.4 %) "
        "en el mismo valor que quien recién empezó a trabajar (≈ 10.7 %). En el bivariado esto se ve: el patrón de `DAYS_EMPLOYED` "
        "por `qcut` es **no lineal (∩)** en vez de monótono. ¿Conviene que el 0 sea un tramo propio o que `flag_sin_empleo` acompañe "
        "a la variable?\n"
        "2. **La regla de tipos y los outliers.** Tras capear en 4, `CNT_CHILDREN` y `AMT_REQ_CREDIT_BUREAU_QRT` quedan con 5 "
        "valores y la regla del notebook (≤ 5 → categórica) las cambiaría de tipo. Se mantuvieron numéricas: ¿un conteo debe "
        "cambiar de tipo por un tratamiento?\n"
        "3. **log(1 + x) no sirve en [0, 1].** La regla del notebook marca 31 variables de vivienda para la vista log, pero en ese rango "
        "la transformación casi no cambia la forma. Si se transforma, debe ser por evidencia, no por regla.\n"
        "4. **`CODE_GENDER` como predictor.** Estadísticamente es una dicotómica bien balanceada, aunque con IV 0.039 no llega al corte "
        "de 0.05 del bivariado. Si hubiera pasado, ¿debería entrar? Usar el género en una decisión de crédito es discutible "
        "regulatoriamente en muchas jurisdicciones."
    )

# ══════════════════════════════════════════════════════════════════════════════
# FILTRO UNIVARIADO
# ══════════════════════════════════════════════════════════════════════════════
with tab_fil:
    MOD = REPO_DIR / "artifacts" / "modelado"
    fu = pd.read_parquet(MOD / "filtro_univariado.parquet")
    gr = pd.read_parquet(MOD / "grupos_cardinalidad.parquet")
    st.markdown(
        "Qué variables **entran al bivariado**. Los criterios los fijó el equipo con el feedback del asistente de docencia; el Gini "
        "se calcula **solo con train** (80 %, estratificado), con los tramos supervisados de OptBinning y el nulo como tramo propio.")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1, st.container(border=True):
        st.markdown(":red-badge[Varianza casi nula]  \nUn valor concentra **≥ 99 %** de los datos no nulos (p. ej. los "
                    "`FLAG_DOCUMENT_*`). No hay variación que pueda separar clases.  \n**Decisión:** eliminar.")
    with c2, st.container(border=True):
        st.markdown(":orange-badge[Nulos altos sin poder]  \nMás de **50 %** de nulos **y** Gini < **0.08** (equivale a IV ≈ 0.02, "
                    "el inicio del rango «Weak»). Si el Gini lo compensa, el nulo es informativo y se conserva.  \n**Decisión:** eliminar.")
    with c3, st.container(border=True):
        st.markdown(":blue-badge[Alta cardinalidad]  \nMás de **15** categorías. Se **agrupan por tasa de default** (OptBinning "
                    "categórico, ≤ 5 grupos, ≥ 5 % por grupo). Si no hay grupos con default distinto, se elimina.  \n**Decisión:** agrupar.")

    fu["motivo_corto"] = np.select([fu["decision"].eq("Agrupa"), fu["motivo"].str.startswith("Varianza"),
                                    fu["motivo"].str.contains("no lo compensa"), fu["motivo"].str.contains("se conserva")],
                                   ["Agrupada por tasa de default", "Elimina · varianza casi nula", "Elimina · nulos sin poder",
                                    "Pasa · nulo informativo"], "Pasa")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Variables evaluadas", len(fu), border=True)
    k2.metric("Pasan al bivariado", int((fu["decision"] != "Elimina").sum()), border=True)
    k3.metric("Eliminadas · varianza", int(fu["motivo_corto"].eq("Elimina · varianza casi nula").sum()), border=True)
    k4.metric("Eliminadas · nulos", int(fu["motivo_corto"].eq("Elimina · nulos sin poder").sum()), border=True)

    DOM = ["Pasa", "Pasa · nulo informativo", "Agrupada por tasa de default", "Elimina · nulos sin poder", "Elimina · varianza casi nula"]
    RNG = [AZUL, "#7fb0ea", VERDE, NARANJA, "#c2410c"]
    f1, f2 = st.columns([3, 2], gap="large")
    base_p = alt.Chart(fu).encode(
        x=alt.X("pct_nulos:Q", title="% de nulos (train)", axis=PCT, scale=alt.Scale(domain=[0, 0.8])),
        y=alt.Y("gini:Q", title="Gini univariado (train)", scale=alt.Scale(type="sqrt", domain=[0, 0.32])))
    puntos = base_p.mark_circle(size=70, opacity=0.85, stroke="white", strokeWidth=0.8).encode(
        color=alt.Color("motivo_corto:N", scale=alt.Scale(domain=DOM, range=RNG), legend=alt.Legend(orient="top", title=None, columns=2, labelLimit=260)),
        tooltip=[alt.Tooltip("variable:N"), alt.Tooltip("motivo_corto:N", title="Decisión"), alt.Tooltip("pct_nulos:Q", format=".1%", title="% nulos"),
                 alt.Tooltip("gini:Q", format=".3f"), alt.Tooltip("pct_dominante:Q", format=".2%", title="% valor dominante")])
    reglas = (alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(strokeDash=[5, 4], color=GRIS).encode(x="x:Q")
              + alt.Chart(pd.DataFrame({"y": [0.08]})).mark_rule(strokeDash=[5, 4], color=GRIS).encode(y="y:Q"))
    zona = alt.Chart(pd.DataFrame({"x": [0.5], "x2": [0.8], "y": [0], "y2": [0.08]})).mark_rect(color=NARANJA, opacity=0.08).encode(
        x="x:Q", x2="x2:Q", y="y:Q", y2="y2:Q")
    f1.altair_chart((zona + reglas + puntos).properties(title={"text": "Nulos vs. poder discriminante",
                    "subtitle": "Zona sombreada: > 50 % de nulos y Gini < 0.08 → se elimina"}), width="stretch", height=420)
    cnt = fu.assign(fam=fu["familia"].map(fam_corta)).groupby(["fam", "familia", "motivo_corto"]).size().reset_index(name="n")
    f2.altair_chart(alt.Chart(cnt).mark_bar(height={"band": 0.72}).encode(
        y=alt.Y("fam:N", sort=cnt.sort_values("familia")["fam"].unique().tolist(), title=None, axis=alt.Axis(labelLimit=200, labelOverlap=False)),
        x=alt.X("n:Q", title="Variables", stack=True),
        color=alt.Color("motivo_corto:N", scale=alt.Scale(domain=DOM, range=RNG), legend=None),
        order=alt.Order("motivo_corto:N"),
        tooltip=[alt.Tooltip("fam:N", title="Familia"), alt.Tooltip("motivo_corto:N", title="Decisión"), alt.Tooltip("n:Q", title="Variables")])
        .properties(title="Decisión por familia"), width="stretch", height=420)
    nota("Lo que más se elimina son **documentos casi nunca entregados** (varianza nula) y las **medidas de vivienda e historial "
         "mensual del buró** con 55–72 % de nulos y Gini bajo. Ojo con el caso contrario: `HC_CARD_MAX_UTILIZACION` tiene 72 % de nulos "
         "pero **se conserva**, porque «no tener tarjeta» separa el riesgo (Gini 0.097). Un nulo no es malo por sí mismo.")

    st.markdown("**Decisión por variable**")
    sel_d = st.multiselect("Filtrar", DOM, default=["Agrupada por tasa de default", "Elimina · nulos sin poder", "Elimina · varianza casi nula",
                                                     "Pasa · nulo informativo"], key="f_uni")
    vf = fu[fu["motivo_corto"].isin(sel_d)] if sel_d else fu
    st.dataframe(vf.sort_values(["motivo_corto", "variable"])[["variable", "tipo", "familia", "pct_nulos", "pct_dominante", "gini", "motivo_corto", "motivo"]]
                 .assign(familia=lambda d: d["familia"].map(fam_corta)), hide_index=True, width="stretch", height=380,
                 column_config={"variable": st.column_config.TextColumn("Variable", pinned=True, width=230),
                                "tipo": st.column_config.TextColumn("Tipo", width=90), "familia": st.column_config.TextColumn("Familia", width=170),
                                "pct_nulos": st.column_config.ProgressColumn("% nulos", format="percent", min_value=0, max_value=1, width=110),
                                "pct_dominante": st.column_config.NumberColumn("% valor dominante", format="percent", width=110),
                                "gini": st.column_config.NumberColumn("Gini", format="%.3f", width=70),
                                "motivo_corto": st.column_config.TextColumn("Decisión", width=190),
                                "motivo": st.column_config.TextColumn("Detalle", width=420)})

    st.markdown("#### Agrupación por tasa de default (alta cardinalidad)")
    vg = st.segmented_control("Variable", sorted(gr["variable"].unique()), default=sorted(gr["variable"].unique())[0], key="var_gr")
    g = gr.query("variable == @vg").copy()
    resumen_g = g.groupby("grupo", as_index=False).agg(n=("n_grupo", "first"), rd=("rd_grupo", "first"), categorias=("categoria", "count"))
    resumen_g = resumen_g.sort_values("rd")
    orden_g = resumen_g["grupo"].tolist()
    q1, q2 = st.columns([2, 3], gap="large")
    base_g = alt.Chart(resumen_g).encode(x=alt.X("grupo:N", sort=orden_g, title=None, axis=alt.Axis(labelAngle=0)))
    barras = base_g.mark_bar(color="#c9c8c3", width={"band": 0.75}).encode(
        y=alt.Y("n:Q", title="Créditos (train)", axis=alt.Axis(titleColor=GRIS)),
        tooltip=[alt.Tooltip("grupo:N"), alt.Tooltip("n:Q", format=","), alt.Tooltip("categorias:Q", title="Categorías")])
    linea = base_g.mark_line(color=AZUL, strokeWidth=2.5, point=alt.OverlayMarkDef(size=80, filled=True, color=AZUL)).encode(
        y=alt.Y("rd:Q", title="Tasa de default", axis=alt.Axis(format="%", titleColor=AZUL)),
        tooltip=[alt.Tooltip("grupo:N"), alt.Tooltip("rd:Q", format=".2%", title="Default")])
    q1.altair_chart(alt.layer(barras, linea).resolve_scale(y="independent").properties(
        title={"text": f"{vg}: grupos resultantes", "subtitle": f"{g['categoria'].nunique()} categorías → {len(resumen_g)} grupos"}),
        width="stretch", height=340)
    g["grupo"] = pd.Categorical(g["grupo"], categories=orden_g, ordered=True)
    q2.altair_chart(alt.Chart(g[g["n"] > 0]).mark_circle(opacity=0.85, stroke="white").encode(
        x=alt.X("rd:Q", title="Tasa de default de la categoría", axis=PCT),
        y=alt.Y("grupo:N", sort=orden_g, title=None),
        size=alt.Size("n:Q", title="Créditos", scale=alt.Scale(range=[20, 900]), legend=None),
        color=alt.Color("grupo:N", sort=orden_g, scale=alt.Scale(scheme="blues"), legend=None),
        tooltip=[alt.Tooltip("categoria:N", title="Categoría"), alt.Tooltip("grupo:N"), alt.Tooltip("n:Q", format=","),
                 alt.Tooltip("rd:Q", format=".2%", title="Default categoría"), alt.Tooltip("rd_grupo:Q", format=".2%", title="Default grupo")])
        .properties(title={"text": "Cada categoría en su grupo", "subtitle": "Tamaño = créditos; pase el cursor para ver la categoría"}),
        width="stretch", height=340)
    with st.expander("Composición de cada grupo", icon=":material/list:"):
        comp = g.groupby("grupo", observed=True).agg(rd=("rd_grupo", "first"), n=("n_grupo", "first"),
                                                       categorias=("categoria", lambda x: ", ".join(sorted(x)))).reset_index()
        tabla_texto(comp.rename(columns={"grupo": "Grupo", "rd": "Default", "n": "Créditos", "categorias": "Categorías"}),
                    {"Default": "{:.2%}", "Créditos": "{:,}"})
    st.caption("Ejemplo de lectura de `ORGANIZATION_TYPE`: `XNA` (sin empleador, sobre todo pensionistas) cae en el grupo de menor "
               "default junto a bancos, policía y universidades; construcción, limpieza y restaurantes, en el de mayor. El nulo de "
               "`OCCUPATION_TYPE` queda como grupo propio «Sin dato».")

st.divider()
st.caption("Fuente: `artifacts/tablon_general.parquet` (ex-ante) y `artifacts/tablon_tratado.parquet` (ex-post). Metodología del "
           "notebook `notebooks/01_eda_univariado.ipynb` (Paso 4); tablas precalculadas con `scripts/build_univariado.py`.")
