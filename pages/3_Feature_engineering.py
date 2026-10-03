"""Feature engineering: indicadores del preprocesamiento, logaritmo de los montos y variables nuevas por criterio experto.

Es la etapa entre el preprocesamiento y los EDA: todo lo que se crea aquí pasa después por los filtros univariado,
bivariado y multivariado, igual que las variables del tablón.
"""
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

REPO_DIR = Path(__file__).resolve().parents[1]
ART = REPO_DIR / "artifacts"
FE = ART / "features"
MOD = ART / "modelado"

AZUL, NARANJA, VERDE, GRIS, TINTA, BARRA = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85", "#2b2b29", "#c9c8c3"
PCT = alt.Axis(format="%")
IV_MIN = 0.05


@st.cache_data(show_spinner=False)
def _leer(ruta: str, version: float) -> pd.DataFrame:
    return pd.read_parquet(ruta)


def t(nombre: str, carpeta: Path = FE) -> pd.DataFrame:
    """Lee un artefacto; la fecha de modificación entra en la llave del caché para que un redeploy no sirva datos viejos."""
    ruta = carpeta / f"{nombre}.parquet"
    return _leer(str(ruta), ruta.stat().st_mtime)


@st.cache_data(show_spinner=False)
def _json(ruta: str, version: float) -> dict:
    return json.load(open(ruta, encoding="utf-8"))


def nota(texto: str, icono: str = ":material/insights:") -> None:
    st.info(texto, icon=icono)


def tabla_texto(df: pd.DataFrame, formatos: dict | None = None) -> None:
    st.table(df.reset_index(drop=True).style.hide(axis="index").format(formatos or {}, na_rep="—"))


def fam_corta(f) -> str:
    return f.split("·", 1)[-1].strip() if isinstance(f, str) else ""


def grafico_tramos(d: pd.DataFrame, titulo: str, subtitulo: str = "", base: float | None = None, alto: int = 330):
    """Mismo gráfico que el bivariado: barras = créditos del tramo (eje izquierdo), línea = tasa de default (eje derecho)."""
    d = d.copy()
    d["etq"] = d["rd"].map(lambda x: f"{x:.1%}")
    orden = d["tramo"].tolist()
    ang = 0 if max(len(str(x)) for x in orden) <= 14 and len(orden) <= 6 else -30
    x = alt.X("tramo:N", sort=orden, title=None, axis=alt.Axis(labelAngle=ang, labelLimit=170, labelOverlap=False))
    tip = [alt.Tooltip("tramo:N", title="Tramo"), alt.Tooltip("n:Q", format=",", title="Créditos (train)"),
           alt.Tooltip("rd:Q", format=".2%", title="Tasa de default"), alt.Tooltip("woe:Q", format=".3f", title="WoE")]
    barras = alt.Chart(d).mark_bar(color=BARRA, size=max(14, min(56, 380 // max(len(d), 1)))).encode(
        x=x, y=alt.Y("n:Q", title="Número de créditos (conteo, train)", axis=alt.Axis(titleColor=GRIS, format="~s")), tooltip=tip)
    cuenta = alt.Chart(d).mark_text(baseline="top", dy=5, color="#5f5e5a", fontSize=10).encode(x=x, y="n:Q", text=alt.Text("n:Q", format=","))
    techo = float(max(d["rd"].max(), base or 0)) * 1.25
    esc = alt.Scale(domain=[0, techo])
    linea = alt.Chart(d).mark_line(color=AZUL, strokeWidth=2.6, point=alt.OverlayMarkDef(color=AZUL, size=70, filled=True)).encode(
        x=x, y=alt.Y("rd:Q", title="Proporción de TARGET (tasa de default)", scale=esc, axis=alt.Axis(format="%", titleColor=AZUL)), tooltip=tip)
    etq = alt.Chart(d).mark_text(dy=-13, color=AZUL, fontSize=11, fontWeight="bold").encode(x=x, y=alt.Y("rd:Q", scale=esc), text="etq:N")
    der = [linea, etq]
    if base is not None:
        der.append(alt.Chart(pd.DataFrame({"b": [base]})).mark_rule(color=GRIS, strokeDash=[2, 3]).encode(y=alt.Y("b:Q", scale=esc)))
    return alt.layer(alt.layer(barras, cuenta), alt.layer(*der)).resolve_scale(y="independent").properties(
        title=alt.TitleParams(titulo, subtitle=subtitulo, anchor="start", offset=10), height=alto,
        padding={"top": 24, "left": 5, "right": 5, "bottom": 5}, autosize=alt.AutoSizeParams(type="fit-x", contains="padding"))


# ── Datos ──────────────────────────────────────────────────────────────────────
cat = t("catalogo")
flags = t("flags")
logc = t("log_comparacion")
val = t("validaciones")
ruta_pf = ART / "parametros_features.json"
PF = _json(str(ruta_pf), ruta_pf.stat().st_mtime)
met = t("metricas_binning", MOD)
fu = t("filtro_univariado", MOD)
bi = t("bivariado_resumen", MOD)
sel = t("multivariado_seleccion", MOD)
bq = t("binning_qcut", MOD)
bo = t("binning_optb", MOD)
dsf = t("dataset_features", MOD)
BASE = float(bo.groupby("variable").apply(lambda d: d["k"].sum() / d["n"].sum()).iloc[0])

NUEVAS = cat[cat["origen"] != "Logaritmo de un monto"]["variable"].tolist()
LOGS = cat[cat["origen"] == "Logaritmo de un monto"]["variable"].tolist()
FINALES = set(sel.query("seleccionada")["variable"])


def recorrido(v: str) -> dict:
    """Qué le pasó a una variable en cada filtro del EDA."""
    r = {"univariado": "—", "iv": np.nan, "gini": np.nan, "bivariado": "—", "multivariado": "—", "final": False}
    f = fu.set_index("variable")
    if v in met["variable"].values:
        m_ = met.set_index("variable").loc[v]
        r["iv"], r["gini"] = float(m_["iv"]), float(m_["gini"])
    if v not in f.index:
        return r
    d, motivo = f.loc[v, "decision"], f.loc[v, "motivo"]
    if d == "Elimina":
        r["univariado"] = ("Sustituida por " + motivo.split("(")[0].replace("Sustituida por", "").strip()) if motivo.startswith("Sustituida") \
            else ("Elimina · varianza" if motivo.startswith("Varianza") else "Elimina · nulos sin poder")
        return r
    r["univariado"] = "Agrupada" if d == "Agrupa" else "Pasa"
    b = bi.set_index("variable")
    if v in b.index:
        r["bivariado"] = "Pasa (IV ≥ 0.05)" if bool(b.loc[v, "pasa"]) else "No pasa"
        if bool(b.loc[v, "pasa"]):
            s_ = sel.set_index("variable").loc[v]
            r["multivariado"] = "Seleccionada" if bool(s_["seleccionada"]) else f"Redundante con {s_['redundante_con']}"
            r["final"] = bool(s_["seleccionada"])
    return r


rec = pd.DataFrame([{"variable": v, **recorrido(v)} for v in cat["variable"]]).merge(
    cat[["variable", "familia", "bloque", "origen", "formula", "lectura", "pct_nulos"]], on="variable")
n_final_fe = int(rec["final"].sum())
n_sustituye = int(logc["gana_log"].sum())
n_base = int(len(fu) - len(cat))

# ── Encabezado ─────────────────────────────────────────────────────────────────
st.title("Feature engineering")
st.markdown(
    "Antes de los EDA, el tablón tratado se enriquece con **variables construidas**. Se combinan tres fuentes: los **indicadores** "
    "que dejó el preprocesamiento, la **transformación logarítmica** de los montos (recomendación de la profesora) y **variables "
    "nuevas por criterio experto**, pensadas desde la capacidad de pago y lo que ya mostraron los resultados. Reglas de la etapa:\n\n"
    "- **Sin fuga de información:** ninguna variable usa el `TARGET`, y todas se calculan fila a fila, igual en train y en test.\n"
    "- **Mismo tratamiento que las variables base:** las nuevas numéricas se capean en p0.1–p99.9 con límites aprendidos **solo en train**.\n"
    "- **Ninguna entra por decreto:** todas pasan por los filtros univariado, bivariado y multivariado, como cualquier otra variable.")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Indicadores (flags)", len(flags), border=True, help="Creados en la construcción del tablón y en el preprocesamiento.")
k2.metric("Logaritmos evaluados", len(LOGS), f"{n_sustituye} sustituyen al monto", delta_color="off", border=True,
          help="De cada par monto / logaritmo sigue uno solo: el que describe mejor el log-odds del default fuera de muestra.")
k3.metric("Variables nuevas", len(NUEVAS), border=True, help="Ratios y agregaciones por criterio experto.")
k4.metric("Variables para el EDA", len(fu), f"+{len(cat)}", delta_color="off", border=True,
          help=f"{n_base} explicativas del tablón tratado + {len(cat)} de feature engineering.")
k5.metric("Llegan al dataset final", n_final_fe, border=True, help="Variables de feature engineering que sobreviven a los tres filtros.")

tab_flag, tab_log, tab_new, tab_rec = st.tabs([":material/flag: Indicadores (flags)", ":material/functions: Logaritmo de los montos",
                                               ":material/add_circle: Variables nuevas", ":material/route: Recorrido por los filtros"])

# ══════════════════════════════════════════════════════════════════════════════
# 1. FLAGS
# ══════════════════════════════════════════════════════════════════════════════
with tab_flag:
    st.markdown(
        "Los indicadores ya existen en el tablón tratado: cuatro los creó el **preprocesamiento** para no imputar faltantes "
        "estructurales y dos vienen de la **construcción del tablón**. No son variables nuevas, pero son *feature engineering*: "
        "convierten una ausencia en información explícita. Se reportan aquí con su tasa de default **en train**.")
    fl = flags.copy()
    fl["dif"] = fl["rd_1"] - fl["rd_0"]
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        largo = fl.melt(id_vars=["variable"], value_vars=["rd_1", "rd_0"], var_name="cond", value_name="rd")
        largo["cond"] = largo["cond"].map({"rd_1": "Indicador = 1", "rd_0": "Indicador = 0"})
        orden_f = fl.sort_values("dif", ascending=False)["variable"].tolist()
        y_ = alt.Y("variable:N", sort=orden_f, title=None, axis=alt.Axis(labelLimit=240, labelOverlap=False, labelFontSize=11))
        reglas = alt.Chart(fl).mark_rule(color=BARRA, strokeWidth=3).encode(y=y_, x=alt.X("rd_0:Q"), x2="rd_1:Q")
        puntos = alt.Chart(largo).mark_circle(size=150, opacity=1, stroke="white", strokeWidth=1.5).encode(
            y=y_, x=alt.X("rd:Q", title="Tasa de default (train)", axis=PCT, scale=alt.Scale(domain=[0.04, 0.115])),
            color=alt.Color("cond:N", scale=alt.Scale(domain=["Indicador = 1", "Indicador = 0"], range=[NARANJA, AZUL]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("variable:N"), alt.Tooltip("cond:N", title="Condición"), alt.Tooltip("rd:Q", format=".2%", title="Default")])
        base_r = alt.Chart(pd.DataFrame({"b": [BASE]})).mark_rule(color=GRIS, strokeDash=[4, 4]).encode(x="b:Q")
        st.altair_chart((reglas + base_r + puntos).properties(
            title={"text": "Default con y sin la condición", "subtitle": f"Línea punteada: tasa base ({BASE:.1%})"}, height=330), width="stretch")
    with c2:
        sb, sp = fl.set_index("variable").loc["flag_sin_buro"], fl.set_index("variable").loc["flag_sin_previas"]
        se = fl.set_index("variable").loc["flag_sin_empleo"]
        st.markdown(
            f"- **Sin buró → más riesgo** ({sb.rd_1:.1%} vs {sb.rd_0:.1%}): no tener historial externo es una señal de riesgo.\n"
            f"- **Sin historial en Home Credit → menos riesgo** ({sp.rd_1:.1%} vs {sp.rd_0:.1%}): sorprende, pero quien vuelve a "
            "pedir suele ser quien más lo necesita.\n"
            f"- **Sin empleo → menos riesgo** ({se.rd_1:.1%} vs {se.rd_0:.1%}): el grupo es casi todo pensionistas, con ingreso estable.\n"
            "- Los pares `flag_sin_buro` / `TIENE_BUREAU` y `flag_sin_previas` / `TIENE_HISTORIAL_HOME_CREDIT` son **complementos "
            "exactos**: dicen lo mismo con el signo cambiado. El multivariado se queda con uno de cada par.")
    tabla_texto(fl.assign(familia=fl["familia"].map(fam_corta),
                          complemento_de=fl["complemento_de"].replace("", "—"))[
        ["variable", "origen", "familia", "regla", "pct_1", "rd_1", "rd_0", "iv", "complemento_de"]].rename(columns={
        "variable": "Indicador", "origen": "Origen", "familia": "Familia", "regla": "Regla", "pct_1": "% con 1",
        "rd_1": "Default si = 1", "rd_0": "Default si = 0", "iv": "IV", "complemento_de": "Complemento exacto de"}),
        {"% con 1": "{:.1%}", "Default si = 1": "{:.2%}", "Default si = 0": "{:.2%}", "IV": "{:.3f}"})
    ind = dsf[dsf["columna"].str.endswith("__nulo")]["columna"].tolist()
    st.markdown("**Indicadores de nulo del dataset final.** Se crean más adelante, al armar el dataset: SMOTE no acepta nulos, así "
                "que en esa versión cada numérica con nulos se imputa con la mediana de train **y** se acompaña de un indicador "
                f"que preserva la señal del faltante ({', '.join(f'`{c}`' for c in ind) or '—'}). En la vista WoE no hacen falta: el "
                "nulo ya es un tramo propio.")
    with st.expander("Recorrido de los indicadores por los filtros", icon=":material/route:"):
        rf = pd.DataFrame([{"variable": v, **recorrido(v)} for v in fl["variable"]])
        tabla_texto(rf[["variable", "univariado", "iv", "bivariado", "multivariado"]].rename(columns={
            "variable": "Indicador", "univariado": "Univariado", "iv": "IV", "bivariado": "Bivariado", "multivariado": "Multivariado"}),
            {"IV": "{:.3f}"})

# ══════════════════════════════════════════════════════════════════════════════
# 2. LOGARITMO
# ══════════════════════════════════════════════════════════════════════════════
with tab_log:
    st.markdown(
        "La profesora recomendó usar el **logaritmo de las variables de gran magnitud** (típicamente montos). ¿Ayuda? La respuesta "
        "depende de **qué modelo** recibe la variable, porque el logaritmo es una transformación **monótona**: cambia la escala, "
        "no el orden de los clientes.")
    a1, a2, a3 = st.columns(3, gap="medium")
    with a1, st.container(border=True, height="stretch"):
        st.markdown(":gray-badge[No cambia nada]  \n**Tramos, WoE, IV, Gini, Spearman y árboles.** Todos dependen solo del orden: "
                    "los mismos clientes caen en los mismos tramos. El IV del monto y el de su logaritmo son **idénticos**.")
    with a2, st.container(border=True, height="stretch"):
        st.markdown(":blue-badge[Cambia la forma]  \n**Asimetría y lectura.** Comprime la cola derecha: la distribución se vuelve "
                    "casi simétrica, la media deja de estar arrastrada por los montos altos y los gráficos se leen mejor.")
    with a3, st.container(border=True, height="stretch"):
        st.markdown(":orange-badge[Puede cambiar el modelo]  \n**Modelos que usan el valor crudo:** logística sin WoE, KNN, redes y "
                    "**SMOTE**, que interpola entre vecinos. Ahí importa en qué escala el riesgo es **lineal** en el log-odds.")

    tabla = logc.copy()
    tabla["sigue_txt"] = np.where(tabla["gana_log"], "Logaritmo", "Monto original")
    st.markdown("#### Evidencia por variable")
    st.dataframe(tabla[["variable", "asim_raw", "asim_log", "iv_raw", "iv_log", "r2mcf_raw", "r2mcf_log",
                        "r2_logit_raw", "r2_logit_log", "sigue_txt"]], hide_index=True, width="stretch",
                 column_config={"variable": st.column_config.TextColumn("Monto", pinned=True, width=190),
                                "asim_raw": st.column_config.NumberColumn("Asimetría", format="%.2f"),
                                "asim_log": st.column_config.NumberColumn("Asim. (log)", format="%.2f"),
                                "iv_raw": st.column_config.NumberColumn("IV", format="%.4f"),
                                "iv_log": st.column_config.NumberColumn("IV (log)", format="%.4f"),
                                "r2mcf_raw": st.column_config.NumberColumn("R² McF.", format="%.4f",
                                                                           help="Logística univariada con el monto: ajuste en train, evaluación en test."),
                                "r2mcf_log": st.column_config.NumberColumn("R² McF. (log)", format="%.4f"),
                                "r2_logit_raw": st.column_config.NumberColumn("Linealidad", format="%.2f",
                                                                              help="R² entre el log-odds de cada decil y su valor medio: 1 = relación lineal."),
                                "r2_logit_log": st.column_config.NumberColumn("Linealidad (log)", format="%.2f"),
                                "sigue_txt": st.column_config.TextColumn("Sigue", width=120, help="Cuál del par pasa a los filtros del EDA.")})
    st.caption("**R² de McFadden** = 1 − log-loss del modelo / log-loss de predecir siempre la tasa base, en test. **Linealidad** = R² de "
               "una recta entre el log-odds de cada decil de train y el valor medio del decil. El AUC de una logística univariada es el "
               "mismo en las dos escalas (solo depende del orden); por eso se compara el **ajuste**, no la discriminación.")

    l1, l2 = st.columns([2, 3], vertical_alignment="bottom")
    var_l = l1.selectbox("Monto", tabla["variable"].tolist(), index=tabla["variable"].tolist().index(
        tabla.loc[tabla["gana_log"], "variable"].iloc[0]) if tabla["gana_log"].any() else 0, key="var_log")
    fila = tabla.set_index("variable").loc[var_l]
    l2.caption(f"`{var_l}` → `{fila['log']}` = {fila['transformacion'].replace('x', var_l)}")
    hl = t("log_hist").query("variable == @var_l")
    dl = t("log_deciles").query("variable == @var_l")
    cl = t("log_curva").query("variable == @var_l")
    g1, g2 = st.columns(2, gap="large")
    for col_, esc, nombre, color in [(g1, "raw", var_l, GRIS), (g2, "log", fila["log"], AZUL)]:
        with col_:
            h = hl[hl["escala"] == esc]
            col_.altair_chart(alt.Chart(h).mark_bar(color=color, binSpacing=0, opacity=0.85).encode(
                x=alt.X("desde:Q", bin="binned", title=nombre, axis=alt.Axis(format="~s")), x2="hasta:Q",
                y=alt.Y("densidad:Q", title="Densidad", axis=alt.Axis(labels=False, ticks=False)),
                tooltip=[alt.Tooltip("desde:Q", format=",.4g", title="Desde"), alt.Tooltip("hasta:Q", format=",.4g", title="Hasta")])
                .properties(title={"text": f"Distribución {'del monto' if esc == 'raw' else 'en logaritmo'}",
                                   "subtitle": f"Asimetría {fila['asim_' + esc]:.2f} (train, vista p0.5–p99.5)"}, height=230), width="stretch")
            pts = dl.assign(x=dl[f"x_{esc}"])
            cv = cl[cl["escala"] == esc]
            col_.altair_chart((alt.Chart(cv).mark_line(color=NARANJA, strokeWidth=2).encode(
                x=alt.X("x:Q", title=f"{nombre} (media del decil)", scale=alt.Scale(zero=False), axis=alt.Axis(format="~s")),
                y=alt.Y("logit:Q", title="log-odds del default", scale=alt.Scale(zero=False)))
                + alt.Chart(pts).mark_circle(size=90, color=color, opacity=1, stroke="white", strokeWidth=1).encode(
                    x="x:Q", y="logit:Q",
                    tooltip=[alt.Tooltip("decil:Q", title="Decil"), alt.Tooltip("x:Q", format=",.4g", title="Media"),
                             alt.Tooltip("rd:Q", format=".2%", title="Default"), alt.Tooltip("n:Q", format=",", title="Créditos")]))
                .properties(title={"text": "¿Es lineal el riesgo en esta escala?",
                                   "subtitle": f"Puntos: deciles de train · línea: logística univariada · linealidad {fila['r2_logit_' + esc]:.2f}"},
                            height=250), width="stretch")
    gana = tabla[tabla["gana_log"]]["variable"].tolist()
    pierde = tabla[~tabla["gana_log"]]["variable"].tolist()
    st.success(
        f"**Decisión: el logaritmo se usa donde la evidencia lo respalda.** De cada par monto / logaritmo sigue **uno solo** (son "
        "redundantes perfectos: ρ de Spearman = 1). Sigue el **logaritmo** en " + (", ".join(f"`{v}`" for v in gana) or "ninguno") +
        ": montos con masa en cero y cola muy larga, donde la relación con el log-odds se vuelve casi lineal. Sigue el **monto "
        "original** en " + (", ".join(f"`{v}`" for v in pierde) or "ninguno") + ": tras el capeo su asimetría ya es moderada y el "
        "logaritmo no mejora el ajuste (en algunos lo empeora). Para la logística sobre WoE y los árboles la elección es **neutra**.",
        icon=":material/check_circle:")
    nota("**Para discutir con la profesora.** Si la convención del curso es «todo monto en logaritmo», basta cambiar "
         "`CRITERIO_LOG = \"siempre\"` en `scripts/pipeline_modelado.py`: el IV y la selección no cambian, solo la escala que verá "
         "SMOTE. Y un hallazgo lateral: SMOTE-NC busca vecinos con distancia euclidiana **sin estandarizar**, así que las variables "
         "en miles (`DAYS_*`, montos) dominan esa distancia. El logaritmo lo atenúa, pero lo correcto sería **estandarizar** antes "
         "de rebalancear.", icono=":material/forum:")

# ══════════════════════════════════════════════════════════════════════════════
# 3. VARIABLES NUEVAS
# ══════════════════════════════════════════════════════════════════════════════
with tab_new:
    st.markdown(
        "Variables construidas con **criterio experto**: cada una responde una pregunta de negocio que ninguna variable del tablón "
        "responde sola. Se organizan en las mismas **familias** del tablón (las fuentes de Home Credit), para que el resto del análisis "
        "las trate igual.")
    nv = rec[rec["variable"].isin(NUEVAS)].copy()
    fams = nv.groupby("familia").size().reset_index(name="n")
    cols_f = st.columns(3, gap="small") + st.columns(3, gap="small")
    for col_, (_, fr) in zip(cols_f, fams.iterrows()):
        n_ok = int(nv[(nv["familia"] == fr["familia"]) & nv["final"]].shape[0])
        col_.metric(fam_corta(fr["familia"]), int(fr["n"]), f"{n_ok} al dataset final", delta_color="off", border=True)

    st.dataframe(nv.assign(familia=nv["familia"].map(fam_corta))[
        ["variable", "familia", "iv", "univariado", "bivariado", "multivariado", "pct_nulos", "formula", "lectura"]],
        hide_index=True, width="stretch", height=38 + 35 * len(nv),
        column_config={"variable": st.column_config.TextColumn("Variable", pinned=True, width=230),
                       "familia": st.column_config.TextColumn("Familia", width=140),
                       "formula": st.column_config.TextColumn("Fórmula", width=260),
                       "lectura": st.column_config.TextColumn("Pregunta de negocio", width=420),
                       "pct_nulos": st.column_config.ProgressColumn("% nulos", format="percent", min_value=0, max_value=1, width=90),
                       "iv": st.column_config.NumberColumn("IV", format="%.3f", width=60),
                       "univariado": st.column_config.TextColumn("Univariado", width=130),
                       "bivariado": st.column_config.TextColumn("Bivariado", width=120),
                       "multivariado": st.column_config.TextColumn("Multivariado", width=200)})

    st.markdown("#### Ficha por variable")
    ops = nv.sort_values("iv", ascending=False)["variable"].tolist()
    var = st.selectbox("Variable nueva (ordenadas por IV)", ops, key="var_fe",
                       format_func=lambda v: f"{v}  ·  IV {rec.set_index('variable').loc[v, 'iv']:.3f}")
    rv, cv_ = rec.set_index("variable").loc[var], cat.set_index("variable").loc[var]
    fam_txt = fam_corta(rv["familia"]) + ("" if rv["bloque"] == fam_corta(rv["familia"]) else f" · {rv['bloque']}")
    st.caption(f"{fam_txt} · `{rv['formula']}`")
    st.markdown(rv["lectura"])
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("IV (OptBinning)", f"{rv['iv']:.3f}", help="Criterio del bivariado: IV ≥ 0.05.")
    m2.metric("Gini", f"{rv['gini']:.3f}", help="Informativo.")
    m3.metric("% nulos", f"{rv['pct_nulos']:.1%}")
    m4.metric("ρ con su insumo", f"{cv_['rho_max_insumo']:.2f}" if pd.notna(cv_.get("rho_max_insumo")) else "—",
              help=f"Spearman con el insumo más parecido ({cv_.get('insumo_mas_correlacionado', '—')}). Cerca de 1 = aporta poco nuevo.")
    m5.metric("Capeados (p0.1–p99.9)", f"{int(cv_['n_capeados']):,}")
    tipo_v = met.set_index("variable").loc[var, "tipo"] if var in met["variable"].values else "numérica"
    q_ = bq[(bq["variable"] == var) & (bq["q"] == (5 if tipo_v == "numérica" else 0))].sort_values("orden")
    o_ = bo[bo["variable"] == var].sort_values("orden")
    tram = st.segmented_control("Trameado", ["qcut 5" if tipo_v == "numérica" else "Categorías", "OptBinning"],
                                default="qcut 5" if tipo_v == "numérica" else "Categorías", key="tram_fe")
    d_ = o_ if tram == "OptBinning" else q_
    f1, f2 = st.columns([3, 2], gap="large")
    with f1:
        if len(d_):
            st.altair_chart(grafico_tramos(d_[d_["n"] > 0], f"{var} vs TARGET", f"{tram} · train · línea gris: tasa base", BASE),
                            width="stretch")
    with f2:
        st.markdown("**Recorrido por los filtros**")
        pasos = [("Univariado", rv["univariado"]), ("Bivariado", rv["bivariado"]), ("Multivariado", rv["multivariado"])]
        for paso, res in pasos:
            color = "green" if res.startswith(("Pasa", "Seleccionada", "Agrupada")) else ("gray" if res == "—" else "orange")
            st.markdown(f"**{paso}** · :{color}-badge[{res}]")
        st.markdown(":green-badge[Llega al dataset final]" if rv["final"] else ":gray-badge[No llega al dataset final]")
        if not rv["final"] and str(rv["multivariado"]).startswith("Redundante"):
            st.caption("Tiene señal, pero repite la de una variable con más IV: el multivariado se queda con una sola.")
        elif rv["bivariado"] == "No pasa":
            st.caption("Su IV no llega a 0.05: la idea de negocio es razonable, pero en estos datos no separa lo suficiente.")

    with st.expander("Ejemplo de cálculo: insumos → variables nuevas (primeros créditos de train)", icon=":material/calculate:"):
        mu = t("muestra")
        st.dataframe(mu, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(c, format="%.3f") for c in mu.columns
                                    if c not in ("SK_ID_CURR",) and mu[c].dtype.kind == "f" and mu[c].abs().max() < 100}
                     | {"SK_ID_CURR": st.column_config.NumberColumn("SK_ID_CURR", format="%d", pinned=True)})

# ══════════════════════════════════════════════════════════════════════════════
# 4. RECORRIDO
# ══════════════════════════════════════════════════════════════════════════════
with tab_rec:
    st.markdown(
        "Crear variables es fácil; que **aporten** es otra cosa. Esta vista sigue a todas las variables de feature engineering "
        "(nuevas y logaritmos) por los tres filtros del EDA y las compara con las variables del tablón.")
    etapas = ["Creadas", "Pasan el univariado", "Pasan el bivariado (IV ≥ 0.05)", "Dataset final"]
    vals = [len(rec), int(rec["univariado"].isin(["Pasa", "Agrupada"]).sum()), int(rec["bivariado"].eq("Pasa (IV ≥ 0.05)").sum()), n_final_fe]
    emb = pd.DataFrame({"etapa": etapas, "n": vals})
    e1, e2 = st.columns([2, 3], gap="large")
    with e1:
        be = alt.Chart(emb).encode(y=alt.Y("etapa:N", sort=etapas, title=None, axis=alt.Axis(labelLimit=220)))
        st.altair_chart((be.mark_bar(color=AZUL, height={"band": 0.62}, cornerRadiusEnd=3).encode(
            x=alt.X("n:Q", title="Variables de feature engineering"), tooltip=[alt.Tooltip("etapa:N"), alt.Tooltip("n:Q", title="Variables")])
            + be.mark_text(align="left", dx=5, fontSize=12, color=TINTA).encode(x="n:Q", text="n:Q"))
            .properties(title="Embudo de las variables de feature engineering", height=220), width="stretch")
    with e2:
        comp = met[["variable", "iv"]].merge(fu[["variable", "decision"]], on="variable", how="left")
        comp = comp[comp["decision"] != "Elimina"].copy()
        comp["grupo"] = np.where(comp["variable"].isin(cat["variable"]), "Feature engineering", "Tablón")
        comp["iv_plot"] = comp["iv"].clip(lower=0.001)
        comp["final"] = np.where(comp["variable"].isin(FINALES), "En el dataset final", "Fuera")
        st.altair_chart((alt.Chart(comp).mark_tick(thickness=2, size=26).encode(
            x=alt.X("iv_plot:Q", title="IV (escala log)", scale=alt.Scale(type="log", domain=[0.001, 1])),
            y=alt.Y("grupo:N", title=None, sort=["Tablón", "Feature engineering"], axis=alt.Axis(labelLimit=200, labelFontSize=11)),
            color=alt.Color("final:N", scale=alt.Scale(domain=["En el dataset final", "Fuera"], range=[AZUL, BARRA]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("variable:N"), alt.Tooltip("iv:Q", format=".3f"), alt.Tooltip("final:N", title="Resultado")])
            + alt.Chart(pd.DataFrame({"x": [IV_MIN]})).mark_rule(color=NARANJA, strokeDash=[5, 4]).encode(x="x:Q"))
            .properties(title={"text": "IV de las variables que llegan al bivariado", "subtitle": "Línea naranja: corte IV = 0.05"},
                        height=200), width="stretch")
    st.dataframe(rec.assign(familia=rec["familia"].map(fam_corta)).sort_values(["final", "iv"], ascending=[False, False])[
        ["variable", "origen", "familia", "iv", "univariado", "bivariado", "multivariado", "final"]],
        hide_index=True, width="stretch", height=38 + 35 * len(rec),
        column_config={"variable": st.column_config.TextColumn("Variable", pinned=True, width=230),
                       "origen": st.column_config.TextColumn("Origen", width=130),
                       "familia": st.column_config.TextColumn("Familia", width=130),
                       "iv": st.column_config.NumberColumn("IV", format="%.3f", width=60),
                       "univariado": st.column_config.TextColumn("Univariado", width=210),
                       "bivariado": st.column_config.TextColumn("Bivariado", width=110),
                       "multivariado": st.column_config.TextColumn("Multivariado", width=250),
                       "final": st.column_config.CheckboxColumn("Dataset final", width=100)})
    st.markdown("#### Validaciones de la etapa")
    v_ = val.copy()
    v_["estado"] = np.where(v_["ok"], "✔ Correcto", "✖ Revisar")
    tabla_texto(v_[["estado", "validacion"]].rename(columns={"estado": "Estado", "validacion": "Validación"}))

st.divider()
st.caption("Fuente: `scripts/pipeline_modelado.py features` (sobre `artifacts/tablon_tratado.parquet`) → `artifacts/tablon_features.parquet`, "
           "`artifacts/parametros_features.json` y `artifacts/features/`. El recorrido por los filtros sale de `artifacts/modelado/`.")
