"""EDA bivariado (Pasos 5 y 6.1–6.4 de la guía): relación de cada variable con el TARGET y filtro único por IV ≥ 0.05."""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

REPO_DIR = Path(__file__).resolve().parents[1]
MOD = REPO_DIR / "artifacts" / "modelado"

AZUL, NARANJA, VERDE, GRIS, TINTA, BARRA = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85", "#2b2b29", "#c9c8c3"
PCT = alt.Axis(format="%")
IV_MIN, IV_SOSP, GINI_SOSP = 0.05, 0.50, 0.38
RANGOS_IV = ["< 0.02 · Not good", "0.02–0.1 · Weak", "0.1–0.3 · Medium", "0.3–0.5 · Strong", "≥ 0.5 · Revisar sobreajuste"]
RANGOS_GINI = ["< 0.08 · Sin poder", "0.08–0.18 · Débil", "0.18–0.30 · Medio", "0.30–0.38 · Fuerte", "≥ 0.38 · Revisar sobreajuste"]
COL_RANGO = ["#c9c8c3", "#9db8d9", "#2a78d6", "#1f4e8c", NARANJA]


@st.cache_data(show_spinner=False)
def _leer(ruta: str, version: float) -> pd.DataFrame:
    return pd.read_parquet(ruta)


def t(nombre: str) -> pd.DataFrame:
    """Lee un artefacto; la fecha de modificación entra en la llave del caché para que un redeploy no sirva datos viejos."""
    ruta = MOD / f"{nombre}.parquet"
    return _leer(str(ruta), ruta.stat().st_mtime)


def nota(texto: str, icono: str = ":material/insights:") -> None:
    st.info(texto, icon=icono)


def tabla_texto(df: pd.DataFrame, formatos: dict | None = None) -> None:
    st.table(df.reset_index(drop=True).style.hide(axis="index").format(formatos or {}, na_rep="—"))


def fam_corta(f) -> str:
    return f.split("·", 1)[-1].strip() if isinstance(f, str) else ""


def grafico_guia(d: pd.DataFrame, titulo: str, subtitulo: str = "", con_test: bool = False, base: float | None = None,
                 alto: int = 360, etiquetas: bool = True, angulo: int | None = None, compacto: bool = False):
    """El gráfico de la guía: barras = frecuencia del tramo (eje izquierdo, gris) y línea = proporción de TARGET (eje derecho, azul)."""
    d = d.copy()
    d["etq"] = d["rd"].map(lambda x: f"{x:.1%}")
    orden = d["tramo"].tolist()
    largo_max = max(len(str(x)) for x in orden)
    ang = angulo if angulo is not None else (0 if largo_max <= 14 and len(orden) <= 6 else -35)
    x = alt.X("tramo:N", sort=orden, title=None, axis=alt.Axis(labelAngle=ang, labelLimit=75 if compacto else 180, labelOverlap=False,
                                                                     labelFontSize=9 if compacto else 11))
    tooltip = [alt.Tooltip("tramo:N", title="Tramo"), alt.Tooltip("n:Q", format=",", title="Créditos (train)"),
               alt.Tooltip("pct:Q", format=".1%", title="% de la muestra"), alt.Tooltip("k:Q", format=",", title="Defaults"),
               alt.Tooltip("rd:Q", format=".2%", title="Tasa de default (train)"), alt.Tooltip("rd_test:Q", format=".2%", title="Tasa de default (test)"),
               alt.Tooltip("woe:Q", format=".3f", title="WoE"), alt.Tooltip("iv:Q", format=".4f", title="Aporte al IV")]
    marca = alt.MarkDef(type="bar", color=BARRA) if compacto else alt.MarkDef(type="bar", color=BARRA, size=max(14, min(60, 420 // max(len(d), 1))))
    barras = alt.Chart(d, mark=marca).encode(
        x=x, y=alt.Y("n:Q", title=None if compacto else "Frecuencia (créditos en train)", axis=alt.Axis(titleColor=GRIS, format="~s")), tooltip=tooltip)
    techo = float(max(d["rd"].max(), d["rd_test"].max() if con_test else 0, base or 0)) * 1.25
    esc_y = alt.Scale(domain=[0, techo])
    linea = alt.Chart(d).mark_line(color=AZUL, strokeWidth=2.6, point=alt.OverlayMarkDef(color=AZUL, size=70, filled=True)).encode(
        x=x, y=alt.Y("rd:Q", title=None if compacto else "Proporción de TARGET (tasa de default)", scale=esc_y, axis=alt.Axis(format="%", titleColor=AZUL)),
        tooltip=tooltip)
    capas_der = [linea]
    if etiquetas:
        capas_der.append(alt.Chart(d).mark_text(dy=-13, color=AZUL, fontSize=11, fontWeight="bold").encode(x=x, y=alt.Y("rd:Q", scale=esc_y), text="etq:N"))
    if con_test:
        capas_der.append(alt.Chart(d).mark_line(color=NARANJA, strokeWidth=1.8, strokeDash=[5, 4],
                                                point=alt.OverlayMarkDef(color=NARANJA, size=30, filled=True)).encode(
            x=x, y=alt.Y("rd_test:Q", scale=esc_y), tooltip=tooltip))
    if base is not None:
        capas_der.append(alt.Chart(pd.DataFrame({"b": [base]})).mark_rule(color=GRIS, strokeDash=[2, 3]).encode(y=alt.Y("b:Q", scale=esc_y)))
    graf = alt.layer(barras, alt.layer(*capas_der)).resolve_scale(y="independent")
    return graf.properties(title=alt.TitleParams(titulo, subtitle=subtitulo or "", anchor="start", offset=10),
                           height=alto, padding={"top": 24, "left": 5, "right": 5, "bottom": 5},
                           autosize=alt.AutoSizeParams(type="fit-x", contains="padding"))


met = t("metricas_binning")
bi = t("bivariado_resumen")
fu = t("filtro_univariado")
bq = t("binning_qcut")
bo = t("binning_optb")
BASE = float(bo.groupby("variable").apply(lambda d: d["k"].sum() / d["n"].sum()).iloc[0])

# ── Encabezado ─────────────────────────────────────────────────────────────────
st.title("EDA bivariado")
st.markdown(
    "Cómo se relaciona cada variable con el `TARGET`. Siguiendo la guía, cada variable se **discretiza** y se compara la "
    "**tasa de default por tramo**: `qcut` en 5 tramos para las continuas (Paso 6.1) y *binning* supervisado con **OptBinning** "
    "(Paso 6.2), que es el que mide el poder predictivo. Todo se aprende **solo con train** (80 %); test (20 %) sirve para "
    "comprobar que el patrón se repite. El filtro (Pasos 6.3–6.4) usa **un único criterio para todos los modelos: IV ≥ 0.05**, "
    "porque el entrenamiento parte de **un solo dataset**. El Gini se sigue reportando como información."
)
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Variables evaluadas", len(bi), border=True, help="Las que pasaron el filtro univariado.")
k2.metric("Pasan (IV ≥ 0.05)", int(bi["pasa"].sum()), border=True, help="Criterio único para la logística y los modelos de ML.")
k3.metric("IV ≥ 0.1 (Medium o más)", int((bi["iv"] >= 0.1).sum()), border=True, help="Cuántas pasarían con el umbral anterior.")
k4.metric("IV ≥ 0.5 (revisar)", int((bi["iv"] >= IV_SOSP).sum()), border=True, help="Posible sobreajuste o fuga de información.")
k5.metric("Inestables train/test", int((~bi["estable"]).sum()), border=True,
          help="Gini de test < 50 % del de train u orden de los tramos distinto (ρ < 0.5). Informativo: ninguna pasa el IV.")

tab_var, tab_rank, tab_pat, tab_sel = st.tabs([":material/bar_chart: Variable vs. TARGET", ":material/leaderboard: Ranking IV y Gini",
                                               ":material/timeline: Patrones", ":material/fact_check: Selección"])

# ══════════════════════════════════════════════════════════════════════════════
# 1. VARIABLE VS TARGET
# ══════════════════════════════════════════════════════════════════════════════
with tab_var:
    f1, f2, f3 = st.columns([5, 3, 4], vertical_alignment="bottom")
    filtro_ds = f2.segmented_control("Mostrar", ["Todas", "Pasan (IV ≥ 0.05)"], default="Todas", key="ds_var")
    lista = bi.copy()
    if filtro_ds == "Pasan (IV ≥ 0.05)":
        lista = lista[lista["pasa"]]
    lista = lista.sort_values("iv", ascending=False)
    ops = lista["variable"].tolist()
    defecto = "EXT_SOURCE_3" if "EXT_SOURCE_3" in ops else ops[0]
    var = f1.selectbox("Variable (ordenadas por IV)", ops, index=ops.index(defecto), key="var_bi",
                       format_func=lambda v: f"{v}  ·  IV {lista.set_index('variable').loc[v, 'iv']:.3f}")
    r = bi.set_index("variable").loc[var]
    es_num = r["tipo"] == "numérica"
    opciones_tram = ["qcut 5", "qcut 10", "OptBinning"] if es_num else ["Categorías", "OptBinning"]
    tram = f3.segmented_control("Trameado", opciones_tram, default=opciones_tram[0], key="tram_bi")
    tram = tram or opciones_tram[0]
    if isinstance(r.get("descripcion"), str) and r["descripcion"]:
        st.caption(f"{fam_corta(r['familia'])} · {r['descripcion']}")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("IV (OptBinning)", f"{r['iv']:.3f}", help="Information Value con los tramos de OptBinning en train: el criterio de selección.")
    m2.metric("Gini (OptBinning)", f"{r['gini']:.3f}", help="2·AUC − 1 usando la tasa de default del tramo como score. Informativo.")
    m3.metric("IV qcut 5", f"{r['iv_q5']:.3f}", help="IV con el trameado básico de la guía (no supervisado).")
    m4.metric("Gini en test", f"{r['gini_test']:.3f}", f"{r['gini_test'] - r['gini']:+.3f}", delta_color="off")
    m5.markdown(f"**Decisión (IV ≥ 0.05)**  \n{':green-badge[Pasa]' if r['pasa'] else ':gray-badge[No pasa]'}  \n{r['rango_iv']}")
    m6.markdown(f"**Gini (informativo)**  \n{r['rango_gini']}")

    if tram == "OptBinning":
        d = bo.query("variable == @var and n > 0").copy()
        sub = "Tramos supervisados (OptBinning, ≤ 5 tramos, ≥ 5 % por tramo, ajustado en train)"
    else:
        q = 5 if tram == "qcut 5" else (10 if tram == "qcut 10" else 0)
        d = bq.query("variable == @var and q == @q").copy()
        if not es_num and len(d) > 15:
            top = d.sort_values("n", ascending=False).head(14)
            resto = d[~d["tramo"].isin(top["tramo"])]
            otras = {"tramo": f"Otras ({len(resto)} categorías)", "n": resto["n"].sum(), "k": resto["k"].sum(),
                     "n_test": resto["n_test"].sum(), "k_test": resto["k_test"].sum(), "iv": resto["iv"].sum()}
            otras.update({"pct": otras["n"] / d["n"].sum(), "rd": otras["k"] / otras["n"], "woe": np.nan,
                          "rd_test": otras["k_test"] / max(otras["n_test"], 1)})
            d = pd.concat([top.sort_values("rd"), pd.DataFrame([otras])], ignore_index=True)
        sub = ("Trameado básico de la guía: pd.qcut sobre train; test con los mismos cortes" if es_num else
               "Categorías originales (sin agrupar), como en el ejemplo de la guía")
    con_test = st.toggle("Superponer la tasa de default en test (línea naranja punteada)", key="test_bi")
    g1, g2 = st.columns([3, 2], gap="large")
    g1.altair_chart(grafico_guia(d, f"Gráfico de barras para {var} vs TARGET", sub, con_test=con_test, base=BASE),
                    width="stretch")
    with g2:
        patron = r["patron"]
        st.markdown(f"**Patrón (qcut 5):** {patron}")
        lect = []
        if r["iv"] >= IV_SOSP:
            lect.append("IV ≥ 0.5: **demasiado bueno**; revisar fuga de información antes de usarla.")
        if patron.startswith("Monótona") or patron.startswith("Casi"):
            sentido = "baja" if "decreciente" in patron else "sube"
            lect.append(f"La tasa de default **{sentido}** a medida que crece la variable: relación ordenada, fácil de usar en una logística.")
        elif patron.startswith("No lineal"):
            lect.append("Relación **no lineal**: el riesgo cambia de dirección. La logística la captura gracias a WoE; los árboles, "
                        "por construcción.")
        elif patron == "Sin patrón claro":
            lect.append("La tasa de default casi no cambia entre tramos: **sin señal** univariada.")
        sd = d[d["tramo"] == "Sin dato"]
        if len(sd) and sd["n"].iloc[0] > 0:
            lect.append(f"El tramo **Sin dato** ({sd['pct'].iloc[0]:.1%} de train) tiene {sd['rd'].iloc[0]:.1%} de default: "
                        f"{'más' if sd['rd'].iloc[0] > BASE else 'menos'} que la base ({BASE:.1%}). El nulo informa.")
        rango = d["rd"].max() - d["rd"].min()
        lect.append(f"Entre el tramo más riesgoso y el menos riesgoso hay **{rango * 100:.1f} p.p.** de diferencia en la tasa de default.")
        st.markdown("\n".join(f"- {x}" for x in lect))
        if var == "DAYS_EMPLOYED":
            st.warning("El último tramo junta a quienes llevan poco tiempo en su empleo con el **0** del centinela (pensionistas, "
                       "default 5.4 %): por eso la curva baja al final. Es el efecto que se discutió en el univariado.",
                       icon=":material/warning:")
        if var == "CODE_GENDER":
            st.warning(f"Con IV {r['iv']:.3f} no llega a 0.05 y queda fuera del dataset. Igual conviene recordar que usar el género en "
                       "una decisión de crédito es discutible regulatoriamente.", icon=":material/gavel:")

    tabla = d[["tramo", "n", "pct", "k", "rd", "woe", "iv", "n_test", "rd_test"]].copy()
    if tram == "OptBinning" and "miembros" in d and d["miembros"].notna().any():
        tabla["miembros"] = d["miembros"].fillna("")
    st.dataframe(tabla, hide_index=True, width="stretch",
                 column_config={"tramo": st.column_config.TextColumn("Tramo", width=220),
                                "n": st.column_config.NumberColumn("Créditos", format="localized"),
                                "pct": st.column_config.ProgressColumn("% muestra", format="percent", min_value=0, max_value=1),
                                "k": st.column_config.NumberColumn("Defaults", format="localized"),
                                "rd": st.column_config.NumberColumn("Tasa default", format="percent"),
                                "woe": st.column_config.NumberColumn("WoE", format="%.3f", help="ln(% buenos / % malos): > 0 menos riesgo."),
                                "iv": st.column_config.NumberColumn("Aporte IV", format="%.4f"),
                                "n_test": st.column_config.NumberColumn("Créditos test", format="localized"),
                                "rd_test": st.column_config.NumberColumn("Tasa test", format="percent"),
                                "miembros": st.column_config.TextColumn("Categorías del tramo", width=380)})

    if es_num:
        bc = t("box_clase").query("variable == @var")
        if len(bc):
            st.markdown("**Separación de clases** (guía, Paso 5.2.1: boxplot por clase · bigotes p5–p95, caja p25–p75, | mediana, ◆ media)")
            yb = alt.Y("clase:N", title=None, sort=["0 · paga", "1 · default"])
            colc = alt.Color("clase:N", scale=alt.Scale(domain=["0 · paga", "1 · default"], range=[AZUL, NARANJA]), legend=None)
            caja = (alt.Chart(bc).mark_rule(strokeWidth=1.5).encode(x=alt.X("p5:Q", title=var, scale=alt.Scale(zero=False)), x2="p95:Q", y=yb, color=colc)
                    + alt.Chart(bc).mark_bar(size=34, opacity=0.35).encode(x="p25:Q", x2="p75:Q", y=yb, color=colc,
                                                                            tooltip=[alt.Tooltip("clase:N"), alt.Tooltip("p25:Q", format=",.4g"),
                                                                                     alt.Tooltip("p50:Q", format=",.4g", title="Mediana"),
                                                                                     alt.Tooltip("p75:Q", format=",.4g"), alt.Tooltip("media:Q", format=",.4g"),
                                                                                     alt.Tooltip("n:Q", format=",", title="Créditos con dato")])
                    + alt.Chart(bc).mark_tick(thickness=3, size=34, color=TINTA).encode(x="p50:Q", y=yb)
                    + alt.Chart(bc).mark_point(shape="diamond", filled=True, size=90, color=TINTA).encode(x="media:Q", y=yb))
            st.altair_chart(caja.properties(height=150), width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# 2. RANKING
# ══════════════════════════════════════════════════════════════════════════════
with tab_rank:
    st.markdown("**¿Cómo se lee cada métrica?** El IV decide; el Gini acompaña, en una escala alineada con la del IV.")
    e1, e2 = st.columns(2, gap="large")
    with e1, st.container(border=True):
        st.markdown("**Information Value (IV)** · criterio **único** de selección")
        tabla_texto(pd.DataFrame({"Rango": ["< 0.02", "0.02 – 0.1", "0.1 – 0.3", "0.3 – 0.5", "≥ 0.5"],
                                  "Interpretación": ["Not good", "Weak", "Medium", "Strong", "Might be over fitting, recheck"],
                                  "Decisión": ["Fuera", "Pasa desde 0.05", "Pasa", "Pasa", "Pasa e informa"]}))
        st.caption("Rangos de Siddiqi (*Credit Risk Scorecards*). Corte en **0.05**, la referencia de la guía: deja fuera lo «Not "
                   "good» y la mitad baja de «Weak».")
    with e2, st.container(border=True):
        st.markdown("**Gini univariado** · informativo")
        tabla_texto(pd.DataFrame({"Rango": ["< 0.08", "0.08 – 0.18", "0.18 – 0.30", "0.30 – 0.38", "≥ 0.38"],
                                  "Interpretación": ["Sin poder", "Débil", "Medio", "Fuerte", "Revisar sobreajuste"],
                                  "IV equivalente": ["< 0.02", "0.02 – 0.1", "0.1 – 0.3", "0.3 – 0.5", "≥ 0.5"]}))
        st.caption("Cortes traducidos desde la escala de IV: si el score se distribuye normal en buenos y malos con separación *d*, "
                   "IV ≈ *d*² y Gini = 2·Φ(*d*/√2) − 1. IV = 0.05 equivale a Gini ≈ 0.13.")

    with st.expander("¿Por qué un único criterio (IV ≥ 0.05) para todos los modelos?", icon=":material/help:", expanded=True):
        st.markdown(
            "1. **Indicación de la profesora:** el entrenamiento parte de **un solo dataset**, el que sale del multivariado. La única "
            "razón para tener dos versiones es el desbalance del `TARGET` (original y rebalanceada con SMOTE), no las variables.\n"
            "2. **IV y Gini dicen casi lo mismo.** Se calculan sobre los mismos tramos y ordenan las variables casi igual (gráfico de "
            "abajo). Usar uno u otro cambia poco; lo que cambiaba la selección era tener **dos umbrales**.\n"
            "3. **¿Y los árboles?** Antes se defendía un umbral más bajo para ML porque un árbol puede aprovechar señales débiles en "
            "combinación. Con un dataset único esa ventaja se renuncia a propósito: se gana un proceso más simple y comparable entre "
            "modelos. Si XGBoost rinde muy por debajo de lo esperado, este es el primer supuesto a revisar."
        )

    rk = bi.copy()
    rk["rango_iv"] = pd.Categorical(rk["rango_iv"], RANGOS_IV, ordered=True)
    rk["rango_gini"] = pd.Categorical(rk["rango_gini"], RANGOS_GINI, ordered=True)
    top_n = st.slider("Variables a mostrar en el ranking", 10, len(rk), 30, step=5, key="top_rank")
    c1, c2 = st.columns(2, gap="large")
    r_iv = rk.nlargest(top_n, "iv")
    reglas_iv = alt.Chart(pd.DataFrame({"x": [0.02, 0.1, 0.3, 0.5]})).mark_rule(strokeDash=[4, 3], color=GRIS).encode(x="x:Q")
    c1.altair_chart((alt.Chart(r_iv).mark_bar(height={"band": 0.75}).encode(
        y=alt.Y("variable:N", sort=r_iv["variable"].tolist(), title=None, axis=alt.Axis(labelLimit=240, labelFontSize=10, labelOverlap=False)),
        x=alt.X("iv:Q", title="Information Value (OptBinning, train)"),
        color=alt.Color("rango_iv:N", scale=alt.Scale(domain=RANGOS_IV, range=COL_RANGO), legend=alt.Legend(orient="top", title=None, columns=2)),
        tooltip=[alt.Tooltip("variable:N"), alt.Tooltip("iv:Q", format=".4f"), alt.Tooltip("rango_iv:N", title="Rango"),
                 alt.Tooltip("iv_test:Q", format=".4f", title="IV en test")]) + reglas_iv)
        .properties(title="Ranking por IV · líneas en 0.02, 0.1, 0.3 y 0.5"), width="stretch", height=max(300, 17 * top_n + 60))
    r_g = rk.nlargest(top_n, "gini")
    reglas_g = alt.Chart(pd.DataFrame({"x": [0.08, 0.18, 0.30, 0.38]})).mark_rule(strokeDash=[4, 3], color=GRIS).encode(x="x:Q")
    c2.altair_chart((alt.Chart(r_g).mark_bar(height={"band": 0.75}).encode(
        y=alt.Y("variable:N", sort=r_g["variable"].tolist(), title=None, axis=alt.Axis(labelLimit=240, labelFontSize=10, labelOverlap=False)),
        x=alt.X("gini:Q", title="Gini univariado (OptBinning, train)"),
        color=alt.Color("rango_gini:N", scale=alt.Scale(domain=RANGOS_GINI, range=COL_RANGO), legend=alt.Legend(orient="top", title=None, columns=2)),
        tooltip=[alt.Tooltip("variable:N"), alt.Tooltip("gini:Q", format=".4f"), alt.Tooltip("rango_gini:N", title="Rango"),
                 alt.Tooltip("gini_test:Q", format=".4f", title="Gini en test")]) + reglas_g)
        .properties(title="Ranking por Gini · líneas en 0.08, 0.18, 0.30 y 0.38"), width="stretch", height=max(300, 17 * top_n + 60))

    s1, s2 = st.columns([3, 2], gap="large")
    curva = pd.DataFrame({"iv": np.linspace(0.0005, 0.7, 200)})
    from scipy.stats import norm
    curva["gini"] = 2 * norm.cdf(np.sqrt(curva["iv"]) / np.sqrt(2)) - 1
    rk["decision"] = np.where(rk["pasa"], "Pasa (IV ≥ 0.05)", "No pasa")
    puntos = alt.Chart(rk).mark_circle(size=60, opacity=0.8, stroke="white", strokeWidth=0.6).encode(
        x=alt.X("iv:Q", title="IV", scale=alt.Scale(type="sqrt")), y=alt.Y("gini:Q", title="Gini", scale=alt.Scale(type="sqrt")),
        color=alt.Color("decision:N", scale=alt.Scale(domain=["Pasa (IV ≥ 0.05)", "No pasa"], range=[AZUL, BARRA]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("variable:N"), alt.Tooltip("iv:Q", format=".4f"), alt.Tooltip("gini:Q", format=".4f")])
    s1.altair_chart((alt.Chart(curva).mark_line(color=NARANJA, strokeDash=[6, 4], strokeWidth=2).encode(x="iv:Q", y="gini:Q")
                     + puntos + alt.Chart(pd.DataFrame({"x": [IV_MIN]})).mark_rule(color=AZUL, strokeDash=[3, 3]).encode(x="x:Q"))
                    .properties(title={"text": "IV vs. Gini: casi la misma información",
                                       "subtitle": "Línea naranja: relación teórica Gini = 2Φ(√IV/√2) − 1 · punteada azul: umbral IV 0.05"}),
                    width="stretch", height=360)
    with s2:
        st.markdown("**¿Cuántas variables pasarían con otro umbral de IV?**")
        umbral = st.select_slider("Umbral de IV", [0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30], value=0.05, key="iv_whatif")
        n_pasa = int((bi["iv"] >= umbral).sum())
        st.metric("Variables con IV ≥ umbral", n_pasa, f"{n_pasa - int(bi['pasa'].sum()):+d} vs. el criterio elegido", delta_color="off")
        st.caption(", ".join(bi.loc[bi["iv"] >= umbral].sort_values("iv", ascending=False)["variable"].head(18)) +
                   (" …" if n_pasa > 18 else ""))
        nota(f"Con **IV ≥ 0.05** pasan **{int(bi['pasa'].sum())} variables**. Con el umbral anterior de 0.1 eran solo "
             f"{int((bi['iv'] >= 0.1).sum())} (los tres scores externos y la antigüedad laboral). Bajar el corte suma variables de "
             "solicitud, ocupación e historial: el modelo depende menos de los scores externos, a cambio de señales individuales más "
             "débiles (todas en el rango «Weak»).", ":material/balance:")

# ══════════════════════════════════════════════════════════════════════════════
# 3. PATRONES
# ══════════════════════════════════════════════════════════════════════════════
with tab_pat:
    st.markdown("Observaciones generales de la guía: cómo se comporta cada variable frente al `TARGET`.")
    pat = bi.copy()
    pat["grupo_guia"] = np.select(
        [pat["iv"] < 0.02, pat["patron"].str.startswith(("Monótona", "Casi")), pat["patron"].str.startswith("No lineal"),
         pat["patron"].eq("Sin patrón claro"), pat["patron"].eq("Dos niveles"), pat["tipo"].eq("categórica")],
        ["Sin patrón claro", "Tendencia monótona", "No lineal (cóncava / convexa)", "Sin patrón claro", "Dos niveles (binaria)",
         "Categórica: diferencias por grupo"], "Sin patrón claro")
    pat.loc[(pat["iv"] >= 0.1), "separacion"] = "Separación clara (IV ≥ 0.1)"
    orden_gg = ["Tendencia monótona", "No lineal (cóncava / convexa)", "Dos niveles (binaria)", "Categórica: diferencias por grupo", "Sin patrón claro"]
    cnt = pat.groupby("grupo_guia").size().reindex(orden_gg).fillna(0).reset_index(name="n")
    p1, p2 = st.columns([2, 3], gap="large")
    p1.altair_chart(alt.Chart(cnt).mark_bar(color=AZUL, height={"band": 0.7}, cornerRadiusEnd=3).encode(
        y=alt.Y("grupo_guia:N", sort=orden_gg, title=None, axis=alt.Axis(labelLimit=260)), x=alt.X("n:Q", title="Variables"),
        tooltip=[alt.Tooltip("grupo_guia:N", title="Patrón"), alt.Tooltip("n:Q", title="Variables")]), width="stretch", height=260)
    with p2:
        st.markdown(
            "- **Separación clara entre clases** (IV ≥ 0.1): `EXT_SOURCE_3`, `EXT_SOURCE_2`, `EXT_SOURCE_1` y `DAYS_EMPLOYED`.\n"
            "- **Tendencia monótona**: los scores externos (más score, menos default), `DAYS_BIRTH` (más joven, más riesgo), la "
            "proporción de pagos tardíos y los créditos activos en el buró (más atraso o más deuda, más riesgo). Es lo esperado del negocio.\n"
            "- **No lineal**: `DAYS_EMPLOYED` (el 0 del centinela baja el riesgo al final) y `AMT_CREDIT` (los créditos medianos "
            "tienen más default que los muy chicos y los muy grandes). La logística los captura vía WoE.\n"
            "- **Sin patrón**: variables operativas como `WEEKDAY_APPR_PROCESS_START` o la mayoría de banderas de contacto: el día de la "
            "solicitud no debería explicar el riesgo, y no lo hace.")
    st.markdown("**Las 12 variables con mayor IV**")
    top12 = bi.nlargest(12, "iv")["variable"].tolist()
    filas = []
    for i in range(0, 12, 3):
        cols = st.columns(3, gap="medium")
        for c, v in zip(cols, top12[i:i + 3]):
            rr = bi.set_index("variable").loc[v]
            dd = bo.query("variable == @v and n > 0")
            c.altair_chart(grafico_guia(dd, v, f"IV {rr['iv']:.3f} · Gini {rr['gini']:.3f} · {rr['patron']}", base=BASE, alto=190,
                                        etiquetas=False, angulo=-30, compacto=True), width="stretch")
    st.caption("Mismo gráfico de la guía con los tramos de OptBinning. Barras grises: frecuencia (eje izquierdo); línea azul: "
               "tasa de default (eje derecho); punteada gris: tasa base. Ejes sin título para ganar espacio.")

# ══════════════════════════════════════════════════════════════════════════════
# 4. SELECCIÓN
# ══════════════════════════════════════════════════════════════════════════════
with tab_sel:
    st.markdown(
        "Primer filtro formal de la guía (Paso 6.4), con **un criterio único: IV ≥ 0.05**. Dos controles adicionales se reportan "
        "como información, sin cambiar la decisión:\n\n"
        "- **Estabilidad train → test.** Con los tramos aprendidos en train, el orden de la tasa de default en test debería parecerse "
        "(ρ de Spearman ≥ 0.5) y el Gini de test no debería caer a menos de la mitad.\n"
        "- **Señal sospechosa.** IV ≥ 0.5 se informa, porque puede indicar sobreajuste o fuga de información.")
    s1, s2 = st.columns([3, 2], gap="large")
    with s1, st.container(border=True):
        sel = bi[bi["pasa"]].sort_values("iv", ascending=False)
        st.markdown(f"**IV ≥ {IV_MIN} → {len(sel)} variables** pasan al multivariado")
        st.dataframe(sel[["variable", "iv", "iv_test", "gini", "patron"]],
                     hide_index=True, width="stretch", height=min(38 + 35 * len(sel), 600),
                     column_config={"variable": st.column_config.TextColumn("Variable", width=230),
                                    "iv": st.column_config.NumberColumn("IV", format="%.3f", width=60),
                                    "iv_test": st.column_config.NumberColumn("IV test", format="%.3f", width=65),
                                    "gini": st.column_config.NumberColumn("Gini", format="%.3f", width=60),
                                    "patron": st.column_config.TextColumn("Patrón (qcut 5)", width=170)})
    with s2:
        est = bi.copy()
        est["estado"] = np.where(~est["estable"], "Inestable", np.where(est["pasa"], "Pasa (IV ≥ 0.05)", "No pasa"))
        diag = pd.DataFrame({"x": [0, 0.32], "y": [0, 0.32]})
        st.altair_chart((alt.Chart(diag).mark_line(color=GRIS, strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
                         + alt.Chart(est).mark_circle(size=60, opacity=0.85, stroke="white").encode(
                             x=alt.X("gini:Q", title="Gini en train"), y=alt.Y("gini_test:Q", title="Gini en test"),
                             color=alt.Color("estado:N", scale=alt.Scale(domain=["Pasa (IV ≥ 0.05)", "No pasa", "Inestable"], range=[AZUL, BARRA, NARANJA]),
                                             legend=alt.Legend(orient="top", title=None)),
                             tooltip=[alt.Tooltip("variable:N"), alt.Tooltip("iv:Q", format=".3f"), alt.Tooltip("gini:Q", format=".3f"),
                                      alt.Tooltip("gini_test:Q", format=".3f"),
                                      alt.Tooltip("rho_tramos_train_test:Q", format=".2f", title="ρ tramos train/test")]))
                        .properties(title="Estabilidad: Gini en train vs. test", height=340), width="stretch")
        inest = bi[~bi["estable"]].sort_values("iv", ascending=False)
        st.caption("Sobre la diagonal, la variable discrimina igual en test que en train. Las inestables (" +
                   ", ".join(f"`{v}`" for v in inest["variable"]) + ") tienen IV < 0.01: ninguna llegaba al corte, así que el control "
                   "no cambia la selección.")
    st.markdown("**Decisión por variable**")
    tabla = bi[["variable", "tipo", "familia", "iv", "rango_iv", "gini", "rango_gini", "estable", "pasa"]].copy()
    tabla["familia"] = tabla["familia"].map(fam_corta)
    st.dataframe(tabla.sort_values("iv", ascending=False), hide_index=True, width="stretch", height=420,
                 column_config={"variable": st.column_config.TextColumn("Variable", pinned=True, width=220),
                                "iv": st.column_config.NumberColumn("IV", format="%.4f"), "gini": st.column_config.NumberColumn("Gini", format="%.4f"),
                                "rango_iv": st.column_config.TextColumn("Rango IV", width=170), "rango_gini": st.column_config.TextColumn("Rango Gini", width=170),
                                "estable": st.column_config.CheckboxColumn("Estable"),
                                "pasa": st.column_config.CheckboxColumn("Pasa (IV ≥ 0.05)")})

st.divider()
st.caption("Fuente: `scripts/pipeline_modelado.py` (etapas `binning` y `seleccion`): qcut y OptBinning (max_n_prebins=20, "
           "min_prebin_size=0.05, max_n_bins=5) ajustados en train; IV, Gini y tasas de test con los mismos tramos.")
