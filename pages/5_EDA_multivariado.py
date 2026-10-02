"""EDA multivariado (Pasos 6.5–7 de la guía): redundancia entre variables con Spearman y conjunto final de cada dataset."""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

REPO_DIR = Path(__file__).resolve().parents[1]
MOD = REPO_DIR / "artifacts" / "modelado"
AZUL, NARANJA, VERDE, GRIS, TINTA, BARRA = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85", "#2b2b29", "#c9c8c3"
RHO_MAX = 0.60


@st.cache_data(show_spinner=False)
def t(nombre: str) -> pd.DataFrame:
    return pd.read_parquet(MOD / f"{nombre}.parquet")


def fam_corta(f) -> str:
    return f.split("·", 1)[-1].strip() if isinstance(f, str) else ""


sel = t("multivariado_seleccion")
pares = t("multivariado_pares")
corr = t("multivariado_corr")
fu = t("filtro_univariado")
bi = t("bivariado_resumen")

st.title("EDA multivariado")
st.markdown(
    "Las variables que pasaron el bivariado tienen señal, pero pueden **decir lo mismo**. Siguiendo la guía (Pasos 6.5–7), se "
    "calcula la correlación de **Spearman** entre las numéricas de cada dataset (mide relaciones monótonas, no solo lineales, y "
    "usa |ρ|: el signo no importa). Si |ρ| > **0.6**, se conserva la variable con **mayor IV** (logística) o **mayor Gini** (ML). "
    "La selección es secuencial, de la más fuerte a la más débil: una variable entra solo si no es redundante con ninguna de las "
    "que ya entraron. Todo se calcula en train."
)

# ── Embudo ─────────────────────────────────────────────────────────────────────
n_tot = len(fu)
n_uni = int((fu["decision"] != "Elimina").sum())
emb = []
for ds, campo, nombre in [("logistica", "pasa_logistica", "Logística (IV)"), ("ml", "pasa_ml", "Machine learning (Gini)")]:
    n_bi = int(bi[campo].sum())
    n_mv = int(sel.query("dataset == @ds and seleccionada").shape[0])
    emb += [{"dataset": nombre, "etapa": e, "orden": i, "n": n} for i, (e, n) in
            enumerate([("Tablón tratado", n_tot), ("Filtro univariado", n_uni), ("Filtro bivariado", n_bi), ("Filtro multivariado", n_mv)])]
emb = pd.DataFrame(emb)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Variables explicativas", n_tot, border=True)
k2.metric("Tras el univariado", n_uni, f"{n_uni - n_tot}", delta_color="off", border=True)
k3.metric("Dataset logístico", int(sel.query("dataset == 'logistica' and seleccionada").shape[0]), border=True)
k4.metric("Dataset ML", int(sel.query("dataset == 'ml' and seleccionada").shape[0]), border=True)
emb["etq"] = emb["n"].astype(str)
base_e = alt.Chart(emb).encode(x=alt.X("etapa:N", sort=["Tablón tratado", "Filtro univariado", "Filtro bivariado", "Filtro multivariado"],
                                       title=None, axis=alt.Axis(labelAngle=0)),
                               xOffset=alt.XOffset("dataset:N"))
st.altair_chart((base_e.mark_bar(size=46, cornerRadiusEnd=3).encode(
    y=alt.Y("n:Q", title="Variables"),
    color=alt.Color("dataset:N", scale=alt.Scale(domain=["Logística (IV)", "Machine learning (Gini)"], range=[AZUL, VERDE]),
                    legend=alt.Legend(orient="top", title=None)),
    tooltip=[alt.Tooltip("dataset:N"), alt.Tooltip("etapa:N"), alt.Tooltip("n:Q", title="Variables")])
    + base_e.mark_text(dy=-8, fontSize=12, color=TINTA).encode(y="n:Q", text="etq:N"))
    .properties(title="Del tablón a cada dataset: cuántas variables sobreviven a cada filtro"), width="stretch", height=300)

ds_lbl = st.segmented_control("Dataset", ["Machine learning (Gini)", "Logística (IV)"], default="Machine learning (Gini)", key="ds_mv")
ds = "ml" if (ds_lbl or "Machine").startswith("Machine") else "logistica"
metrica = "Gini" if ds == "ml" else "IV"
s_ds = sel.query("dataset == @ds").sort_values("metrica", ascending=False)
c_ds = corr.query("dataset == @ds")
p_ds = pares.query("dataset == @ds")

tab_mat, tab_par, tab_fin = st.tabs([":material/grid_on: Matriz de Spearman", ":material/compare: Pares redundantes",
                                     ":material/checklist: Conjunto final"])

with tab_mat:
    orden = [v for v in s_ds["variable"] if v in set(c_ds["v1"])]
    solo_sel = st.toggle("Mostrar solo las variables seleccionadas", key="solo_sel_mv", value=False)
    if solo_sel:
        orden = [v for v in orden if v in set(s_ds.query("seleccionada")["variable"])]
    cm = c_ds[c_ds["v1"].isin(orden) & c_ds["v2"].isin(orden)].copy()
    cm["alta"] = (cm["rho"].abs() > RHO_MAX) & (cm["v1"] != cm["v2"])
    cm["txt"] = cm["rho"].map(lambda x: f"{x:.2f}")
    base_m = alt.Chart(cm).encode(
        x=alt.X("v2:N", sort=orden, title=None, axis=alt.Axis(labelAngle=-60, labelLimit=200, labelFontSize=9, labelOverlap=False)),
        y=alt.Y("v1:N", sort=orden, title=None, axis=alt.Axis(labelLimit=240, labelFontSize=9, labelOverlap=False)))
    calor = base_m.mark_rect(stroke="white", strokeWidth=0.6).encode(
        color=alt.Color("rho:Q", scale=alt.Scale(domain=[-1, 0, 1], range=[AZUL, "#f2f1ee", NARANJA], interpolate="lab"), title="ρ de Spearman"),
        tooltip=[alt.Tooltip("v1:N", title="Variable 1"), alt.Tooltip("v2:N", title="Variable 2"), alt.Tooltip("rho:Q", format=".3f", title="ρ")])
    borde = alt.Chart(cm[cm["alta"]]).mark_rect(fill=None, stroke=TINTA, strokeWidth=1.6).encode(
        x=alt.X("v2:N", sort=orden), y=alt.Y("v1:N", sort=orden))
    capas = calor + borde
    if len(orden) <= 16:
        capas = capas + base_m.mark_text(fontSize=10).encode(text="txt:N", color=alt.condition("abs(datum.rho) > 0.6", alt.value("white"), alt.value(TINTA)))
    lado = max(320, 24 * len(orden) + 120)
    st.altair_chart(capas.properties(title={"text": f"Correlación de Spearman · dataset {'ML' if ds == 'ml' else 'logístico'}",
                                            "subtitle": f"Ordenadas por {metrica} (de mayor a menor) · recuadro negro: |ρ| > 0.6"}),
                    width="stretch", height=lado)
    st.caption("Entran numéricas, dicotómicas (codificadas 0/1) y categóricas ordinales numéricas. Las categóricas nominales no "
               "entran a Spearman (guía); entre ellas se revisó la V de Cramér con el mismo umbral y ningún par lo supera.")
    if ds == "ml":
        st.info("El bloque naranja es **vivienda**: las versiones _AVG / _MEDI / _MODE y las distintas medidas del edificio (área, "
                "pisos, ascensores) están correlacionadas entre sí por encima de 0.7. De ese bloque sobreviven solo `FLOORSMAX_AVG`, "
                "`YEARS_BEGINEXPLUATATION_MODE` y `ENTRANCES_AVG`.", icon=":material/apartment:")

with tab_par:
    if p_ds.empty:
        st.success(f"Ningún par del dataset {'logístico' if ds == 'logistica' else 'ML'} supera |ρ| > 0.6: las variables "
                   "seleccionadas aportan información distinta.", icon=":material/check_circle:")
    else:
        p = p_ds.copy()
        p["abs"] = p["rho"].abs()
        decisivos = p[p["resultado"].str.startswith("Conserva")]
        st.markdown(f"**{len(p)} pares** superan |ρ| > 0.6; **{len(decisivos)}** deciden una exclusión. El resto son pares entre "
                    "variables que ya habían salido por una variable más fuerte.")
        ver_todos = st.toggle("Ver también los pares entre variables ya excluidas", key="todos_pares")
        vista = p if ver_todos else decisivos
        st.dataframe(vista.sort_values("abs", ascending=False)[["medida", "variable_1", "variable_2", "rho", "metrica_1", "metrica_2", "resultado"]],
                     hide_index=True, width="stretch", height=min(38 + 35 * len(vista), 520),
                     column_config={"medida": st.column_config.TextColumn("Medida", width=90),
                                    "variable_1": st.column_config.TextColumn("Variable 1", width=210),
                                    "variable_2": st.column_config.TextColumn("Variable 2", width=210),
                                    "rho": st.column_config.NumberColumn("ρ", format="%.3f"),
                                    "metrica_1": st.column_config.NumberColumn(f"{metrica} 1", format="%.4f"),
                                    "metrica_2": st.column_config.NumberColumn(f"{metrica} 2", format="%.4f"),
                                    "resultado": st.column_config.TextColumn("Según la regla", width=380)})
        ch = decisivos.copy()
        ch["par"] = [f"{a} ↔ {b}" for a, b in zip(ch["variable_1"], ch["variable_2"])]
        ch["ini"] = 0.5
        st.altair_chart(alt.Chart(ch, title=alt.TitleParams("Pares que deciden una exclusión", subtitle=f"|ρ| desde 0.5; línea punteada: umbral {RHO_MAX}",
                                                            anchor="start")).mark_bar(color=NARANJA, height={"band": 0.7}, cornerRadiusEnd=3).encode(
            y=alt.Y("par:N", sort="-x", title=None, axis=alt.Axis(labelLimit=420, labelFontSize=10)),
            x=alt.X("abs:Q", title="|ρ de Spearman|", scale=alt.Scale(domain=[0.5, 1])), x2="ini:Q",
            tooltip=[alt.Tooltip("par:N"), alt.Tooltip("rho:Q", format=".3f"), alt.Tooltip("resultado:N")])
            + alt.Chart(pd.DataFrame({"x": [RHO_MAX]})).mark_rule(color=TINTA, strokeDash=[4, 3]).encode(x="x:Q"),
            width="stretch", height=max(200, 22 * len(ch) + 40))
        if ds == "ml":
            st.warning(
                "**Un par para pensar: `EXT_SOURCE_1` ↔ `DAYS_BIRTH` (ρ = −0.79).** En los datos originales (sin imputar) la "
                "correlación es −0.60; entre los créditos **imputados** sube a −0.95, porque `EXT_SOURCE_1` se imputó con la "
                "mediana por **grupo de edad** × tipo de ingreso. Parte de la redundancia la creó el preprocesamiento. Con los "
                "datos originales el par seguiría superando 0.6 (por poco), así que la decisión no cambia, pero conviene saberlo.",
                icon=":material/psychology:")

with tab_fin:
    f1, f2 = st.columns([3, 2], gap="large")
    fin = s_ds.copy()
    fin["estado"] = np.where(fin["seleccionada"], "Seleccionada", "Excluida por redundancia")
    f1.altair_chart(alt.Chart(fin).mark_bar(height={"band": 0.75}).encode(
        y=alt.Y("variable:N", sort=fin["variable"].tolist(), title=None, axis=alt.Axis(labelLimit=260, labelFontSize=10, labelOverlap=False)),
        x=alt.X("metrica:Q", title=metrica),
        color=alt.Color("estado:N", scale=alt.Scale(domain=["Seleccionada", "Excluida por redundancia"], range=[AZUL if ds == "logistica" else VERDE, BARRA]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("variable:N"), alt.Tooltip("metrica:Q", format=".4f", title=metrica), alt.Tooltip("estado:N"),
                 alt.Tooltip("redundante_con:N", title="Redundante con"), alt.Tooltip("medida_redundancia:Q", format=".3f", title="ρ")]),
        width="stretch", height=max(220, 20 * len(fin) + 60))
    with f2:
        fs = fin[fin["seleccionada"]]
        st.markdown(f"**{len(fs)} variables finales** · dataset {'logístico' if ds == 'logistica' else 'ML'}")
        st.dataframe(fs.assign(familia=fs["familia"].map(fam_corta))[["variable", "metrica", "familia"]],
                     hide_index=True, width="stretch", height=min(38 + 35 * len(fs), 20 * len(fin) + 60),
                     column_config={"variable": st.column_config.TextColumn("Variable", width=215),
                                    "familia": st.column_config.TextColumn("Familia", width=150),
                                    "metrica": st.column_config.NumberColumn(metrica, format="%.3f", width=60)})
    if ds == "logistica":
        st.markdown("**Complemento para la logística: correlación y VIF de las variables en WoE**")
        v1, v2 = st.columns(2, gap="large")
        vif = t("logistica_vif")
        v1.altair_chart(alt.Chart(vif).mark_bar(color=AZUL, height={"band": 0.6}).encode(
            y=alt.Y("columna:N", title=None), x=alt.X("vif:Q", title="VIF", scale=alt.Scale(domain=[0, max(5, vif["vif"].max() * 1.1)])),
            tooltip=[alt.Tooltip("columna:N"), alt.Tooltip("vif:Q", format=".2f")])
            + alt.Chart(pd.DataFrame({"x": [5]})).mark_rule(color=NARANJA, strokeDash=[4, 3]).encode(x="x:Q"), width="stretch", height=200)
        cw = t("logistica_corr_woe")
        v2.altair_chart((alt.Chart(cw).mark_rect().encode(x=alt.X("v2:N", title=None, axis=alt.Axis(labelAngle=-30)), y=alt.Y("v1:N", title=None),
                                                         color=alt.Color("rho:Q", scale=alt.Scale(domain=[-1, 0, 1], range=[AZUL, "#f2f1ee", NARANJA]), legend=None))
                         + alt.Chart(cw).mark_text(fontSize=11).encode(x="v2:N", y="v1:N", text=alt.Text("rho:Q", format=".2f"))),
                        width="stretch", height=200)
        st.caption("VIF < 5 (línea naranja): no hay multicolinealidad que distorsione los coeficientes de la logística.")
    else:
        st.markdown(
            "- Del dataset ML salen sobre todo **variables que miden lo mismo con otra unidad**: `AMT_CREDIT` y `AMT_ANNUITY` frente a "
            "`AMT_GOODS_PRICE` (ρ = 0.98 y 0.83), conteos frente a proporciones (`HC_N_PAGOS_TARDE` frente a `HC_PROP_PAGOS_TARDE`) y "
            "las versiones de vivienda.\n"
            "- `REGION_RATING_CLIENT` sale frente a `REGION_RATING_CLIENT_W_CITY` (la segunda incluye la ciudad y tiene más Gini).\n"
            "- Se conserva una de cada «familia de información»: así un árbol no reparte la importancia entre gemelas y la "
            "interpretación (Paso 12) queda limpia.")

st.divider()
st.caption("Fuente: `scripts/pipeline_modelado.py seleccion` · Spearman = Pearson sobre rangos, con nulos por pares (mínimo 1,000 "
           "observaciones comunes) · umbral |ρ| > 0.6 (la guía sugiere 0.8; el equipo eligió uno más estricto).")
