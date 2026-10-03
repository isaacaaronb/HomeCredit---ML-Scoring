"""EDA multivariado (Pasos 6.5–7 de la guía): redundancia con Spearman (sale la de menor IV) y dataset de entrenamiento único."""
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
def _leer(ruta: str, version: float) -> pd.DataFrame:
    return pd.read_parquet(ruta)


def t(nombre: str) -> pd.DataFrame:
    """Lee un artefacto; la fecha de modificación entra en la llave del caché para que un redeploy no sirva datos viejos."""
    ruta = MOD / f"{nombre}.parquet"
    return _leer(str(ruta), ruta.stat().st_mtime)


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
    "calcula la correlación de **Spearman** entre las numéricas (mide relaciones monótonas, no solo lineales, y usa |ρ|: el signo "
    "no importa). Si |ρ| > **0.6**, de cada par **sale la variable de menor IV**. La selección es secuencial, de mayor a menor IV: "
    "una variable entra solo si no es redundante con ninguna de las que ya entraron. Todo se calcula en train y el resultado es el "
    "**dataset de entrenamiento único** que usarán todos los modelos."
)

# ── Embudo ─────────────────────────────────────────────────────────────────────
n_tot = len(fu)
n_uni = int((fu["decision"] != "Elimina").sum())
n_bi = int(bi["pasa"].sum())
n_mv = int(sel["seleccionada"].sum())
emb = pd.DataFrame({"etapa": ["Tablón tratado", "Filtro univariado", "Filtro bivariado (IV ≥ 0.05)", "Filtro multivariado"],
                    "n": [n_tot, n_uni, n_bi, n_mv]})
k1, k2, k3, k4 = st.columns(4)
k1.metric("Variables explicativas", n_tot, border=True)
k2.metric("Tras el univariado", n_uni, f"{n_uni - n_tot}", delta_color="off", border=True)
k3.metric("Tras el bivariado", n_bi, f"{n_bi - n_uni}", delta_color="off", border=True)
k4.metric("Dataset final", n_mv, f"{n_mv - n_bi}", delta_color="off", border=True)
emb["etq"] = emb["n"].astype(str)
base_e = alt.Chart(emb).encode(y=alt.Y("etapa:N", sort=emb["etapa"].tolist(), title=None, axis=alt.Axis(labelLimit=260)))
st.altair_chart((base_e.mark_bar(color=AZUL, height={"band": 0.62}, cornerRadiusEnd=3).encode(
    x=alt.X("n:Q", title="Variables"), tooltip=[alt.Tooltip("etapa:N"), alt.Tooltip("n:Q", title="Variables")])
    + base_e.mark_text(align="left", dx=5, fontSize=12, color=TINTA).encode(x="n:Q", text="etq:N"))
    .properties(title="Del tablón al dataset de entrenamiento: cuántas variables sobreviven a cada filtro"), width="stretch", height=210)

s_ds = sel.sort_values("iv", ascending=False)
tab_mat, tab_par, tab_fin = st.tabs([":material/grid_on: Matriz de Spearman", ":material/compare: Pares redundantes",
                                     ":material/checklist: Conjunto final"])

with tab_mat:
    orden = [v for v in s_ds["variable"] if v in set(corr["v1"])]
    solo_sel = st.toggle("Mostrar solo las variables seleccionadas", key="solo_sel_mv", value=False)
    if solo_sel:
        orden = [v for v in orden if v in set(s_ds.query("seleccionada")["variable"])]
    cm = corr[corr["v1"].isin(orden) & corr["v2"].isin(orden)].copy()
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
    lado = max(320, 30 * len(orden) + 140)
    st.altair_chart(capas.properties(title={"text": "Correlación de Spearman entre las variables que pasaron el bivariado",
                                            "subtitle": "Ordenadas por IV (de mayor a menor) · recuadro negro: |ρ| > 0.6"}),
                    width="stretch", height=lado)
    st.caption("Entran numéricas, dicotómicas (codificadas 0/1) y categóricas ordinales numéricas. Las categóricas nominales "
               "(`OCCUPATION_TYPE`, `ORGANIZATION_TYPE`, `NAME_INCOME_TYPE`) no entran a Spearman (guía); entre ellas se revisó la "
               "V de Cramér con el mismo umbral y ningún par lo supera.")

with tab_par:
    if pares.empty:
        st.success("Ningún par supera |ρ| > 0.6: las variables seleccionadas aportan información distinta.", icon=":material/check_circle:")
    else:
        p = pares.copy()
        p["abs"] = p["rho"].abs()
        decisivos = p[p["resultado"].str.startswith("Conserva")]
        n_dif = int(p["gini_elegiria_otra"].sum())
        st.markdown(f"**{len(p)} pares** superan |ρ| > 0.6 y **{len(decisivos)}** deciden una exclusión. Criterio: sale la de "
                    f"**menor IV**. Como control, el criterio anterior (mayor Gini) habría elegido distinto en **{n_dif}** de ellos.")
        st.dataframe(p.sort_values("abs", ascending=False)[["medida", "variable_1", "variable_2", "rho", "iv_1", "iv_2", "gini_1", "gini_2", "resultado"]],
                     hide_index=True, width="stretch", height=min(38 + 35 * len(p), 520),
                     column_config={"medida": st.column_config.TextColumn("Medida", width=90),
                                    "variable_1": st.column_config.TextColumn("Variable 1", width=210),
                                    "variable_2": st.column_config.TextColumn("Variable 2", width=170),
                                    "rho": st.column_config.NumberColumn("ρ", format="%.3f"),
                                    "iv_1": st.column_config.NumberColumn("IV 1", format="%.4f"),
                                    "iv_2": st.column_config.NumberColumn("IV 2", format="%.4f"),
                                    "gini_1": st.column_config.NumberColumn("Gini 1", format="%.3f", help="Informativo."),
                                    "gini_2": st.column_config.NumberColumn("Gini 2", format="%.3f", help="Informativo."),
                                    "resultado": st.column_config.TextColumn("Según la regla (menor IV sale)", width=380)})
        ch = decisivos.copy()
        ch["par"] = [f"{a} ↔ {b}" for a, b in zip(ch["variable_1"], ch["variable_2"])]
        ch["ini"] = 0.5
        st.altair_chart(alt.Chart(ch, title=alt.TitleParams("Pares que deciden una exclusión", subtitle=f"|ρ| desde 0.5; línea punteada: umbral {RHO_MAX}",
                                                            anchor="start")).mark_bar(color=NARANJA, height={"band": 0.6}, cornerRadiusEnd=3).encode(
            y=alt.Y("par:N", sort="-x", title=None, axis=alt.Axis(labelLimit=420, labelFontSize=11, labelOverlap=False)),
            x=alt.X("abs:Q", title="|ρ de Spearman|", scale=alt.Scale(domain=[0.5, 1])), x2="ini:Q",
            tooltip=[alt.Tooltip("par:N"), alt.Tooltip("rho:Q", format=".3f"), alt.Tooltip("resultado:N")])
            + alt.Chart(pd.DataFrame({"x": [RHO_MAX]})).mark_rule(color=TINTA, strokeDash=[4, 3]).encode(x="x:Q"),
            width="stretch", height=max(180, 48 * len(ch) + 70))
        st.markdown(
            "- **`AMT_CREDIT` sale frente a `AMT_GOODS_PRICE`** (ρ = 0.98): el monto del crédito es casi el precio del bien.\n"
            "- **`HC_N_RECHAZADAS` sale frente a `HC_PROP_SOLICITUDES_RECHAZADAS`**: el conteo y la proporción de rechazos miden lo mismo; "
            "la proporción tiene más IV porque no depende de cuántas solicitudes hizo el cliente.")
        par_eb = p[(p["variable_1"] == "EXT_SOURCE_1") & (p["variable_2"] == "DAYS_BIRTH")]
        if len(par_eb):
            rho_eb = float(par_eb["rho"].iloc[0])
            st.warning(
                f"**Un par para pensar: `EXT_SOURCE_1` ↔ `DAYS_BIRTH` (ρ = {rho_eb:.3f}).** Ahora que `EXT_SOURCE_1` ya no se imputa, la "
                "correlación es la de los datos originales (cuando se imputaba con la mediana por grupo de edad llegaba a −0.79: la "
                f"imputación fabricaba parte de la redundancia). Con |ρ| = {abs(rho_eb):.3f} el par supera el umbral de 0.6 **por "
                "milésimas**, y además se mide solo sobre el 44 % de créditos que tienen `EXT_SOURCE_1`. La regla excluye "
                "`DAYS_BIRTH`, pero es la decisión más frágil del filtro: un umbral de 0.61 la conservaría.",
                icon=":material/psychology:")

with tab_fin:
    f1, f2 = st.columns([3, 2], gap="large")
    fin = s_ds.copy()
    fin["estado"] = np.where(fin["seleccionada"], "Seleccionada", "Excluida por redundancia")
    f1.altair_chart(alt.Chart(fin).mark_bar(height={"band": 0.7}, cornerRadiusEnd=3).encode(
        y=alt.Y("variable:N", sort=fin["variable"].tolist(), title=None, axis=alt.Axis(labelLimit=260, labelFontSize=11, labelOverlap=False)),
        x=alt.X("iv:Q", title="IV"),
        color=alt.Color("estado:N", scale=alt.Scale(domain=["Seleccionada", "Excluida por redundancia"], range=[AZUL, BARRA]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("variable:N"), alt.Tooltip("iv:Q", format=".4f", title="IV"), alt.Tooltip("gini:Q", format=".3f", title="Gini"),
                 alt.Tooltip("estado:N"), alt.Tooltip("redundante_con:N", title="Redundante con"),
                 alt.Tooltip("medida_redundancia:Q", format=".3f", title="ρ")])
        + alt.Chart(pd.DataFrame({"x": [0.05]})).mark_rule(color=GRIS, strokeDash=[4, 3]).encode(x="x:Q"),
        width="stretch", height=max(240, 26 * len(fin) + 60))
    with f2:
        fs = fin[fin["seleccionada"]]
        st.markdown(f"**{len(fs)} variables finales** · dataset de entrenamiento único")
        st.dataframe(fs.assign(familia=fs["familia"].map(fam_corta))[["variable", "iv", "familia"]],
                     hide_index=True, width="stretch", height=38 + 35 * len(fs),
                     column_config={"variable": st.column_config.TextColumn("Variable", width=215),
                                    "familia": st.column_config.TextColumn("Familia", width=160),
                                    "iv": st.column_config.NumberColumn("IV", format="%.3f", width=60)})
    st.markdown("**Control adicional: correlación y VIF de las variables finales en WoE** (como las verá la regresión logística)")
    v1, v2 = st.columns([2, 3], gap="large")
    vif = t("woe_vif")
    v1.altair_chart(alt.Chart(vif).mark_bar(color=AZUL, height={"band": 0.6}).encode(
        y=alt.Y("columna:N", title=None, sort="-x", axis=alt.Axis(labelLimit=260, labelFontSize=10, labelOverlap=False)),
        x=alt.X("vif:Q", title="VIF", scale=alt.Scale(domain=[0, max(5, vif["vif"].max() * 1.1)])),
        tooltip=[alt.Tooltip("columna:N"), alt.Tooltip("vif:Q", format=".2f")])
        + alt.Chart(pd.DataFrame({"x": [5]})).mark_rule(color=NARANJA, strokeDash=[4, 3]).encode(x="x:Q"),
        width="stretch", height=max(220, 24 * len(vif) + 40))
    cw = t("woe_corr")
    orden_w = vif.sort_values("vif", ascending=False)["columna"].tolist()
    v2.altair_chart((alt.Chart(cw).mark_rect().encode(
        x=alt.X("v2:N", title=None, sort=orden_w, axis=alt.Axis(labelAngle=-60, labelLimit=200, labelFontSize=9, labelOverlap=False)),
        y=alt.Y("v1:N", title=None, sort=orden_w, axis=alt.Axis(labelLimit=220, labelFontSize=9, labelOverlap=False)),
        color=alt.Color("rho:Q", scale=alt.Scale(domain=[-1, 0, 1], range=[AZUL, "#f2f1ee", NARANJA], interpolate="lab"), legend=None),
        tooltip=[alt.Tooltip("v1:N"), alt.Tooltip("v2:N"), alt.Tooltip("rho:Q", format=".3f")])
        + alt.Chart(cw).mark_text(fontSize=9).encode(x=alt.X("v2:N", sort=orden_w), y=alt.Y("v1:N", sort=orden_w),
                                                    text=alt.Text("rho:Q", format=".1f"))),
                    width="stretch", height=max(260, 24 * len(vif) + 120))
    st.caption(f"VIF máximo: {vif['vif'].max():.2f}. Por debajo de 5 (línea naranja) no hay multicolinealidad que distorsione los "
               "coeficientes de la logística.")

st.divider()
st.caption("Fuente: `scripts/pipeline_modelado.py seleccion` · Spearman = Pearson sobre rangos, con nulos por pares (mínimo 1,000 "
           "observaciones comunes) · umbral |ρ| > 0.6 (la guía sugiere 0.8; el equipo eligió uno más estricto) · de cada par sale la "
           "de menor IV.")
