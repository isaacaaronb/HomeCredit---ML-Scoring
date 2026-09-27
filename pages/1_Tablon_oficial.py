"""Tablón oficial: diccionario de variables organizado por familias, con vista de los datos."""
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

REPO_DIR = Path(__file__).resolve().parents[1]
TABLON = REPO_DIR / "artifacts" / "tablon_general.parquet"
MUESTRA = REPO_DIR / "artifacts" / "tablon_muestra.parquet"
DICCIONARIO = REPO_DIR / "data_dictionary" / "diccionario_tablon.csv"
FAMILIAS = REPO_DIR / "data_dictionary" / "familias.csv"

COLOR_TIPO = {  # badge de Streamlit por tipo de dato
    "Numérica continua": "blue", "Numérica discreta": "violet", "Dicotómica": "green",
    "Categórica nominal": "orange", "Categórica ordinal": "orange", "Identificador": "gray",
    "Objetivo (dicotómica)": "red",
}


@st.cache_data(show_spinner=False)
def cargar():
    dicc = pd.read_csv(DICCIONARIO)
    fam = pd.read_csv(FAMILIAS)
    resumen = (dicc.groupby("familia")
               .agg(variables=("variable", "size"),
                    construidas=("origen", lambda s: int((s == "Construida").sum())),
                    fill_rate=("fill_rate", "mean"))
               .reset_index())
    fam = fam.merge(resumen, on="familia", how="left")
    muestra = pd.read_parquet(MUESTRA)
    filas = pq.read_metadata(TABLON).num_rows
    return dicc, fam, muestra, filas


def pct(v: float) -> str:
    return "100%" if v >= 0.9995 else f"{v:.1%}"


def fmt_valor(v) -> str:
    """Valor legible para la ficha de un crédito."""
    if pd.isna(v):
        return "—"
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if float(v).is_integer():
            return f"{v:,.0f}"
        return f"{v:,.2f}" if abs(v) >= 100 else f"{v:.4g}"
    return str(v)


def etiqueta_familia(fam: pd.DataFrame):
    nombres = dict(zip(fam["familia"], fam["nombre_corto"]))
    return lambda f: f"{f.split(' · ')[0]} · {nombres[f]}"


COLUMNAS_DICC = {
    "variable": st.column_config.TextColumn("Variable", width=200, pinned=True),
    "descripcion": st.column_config.TextColumn("Descripción", width=320, help="Pasa el cursor para ver el texto completo."),
    "tipo_dato": st.column_config.TextColumn("Tipo de dato", width=130),
    "formato": st.column_config.TextColumn("Formato", width=120),
    "fuente": st.column_config.TextColumn("Fuente", width=180),
    "ventana": st.column_config.TextColumn("Ventana temporal", width="medium"),
    "fill_rate": st.column_config.ProgressColumn("Fill rate", format="percent", min_value=0, max_value=1, width=100),
    "ejemplos": st.column_config.TextColumn("Valores de ejemplo", width="medium"),
    "calculo": st.column_config.TextColumn("Cómo se construye", width="large"),
    "familia": st.column_config.TextColumn("Familia", width=190),
    "origen": st.column_config.TextColumn("Origen", width=100),
}

dicc, fam, muestra, n_filas = cargar()
fmt_familia = etiqueta_familia(fam)

# ── Encabezado ─────────────────────────────────────────────────────────────────
st.title("Tablón oficial")
st.markdown(
    "La base de modelización tiene **una fila por crédito objetivo** y reúne en 148 columnas todo lo que se sabe del "
    "solicitante al momento de pedir el crédito: la solicitud misma, su perfil y su historial en el buró y en Home Credit. "
    "Para leerla con sentido de negocio, las variables se organizan en **14 familias**: cada una responde una pregunta "
    "distinta sobre el riesgo del solicitante."
)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Créditos", f"{n_filas:,}", border=True)
m2.metric("Variables", f"{len(dicc)}", border=True)
m3.metric("Familias", f"{len(fam)}", border=True)
m4.metric("Construidas", f"{(dicc.origen == 'Construida').sum()}", border=True,
          help=f"Variables agregadas desde las tablas históricas; las otras {(dicc.origen == 'Original').sum()} vienen de application_train.")
m5.metric("Fill rate promedio", f"{dicc['fill_rate'].mean():.1%}", border=True,
          help="Promedio, entre variables, del % de créditos con dato no nulo.")

with st.expander("Claves para leer el tablón", icon=":material/lightbulb:", expanded=True):
    k1, k2, k3, k4 = st.columns(4, gap="medium")
    k1.markdown("**Unidad de análisis**  \nCada fila es un **crédito** (`SK_ID_CURR`), no una persona. `TARGET = 1` marca "
                "dificultades de pago en las primeras cuotas.")
    k2.markdown("**Tiempo relativo**  \nLas variables `DAYS_*` cuentan días **antes de la solicitud** (−365 ≈ un año antes). "
                "`DAYS_EMPLOYED = 365243` es un valor centinela, no un dato real.")
    k3.markdown("**Nulo ≠ cero**  \nEn las familias de historial, un nulo significa **sin registros en esa fuente**. "
                "`TIENE_BUREAU` y `TIENE_HISTORIAL_HOME_CREDIT` lo hacen explícito.")
    k4.markdown("**Datos ya normalizados**  \nLas variables de edificio (`_AVG`, `_MODE`, `_MEDI`) y los `EXT_SOURCE_*` "
                "llegan escalados por Home Credit; no están en unidades físicas.")


def ficha_variable(var: str) -> None:
    """Tarjeta con todo lo que se sabe de una variable."""
    v = dicc.set_index("variable").loc[var]
    with st.container(border=True):
        st.markdown(f"#### `{var}`")
        st.markdown(" ".join([f":{COLOR_TIPO.get(v['tipo_dato'], 'gray')}-badge[{v['tipo_dato']}]",
                              f":gray-badge[{fmt_familia(v['familia'])}]",
                              f":gray-badge[{v['origen']}]"]))
        st.markdown(v["descripcion"])
        a, b = st.columns([3, 2], gap="large")
        with a:
            for campo, titulo in (("lectura", "Cómo leerla"), ("calculo", "Cómo se construye"),
                                  ("observaciones", "Observación")):
                if isinstance(v[campo], str) and v[campo]:
                    st.markdown(f"**{titulo}:** {v[campo]}")
            st.markdown(f"**Valores de ejemplo:** {v['ejemplos']}")
        with b:
            st.markdown(f"**Fuente:** `{v['fuente']}`  \n**Formato:** {v['formato']}  \n**Ventana:** {v['ventana']}")
            x1, x2, x3 = st.columns(3)
            x1.metric("Fill rate", pct(v["fill_rate"]))
            x2.metric("Nulos", f"{int(v['n_nulos']):,}")
            x3.metric("Valores únicos", f"{int(v['n_unicos']):,}")


def tabla_seleccionable(tabla: pd.DataFrame, clave: str, alto: int) -> str | None:
    """Dataframe con selección de una fila; devuelve la variable elegida."""
    evento = st.dataframe(tabla, hide_index=True, width="stretch", height=alto, column_config=COLUMNAS_DICC,
                          on_select="rerun", selection_mode="single-row", key=clave)
    filas = evento.selection.rows
    return tabla.iloc[filas[0]]["variable"] if filas else None


tab_fam, tab_dicc, tab_datos = st.tabs([":material/category: Familias de variables",
                                        ":material/menu_book: Diccionario completo",
                                        ":material/table_rows: Vista de los datos"])

# ── Pestaña 1: familias ────────────────────────────────────────────────────────
with tab_fam:
    st.subheader("Panorama de las familias")
    panorama = fam[["familia", "variables", "construidas", "fuente", "fill_rate", "pregunta_negocio"]].copy()
    panorama["familia"] = panorama["familia"].map(fmt_familia)
    st.dataframe(
        panorama, hide_index=True, width="stretch", height=38 + 35 * len(panorama),
        column_config={
            "familia": st.column_config.TextColumn("Familia", width=260),
            "variables": st.column_config.NumberColumn("Variables", width="small"),
            "construidas": st.column_config.NumberColumn("Construidas", width=95,
                                                         help="Variables agregadas desde tablas históricas."),
            "fuente": st.column_config.TextColumn("Tabla(s) fuente", width="medium"),
            "fill_rate": st.column_config.ProgressColumn("Fill rate promedio", format="percent",
                                                         min_value=0, max_value=1, width="small"),
            "pregunta_negocio": st.column_config.TextColumn("Pregunta de negocio que responde", width="large"),
        },
    )

    st.subheader("Explorar una familia")
    elegida = st.selectbox("Familia", fam["familia"], index=len(fam) - 2, format_func=fmt_familia,
                           label_visibility="collapsed")
    f = fam.set_index("familia").loc[elegida]
    vars_f = dicc[dicc["familia"] == elegida]

    with st.container(border=True):
        st.markdown(f"### {f['icono']} {f['nombre_corto']}")
        izq, der = st.columns([3, 2], gap="large")
        with izq:
            st.markdown(f["descripcion"])
            st.markdown(f"**Cómo se obtiene:** {f['construccion']}")
        with der:
            st.markdown(f"**Pregunta de negocio**  \n*{f['pregunta_negocio']}*")
            st.markdown(f"**Fuente:** `{f['fuente']}`")
            tipos = vars_f["tipo_dato"].value_counts()
            st.markdown("**Tipos de dato:** " + " ".join(
                f":{COLOR_TIPO.get(t, 'gray')}-badge[{t} · {n}]" for t, n in tipos.items()))
        c1, c2, c3 = st.columns(3)
        c1.metric("Variables", len(vars_f))
        c2.metric("Fill rate promedio", pct(vars_f["fill_rate"].mean()))
        c3.metric("Fill rate mínimo", pct(vars_f["fill_rate"].min()),
                  help=f"Variable con menos datos: {vars_f.loc[vars_f['fill_rate'].idxmin(), 'variable']}")

    st.caption(":material/touch_app: Haz clic en una fila para ver la ficha completa de la variable.")
    elegida_var = tabla_seleccionable(vars_f[["variable", "descripcion", "tipo_dato", "formato", "fill_rate"]],
                                      clave=f"tabla_{elegida[:2]}", alto=min(38 + 35 * len(vars_f), 460))
    if elegida_var:
        ficha_variable(elegida_var)

# ── Pestaña 2: diccionario completo ────────────────────────────────────────────
with tab_dicc:
    f1, f2, f3, f4 = st.columns([2.2, 2, 1.6, 1.4], vertical_alignment="bottom")
    texto = f1.text_input("Buscar", placeholder="Nombre o descripción (p. ej. atraso)",
                          icon=":material/search:")
    sel_fam = f2.multiselect("Familia", fam["familia"], format_func=fmt_familia, placeholder="Todas")
    sel_tipo = f3.multiselect("Tipo de dato", sorted(dicc["tipo_dato"].unique()), placeholder="Todos")
    sel_origen = f4.segmented_control("Origen", ["Original", "Construida"], selection_mode="multi",
                                      default=["Original", "Construida"])

    filtro = dicc.copy()
    if texto:
        t = texto.strip().lower()
        filtro = filtro[filtro["variable"].str.lower().str.contains(t, regex=False)
                        | filtro["descripcion"].str.lower().str.contains(t, regex=False)]
    if sel_fam:
        filtro = filtro[filtro["familia"].isin(sel_fam)]
    if sel_tipo:
        filtro = filtro[filtro["tipo_dato"].isin(sel_tipo)]
    filtro = filtro[filtro["origen"].isin(sel_origen or [])]

    cab, desc = st.columns([4, 1], vertical_alignment="center")
    cab.caption(f"{len(filtro)} de {len(dicc)} variables")
    desc.download_button("Descargar CSV", dicc.to_csv(index=False).encode("utf-8"),
                         file_name="diccionario_tablon.csv", mime="text/csv", icon=":material/download:",
                         width="stretch")

    vista = filtro[["variable", "familia", "descripcion", "tipo_dato", "fill_rate"]].copy()
    vista["familia"] = vista["familia"].map(fmt_familia)
    if filtro.empty:
        st.info("Ninguna variable coincide con los filtros.", icon=":material/filter_alt_off:")
    else:
        st.caption(":material/touch_app: Haz clic en una fila para ver la ficha completa de la variable.")
        elegida_var = tabla_seleccionable(vista, clave="tabla_diccionario", alto=460)
        if elegida_var:
            ficha_variable(elegida_var)

# ── Pestaña 3: vista de los datos ──────────────────────────────────────────────
with tab_datos:
    st.markdown(
        f"Muestra aleatoria de **{len(muestra)} créditos** del tablón (semilla fija). Elige qué familias ver para no "
        "perderte entre 148 columnas; `SK_ID_CURR` y `TARGET` siempre se muestran."
    )
    d1, d2 = st.columns([4, 1.3], vertical_alignment="bottom")
    por_defecto = [x for x in fam["familia"] if x.startswith(("01", "07", "10", "13"))]
    ver_fam = d1.multiselect("Familias a mostrar", fam["familia"], default=por_defecto, format_func=fmt_familia)
    solo_default = d2.toggle("Solo TARGET = 1", help="Muestra solo los créditos con dificultades de pago.")

    cols = ["SK_ID_CURR", "TARGET"] + [c for c in dicc.loc[dicc["familia"].isin(ver_fam), "variable"]
                                       if c not in ("SK_ID_CURR", "TARGET")]
    datos = muestra[muestra["TARGET"] == 1] if solo_default else muestra
    st.dataframe(
        datos[cols], hide_index=True, width="stretch", height=420,
        column_config={
            "SK_ID_CURR": st.column_config.NumberColumn("SK_ID_CURR", format="%d", pinned=True),
            "TARGET": st.column_config.NumberColumn("TARGET", format="%d", pinned=True),
        },
    )
    st.caption(f"{len(datos)} créditos × {len(cols)} columnas. Las celdas vacías son nulos.")

    st.subheader("Un crédito, variable por variable")
    st.markdown("Selecciona un crédito para leer cada valor junto a su significado.")
    credito = st.selectbox("Crédito (SK_ID_CURR)", datos["SK_ID_CURR"], label_visibility="collapsed",
                           format_func=lambda i: f"SK_ID_CURR {i}")
    fila = datos.set_index("SK_ID_CURR").loc[credito]
    ficha = dicc[dicc["variable"].isin(cols) & (dicc["variable"] != "SK_ID_CURR")][
        ["variable", "familia", "descripcion"]].copy()
    ficha.insert(1, "valor", [fmt_valor(fila[c]) for c in ficha["variable"]])
    ficha["familia"] = ficha["familia"].map(fmt_familia)
    estado = "con dificultades de pago (TARGET = 1)" if fila["TARGET"] == 1 else "sin dificultades de pago (TARGET = 0)"
    st.markdown(f"Crédito **{credito}** · {estado}")
    st.dataframe(ficha, hide_index=True, width="stretch", height=min(38 + 35 * len(ficha), 520),
                 column_config={**COLUMNAS_DICC, "valor": st.column_config.TextColumn("Valor", width="small")})

st.divider()
st.caption("Fuentes: diccionario oficial del curso (`data_dictionary/HomeCredit_diccionario.xlsx`) y descripción de columnas "
           "de Kaggle. Familias, formatos y ventanas: elaboración propia (`data_dictionary/familias.csv`, "
           "`variables_metadata.csv`). El diccionario se regenera con `scripts/build_diccionario.py`.")
