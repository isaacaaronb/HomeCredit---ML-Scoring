"""Contexto del proyecto: Home Credit, el problema de negocio, los datos y el enfoque de trabajo."""
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

REPO_DIR = Path(__file__).resolve().parents[1]
TABLON = REPO_DIR / "artifacts" / "tablon_general.parquet"
URL_REPO = "https://github.com/isaacaaronb/HomeCredit---ML-Scoring"
URL_KAGGLE = "https://www.kaggle.com/competitions/home-credit-default-risk"


@st.cache_data(show_spinner=False)
def resumen_tablon(ruta: str) -> dict:
    """Dimensiones del tablón oficial y tasa de default (solo lee la columna TARGET)."""
    columnas = pq.read_schema(ruta).names
    target = pd.read_parquet(ruta, columns=["TARGET"])["TARGET"]
    return {"filas": len(target), "columnas": len(columnas),
            "defaults": int(target.sum()), "tasa": float(target.mean())}


# ── Encabezado ─────────────────────────────────────────────────────────────────
st.title("Scoring de riesgo de crédito · Home Credit")
st.markdown(
    "Un modelo de **probabilidad de incumplimiento** (*probability of default*, PD) para solicitantes con poco "
    "o ningún historial crediticio, construido de punta a punta: integración de datos, diagnóstico de calidad, "
    "análisis exploratorio, selección de variables y modelos de clasificación."
)

r = resumen_tablon(str(TABLON))
c1, c2, c3, c4 = st.columns(4)
c1.metric("Créditos en la muestra", f"{r['filas']:,}", border=True)
c2.metric("Tasa de default", f"{r['tasa']:.2%}", border=True,
          help=f"Proporción de TARGET = 1: {r['defaults']:,} créditos con dificultades de pago.")
c3.metric("Variables en el tablón", f"{r['columnas']}", border=True,
          help="122 originales de application_train + 8 de Bureau + 16 de historial Home Credit + 2 indicadores de cobertura.")
c4.metric("Tablas fuente integradas", "7", border=True,
          help="application_train, bureau, bureau_balance, previous_application, POS_CASH_balance, "
               "credit_card_balance e installments_payments.")

st.write("")

# ── El problema ────────────────────────────────────────────────────────────────
st.header("El problema", divider="gray")
izq, der = st.columns([3, 2], gap="large")

with izq:
    st.subheader("Home Credit y la inclusión financiera")
    st.markdown(
        """
Home Credit es una financiera de consumo internacional especializada en prestar a personas con historial
crediticio **escaso o inexistente** (población *unbanked* o *thin-file*). En ese segmento el scoring tradicional,
basado en el buró de crédito, pierde poder: sin historial no hay señal. El resultado habitual es un rechazo
o, peor, que el solicitante termine en manos de prestamistas informales en condiciones abusivas.

Para ampliar el acceso al crédito sin deteriorar la calidad de la cartera, Home Credit complementa la
información de buró con **datos alternativos** (por ejemplo, comportamiento transaccional y de pago) y modelos
estadísticos. En 2018 abrió una parte de sus datos en una
[competencia de Kaggle]({url}) con una pregunta concreta:

> **¿Qué tan capaz es cada solicitante de devolver su préstamo?**
""".format(url=URL_KAGGLE)
    )

with der:
    with st.container(border=True):
        st.markdown("**Por qué importa ordenar bien el riesgo**")
        st.markdown(
            """
- **Aprobar a quien no paga** genera pérdida: la pérdida esperada es **EL = PD × LGD × EAD**.
- **Rechazar a quien sí pagaría** es ingreso perdido y exclusión financiera.
- Un buen *ranking* de PD permite fijar **puntos de corte**, **precios según riesgo** y **límites** de exposición.
"""
        )
    with st.container(border=True):
        st.markdown("**Definición del evento (`TARGET`)**")
        st.markdown(
            """
`TARGET = 1` si el cliente tuvo **dificultades de pago**: un atraso de más de *X* días en al menos una de las
primeras *Y* cuotas del crédito; `TARGET = 0` en cualquier otro caso. Home Credit no publica los valores de *X* e *Y*.
"""
        )

st.warning(
    f"**Clases desbalanceadas:** solo el {r['tasa']:.1%} de los créditos incumple. Un modelo que prediga "
    f"\"nadie incumple\" acertaría el {1 - r['tasa']:.1%} de las veces y no serviría para nada; por eso se evalúa "
    "la capacidad de **discriminar** (AUC-ROC, métrica oficial de la competencia, y su equivalente Gini = 2·AUC − 1) "
    "además de la exactitud por clase.",
    icon=":material/balance:",
)

# ── Los datos ──────────────────────────────────────────────────────────────────
st.header("Los datos", divider="gray")
st.markdown(
    "La unidad de análisis es el **crédito objetivo** (`SK_ID_CURR`) de `application_train`. Las tablas históricas "
    "tienen otra granularidad (créditos externos, meses, pagos), así que **cada una se agrega por separado a una fila "
    "por `SK_ID_CURR`** y recién entonces se une con `LEFT JOIN`. Unir directamente las tablas originales "
    "multiplicaría filas y distorsionaría cualquier estadístico."
)

DIAGRAMA = """
digraph {
  rankdir=TB; bgcolor="transparent"; nodesep=0.3; ranksep=0.45;
  node [shape=box, style="rounded,filled", fillcolor="#EAF1FB", color="#2A78D6", fontname="Helvetica", fontsize=13, fontcolor="#0B0B0B", margin="0.15,0.07"];
  edge [color="#8A8A85", arrowsize=0.7];

  bb   [label="bureau_balance\\n27.3 M filas · mensual"];
  bur  [label="bureau\\n1.7 M créditos externos\\n→ 8 variables BUREAU_*"];
  prev [label="previous_application\\n1.7 M solicitudes previas\\n→ 5 variables HC_*"];
  pos  [label="POS_CASH_balance\\n10.0 M filas · mensual\\n→ 3 variables HC_*"];
  cc   [label="credit_card_balance\\n3.8 M filas · mensual\\n→ 4 variables HC_*"];
  ins  [label="installments_payments\\n13.6 M filas · por pago\\n→ 4 variables HC_*"];
  app  [label="application_train  ·  307,511 créditos objetivo  ·  122 variables", fillcolor="#2A78D6", fontcolor="white"];
  tab  [label="Tablón de modelización  ·  307,511 × 148  (+ TIENE_BUREAU, TIENE_HISTORIAL_HOME_CREDIT)", fillcolor="#1BAF7A", color="#1BAF7A", fontcolor="white"];

  bb -> bur;
  {bur prev pos cc ins} -> app;
  app -> tab;
}
"""
st.graphviz_chart(DIAGRAMA, width="content")

d1, d2, d3 = st.columns(3, gap="medium")
with d1, st.container(border=True):
    st.markdown("**Solicitud actual**")
    st.caption("Perfil sociodemográfico, ingresos, montos del crédito, empleo, vivienda y scores externos "
               "(`EXT_SOURCE_1/2/3`) al momento de la solicitud.")
with d2, st.container(border=True):
    st.markdown("**Buró de crédito**")
    st.caption("Créditos en otras entidades y su historial mensual: número de créditos activos, deuda total, "
               "días máximos de atraso y proporción de meses con atraso.")
with d3, st.container(border=True):
    st.markdown("**Historial con Home Credit**")
    st.caption("Solicitudes previas (aprobadas y rechazadas), operaciones POS/Cash, tarjetas de crédito y pagos de "
               "cuotas: atrasos, utilización de línea y pagos tardíos.")

# ── Enfoque ────────────────────────────────────────────────────────────────────
st.header("Nuestro enfoque", divider="gray")
st.markdown(
    "El trabajo sigue la guía metodológica del curso, con un flujo **reproducible y auditable**: cada etapa deja "
    "evidencia (tablas, gráficos y parámetros exportados) y las decisiones de tratamiento se documentan antes de modelar."
)

ETAPAS = [
    ("Construcción del tablón", "Traducción a SQL (DuckDB) de la lógica oficial; reproduce la base oficial con 0 diferencias en sus 148 columnas.", "green", "Completado"),
    ("Análisis estadístico inicial (descriptivo)", "Separación de variables numéricas, categóricas y dicotómicas; métricas de forma, colas y valores centinela.", "green", "Completado"),
    ("Análisis de calidad (diagnóstico)", "Nulos estructurales vs. informativos, centinelas (p. ej. `DAYS_EMPLOYED = 365243`), outliers y alertas por variable.", "green", "Completado"),
    ("Preprocesamiento", "Imputación solo de faltantes despreciables (moda, mediana, razón de montos) e indicadores de ausencia; los scores externos conservan su nulo (tramo propio en el WoE); outliers: valores sin sentido a nulo, filas super extremas eliminadas y capeo p0.1–p99.9 (aprendido en train, partición 80/20).", "green", "Completado"),
    ("EDA univariado", "Distribuciones, ajuste de distribuciones teóricas, boxplots y Q-Q; filtro por varianza, nulos y cardinalidad.", "green", "Completado"),
    ("EDA bivariado vs. TARGET", "Tasa de default por tramos (`qcut` 5 y OptBinning); criterio único para todos los modelos: IV ≥ 0.05 (el Gini se reporta como información).", "green", "Completado"),
    ("EDA multivariado y dataset final", "Redundancia con Spearman (|ρ| > 0.6, sale la de menor IV); dataset de entrenamiento único en dos versiones: original (logística) y SMOTE (ML).", "green", "Completado"),
    ("Modelos", "Regresión logística sobre WoE (referencia), árbol de decisión y Random Forest.", "gray", "Pendiente"),
    ("Evaluación e interpretación", "Matriz de confusión, curvas ROC / AUC, exactitud por clase e importancia de variables.", "gray", "Pendiente"),
]
for i, (etapa, detalle, color, estado) in enumerate(ETAPAS, start=1):
    col_n, col_txt, col_estado = st.columns([0.4, 6, 1.4], vertical_alignment="center")
    col_n.markdown(f"**{i:02d}**")
    col_txt.markdown(f"**{etapa}** — {detalle}")
    col_estado.markdown(f":{color}-badge[{estado}]")

# ── Navegación ─────────────────────────────────────────────────────────────────
st.header("Cómo recorrer esta app", divider="gray")
n1, n2, n3 = st.columns(3, gap="medium")
with n1, st.container(border=True):
    st.markdown("**Datos**")
    st.page_link("pages/1_Tablon_oficial.py", label="Tablón oficial", icon=":material/table_chart:")
    st.page_link("pages/2_Calidad_y_preprocesamiento.py", label="Calidad y preprocesamiento", icon=":material/rule:")
with n2, st.container(border=True):
    st.markdown("**Análisis exploratorio**")
    st.page_link("pages/3_EDA_univariado.py", label="EDA univariado", icon=":material/bar_chart:")
    st.page_link("pages/4_EDA_bivariado.py", label="EDA bivariado", icon=":material/compare_arrows:")
    st.page_link("pages/5_EDA_multivariado.py", label="EDA multivariado", icon=":material/hub:")
with n3, st.container(border=True):
    st.markdown("**Modelamiento**")
    st.page_link("pages/6_Dataset_final.py", label="Dataset final", icon=":material/dataset:")
    st.page_link("pages/7_Modelos.py", label="Modelos", icon=":material/model_training:")
    st.page_link(URL_REPO, label="Código en GitHub", icon=":material/code:")

# ── Pie ────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"Proyecto académico y de portafolio · Curso *Gestión de Riesgo Avanzada* (1FIN51), PUCP · "
    f"Autor: Isaac Benavides · [Código]({URL_REPO}) · Datos: [Home Credit Default Risk]({URL_KAGGLE}) (Kaggle)."
)
