"""HomeCredit - ML Scoring · punto de entrada de la app Streamlit (enrutador de páginas)."""
import streamlit as st

st.set_page_config(
    page_title="HomeCredit · ML Scoring",
    page_icon=":material/credit_score:",
    layout="wide",
)

PAGINAS = {
    "Proyecto": [
        st.Page("pages/0_Contexto.py", title="Contexto del proyecto", icon=":material/home:", default=True),
    ],
    "Datos": [
        st.Page("pages/1_Tablon_oficial.py", title="Tablón oficial", icon=":material/table_chart:"),
        st.Page("pages/2_Calidad_y_preprocesamiento.py", title="Calidad y preprocesamiento", icon=":material/rule:"),
        st.Page("pages/3_Feature_engineering.py", title="Feature engineering", icon=":material/tune:"),
    ],
    "Análisis exploratorio": [
        st.Page("pages/4_EDA_univariado.py", title="EDA univariado", icon=":material/bar_chart:"),
        st.Page("pages/5_EDA_bivariado.py", title="EDA bivariado", icon=":material/compare_arrows:"),
        st.Page("pages/6_EDA_multivariado.py", title="EDA multivariado", icon=":material/hub:"),
    ],
    "Modelamiento": [
        st.Page("pages/7_Dataset_final.py", title="Dataset final", icon=":material/dataset:"),
        st.Page("pages/8_Modelos.py", title="Modelos", icon=":material/model_training:"),
    ],
}

st.navigation(PAGINAS).run()
