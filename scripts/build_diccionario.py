"""Genera el diccionario del tablón oficial que consume la app (página «Tablón oficial»).

Entradas (versionadas en el repo):
  - data_dictionary/HomeCredit_diccionario.xlsx   diccionario oficial (definiciones, fuente, subtipo, cálculo)
  - data_dictionary/variables_metadata.csv        familia, formato y ventana temporal de cada variable (curado)
  - data_dictionary/familias.csv                  descripción de negocio de cada familia (curado)
  - artifacts/tablon_general.parquet              tablón oficial (307,511 × 148)

Salidas:
  - data_dictionary/diccionario_tablon.csv        una fila por variable, con fill rate y ejemplos
  - artifacts/tablon_muestra.parquet              muestra aleatoria de 200 créditos para la vista de datos

Uso:  python scripts/build_diccionario.py
"""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DD = REPO / "data_dictionary"
TABLON = REPO / "artifacts" / "tablon_general.parquet"
SEMILLA = 42


def tipo_de_dato(var: str, tipo: str, subtipo: str, n_unicos: int) -> str:
    """Clase de dato legible, a partir del diccionario oficial y de los datos."""
    if var == "SK_ID_CURR":
        return "Identificador"
    if var == "TARGET":
        return "Objetivo (dicotómica)"
    if n_unicos == 2:
        return "Dicotómica"
    if tipo == "Categórica":
        return "Categórica ordinal" if subtipo == "Ordinal" else "Categórica nominal"
    if subtipo == "Ordinal":
        return "Categórica ordinal"
    if "Continua" in subtipo or "Score" in subtipo:
        return "Numérica continua"
    return "Numérica discreta"


def ejemplos(s: pd.Series, clase: str) -> str:
    """Valores típicos: categorías más frecuentes (con %) o percentiles 10, 50 y 90."""
    x = s.dropna()
    if x.empty:
        return "—"
    if clase.startswith(("Dicotómica", "Categórica", "Objetivo")):
        vc = x.value_counts(normalize=True).head(3)
        return " · ".join(f"{k} ({v:.0%})" if v >= 0.01 else f"{k} (<1%)" for k, v in vc.items())
    q = x.quantile([0.10, 0.50, 0.90])
    fmt = lambda v: f"{v:,.0f}" if abs(v) >= 1000 or float(v).is_integer() else f"{v:,.3g}"
    return f"p10: {fmt(q[0.10])} · mediana: {fmt(q[0.50])} · p90: {fmt(q[0.90])}"


def main() -> None:
    df = pd.read_parquet(TABLON)
    oficial = pd.read_excel(DD / "HomeCredit_diccionario.xlsx", sheet_name="Diccionario").set_index("Variable final")
    meta = pd.read_csv(DD / "variables_metadata.csv").set_index("variable")
    familias = pd.read_csv(DD / "familias.csv")

    assert list(meta.index) == list(df.columns), "variables_metadata.csv debe seguir el orden del tablón"
    assert set(meta["familia"]) == set(familias["familia"]), "familias sin describir o sobrantes"

    filas = []
    for i, var in enumerate(df.columns, start=1):
        o, m, s = oficial.loc[var], meta.loc[var], df[var]
        n_unicos = int(s.nunique(dropna=True))
        clase = tipo_de_dato(var, str(o["Tipo"]), str(o["Subtipo"]), n_unicos)
        texto = lambda c: "" if pd.isna(o[c]) or str(o[c]).strip() in ("-", "—") else str(o[c]).strip()
        filas.append({
            "orden": i, "variable": var, "familia": m["familia"],
            "origen": "Original" if o["Origen"] == "Original" else "Construida",
            "tipo_dato": clase, "formato": m["formato"], "fuente": o["Tabla fuente"], "ventana": m["ventana"],
            "descripcion": texto("Definición"), "lectura": texto("Interpretación / lectura"),
            "calculo": texto("Cómo se obtiene / cálculo"), "observaciones": texto("Observaciones"),
            "fill_rate": round(float(s.notna().mean()), 6), "n_nulos": int(s.isna().sum()), "n_unicos": n_unicos,
            "ejemplos": ejemplos(s, clase),
        })
    dicc = pd.DataFrame(filas)
    dicc.to_csv(DD / "diccionario_tablon.csv", index=False, encoding="utf-8")

    muestra = df.sample(200, random_state=SEMILLA).sort_values("SK_ID_CURR")
    muestra.to_parquet(REPO / "artifacts" / "tablon_muestra.parquet", compression="zstd", index=False)

    print(f"diccionario_tablon.csv: {dicc.shape} | tablon_muestra.parquet: {muestra.shape}")
    print(dicc.groupby("familia").size().to_string())
    print(dicc["tipo_dato"].value_counts().to_string())


if __name__ == "__main__":
    main()
