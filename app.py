import os

import certifi
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.datasets import fetch_california_housing


st.set_page_config(
    page_title="Viviendas de California",
    page_icon="🏠",
    layout="wide",
)


@st.cache_data
def cargar_datos():
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    datos = fetch_california_housing()
    df = pd.DataFrame(datos.data, columns=datos.feature_names)
    df["MedHouseVal"] = datos.target
    return df


st.title("Análisis interactivo de viviendas de California")
st.write(
    "Explora los datos del censo de California de 1990. "
    "El valor objetivo `MedHouseVal` está expresado en cientos de miles de dólares."
)

df_california = cargar_datos()

st.header("Fase 1: carga y preparación de datos")
st.subheader("Primeras cinco filas")
st.dataframe(df_california.head(5), width="stretch")

st.subheader("Tipo de dato de cada columna")
st.write(df_california.dtypes.astype(str))

st.subheader("Valores faltantes por columna")
valores_faltantes = df_california.isnull().sum()
st.write(valores_faltantes)
if valores_faltantes.sum() == 0:
    st.success("El conjunto de datos no contiene valores faltantes.")

st.sidebar.markdown(
    """
    ## Filtros
    Selecciona la edad mediana de las viviendas y, opcionalmente,
    una latitud mínima para explorar distintos vecindarios.
    """
)

st.sidebar.subheader("Edad mediana de la vivienda")
edad_minima = float(df_california["HouseAge"].min())
edad_maxima = float(df_california["HouseAge"].max())
rango_edad = st.sidebar.slider(
    "Rango de HouseAge",
    min_value=edad_minima,
    max_value=edad_maxima,
    value=(edad_minima, edad_maxima),
    step=1.0,
)

activar_filtro_latitud = st.sidebar.checkbox("Activar filtro de latitud mínima")
if activar_filtro_latitud:
    latitud_minima = st.sidebar.number_input(
        "Latitud mínima",
        min_value=float(df_california["Latitude"].min()),
        max_value=float(df_california["Latitude"].max()),
        value=float(df_california["Latitude"].min()),
        step=0.1,
        format="%.2f",
    )

df_filtrado = df_california[
    df_california["HouseAge"].between(rango_edad[0], rango_edad[1])
].copy()
if activar_filtro_latitud:
    df_filtrado = df_filtrado[df_filtrado["Latitude"] >= latitud_minima]

st.header("Fase 2: análisis descriptivo interactivo")
st.write(f"Registros después de aplicar los filtros: {len(df_filtrado):,}")

if df_filtrado.empty:
    st.info("No hay viviendas que coincidan con los filtros seleccionados.")
else:
    valor_mediano = df_filtrado["MedHouseVal"].median()
    rango_valor = (
        df_filtrado["MedHouseVal"].max() - df_filtrado["MedHouseVal"].min()
    )
    columna_mediana, columna_rango = st.columns(2)
    columna_mediana.metric("Mediana de MedHouseVal", f"{valor_mediano:.3f}")
    columna_rango.metric("Rango de MedHouseVal (máximo - mínimo)", f"{rango_valor:.3f}")

    st.header("Fase 3: visualización dinámica")
    columna_histograma, columna_dispersion = st.columns(2)

    with columna_histograma:
        st.subheader("Distribución del valor mediano de la vivienda")
        figura_histograma, eje_histograma = plt.subplots()
        eje_histograma.hist(df_filtrado["MedHouseVal"], bins=30, color="steelblue")
        eje_histograma.set_xlabel("MedHouseVal (cientos de miles de dólares)")
        eje_histograma.set_ylabel("Cantidad de grupos de bloques")
        eje_histograma.set_title("Distribución de MedHouseVal")
        figura_histograma.tight_layout()
        st.pyplot(figura_histograma)
        plt.close(figura_histograma)

    with columna_dispersion:
        st.subheader("Ingresos y valor mediano de la vivienda")
        figura_dispersion, eje_dispersion = plt.subplots()
        eje_dispersion.scatter(
            df_filtrado["MedInc"],
            df_filtrado["MedHouseVal"],
            alpha=0.35,
            s=12,
            color="darkorange",
        )
        eje_dispersion.set_xlabel("MedInc (ingreso mediano)")
        eje_dispersion.set_ylabel("MedHouseVal (cientos de miles de dólares)")
        eje_dispersion.set_title("Ingresos vs. valor de la vivienda")
        figura_dispersion.tight_layout()
        st.pyplot(figura_dispersion)
        plt.close(figura_dispersion)

    st.subheader("Mapa geográfico de los grupos de bloques filtrados")
    st.map(
        df_filtrado[["Latitude", "Longitude"]].rename(
            columns={"Latitude": "latitude", "Longitude": "longitude"}
        )
    )
