import streamlit as st
from streamlit_folium import st_folium
import folium
from src.prediction import (
    load_grilla,
    load_model,
    predict_microplastics_at_point
)

# CONFIGURACIÓN BÁSICA

st.set_page_config(
    page_title="Título de la App",
    layout="wide"
)

st.title("Título de la App")
st.write(
    "Selecciona un punto del océano para estimar la concentración "
    "esperada de microplásticos a partir de las condiciones oceánicas reales."
)

# CARGA DE RECURSOS

@st.cache_resource
def get_model():
    return load_model()

@st.cache_data
def get_grilla():
    return load_grilla()

model = get_model()
grilla = get_grilla()

FEATURES = ['temperature', 'salinity', 'chlorophyll', 'nitrate',
            'phosphate', 'oxygen_dissolved', 'oxygen_utilization']

# CAPA 1  - MAPA INTERACTIVO

st.header("Selección espacial")

# Mapa base (centrado global)
m = folium.Map(
    location=[0, 0],
    zoom_start=2,
    tiles="CartoDB positron"
)

# Render del mapa en Streamlit
map_data = st_folium(
    m,
    width=700,
    height=450
)
# Obtener coordenadas del clic del usuario
if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

    st.success(f"Punto seleccionado: lat={lat:.3f}, lon={lon:.3f}")

    # PREDICCIÓN
    result = predict_microplastics_at_point(
        lon=lon,
        lat=lat,
        grilla=grilla,
        model=model,
        feature_names=FEATURES
    )

    # OUTPUTS

    st.subheader("Resultado")

    if result.get("status") == "water":
        st.subheader("Concentración esperada de microplásticos")

        st.metric(
            label="Estimación",
            value=f"{result['microplastics_real']:.2f} items/m³"
    )

        st.subheader("Variables ambientales asociadas")
        st.json(result["environmental_variables"])

        with st.expander("Ver valor en escala logarítmica"):
            st.write(f"{result['microplastics_log']:.3f}")
    
    elif result.get("status") == "land":
        st.warning(result.get(("message")))
        st.info("Por favor, selecciona un punto dentro del océano para obtener una predicción.")
else:
    st.error("Resultado inesperado.")


