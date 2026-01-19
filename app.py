import streamlit as st
import geopandas as gpd
import branca.colormap as cm
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

@st.cache_data
def get_mapa_continuo():
    return gpd.read_file("./data/Predicciones/mapa_continuo.gpkg")

model = get_model()
grilla = get_grilla()
gdf_continuo = get_mapa_continuo()

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
# Colormap
vmin = gdf_continuo["microplastics_log_est"].min()
vmax = gdf_continuo["microplastics_log_est"].max()
colormap = cm.linear.YlOrRd_09.scale(vmin, vmax)
colormap.caption = "Concentración estimada de microplásticos (log items/m³)"

# Subsample
gdf_plot = gdf_continuo.sample(5000, random_state=42)

# Overlay de microplásticos
for _, row in gdf_plot.iterrows():
    lat = row.geometry.y
    lon = row.geometry.x
    color = colormap(row["microplastics_log_est"])
    folium.CircleMarker(
        location=[lat, lon],
        radius=5,
        fill=True,
        fill_color=color,
        fill_opacity=0.6,
    color=None
    ).add_to(m)
# Leyenda
colormap.add_to(m)   

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
        st.warning(result.get("message"))
        st.info("Por favor, selecciona un punto dentro del océano para obtener una predicción.")
    else:
        st.error("Resultado inesperado.")

else:
    st.info("Haz clic en el mapa para seleccionar un punto del océano.")

