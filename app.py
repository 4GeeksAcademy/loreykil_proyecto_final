import streamlit as st
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import branca.colormap as cm
from streamlit_folium import st_folium
import folium
import plotly.graph_objects as go
from shapely.geometry import Point

from src.prediction import (
    load_grilla,
    load_model,
    predict_microplastics_at_point
)
from src.hazard import (
    compute_hazard_pressure,
    compute_hazard_morphology,
    compute_hazard_index,
    hazard_label
)

if "mp_real" not in st.session_state:
    st.session_state.mp_real = None

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

@st.cache_data
def get_hazard_gdf():
    return gpd.read_file("./data/grid_indice/hazard_index_grid_final.gpkg")

@st.cache_data
def get_mp_gdf():
    return gpd.read_file("./data/GeoDataFrame/gdf_microplastics.gpkg")

model = get_model()
grilla = get_grilla()
gdf_continuo = get_mapa_continuo()
hazard_gdf = get_hazard_gdf()
mp_gdf = get_mp_gdf()
mp_observed = mp_gdf["microplastics_measurement"].dropna().values
MP_REF_P95 = np.percentile(mp_observed, 95)

FEATURES = ['temperature', 'salinity', 'chlorophyll', 'nitrate',
            'phosphate', 'oxygen_dissolved', 'oxygen_utilization']


RANGES = {
    "temperature": (-2, 30),
    "salinity": (5, 40),
    "chlorophyll": (-1, 1),
    "nitrate": (0, 37),
    "phosphate": (0, 5),
    "oxygen_dissolved": (30, 400),
    "oxygen_utilization": (0, 300)
}

def normalize(value, vmin, vmax):
    return max(0, min(1, (value - vmin) / (vmax - vmin)))

# CAPA 1  - MAPA INTERACTIVO

st.header("Selección espacial")

def get_ocean_profile_at_point(gdf_continuo, lon, lat):
    point = Point(lon, lat)

    # Asegurar CRS coherente para distancias
    if gdf_continuo.crs.is_geographic:
        gdf_proj = gdf_continuo.to_crs(epsg=3857)
        point_proj = gpd.GeoSeries([point], crs=4326).to_crs(epsg=3857).iloc[0]
    else:
        gdf_proj = gdf_continuo
        point_proj = point

    distances = gdf_proj.geometry.distance(point_proj)
    idx = distances.idxmin()

    return gdf_continuo.loc[idx, "profile_eco"]

def plot_oceanographic_radar(env_vars):
    labels = list(env_vars.keys())

    values = [
        normalize(env_vars[k], *RANGES[k])
        for k in labels
        if k in RANGES
    ]

    labels = labels + [labels[0]]
    values = values + [values[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        name='Perfil oceanográfico'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        margin=dict(l=40, r=40, b=40)
    )
    return fig

@st.cache_resource
def build_map(_gdf_continuo):
    # Mapa base (centrado global)
    m = folium.Map(
        location=[0, 0],
        zoom_start=2,
        tiles="CartoDB voyager"
    )
    # Colormap
    vmin = _gdf_continuo["microplastics_log_est"].min()
    vmax = _gdf_continuo["microplastics_log_est"].max()
    colormap = cm.linear.YlOrRd_09.scale(vmin, vmax)
    colormap.caption = "Concentración estimada de microplásticos (log items/m³)"
    # Subsample
    gdf_plot = _gdf_continuo.sample(5000, random_state=42)

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
            fill_opacity=0.8,
        color=None
        ).add_to(m)
    # Leyenda
    colormap.add_to(m)
    return m

m = build_map(gdf_continuo)

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
        st.session_state.mp_real = result["microplastics_real"]

        st.subheader("Variables ambientales asociadas")
        # Extraer perfil oceanográfico desde el mapa contínuo
        ocean_profile = get_ocean_profile_at_point(
            gdf_continuo,
            lon=lon,
            lat=lat
        )
        # Añadirlo como una variable más
        env_vars = result['environmental_variables'].copy()
        env_vars["ocean_profile"] = ocean_profile
        fig = plot_oceanographic_radar(
            {k: v for k, v in env_vars.items() if k in FEATURES}
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Ver valores numéricos"):
            st.json(env_vars)

        with st.expander("Ver valor en escala logarítmica"):
            st.write(f"{result['microplastics_log']:.3f}")
    
    elif result.get("status") == "land":
        st.warning(result.get("message"))
        st.info("Por favor, selecciona un punto dentro del océano para obtener una predicción.")
    else:
        st.error("Resultado inesperado.")

else:
    st.info("Haz clic en el mapa para seleccionar un punto del océano.")

# CAPA 2  - Hazard Index

st.header("Índice de riesgo por microplásticos")

st.write(
    "Dado el mismo nivel de microplásticos, explora cómo cambia"
    "el riesgo potencial al modificar su composición morfológica"
)

st.subheader("Composición morfológica (%)")

if st.session_state.mp_real is None:
    st.info("Selecciona un punto en el mapa para obtener la concentración de microplásticos.")
    st.stop()
mp_real = st.session_state.mp_real
fibers = st.slider("Fibras", 0, 100, 40)
fragments = st.slider("Fragmentos", 0, 100, 30)
spheres = st.slider("Esferas", 0, 100, 20)
others = st.slider("Otros", 0, 100, 10)

total = fibers + fragments + spheres + others

# Cálculo de Hazard Index
if total != 100:
    st.warning("Las proporciones deben sumar 100 %.")
    st.stop()
else:
    proportions = {
        "fibers": fibers / 100,
        "fragments": fragments / 100,
        "spheres": spheres / 100,
        "others": others / 100
    }

    hazard_pressure = compute_hazard_pressure(mp_concentration=mp_real, ref_max=MP_REF_P95)
    hazard_morphology = compute_hazard_morphology(proportions)
    hazard_index = compute_hazard_index(hazard_pressure, hazard_morphology)
    label = hazard_label(hazard_index)

# OUTPUTS
 # Hazard Index + etiqueta
    
st.subheader("Hazard Index")

st.metric(
    label="Riesgo potencial por microplásticos",
    value=f"{hazard_index:.2f}",
    help="Índice normalizado entre 0 y 1"
)

st.write(f"**Nivel:** {label}")

# OUTPUTS
 # Barra horizontal de colores
    
st.progress(hazard_index)
# Gráfico de indicador
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=hazard_index,
    domain={'x': [0, 1], 'y': [0, 1]},
    gauge={
        'axis': {'range': [0, 1]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 0.33], 'color': "green"},
            {'range': [0.33, 0.66], 'color': "orange"},
            {'range': [0.66, 1], 'color': "red"}
        ],
    }
))

st.plotly_chart(fig, use_container_width=True)

# Gráfico de distribución del Hazard Index observado

hazard_values = hazard_gdf["hazard_index"].values
percentile = (hazard_values < hazard_index).mean() * 100

fig, ax = plt.subplots()
ax.hist(hazard_values, bins=30, alpha=0.7)
ax.axvline(hazard_index, color="red", linewidth=2)
ax.set_xlabel("Hazard Index observado")
ax.set_ylabel("Frecuencia")

st.pyplot(fig)

st.write(
    f"Este valor de riesgo se sitúa en el percentil **{100 - percentile:.1f}** del Hazard Index observado"
)

