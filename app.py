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


# CAPA 3
ECO_PERC = {
    "eco_count": {
        "low": 53,
        "medium": 147,
        "high": 464,
    }
}
@st.cache_data
def get_iucn_risk_distribution():
    gdf_risk = gpd.read_file(
        "./data/Predicciones/risk_prediction_with_confidence.gpkg"
    )
    return gdf_risk["iucn_mean_risk"].dropna().values

# Imports nuevos
from src.ecology import (
    load_ecology_models,
    build_cell_dataframe,
    predict_ecological_impact
)

# Cargar modelos (cacheados)
@st.cache_resource
def get_ecology_models():
    return load_ecology_models()

rf_risk, rf_species = get_ecology_models()

# Checkbox para activar exploración
st.header("Implicaciones ecológicas esperadas")

explore_ecology = st.checkbox(
    "Explorar escenarios ecológicos alternativos",
    help=(
        "Permite explorar cómo cambiarían las implicaciones ecológicas "
        "si el contexto ecológico local fuese diferente al observado."
    )
)
#Solo se exponen al usuario aquellas variables que puede interpretar 
# y para las que existen rangos empíricos claros. El resto se mantiene
#  en su valor de ausencia, reproduciendo el comportamiento por defecto del modelo.

# Inputs semánticos (solo si el checkbox está activo)
# Valores por defecto (modo no exploratorio)
richness_level = "Media"
complexity_level = 2
eco_overrides = {}

if explore_ecology:
    st.subheader("Escenario ecológico hipotético")

    richness_level = st.selectbox(
        "Presencia de especies en riesgo",
        ["Baja", "Media", "Alta"],
        index=1
    )

    complexity_level = st.selectbox(
        "Cantidad de formas de microplásticos presentes",
        [2, 3, 4],
        index=1
    )

# Mapeo de escenarios a valores reales
level_map_richness = {
    "Baja": "low",
    "Media": "medium",
    "Alta": "high"
}

eco_overrides = {
    "eco_count": ECO_PERC["eco_count"][level_map_richness[richness_level]],
    "eco_shape_richness": complexity_level,
}
cell_df = build_cell_dataframe(
    hazard_index=hazard_index,
    mp_real=mp_real,
    morphology={
        "fibers": proportions["fibers"],
        "fragments": proportions["fragments"],
        "spheres": proportions["spheres"],
        "others": proportions["others"],
    },
    env_vars=env_vars,
    extra_ecology={
        "eco_count": eco_overrides.get("eco_count", 0),
        "eco_shape_richness": eco_overrides.get("eco_shape_richness", 2),  # 👈 baseline correcto
        "eco_mean_size": 0,
        "eco_small_ratio": 0,
        "log_dist_m": 0,
        "eco_dist_m": 0,
        "noaa_mp_mean": mp_real,
        "noaa_mp_max": mp_real,
        "noaa_mp_count": 1,
    },
)

with st.expander("¿Qué variables se están usando ahora?"):

    st.markdown("### Variables ambientales (observadas)")
    st.write(
        "- Temperatura\n"
        "- Salinidad\n"
        "- Clorofila\n"
        "- Nutrientes\n"
        "- Oxígeno"
    )

    st.markdown("### Presión por microplásticos")
    st.write(
        f"- Concentración estimada: {mp_real:.2f} items/m³\n"
        f"- Hazard index: {hazard_index:.2f}"
    )

    if explore_ecology:
        st.markdown("### Contexto ecológico (escenario definido por el usuario)")
        st.write(
            f"- Presencia de especies vulnerables en riesgo (eco_count): {eco_overrides['eco_count']:.0f}\n"
            f"- Diversidad morfológica (eco_shape_richness): {eco_overrides['eco_shape_richness']}"
        )
    else:
        st.markdown("### Contexto ecológico (no observado)")
        st.write(
            "- Presencia de especies en riesgo: asumida como ausente\n"
            "- Diversidad morfológica: mínima plausible (2 formas)"
        )

    st.markdown("### Variables no utilizadas activamente")
    st.write(
        "- Tamaño medio de organismos\n"
        "- Proporción de organismos pequeños\n"
        "- Distancias ecológicas\n"
        "\nEstas variables no informan la predicción en el estado actual."
    )

# ------------------------------------
# Indicador de modo de predicción
# ------------------------------------

if explore_ecology:
    mode_icon = "🟡"
    mode_title = "Escenario ecológico exploratorio"
    mode_text = (
        "El resultado se basa en un escenario ecológico definido por el usuario, "
        "acotado por valores observados reales."
    )
else:
    mode_icon = "🔵"
    mode_title = "Predicción por extrapolación ambiental"
    mode_text = (
        "El resultado se basa en variables ambientales y presión por microplásticos. "
        "No hay información ecológica local observada."
    )

st.markdown(
    f"""
    ### {mode_icon} {mode_title}
    {mode_text}
    """
)

# Predicción
eco_result = predict_ecological_impact(
    cell_df,
    rf_risk,
    rf_species
)

# Calcular percentil del valor predicho
risk_dist = get_iucn_risk_distribution()
risk_pred = eco_result["risk_mean"]

percentile = (risk_dist < risk_pred).mean() * 100

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Riesgo ecológico por presión medio esperado",
        f"{eco_result['risk_mean']:.2f} / 4"
    )

with col2:
    st.metric(
        "Especies vulnerables en riesgo por presión esperadas",
        f"{eco_result['species_at_risk']:.1f}"
    )

st.caption(
    "Resultados basados en patrones globales. "
    "Interpretar a escala regional."    
)
st.caption(
    "Este resultado no representa un efecto causal directo de los microplásticos sobre las especies, sino una estimación de las implicaciones ecológicas potenciales asociadas a niveles de presión por microplásticos bajo condiciones ambientales similares a las observadas"
)
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.hist(risk_dist, bins=30, alpha=0.7)
ax.axvline(risk_pred, color="red", linewidth=2)
ax.set_xlabel("iucn_mean_risk observado")
ax.set_ylabel("Frecuencia")
ax.set_title("Distribución global del riesgo ecológico")

st.pyplot(fig)

st.write(
    f"Este valor se sitúa en el percentil **{percentile:.1f}** "
    "del riesgo ecológico observado."
)

if explore_ecology:
    st.warning(
        "Este resultado corresponde a un escenario ecológico exploratorio "
        "definido por el usuario y no a una predicción basada únicamente en datos observados."
    )
else:
    st.info(
        "Este resultado se basa en patrones aprendidos a partir de datos observados "
        "y extrapolación ambiental en regiones sin información ecológica directa."
    )

# CAPA 3 toma el hazard que ya calculamos y lo traduce en consecuencias ecológicas esperadas.