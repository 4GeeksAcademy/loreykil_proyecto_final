import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import branca.colormap as cm
from streamlit_folium import st_folium
import folium
import plotly.graph_objects as go
from shapely.geometry import Point
from statsmodels.nonparametric.smoothers_lowess import lowess

from src.prediction import (
    load_grilla,
    load_model,
    predict_microplastics_at_point
)
from src.hazard import (
    compute_hazard_pressure,
    compute_hazard_morphology,
    compute_hazard_index,
    hazard_label,
    compute_hazard_morphology_from_props
)
from src.ecology import (
    load_ecology_models,
    build_cell_dataframe,
    predict_ecological_impact,
    predict_ecological_impact_index,
    predict_hazard_coherence,
    load_ecology_models_index
)
# CONFIGURACIÓN BÁSICA

if "mp_real" not in st.session_state:
    st.session_state.mp_real = None

if "hazard_index" not in st.session_state:
    st.session_state.hazard_index = None

st.set_page_config(
    page_title="Título de la App",
    layout="wide"
)

st.title("Título de la App")
mode = st.radio(
    "",
    ["Flujo interactivo", "Análisis por capas"],
    horizontal=True
)
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
mp_gdf = get_mp_gdf()
hazard_gdf = get_hazard_gdf()

mp_observed = mp_gdf["microplastics_measurement"].dropna().values
MP_REF_P90 = np.percentile(mp_observed, 90)



# CONFIGURACIÖN SEMÁNTICA

FEATURES = ['temperature', 'salinity', 'chlorophyll', 'nitrate',
            'phosphate', 'oxygen_dissolved', 'oxygen_utilization'
]

RANGES = {
    "temperature": (-2, 30),
    "salinity": (5, 40),
    "chlorophyll": (-1, 1),
    "nitrate": (0, 37),
    "phosphate": (0, 5),
    "oxygen_dissolved": (30, 400),
    "oxygen_utilization": (0, 300)
}

ENV_VARS_META = {
    "temperature": {
        "label": "Temperatura del agua",
        "unit": "°C"
    },
    "salinity": {
        "label": "Salinidad",
        "unit": None
    },
    "chlorophyll": {
        "label": "Clorofila-a",
        "unit": "mg/m³"
    },
    "nitrate": {
        "label": "Nitrato",
        "unit": "µmol/L"
    },
    "phosphate": {
        "label": "Fosfato",
        "unit": "µmol/L"
    },
    "oxygen_dissolved": {
        "label": "Oxígeno disuelto",
        "unit": "µmol/kg"
    },
    "oxygen_utilization": {
        "label": "Utilización de oxígeno",
        "unit": "µmol/kg"
    }
}

PROFILE_DESCRIPTIONS = {
    "subpolar_open_sea": (
        "Aguas frías de mar abierto con alta disponibilidad de nutrientes "
        "pero baja producción primaria. "
        "Característico de regiones subpolares con fuerte mezcla vertical "
        "y consumo elevado de oxígeno."
    ),
    "open_sea_temperate": (
        "Región oceánica alejada de la costa, con temperaturas "
        "templadas y baja productividad primaria. "
        "Aguas bien oxigenadas, con concentraciones moderadas de "
        "nutrientes, típicas de sistemas oligotróficos de mar abierto."
    ),
    "cold_open_sea": (
        "Región oceánica remota y fría, con alta disponibilidad de "
        "nutrientes y productividad relativamente elevada. "
        "Aguas bien oxigenadas, asociadas a sistemas productivos de "
        "latitudes altas o zonas de afloramiento lejano."
    ),
    "cold_shelf": (
        "Zona costera fría, influenciada por aportes continentales y "
        "mezcla costera. "
        "Presenta mayor productividad relativa y aguas muy oxigenadas, "
        "características de plataformas frías y dinámicas."
    ),
    "warm_coastal": (
        "Zona costera cálida, cercana a tierra, con productividad "
        "moderada y bajos nutrientes. "
        "Representa sistemas costeros oligotróficos influenciados por "
        "temperaturas elevadas"
    ),
}

PROFILE_LABELS = {
    "subpolar_open_sea": "Mar abierto subpolar",
    "open_sea_temperate": "Mar abierto templado",
    "cold_open_sea": "Mar abierto frío",
    "cold_shelf": "Plataforma continental fría",
    "warm_coastal": "Costa cálida"
}

PROFILE_COLORS = {
    "subpolar_open_sea": "#1f77b4",     # azul
    "open_sea_temperate": "#ff7f0e",  # naranja
    "cold_open_sea": "#2ca02c",      # verde
    "cold_shelf": "#9467bd",      # morado
    "warm_coastal": "#189E93"
}
ECO_PERC = {
    "eco_count": {
        "low": 53,
        "medium": 147,
        "high": 464,
    }
}
WEIGHTS = {
    "fibers": 1.0,
    "fragments": 0.7,
    "spheres": 0.4,
    "others": 0.2
}

# FUNCIONES AUXILIARES

def normalize(value, vmin, vmax):
    return max(0, min(1, (value - vmin) / (vmax - vmin)))

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

@st.cache_data
def compute_profile_means(_gdf_continuo, profile_col="profile_eco"):
    return (
        _gdf_continuo
        .groupby(profile_col)[FEATURES]
        .mean()
    )
profile_means = compute_profile_means(gdf_continuo)


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

def plot_oceanographic_radar(env_point, env_mean, profile_name):
    tech_labels = FEATURES
    display_labels = [ENV_VARS_META[k]["label"] for k in FEATURES]
    values_point = [
        normalize(env_point[k], *RANGES[k])
        for k in tech_labels
    ]
    values_mean = [
        normalize(env_mean[k], *RANGES[k])
        for k in tech_labels
    ]

    # cerrar polígonos
    display_labels_closed = display_labels + [display_labels[0]]
    values_point += [values_point[0]]
    values_mean += [values_mean[0]]

    profile_label = PROFILE_LABELS.get(profile_name, profile_name)
    color_profile = PROFILE_COLORS.get(profile_name, "#999999")
    
    fig = go.Figure()

    # Perfil medio (sombreado)
    fig.add_trace(go.Scatterpolar(
        r=values_mean,
        theta=display_labels_closed,
        fill='toself',
        fillcolor=color_profile,
        line=dict(color=color_profile),
        opacity=0.3,
        name=f'Perfil oceanográfico ({profile_label})'
    ))

    # Punto seleccionado (línea)
    fig.add_trace(go.Scatterpolar(
        r=values_point,
        theta=display_labels_closed,
        fill=None,
        line=dict(
            width=2,
            color="black"
        ),
        marker=dict(size=6),
        name="Punto seleccionado"
    ))

    return fig
@st.cache_data
def get_iucn_risk_distribution():
    gdf_risk = gpd.read_file(
        "./data/Predicciones/risk_prediction_with_confidence.gpkg"
    )
    return gdf_risk["iucn_mean_risk"].dropna().values

@st.cache_resource
def get_ecology_models():
    return load_ecology_models()

# INTERACCIÓN CAPA 1
if mode == "Flujo interactivo":
    st.header("Selección espacial")
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
            # Microplásticos
            st.subheader("Concentración esperada de microplásticos")
            st.metric(
                label="Estimación",
                value=f"{result['microplastics_real']:.2f} items/m³"
            )
            st.session_state.mp_real = result["microplastics_real"]
            # 🔴 MUY IMPORTANTE: invalidar hazard al cambiar de punto
            st.session_state.hazard_index = None
            with st.expander("Ver valor en escala logarítmica"):
                st.write(f"{result['microplastics_log']:.3f}")

            # Extraer perfil oceanográfico desde el mapa contínuo
            ocean_profile = get_ocean_profile_at_point(
                gdf_continuo,
                lon=lon,
                lat=lat
            )
            # Perfil ecológico
            profile_label = PROFILE_LABELS.get(ocean_profile, ocean_profile)
            color = PROFILE_COLORS.get(ocean_profile, "#999999")
            
            st.subheader("Perfil oceanográfico asociado al punto")
            st.markdown(
                f"""
                <div style="
                color:{color};
                font-weight:600;
                font-size:16px;
                ">
                    {profile_label}
                </div>
                """,
                unsafe_allow_html=True
                )
            description = PROFILE_DESCRIPTIONS.get(
                ocean_profile,
                "No hay descripción disponible para este perfil."
            )
            st.markdown(
                f"""
                <div style="
                    border-left:6px solid {color};
                    background-color:rgba(0,0,0,0.03);
                    padding:12px 14px;
                    border-radius:6px;
                    font-size:14px;
                    line-height:1.5;
                    margin-top:6px;
                    margin-bottom:12px;
                ">
                    {description}
                </div>
                """,
                unsafe_allow_html=True
            )


            # Añadirlo como una variable más
            env_vars = result['environmental_variables'].copy()
            env_vars["ocean_profile"] = ocean_profile
            st.session_state.env_vars = env_vars
            profile_name = ocean_profile
            env_mean = profile_means.loc[profile_name].to_dict()
            fig = plot_oceanographic_radar(
                env_point=env_vars,
                env_mean=env_mean,
                profile_name=profile_name
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Los valores están normalizados (0–1) y no representan unidades físicas.")

            with st.expander("Ver valores numéricos"):
                pretty_env_vars = pd.DataFrame([
                    {
                        "Variable": ENV_VARS_META[k]["label"],
                        "Valor": round(float(env_vars[k]), 3),
                        "Unidad": ENV_VARS_META[k]["unit"] or "-"
                    }
                    for k in FEATURES
                    
                ])
                st.table(pretty_env_vars)

            
        
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

    st.sidebar.subheader("Composición morfológica dominante")
    
    profile = st.sidebar.selectbox(
        "Selecciona un perfil",
        [
            "Dominio de fibras",
            "Dominio de fragmentos",
            "Dominio de esferas",
            "Mezcla equilibrada",
            "Personalizado"
        ]
    )
    
    proportions_ready = True

    if profile == "Dominio de fibras":
        proportions = {"fibers": 0.7, "fragments": 0.15, "spheres": 0.1, "others": 0.05}

    elif profile == "Dominio de fragmentos":
        proportions = {"fibers": 0.15, "fragments": 0.7, "spheres": 0.1, "others": 0.05}

    elif profile == "Dominio de esferas":
        proportions = {"fibers": 0.1, "fragments": 0.15, "spheres": 0.7, "others": 0.05}

    elif profile == "Mezcla equilibrada":
        proportions = {"fibers": 0.25, "fragments": 0.25, "spheres": 0.25, "others": 0.25}

    elif profile == "Personalizado":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            fibers = st.sidebar.slider("Fibras", 0, 100, 40)
            fragments = st.sidebar.slider("Fragmentos", 0, 100, 30)
        with col2:
            spheres = st.sidebar.slider("Esferas", 0, 100, 20)
            others = st.sidebar.slider("Otros", 0, 100, 10)
        total = fibers + fragments + spheres + others
        if total != 100:
            proportions_ready = False
        else:
            proportions = {
            "fibers": fibers / 100,
            "fragments": fragments / 100,
            "spheres": spheres / 100,
            "others": others / 100
        }

    hazard_ready = (
        st.session_state.mp_real is not None
        and proportions_ready
    )

    if not hazard_ready:
        if st.session_state.mp_real is None:
            st.info("Selecciona un punto en el mapa para obtener microplásticos.")
        if not proportions_ready:
            st.sidebar.warning("La suma de las proporciones debe ser 100%.")
    else:
        # 🔄 recalcular solo si está invalidado
        if st.session_state.hazard_index is None:
            mp_real = st.session_state.mp_real

            hazard_pressure = compute_hazard_pressure(
                mp_concentration=mp_real,
                ref_max=MP_REF_P90
            )
            
            hazard_morphology = compute_hazard_morphology(proportions)
            st.session_state.hazard_index = compute_hazard_index(
                hazard_pressure,
                hazard_morphology
            )
            st.write("DEBUG hazard_pressure:", hazard_pressure)
            st.write("DEBUG hazard_morphology:", hazard_morphology)

        hazard_index = st.session_state.hazard_index
        label = hazard_label(hazard_index)
    # Visualización
        st.subheader("Hazard Index")

        st.metric(
            label="Riesgo potencial por microplásticos",
            value=f"{hazard_index:.2f}",
            help="Índice normalizado entre 0 y 1"
        )

        st.write(f"**Nivel:** {label}")
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
    st.header("Implicaciones ecológicas esperadas")

    eco_ready = (
        "env_vars" in st.session_state
        and st.session_state.mp_real is not None
    )
    
    hazard_index = st.session_state.hazard_index

    if not eco_ready:
        st.info(
            "Selecciona un punto en el mapa para estimar las implicaciones "
            "ecológicas asociadas a la presión por microplásticos."
        )
        # ⛔ No se calcula nada, pero el título ya está visible
        st.stop()
        
        if st.session_state.mp_real is None:
            st.info(
                "Para activar esta capa, primero selecciona un punto "
                "en el mapa y calcula el índice de riesgo por microplásticos."
            )
            st.stop()

    env_vars = st.session_state.env_vars
    mp_real = st.session_state.mp_real
    rf_risk, rf_species = get_ecology_models()
    # Checkbox para activar exploración

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
    richness_level = "58-147 µm"
    complexity_level = 2
    eco_overrides = {}

    if explore_ecology:
        st.subheader("Escenario ecológico hipotético")

        richness_level = st.selectbox(
            "Tamaño relativo de los ítems presentes",
            ["0-57 µm", "58-147 µm", "148-464 µm"],
            index=1
        )

        complexity_level = st.selectbox(
            "Cantidad de formas de microplásticos presentes",
            [2, 3, 4],
            index=1
        )

    # Mapeo de escenarios a valores reales
    level_map_richness = {
        "0-57 µm": "low",
        "58-147 µm": "medium",
        "148-464 µm": "high"
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

elif mode == "Análisis por capas":

    st.sidebar.header("Capas")

    capa = st.sidebar.radio(
        "",
        ["Microplásticos", "Hazard", "Ecología"]
    )

    if capa == "Microplásticos":
        st.header("Análisis de microplásticos")

        fig, ax = plt.subplots()
        ax.hist(mp_gdf["microplastics_measurement"].dropna(), bins=40)
        st.pyplot(fig)

        st.dataframe(mp_gdf.head(200))

    elif capa == "Hazard":

        st.header("Análisis del Hazard Index")

        st.markdown(
            """
            En esta sección se analiza cómo se comporta el **Hazard Index**, un indicador
            que resume el riesgo potencial asociado a los microplásticos en el océano.

            El objetivo no es evaluar un punto concreto, sino **entender qué valores son habituales,
            de qué depende el índice y cómo debe interpretarse**.
            """
        )

        # =========================================================
        # 1. ¿Qué es el Hazard Index?
        # =========================================================

        st.subheader("¿Qué es el Hazard Index?")

        st.markdown(
            """
            El **Hazard Index** es una medida normalizada entre **0 (riesgo bajo)** y
            **1 (riesgo alto)** que combina:

            - La **cantidad de microplásticos** presentes (presión)
            - La **composición morfológica** de esos microplásticos

            Permite comparar regiones oceánicas en términos de **riesgo potencial**,
            pero **no representa un impacto ecológico directo**.
            """
        )

        # =========================================================
        # 2. Distribución global
        # =========================================================

        st.subheader("¿Qué valores del Hazard Index son habituales?")

        hazard_values = hazard_gdf["hazard_index"].dropna().values

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(hazard_values, bins=30, alpha=0.75)
        ax.axvline(np.percentile(hazard_values, 75), linestyle="--", label="Percentil 75")
        ax.axvline(np.percentile(hazard_values, 90), linestyle=":", label="Percentil 90")
        ax.set_xlabel("Hazard Index")
        ax.set_ylabel("Frecuencia")
        ax.legend()

        st.pyplot(fig)

        st.markdown(
            """
            La mayoría de las regiones oceánicas presentan valores bajos o moderados.
            Los valores altos son menos frecuentes y representan situaciones más extremas.
            """
        )

        # =========================================================
        # 3. Dependencia con presión y morfología
        # =========================================================

        st.subheader("¿De qué depende el Hazard Index?")

        col1, col2 = st.columns(2)

        # ---------------------------------------------------------
        # 3.1 Presión
        # ---------------------------------------------------------

        with col1:
            st.markdown("### Relación con la cantidad de microplásticos")

            x = hazard_gdf["hazard_pressure"]
            y = hazard_gdf["hazard_index"]

            fig, ax = plt.subplots()
            ax.scatter(x, y, s=10, alpha=0.4)

            # LOWESS
            smoothed = lowess(y, x, frac=0.3)

            ax.plot(smoothed[:, 0], smoothed[:, 1], color="red", linewidth=2)
            ax.set_xlabel("Presión por microplásticos (normalizada)")
            ax.set_ylabel("Hazard Index")

            st.pyplot(fig)

            st.caption(
                "A mayor presión por microplásticos, mayor Hazard Index en promedio, "
                "aunque con variabilidad."
            )

        # ---------------------------------------------------------
        # 3.2 Morfología
        # ---------------------------------------------------------

        with col2:
            st.markdown("### Influencia de la composición morfológica")

            x = hazard_gdf["hazard_morphology"]
            y = hazard_gdf["hazard_index"]

            fig, ax = plt.subplots()
            ax.scatter(x, y, s=10, alpha=0.4, color="darkgreen")

            smoothed = lowess(y, x, frac=0.3)
            ax.plot(smoothed[:, 0], smoothed[:, 1], color="black", linewidth=2)

            ax.set_xlabel("Composición morfológica (índice)")
            ax.set_ylabel("Hazard Index")

            st.pyplot(fig)

            st.caption(
                "La composición morfológica introduce diferencias claras en el nivel de riesgo, "
                "incluso con cantidades similares de microplásticos."
            )

        # =========================================================
        # 4. BLOQUE INTERACTIVO — MORFOLOGÍA (CON PESOS EXPLÍCITOS)
        # =========================================================

        st.subheader("🧩 ¿Cómo influye la composición de formas?")

        st.markdown(
            """
            La **composición morfológica** depende de la proporción relativa
            de distintas formas de microplásticos.

            No todas las formas contribuyen por igual al riesgo potencial:
            algunas tienen **mayor peso** que otras.
            """
        )

        col1, col2 = st.columns(2)

        # -------------------------------
        # Sliders (suma automática = 100)
        # -------------------------------
        with col1:
            fibers = st.slider("Fibras", 0, 100, 40)
            fragments = st.slider("Fragmentos", 0, 100 - fibers, 30)
            spheres = st.slider(
                "Esferas",
                0,
                100 - fibers - fragments,
                20
            )

        others = 100 - (fibers + fragments + spheres)

        proportions = {
            "fibers": fibers / 100,
            "fragments": fragments / 100,
            "spheres": spheres / 100,
            "others": others / 100,
        }

        # -------------------------------
        # Índice morfológico
        # -------------------------------
        hazard_morph_user = compute_hazard_morphology_from_props(proportions)

        # -------------------------------
        # Contribución ponderada
        # -------------------------------
        weighted_contrib = {
            k: proportions[k] * WEIGHTS[k]
            for k in proportions
        }

        with col2:
            st.metric(
                "Índice morfológico resultante",
                f"{hazard_morph_user:.2f}"
            )

            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar(
                weighted_contrib.keys(),
                weighted_contrib.values(),
                color=["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
            )
            ax.set_ylabel("Contribución al riesgo morfológico")
            ax.set_ylim(0, 1)
            ax.set_title("Contribución ponderada por forma")

            st.pyplot(fig)

        # -------------------------------
        # Texto explicativo NO TÉCNICO
        # -------------------------------

        st.markdown(
            """
            **¿Por qué algunas barras llegan más alto que otras?**

            Aunque dos formas estén presentes en la misma proporción,
            **no contribuyen igual al riesgo potencial**.

            - Las **fibras** tienen el mayor peso relativo, por lo que
            su contribución puede alcanzar valores cercanos al **100%**
            del riesgo morfológico.
            - Los **fragmentos** tienen un peso intermedio, por lo que su
            contribución máxima es menor (alrededor del **60–70%**).
            - Las **esferas** y otros tipos tienen pesos más bajos y,
            por tanto, su contribución al riesgo es menor, incluso si
            están presentes en cantidades similares.

            Este enfoque permite reflejar que **no todas las formas de
            microplásticos tienen el mismo potencial de riesgo**.
            """
        )

        st.info(
            "Este análisis muestra únicamente el **componente morfológico del riesgo**. "
            "El Hazard Index completo combina este componente con la cantidad total "
            "de microplásticos presente."
        )


        # =========================================================
        # 5. CONTRIBUCIÓN RELATIVA (GRÁFICO REAL)
        # =========================================================

        st.subheader("⚖️ Contribución relativa de presión y morfología")

        st.markdown(
            """
            El Hazard Index surge de la combinación de **presión por microplásticos**
            y **composición morfológica**.

            Las curvas siguientes muestran cómo varía el Hazard Index observado
            cuando cambia cada componente.
            """
        )

        fig, ax = plt.subplots(figsize=(5, 3))

        sm_p = lowess(
            hazard_gdf["hazard_index"],
            hazard_gdf["hazard_pressure"],
            frac=0.3
        )
        sm_m = lowess(
            hazard_gdf["hazard_index"],
            hazard_gdf["hazard_morphology"],
            frac=0.3
        )

        ax.plot(sm_p[:, 0], sm_p[:, 1], label="Variación con la presión", linewidth=2)
        ax.plot(sm_m[:, 0], sm_m[:, 1], label="Variación con la morfología", linewidth=2)

        ax.set_xlabel("Componente del índice (escala propia)")
        ax.set_ylabel("Hazard Index")
        ax.legend()

        st.pyplot(fig)

        st.caption(
            "Ambos factores contribuyen al riesgo potencial, de forma complementaria."
        )
        # =========================================================
        # 6. CALCULADORA INTERACTIVA DEL HAZARD INDEX
        # =========================================================

        st.subheader("🧮 Calculadora del Hazard Index")

        st.markdown(
            """
            En este apartado puedes **explorar cómo se obtiene un valor del Hazard Index**
            a partir de sus dos componentes principales:

            - **Presión por microplásticos** (cantidad)
            - **Composición morfológica** (tipo de microplásticos)

            El valor calculado se compara con los valores **realmente observados**
            en el océano.
            """
        )

        # ---------------------------------------------------------
        # Input: presión por microplásticos
        # ---------------------------------------------------------

        st.markdown("### 1️⃣ Presión por microplásticos")

        # Usamos el rango observado real
        p_min = hazard_gdf["hazard_pressure"].min()
        p_max = hazard_gdf["hazard_pressure"].max()

        hazard_pressure_user = st.slider(
            "Nivel de presión por microplásticos (normalizado)",
            min_value=float(p_min),
            max_value=float(p_max),
            value=float(np.median(hazard_gdf["hazard_pressure"])),
            step=0.01,
        )

        st.caption(
            "Este valor representa la cantidad relativa de microplásticos, "
            "normalizada a partir de observaciones reales."
        )

        # ---------------------------------------------------------
        # Input: morfología (ya calculada antes)
        # ---------------------------------------------------------

        st.markdown("### 2️⃣ Composición morfológica")

        st.write(
            f"Índice presión morfológico seleccionado: **{hazard_morph_user:.2f}**"
        )

        # ---------------------------------------------------------
        # Cálculo del Hazard Index
        # ---------------------------------------------------------

        # ⚠️ IMPORTANTE:
        # Usamos la misma función lógica que en la construcción del índice.
        # Aquí asumimos combinación multiplicativa normalizada.
        hazard_morph_user = float(hazard_morph_user)
        hazard_pressure_user = float(hazard_pressure_user)

        hazard_index_user = (
            0.5 * hazard_pressure_user
            + 0.5 * hazard_morph_user
        )

        # ---------------------------------------------------------
        # Interpretación
        # ---------------------------------------------------------

        st.markdown("### 📊 Resultado")

        # Percentil respecto a valores observados
        hazard_dist = hazard_gdf["hazard_index"].dropna().values
        percentile = (hazard_dist < hazard_index_user).mean() * 100

        st.metric(
            label="Hazard Index estimado",
            value=f"{hazard_index_user:.2f}",
        )

        # Clasificación semántica (coherente con Jenks)
        if hazard_index_user <= np.percentile(hazard_dist, 33):
            label = "Bajo"
            color = "green"
        elif hazard_index_user <= np.percentile(hazard_dist, 66):
            label = "Medio"
            color = "orange"
        else:
            label = "Alto"
            color = "red"

        st.markdown(
            f"""
            **Nivel estimado:**  
            <span style="color:{color}; font-weight:600;">{label}</span>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            Este valor se sitúa aproximadamente en el **percentil {percentile:.1f}**
            del Hazard Index observado en el océano.
            """
        )

        # ---------------------------------------------------------
        # Nota explicativa final
        # ---------------------------------------------------------

        st.info(
            """
            Este resultado es **explicativo**, no predictivo.

            Muestra cómo la combinación de presión y composición morfológica
            se traduce en un valor del Hazard Index, utilizando la misma lógica
            empleada en su construcción original.
            """
        )

    ###############################################
    elif capa == "Ecología":

        st.header("Implicaciones ecológicas potenciales")

        st.markdown(
            """
            En esta sección puedes explorar las implicaciones ecológicas
            asociadas a **un determinado nivel de Hazard Index**.

            No es necesario haber calculado previamente el índice:
            puedes introducir directamente un valor.
            """
        )

        # =========================================================
        # INPUT PRINCIPAL — HAZARD INDEX
        # =========================================================

        hazard_values = hazard_gdf["hazard_index"].dropna().values

        hazard_index = st.slider(
            "Hazard Index",
            min_value=float(hazard_values.min()),
            max_value=float(hazard_values.max()),
            value=float(np.median(hazard_values)),
            step=0.01,
            help="Índice normalizado entre 0 (bajo) y 1 (alto)"
        )

        # =========================================================
        # VARIABLES ECOLÓGICAS BASE (MODO NO EXPLORATORIO)
        # =========================================================

        eco_count = 0
        eco_shape_richness = 2
        eco_mean_size = 0
        eco_small_ratio = 0
        log_dist_m = 0
        eco_dist_m = 0

        # =========================================================
        # PREDICCIÓN ECOLÓGICA BASE (MODELO INDEX)
        # =========================================================

        eco_result = predict_ecological_impact_index(
            hazard_index=hazard_index,
            eco_count=eco_count,
            eco_shape_richness=eco_shape_richness,
            eco_mean_size=eco_mean_size,
            eco_small_ratio=eco_small_ratio,
            log_dist_m=log_dist_m,
            eco_dist_m=eco_dist_m,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Riesgo ecológico medio esperado",
                f"{eco_result['iucn_mean_risk']:.2f} / 4"
            )

        with col2:
            st.metric(
                "Especies vulnerables potencialmente afectadas",
                f"{eco_result['species_count']:.1f}"
            )

        st.caption(
            "Resultados basados en patrones globales. "
            "Interpretar a escala regional."
        )

        # =========================================================
        # EXPLORACIÓN DE ESCENARIOS ECOLÓGICOS
        # =========================================================

        st.subheader("🔎 Exploración de escenarios ecológicos")

        explore = st.checkbox(
            "Explorar escenarios ecológicos alternativos",
            help=(
                "Permite explorar cómo cambiarían las implicaciones ecológicas "
                "si el contexto ecológico local fuese distinto al observado."
            )
        )

        if explore:
            st.markdown(
                """
                ⚠️ **Modo exploratorio activo**

                Los valores definidos a continuación representan **escenarios
                hipotéticos plausibles**, no observaciones reales.
                """
            )

            col1, col2 = st.columns(2)

            with col1:
                eco_count = st.slider(
                    "Presencia de especies vulnerables (eco_count)",
                    0, 500, 150
                )

            with col2:
                eco_shape_richness = st.slider(
                    "Diversidad morfológica de microplásticos",
                    2, 4, 3
                )

            eco_result = predict_ecological_impact_index(
                hazard_index=hazard_index,
                mp_real=hazard_index,  # proxy si no quieres exponer mp_real
                morphology={
                    "fibers": 0.25,
                    "fragments": 0.25,
                    "spheres": 0.25,
                    "others": 0.25,
                },
                eco_count=eco_count,
                eco_shape_richness=eco_shape_richness,
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Riesgo ecológico (escenario exploratorio)",
                    f"{eco_result['iucn_mean_risk']:.2f} / 4"
                )

            with col2:
                st.metric(
                    "Especies vulnerables esperadas",
                    f"{eco_result['species_count']:.1f}"
                )

        # =========================================================
        # CONTEXTUALIZACIÓN GLOBAL
        # =========================================================

        st.subheader("📊 Contextualización global del riesgo")

        risk_dist = hazard_gdf["iucn_mean_risk"].dropna().values
        percentile = (risk_dist < eco_result["iucn_mean_risk"]).mean() * 100

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(risk_dist, bins=30, alpha=0.7)
        ax.axvline(eco_result["iucn_mean_risk"], color="red", linewidth=2)
        ax.set_xlabel("Riesgo ecológico medio observado")
        ax.set_ylabel("Frecuencia")

        st.pyplot(fig)

        st.markdown(
            f"""
            Este valor se sitúa aproximadamente en el
            **percentil {percentile:.1f}** del riesgo ecológico observado
            a escala global.
            """
        )

        # =========================================================
        # COHERENCIA ECOLÓGICA OBSERVADA (MODELO A)
        # =========================================================

        st.subheader("🧩 Coherencia ecológica observada")

        hazard_prob = predict_hazard_coherence(
            eco_shape_richness=eco_shape_richness,
            eco_count=eco_count,
        )

        st.metric(
            "Probabilidad de hazard elevado",
            f"{hazard_prob:.2f}"
        )

        st.caption(
            "Este resultado refleja patrones observados de co-ocurrencia "
            "entre contexto ecológico y presión por microplásticos. "
            "No implica causalidad."
        )
