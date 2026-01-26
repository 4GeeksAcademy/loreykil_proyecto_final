import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
import folium
import json
from shapely.geometry import Point
import branca.colormap as cm
import pydeck as pdk
import plotly.graph_objects as go
from statsmodels.nonparametric.smoothers_lowess import lowess
import psutil
import os
from folium.plugins import HeatMap

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
)
# CONFIGURACIÓN BÁSICA

if "mp_real" not in st.session_state:
    st.session_state.mp_real = None

if "clicked_point" not in st.session_state:
    st.session_state.clicked_point = None

if "hazard_index" not in st.session_state:
    st.session_state.hazard_index = None

st.set_page_config(
    page_title="Título de la App",
    layout="wide"
)

st.title("Título de la App")
st.sidebar.markdown("## Modo de visualización")
mode = st.sidebar.radio(
    "",
    ["Flujo interactivo", "Análisis por capas"],
)



# CARGA DE RECURSOS

@st.cache_resource
def get_model():
    return load_model()

@st.cache_resource
def get_grilla():
    return load_grilla()

@st.cache_resource
def get_microplastics_overlay():
    with open("data/Predicciones/microplastics_bounds.json") as f:
        bounds = json.load(f)

    image_path = "data/Predicciones/microplastics_log_est.png"
    return image_path, bounds

@st.cache_resource
def get_hazard_gdf():
    return gpd.read_file(
        "./data/grid_indice/hazard_index_grid_final.gpkg",
        columns=[
            "geometry",
            "hazard_index",
            "hazard_pressure",
            "hazard_morphology",
            "mp_pieces_m3",
            "iucn_mean_risk"
        ]
    )

@st.cache_resource
def get_mp_gdf():
    return gpd.read_file(
        "./data/grid_ocean/gdf_microplastics_with_env.gpkg",
        columns=[
            "geometry",
            "lat",
            "lon",
            "microplastics_measurement",
            "profile_eco",
            "distance_to_coast_km"
        ]
    )

@st.cache_resource
def get_mp_parquet():
    return pd.read_parquet(
        "./data/grid_ocean/NOAA_encology.parquet",
    )



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

HAZARD_CARD = """
<div style="border:1px solid #e6e6e6;border-radius:10px;padding:20px;min-height:220px;">
  <div style="font-size:14px;color:#666;">Riesgo potencial por microplásticos</div>
  <div style="font-size:48px;font-weight:600;margin:10px 0;">{value:.2f}</div>
  <div style="font-size:16px;"><b>Nivel:</b> {label}</div>
</div>
"""
MORPHOLOGY_VISUALS = {
    "Dominio de fibras": {
        "title": "Fibras",
        "description": "Microplásticos alargados, asociados frecuentemente a textiles sintéticos.",
        "image": "data/ecotaxa/images/fibra.jpg"
    },
    "Dominio de fragmentos": {
        "title": "Fragmentos",
        "description": "Partículas irregulares procedentes de la fragmentación de plásticos mayores.",
        "image": "data/ecotaxa/images/fragmento.jpg"
    },
    "Dominio de esferas": {
        "title": "Esferas",
        "description": "Microesferas plásticas, históricamente usadas en cosméticos y abrasivos.",
        "image": "data/ecotaxa/images/esfera.jpg"
    },
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


@st.cache_data
def get_profile_means():
    gdf = get_grilla()
    return gdf.groupby("profile_eco")[FEATURES].mean()



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

def show_morphology_mix_images():
    st.sidebar.markdown("**Mezcla de morfologías**")
    st.sidebar.caption(
        "Distribución combinada de las principales formas de microplásticos."
    )

    cols = st.sidebar.columns(2)

    images = [
        ("Fibras", "data/ecotaxa/images/fibra.jpg"),
        ("Fragmentos", "data/ecotaxa/images/fragmento.jpg"),
        ("Esferas", "data/ecotaxa/images/esfera.jpg"),
        ("Otros", "data/ecotaxa/images/otro.jpg"),
    ]

    for i, (label, path) in enumerate(images):
        with cols[i % 2]:
            st.image(path, caption=label, width=130)

@st.cache_data
def get_iucn_risk_distribution():
    gdf_risk = gpd.read_file(
        "./data/Predicciones/risk_prediction_with_confidence.gpkg"
    )
    return gdf_risk["iucn_mean_risk"].dropna().values
@st.cache_data

def get_global_ecology_gdf():
    return gpd.read_file(
        "./data/Predicciones/species_count_prediction_with_confidence.gpkg"
    )

@st.cache_resource
def get_ecology_models():
    return load_ecology_models()

@st.cache_data
def get_land_polygons():
    return gpd.read_file("./data/ocean/ne_10m_land.shp").to_crs(epsg=4326)


def memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

st.sidebar.markdown("### :brain: Uso de memoria")
st.sidebar.write(f"{memory_usage_mb():.1f} MB")
import psutil
import os

def memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

st.sidebar.markdown("### 🧠 Uso de memoria")
st.sidebar.write(f"{memory_usage_mb():.1f} MB")

if mode == "Flujo interactivo":
    st.header("Selección espacial")
    st.sidebar.markdown(
        "<h2 style=color:#1f77b4;'>Inputs del escenario</h2>",
        unsafe_allow_html=True
    )

    st.sidebar.markdown(
        """
        <div style="
            font-size: 12px;
            color: #777777;
            line-height: 1.4;
        ">
            Define aquí el escenario de riesgo asociado al punto seleccionado.
            <br><br>
            <ul style="padding-left: 16px; margin: 0;">
                <li>Selecciona un punto en el mapa.</li>
                <li>Revisa la concentración estimada de microplásticos.</li>
                <li>Elige la composición morfológica.</li>
                <li>Interpreta riesgo e implicaciones ecológicas.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.divider()
    # INTERACCIÓN CAPA 1¡
    st.write(
    "Selecciona un punto del océano para estimar la concentración "
    "esperada de microplásticos a partir de las condiciones oceánicas reales."
    )
    # Estado
    if "clicked_point" not in st.session_state:
        st.session_state.clicked_point = None

    if "last_processed_click" not in st.session_state:
        st.session_state.last_processed_click = None

    # mapa base
    image_path, raster_bounds = get_microplastics_overlay()

    m = folium.Map(
        location=[0, 0],
        zoom_start=1,
        tiles=None,
        crs="EPSG4326",
        no_wrap=True,
    )


    folium.raster_layers.ImageOverlay(
        image=image_path,
        bounds=raster_bounds,
        opacity=0.6,
        name="Microplásticos (log)"
    ).add_to(m)

    control_points = {
        "Madrid": (40.4168, -3.7038),
        "Quito": (0.0, -78.4678),
        "Greenwich": (51.48, 0.0),
        "Cabo": (-33.92, 18.42)
    }

    for name, (lat, lon) in control_points.items():
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color="black",
            fill=True,
            fill_opacity=1,
            tooltip=name
        ).add_to(m)

    
    if st.session_state.clicked_point is not None:
        folium.Marker(
            location=[
                st.session_state.clicked_point["lat"],
                st.session_state.clicked_point["lon"]
            ],
            icon=folium.Icon(color="black", icon="dot-circle-o")
        ).add_to(m)


    # Renderizar mapa
    map_data = st_folium(
    m,
    width=900,
    height=520,
    key="mapa_interactivo"
    )

    # Procesar click
    if map_data and map_data.get("last_clicked"):
        new_point = (
            round(map_data["last_clicked"]["lat"], 6),
            round(map_data["last_clicked"]["lng"], 6),
        )

        if st.session_state.last_processed_click != new_point:
            st.session_state.last_processed_click = new_point

            st.session_state.clicked_point = {
                "lat": new_point[0],
                "lon": new_point[1],
            }

            st.session_state.hazard_index = None
            st.rerun()
        
    if st.session_state.clicked_point is not None:
        lat = st.session_state.clicked_point["lat"]
        lon = st.session_state.clicked_point["lon"]

        st.success(f"Punto seleccionado: lat={lat:.3f}, lon={lon:.3f}")   

        # PREDICCIÓN
        grilla = get_grilla()
        model = get_model()
        result = predict_microplastics_at_point(
            lon=lon,
            lat=lat,
            grilla=grilla,
            model=model,
            feature_names=FEATURES
        )
        ocean_profile = result["profile_eco"]
        st.write("Índice de celda:", result["cell_index"])
        st.write("Perfil:", result["profile_eco"])
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
            st.session_state.env_vars = env_vars
            profile_name = ocean_profile
            profile_means = get_profile_means()

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
        "Dado el mismo nivel de microplásticos, explora cómo cambia "
        "el riesgo potencial al modificar su composición morfológica"
    )
    
    has_point = st.session_state.get("clicked_point") is not None
    if not has_point:
        st.info("Selecciona un punto en el mapa para obtener microplásticos.")
        hazard_index = None
    else:
        st.sidebar.header("Hazard Index")
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
        # Proporciones por defecto (seguras)
        proportions = {
            "fibers": 0.25,
            "fragments": 0.25,
            "spheres": 0.25,
            "others": 0.25
        }

        st.sidebar.markdown("---")


        if profile in MORPHOLOGY_VISUALS:
            visual = MORPHOLOGY_VISUALS[profile]

            st.sidebar.markdown(f"**{visual['title']}**")
            st.sidebar.image(
                visual["image"],
                width=260
            )
            st.sidebar.caption(visual["description"])

        elif profile in ["Mezcla equilibrada", "Personalizado"]:
            show_morphology_mix_images()


    
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

        # Validación de estados
        if st.session_state.mp_real is None:
            hazard_index = None
        elif not proportions_ready:
            st.sidebar.warning("La suma de las proporciones debe ser 100%.")
            hazard_index = None
        else:
            # recalcular solo si está invalidado
            if st.session_state.hazard_index is None:
                mp_real = st.session_state.mp_real
                mp_gdf = get_mp_gdf()
                mp_observed = mp_gdf["microplastics_measurement"].dropna().values
                MP_REF_P90 = np.percentile(mp_observed, 90)

                hazard_pressure = compute_hazard_pressure(
                    mp_concentration=mp_real,
                    ref_max=MP_REF_P90
                )
                
                hazard_morphology = compute_hazard_morphology(proportions)

                st.session_state.hazard_index = compute_hazard_index(
                    hazard_pressure,
                    hazard_morphology
                )
                
                mp_gdf = None

            hazard_index = st.session_state.hazard_index
            label = hazard_label(hazard_index)
    # Visualización
        if hazard_index is not None:
            st.subheader("Hazard Index")

            col1, col2 = st.columns([1,1])

            with col1:
                st.markdown(
                    HAZARD_CARD.format(value=hazard_index, label=label),
                    unsafe_allow_html=True
                )
                    
            
            with col2:
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
                fig.update_layout(
                    height=220,
                    margin=dict(t=20, b=20, l=10, r=10)
                )


                st.plotly_chart(fig, use_container_width=True)
            
            

            # Gráfico de distribución del Hazard Index observado

            hazard_gdf = get_hazard_gdf()
            hazard_values = hazard_gdf["hazard_index"].values
            percentile = (hazard_values < hazard_index).mean() * 100

            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                fig, ax = plt.subplots(figsize=(3, 2), dpi=120)
                ax.hist(hazard_values, bins=30, alpha=0.7)
                ax.axvline(hazard_index, color="red", linewidth=1)
                ax.set_xlabel("Hazard Index observado", fontsize=6)
                ax.set_ylabel("Frecuencia", fontsize=6)
                ax.tick_params(axis="both", labelsize=5)

                st.pyplot(fig)
                plt.close(fig)


            st.write(
                f"Este valor de riesgo se sitúa en el percentil **{100 - percentile:.1f}** del Hazard Index observado"
            )

            del hazard_gdf
            import gc
            gc.collect()

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
    
    st.sidebar.markdown("## Escenario ecológico")

    explore_ecology = st.sidebar.checkbox(
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
        st.sidebar.markdown("### Escenario ecológico hipotético")

        richness_level = st.sidebar.selectbox(
            "Tamaño relativo de los ítems presentes",
            ["0-57 µm", "58-147 µm", "148-464 µm"],
            index=1
        )

        complexity_level = st.sidebar.selectbox(
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
    

    fig, ax = plt.subplots()
    ax.hist(risk_dist, bins=30, alpha=0.7)
    ax.axvline(risk_pred, color="red", linewidth=2)
    ax.set_xlabel("iucn_mean_risk observado")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución global del riesgo ecológico")

    st.pyplot(fig)
    plt.close(fig)


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


    capa = st.radio(
        "",
        ["Microplásticos", "Hazard", "Ecología"],
        horizontal=True
    )

    if capa == "Microplásticos":
        mp_gdf = get_mp_gdf()
        hazard_gdf = get_hazard_gdf()
        st.header("Análisis de microplásticos")
        st.markdown(
            """
            Esta sección explor los datos observados de microplásticos en el océano.
            """
        )

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
           "Mapa observado",
            "Distribución",
            "Costa",
            "Ambiente",
            "Clusters"
        ])

        # TAB 1: Mapa observado
        with tab1:
            st.subheader("Distribución espacial observada")

            st.markdown(
                """
                Este mapa muestra la localización de las muestras de microplásticos utilizados en el análisis.
                Cada punto representa una observación puntual.
                """
            )

            # Preparar datos
            mp_parquet = get_mp_parquet().copy()
            mp_parquet = mp_parquet[mp_gdf["microplastics_measurement"] > 0].copy()
            mp_parquet["log_microplastics"] = np.log10(mp_parquet["microplastics_measurement"])

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=mp_parquet,
                get_position="[lon, lat]",
                get_radius=30000,
                get_fill_color="[log_microplastics * 30 + 100, 100, 160, 160]",
                pickable=True,
            )

            view_state = pdk.ViewState(
                latitude=mp_parquet["lat"].mean(),
                longitude=mp_parquet["lon"].mean(),
                zoom=1,
            )

            st.pydeck_chart(
                pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    tooltip={
                        "text": "Microplásticos (log): {log_microplastics}"
                    },
                )
            )
            with st.expander("Ver tabla de datos"):
                st.dataframe(
                    mp_parquet[[
                        "lat",
                        "lon",
                        "microplastics_measurement",
                        "log_microplastics"
                    ]].head(200)
                )

        # TAB 2: Distribución
        with tab2:
            st.subheader("Distribución de concentraciones")
            scale = st.radio(
                "Escala",
                ["Lineal", "Logarítmica"],
                horizontal=True,
                help=(
                    "La escala logarítmica se utiliza porque las concentraciones de microplásticos "
                    "presentan valores extremos. Esta escala permite visualizar mejor "
                    "la distribución general sin que los valores muy altos dominen el gráfico."

                )
            )
            
            data = mp_parquet["microplastics_measurement"].dropna()
            if scale == "Logarítmica":
                data = mp_parquet["log_microplastics"]
                plot_data = mp_parquet["log_microplastics"]
                xlabel = "Log(Microplásticos items/m³)"
            else:
                plot_data = mp_parquet["microplastics_measurement"]
                xlabel = "Microplásticos items/m³"
            
            fig, ax = plt.subplots()
            ax.hist(
                plot_data,
                bins=40,
                color="#1f77b4",
                alpha=0.8
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Frecuencia")

            st.pyplot(fig)
            plt.close(fig)


            with st.expander("¿Cómo interpretar este histograma?"):
                st.markdown(
                    """
                    - La mayoría de las observaciones se concentran en valores bajos.
                    - Existen valores extremos (hotspots) con concentraciones muy elevadas.
                    - La escala logarítmica ayuda a visualizar mejor la distribución general.
                    """
                )

            st.markdown(
                f"""
                - Número de muestras: **{len(mp_parquet["microplastics_measurement"].dropna())}**
                - Concentración media: **{mp_parquet["microplastics_measurement"].mean():.2f} items/m³**
                - Concentración mediana: **{mp_parquet["microplastics_measurement"].median():.2f} items/m³**
                - Percentil 90: **{np.percentile(mp_parquet["microplastics_measurement"].dropna(), 90):.2f} items/m³**
                """
            )

        
            st.dataframe(mp_parquet.head(200))

        # TAB 3: Costa
        with tab3:
            st.subheader("Microplásticos y distancia a la costa")
            
            st.markdown(
                "Comparación de concentraciones según la proximidad a la costa."
            )

            bins = [0, 50, 200, np.inf]
            labels = ["0-50 km", "51-200 km", ">200 km"]

            mp_parquet = mp_parquet.copy()
            mp_parquet["coastal_band"] = pd.cut(
                mp_parquet["distance_to_coast_km"],
                bins=bins,
                labels=labels,
            )

            fig, ax = plt.subplots()
            
            mp_parquet.boxplot(
                column="log_microplastics",
                by="coastal_band",
                ax=ax,
                grid=False,
                showfliers=True
            )

            ax.set_xlabel("Distancia a la costa")
            ax.set_ylabel("Log(Microplásticos items/m³)")
            ax.set_title("Concentración de microplásticos según distancia a la costa")
            plt.suptitle("")

            st.pyplot(fig)
            plt.close(fig)


            with st.expander("¿Cómo interpretar este gráfico?"):
                st.markdown(
                    """
                    - Las tres categorías muestran concentraciones bajas en la mayoría de observaciones
                    - Sin embargo, existen valores extremos en todas las distancias.
                    - Distribución asimétrica. Hotspots localizados.                   
                    """
                )

        # TAB 4: Ambiente
        with tab4:
            st.subheader("Microplásticos y variables ambientales")

            st.markdown(
                "Explora las concentraciones de microplásticos "
                "en función de las variables ambientales medidas."
            )

            env_var = st.selectbox(
                "Selecciona una variable ambiental",
                FEATURES,
                index=0
            )

            fig, ax = plt.subplots()

            ax.scatter(
                mp_parquet[env_var],
                mp_parquet["log_microplastics"],
                alpha=0.5,
                s=20,
                color="#1f77b4"
            )
            ax.set_xlabel(f"{ENV_VARS_META[env_var]['label']}")
            ax.set_ylabel("Log(Microplásticos items/m³)")
            ax.set_title(
                f"Microplásticos vs {ENV_VARS_META[env_var]['label']}"
            )

            st.pyplot(fig)
            plt.close(fig)


            with st.expander("¿Cómo interpretar este gráfico?"):
                st.markdown(
                    """
                    - Las concentraciones de microplásticos no están controladas por una única 
                    variable ambiental. Su distribución refleja la superposición de 
                    múltiples procesos, fuentes, transporte, mezcla y acumulación, que generan 
                    patrones complejos y heterogéneos.
                    """
                )

        # TAB 5: Clusters
        with tab5:
            st.subheader("Microplásticos y perfiles ambientales")

            st.markdown(
                """
                Explora las concentraciones de microplásticos
                según los perfiles ambientales oceánicos definidos.
                """
            )
            
            fig, ax = plt.subplots()

            # Ordenar clusters para que el gráfico sea estable
            clusters = sorted(mp_parquet["profile_eco"].dropna().unique())

            ax.boxplot(
                [
                    mp_parquet.loc[mp_parquet["profile_eco"] == c, "log_microplastics"]
                    for c in clusters
                ],
                showfliers=True
            )
            ax.set_xticklabels(
                [PROFILE_LABELS.get(c, c) for c in clusters],
                rotation=30,
                ha="right"
            )

            ax.set_xlabel("Perfil ambiental oceánico")
            ax.set_ylabel("Log(Microplásticos items/m³)")
            ax.set_title("Concentración de microplásticos según perfil ambiental")

            st.pyplot(fig)
            plt.close(fig)


            with st.expander("¿Cómo interpretar este gráfico?"):
                st.markdown(
                    """
                     - Los clusters agrupan puntos con condiciones ambientales similares.
                     - Algunos perfiles muestran concentraciones típicas más altas o mayor variabilidad.
                     - Estas diferencias no implican causalidad directa, sino asociaciones contextuales.
                    """
                )

            # Tabla resumen
            summary = (
                mp_parquet
                .groupby("profile_eco")["microplastics_measurement"]
                .agg(
                    n_observations="count",
                    median="median",
                    mean="mean",
                )
                .reset_index()
            )
            summary["Perfil ambiental"] = summary["profile_eco"].map(
                lambda x: PROFILE_LABELS.get(x, x)
            )

            summary = summary[
                [
                    "Perfil ambiental",
                    "n_observations",
                    "median",
                    "mean"
                ]
            ]
            summary = summary.rename(
                columns={
                    "n_observations": "N",
                    "median": "Mediana (items/m³)",
                    "mean": "Media (items/m³)"
                }
            )


            with st.expander("Ver resumen por cluster"):
                st.dataframe(summary)       
        
            
    elif capa == "Hazard":

        hazard_gdf = get_hazard_gdf()

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
        plt.close(fig)


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
            plt.close(fig)


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
            plt.close(fig)


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
            plt.close(fig)


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
        plt.close(fig)


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

        del hazard_gdf
        import gc
        gc.collect()


    ###############################################

    elif capa == "Ecología":

        hazard_gdf = get_hazard_gdf()

        st.header("Implicaciones ecológicas potenciales")

        st.markdown(
            """
            En esta sección se exploran **implicaciones ecológicas potenciales**
            bajo distintos **escenarios ambientales y de presión por microplásticos**.

            El usuario define un escenario y el sistema estima dos indicadores
            ecológicos complementarios.
            """
        )

        # =========================================================
        # DEFINICIÓN DEL ESCENARIO ECOLÓGICO
        # =========================================================

        st.subheader("🔧 Definir escenario ecológico")

        col1, col2 = st.columns(2)

        # ---------------------------------------------------------
        # Presión por microplásticos (para mean_risk)
        # ---------------------------------------------------------
        with col1:
            mp_dist = hazard_gdf["mp_pieces_m3"].dropna().values

            mp_percentile = st.slider(
                "Presión relativa por microplásticos",
                min_value=1,
                max_value=99,
                value=50,
                step=1,
                help="Percentil observado de concentración de microplásticos"
            )

            mp_real = float(np.percentile(mp_dist, mp_percentile))

        # ---------------------------------------------------------
        # Hazard Index (para species_count)
        # ---------------------------------------------------------
        with col2:
            hazard_values = hazard_gdf["hazard_index"].dropna().values

            hazard_index = st.slider(
                "Hazard Index",
                min_value=float(hazard_values.min()),
                max_value=float(hazard_values.max()),
                value=float(np.median(hazard_values)),
                step=0.01,
                help="Índice sintético de riesgo por microplásticos"
            )

        # ---------------------------------------------------------
        # Contexto ecológico (tamaño del microplástico y complejidad)
        # ---------------------------------------------------------

        col3, col4 = st.columns(2)

        with col3:
            eco_size_label = st.selectbox(
                "Tamaño relativo de los microplásticos",
                ["0–57 µm", "58–147 µm", "148–464 µm"],
                index=1,
                help="Rangos basados en percentiles observados"
            )

            ECO_SIZE_MAP = {
                "0–57 µm": 53,
                "58–147 µm": 147,
                "148–464 µm": 464,
            }

            eco_count = ECO_SIZE_MAP[eco_size_label]

        with col4:
            eco_shape_richness = st.slider(
                "Diversidad morfológica de microplásticos",
                min_value=2,
                max_value=4,
                value=3,
                help="Número de formas distintas presentes"
            )

        # =========================================================
        # MORFOLOGÍA — PROPORCIONES (suma automática = 100)
        # =========================================================

        st.subheader("🧩 Composición morfológica de los microplásticos")

        col_left, col_right = st.columns([2, 1])  # sliders más espacio que el gráfico

        with col_left:

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

            st.caption(
                f"Otros se ajusta automáticamente: **{others}%**"
            )

        # -------------------------------------------------
        # Contribución ponderada (con pesos)
        # -------------------------------------------------

        weighted_morphology = {
            k: proportions[k] * WEIGHTS[k]
            for k in proportions
        }

        with col_right:
            fig, ax = plt.subplots(figsize=(3, 2))
            ax.bar(
                weighted_morphology.keys(),
                weighted_morphology.values(),
                color=["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
            )
            ax.set_ylim(0, 1)
            ax.set_title("Contribución por forma", fontsize=10)
            ax.tick_params(axis="x", labelsize=8)
            ax.tick_params(axis="y", labelsize=8)

            st.pyplot(fig, use_container_width=False)
            plt.close(fig)



        st.caption(
            "La contribución depende tanto de la proporción como del peso asignado "
            "a cada forma. No todas las morfologías tienen el mismo potencial de riesgo."
        )

        # =========================================================
        # VARIABLES FIJAS (baseline)
        # =========================================================

        eco_mean_size = 0
        eco_small_ratio = 0
        log_dist_m = 0
        eco_dist_m = 0

        # =========================================================
        # RESULTADOS ECOLÓGICOS
        # =========================================================

        st.subheader("🌱 Resultados ecológicos esperados")

        col7, col8 = st.columns(2)

        # ---------------------------------------------------------
        # 🅰️ Riesgo ecológico medio (modelo RAW)
        # ---------------------------------------------------------
        with col7:
            eco_result_risk = predict_ecological_impact_index(
                hazard_index=hazard_index,
                mp_real=mp_real,
                morphology=proportions,
                eco_count=eco_count,
                eco_shape_richness=eco_shape_richness,
                eco_mean_size=eco_mean_size,
                eco_small_ratio=eco_small_ratio,
                log_dist_m=log_dist_m,
                eco_dist_m=eco_dist_m,
            )

            st.metric(
                "Riesgo ecológico medio esperado",
                f"{eco_result_risk['iucn_mean_risk']:.2f} / 4"
            )

            st.caption(
                "Este indicador responde principalmente a la presión por microplásticos "
                "y a su composición."
            )

        # ---------------------------------------------------------
        # 🅱️ Especies vulnerables (modelo INDEX)
        # ---------------------------------------------------------
        with col8:
            eco_result_species = predict_ecological_impact_index(
                hazard_index=hazard_index,
                eco_count=eco_count,
                eco_shape_richness=eco_shape_richness,
                eco_mean_size=eco_mean_size,
                eco_small_ratio=eco_small_ratio,
                log_dist_m=log_dist_m,
                eco_dist_m=eco_dist_m,
            )

            st.metric(
                "Especies vulnerables potencialmente afectadas",
                f"{eco_result_species['species_count']:.1f}"
            )

            st.caption(
                "Este indicador depende del nivel global de Hazard Index "
                "y del contexto ecológico."
            )

        # =========================================================
        # EXPLICACIÓN FINAL (NO TÉCNICA)
        # =========================================================

        st.info(
            """
            **Cómo interpretar estos resultados**

            • El **riesgo ecológico medio** refleja la intensidad potencial del impacto,
            y responde principalmente a la **cantidad y tipo de microplásticos**.

            • El número de **especies vulnerables** se asocia al **nivel global de hazard**
            y a patrones ecológicos observados.

            Ambos indicadores representan **dimensiones complementarias**
            del impacto ecológico potencial.
            """
        )


        # =========================================================
        # CONTEXTUALIZACIÓN GLOBAL DEL RIESGO
        # =========================================================

        st.subheader("📊 Contextualización global del riesgo")

        risk_dist = hazard_gdf["iucn_mean_risk"].dropna().values
        risk_value = eco_result_risk["iucn_mean_risk"]

        percentile = (risk_dist < risk_value).mean() * 100

        fig, ax = plt.subplots(figsize=(2.8, 1.9))

        ax.hist(
            risk_dist,
            bins=22,
            alpha=0.75,
            color="#b0c4de"
        )
        ax.axvline(
            risk_value,
            color="red",
            linewidth=1.2
        )

        ax.set_xlabel("Riesgo ecológico medio", fontsize=8)
        ax.set_ylabel("Frecuencia", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)

        st.pyplot(fig, use_container_width=False)
        plt.close(fig)


        st.markdown(
            f"""
            Este valor se sitúa aproximadamente en el
            **percentil {percentile:.1f}** del riesgo ecológico observado
            a escala global.
            """
        )

        st.caption(
            "Distribución basada en celdas con información ecológica observada."
        )

        # =========================================================
        # COHERENCIA ECOLÓGICA OBSERVADA (MODELO A)
        # =========================================================
        st.subheader(("🧩 COHERENCIA ECOLÓGICA OBSERVADA"))

        st.markdown("¿Es habitual observar hazard alto en este contexto ecológico?")

        st.markdown(
            """
            Este bloque evalúa si, **dado un contexto ecológico concreto**,
            es habitual observar **niveles elevados de presión por microplásticos**.

            El resultado se basa en **patrones observados a escala global**
            y **no implica causalidad directa**.
            """
        )

        # ---------------------------------------------------------
        # INPUTS ECOLÓGICOS DEL MODELO
        # ---------------------------------------------------------

        col_left, col_right = st.columns([2, 1])

        with col_left:
            eco_shape_richness = st.slider(
                "Diversidad morfológica de microplásticos",
                min_value=2,
                max_value=4,
                value=eco_shape_richness,
                help="Número de formas distintas de microplásticos presentes"
            )

            vuln_level = st.slider(
                "Nivel de amenaza (IUCN Red List)",  
                min_value=1,
                max_value=4,
                value=2,
                step=1,
                help="Nivel de amenaza de las especies presentes"
            )
            # 🔑 Traducción ecológica coherente
            ecotaxa_present = 1 if vuln_level >= 1 else 0
            
        with col_right:
            vuln_table = pd.DataFrame({
                "Código": [4, 3, 2, 1],
                "Categoría IUCN": [
                    "CR – Critically Endangered",
                    "EN – Endangered",
                    "VU – Vulnerable",
                    "NT – Near Threatened"
                ]
            })

            st.markdown("**Equivalencia IUCN**")
            st.table(vuln_table)

        # ---------------------------------------------------------
        # CONSTRUCCIÓN DE FEATURES (MODELO A)
        # ---------------------------------------------------------

        hazard_prob = predict_hazard_coherence(
            eco_shape_richness=eco_shape_richness,
            ecotaxa_present=ecotaxa_present,
            vuln=vuln_level,
        )

        # ---------------------------------------------------------
        # OUTPUT
        # ---------------------------------------------------------

        st.metric(
            "Probabilidad de observar hazard elevado",
            f"{hazard_prob:.2f}"
        )

        st.caption(
            """
            Esta probabilidad refleja **patrones de co-ocurrencia observados**
            entre contexto ecológico y presión por microplásticos.

            No representa un efecto causal ni una predicción de impacto ecológico.
            """
        )

        st.info(
            """
            Una probabilidad baja no implica ausencia de riesgo ecológico.

            Indica que, en los datos observados, los contextos ecológicos
            más diversos y con especies amenazadas **no suelen coincidir**
            con niveles elevados de presión por microplásticos.

            Este bloque evalúa **co-ocurrencia observada**, no impacto potencial.
            """
        )

        # =========================================================
        # BLOQUE FINAL — PROYECCIÓN ECOLÓGICA GLOBAL
        # =========================================================

        st.subheader("🌍 Proyección ecológica global")

        st.markdown(
            """
            Este bloque muestra una **proyección espacial global** de las implicaciones
            ecológicas potenciales asociadas a la presión por microplásticos.

            A diferencia de los bloques anteriores, aquí no se exploran escenarios
            hipotéticos, sino patrones espaciales aprendidos a partir de
            **condiciones ambientales reales**.
            """
        )

        # =========================================================
        # CARGA Y PREPARACIÓN DEL DATASET GLOBAL
        # =========================================================

        gdf_global = get_global_ecology_gdf()
        land = get_land_polygons()
    
        # 👉 SOLO OCÉANO + valores válidos
        gdf_global = gdf_global[
            (gdf_global["pred_iucn_mean_risk"].notna())
        ].copy()

        # 🔑 CLAVE: reproyección para Folium
        if gdf_global.crs.to_epsg() != 4326:
            gdf_global = gdf_global.to_crs(epsg=4326)

        # =========================================================
        # FILTRAR SOLO OCÉANO (QUITAR TIERRA)
        # =========================================================

        gdf_global["geometry_point"] = gdf_global.geometry.centroid

        gdf_global = gdf_global[
            ~gdf_global["geometry_point"].apply(
                lambda p: land.contains(p).any()
            )
        ].copy()

        # =========================================================
        # SELECTOR DE VARIABLE PARA EL MAPA
        # =========================================================

        st.markdown("### 🎨 Variable mostrada en el mapa")

        color_var_label = st.radio(
            "",
            [
                "Riesgo ecológico medio (IUCN)",
                "Especies vulnerables (escala logarítmica)"
            ],
            horizontal=True
        )

        if color_var_label == "Riesgo ecológico medio (IUCN)":
            COLOR_COL = "pred_iucn_mean_risk"
            COLOR_LABEL = "Riesgo ecológico medio proyectado (IUCN)"
        else:
            COLOR_COL = "pred_log_iucn_species_count"
            COLOR_LABEL = "Especies vulnerables potencialmente afectadas (log)"

        # =========================================================
        # COLORMAP (ROBUSTO)
        # =========================================================

        vmin = gdf_global[COLOR_COL].quantile(0.02)
        vmax = gdf_global[COLOR_COL].quantile(0.98)

        colormap = cm.linear.YlOrRd_09.scale(vmin, vmax)
        colormap.caption = COLOR_LABEL

        # =========================================================
        # CONSTRUCCIÓN DEL MAPA (UNA SOLA VEZ)
        # =========================================================

        m = folium.Map(
            location=[0, 0],
            zoom_start=2,
            tiles="CartoDB voyager"
        )

        # Submuestreo para rendimiento
        gdf_plot = gdf_global.sample(
            min(4000, len(gdf_global)),
            random_state=42
        )

        # =========================================================
        # HEATMAP (GRADIENTE CONTINUO)
        # =========================================================

        heat_data = []

        for _, row in gdf_plot.iterrows():
            centroid = row.geometry.centroid

            value = float(row[COLOR_COL])
            if not np.isnan(value):
                heat_data.append([
                    centroid.y,
                    centroid.x,
                    value
                ])

        HeatMap(
            heat_data,
            gradient={
                0.0: "#2c7bb6",
                0.4: "#abd9e9",
                0.6: "#ffffbf",
                0.8: "#fdae61",
                1.0: "#d7191c",
            },
            radius=25,
            blur=30,
            min_opacity=0.3,
        ).add_to(m)


        # =========================================================
        # MARCADOR DEL CLICK (si existe)
        # =========================================================

        if "global_click" in st.session_state:
            folium.CircleMarker(
                location=[
                    st.session_state.global_click["lat"],
                    st.session_state.global_click["lng"]
                ],
                radius=8,
                color="black",
                weight=2,
                fill=True,
                fill_color="cyan",
                fill_opacity=1,
            ).add_to(m)

        colormap.add_to(m)

        # =========================================================
        # RENDER DEL MAPA (UNA SOLA VEZ)
        # =========================================================

        map_data = st_folium(
            m,
            width=900,
            height=520,
        )

        # =========================================================
        # GESTIÓN DEL CLICK
        # =========================================================

        if map_data and map_data.get("last_clicked"):
            new_click = {
                "lat": map_data["last_clicked"]["lat"],
                "lng": map_data["last_clicked"]["lng"]
            }

            if st.session_state.get("global_click") != new_click:
                st.session_state.global_click = new_click
                st.rerun()

        # =========================================================
        # OUTPUT LOCAL + MINI RADAR AMBIENTAL 🔥
        # =========================================================

        if "global_click" in st.session_state:

            lat = st.session_state.global_click["lat"]
            lng = st.session_state.global_click["lng"]

            point = gpd.GeoSeries(
                [Point(lng, lat)],
                crs=4326
            )

            distances = gdf_global.geometry.distance(point.iloc[0])
            idx = distances.idxmin()
            row = gdf_global.loc[idx]

            st.markdown("### 📍 Resultado local")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Riesgo ecológico medio proyectado",
                    f"{row['pred_iucn_mean_risk']:.2f} / 4"
                )

            with col2:
                st.metric(
                    "Especies vulnerables potencialmente afectadas",
                    f"{row['pred_iucn_species_count']:.1f}"
                )

            # Percentil global
            risk_dist = gdf_global["pred_iucn_mean_risk"].values
            percentile = (risk_dist < row["pred_iucn_mean_risk"]).mean() * 100

            st.markdown(
                f"""
                Este valor se sitúa aproximadamente en el
                **percentil {percentile:.1f}** del riesgo ecológico proyectado
                a escala global.
                """
            )

            # -----------------------------
            # TABLA DE VARIABLES AMBIENTALES
            # -----------------------------

            st.markdown("### 🌊 Contexto ambiental local")

            env_table = []

            for k in FEATURES:
                if k in row and k in gdf_global.columns:
                    env_table.append({
                        "Variable": ENV_VARS_META.get(k, {}).get("label", k),
                        "Valor local": round(float(row[k]), 3),
                        "Promedio global": round(float(gdf_global[k].mean()), 3),
                        "Unidad": ENV_VARS_META.get(k, {}).get("unit", "-"),
                    })

            env_df = pd.DataFrame(env_table)

            st.dataframe(
                env_df,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "La tabla muestra las condiciones ambientales en el punto seleccionado "
                "comparadas con el promedio global oceánico."
            )

        # =========================================================
        # AVISO FINAL
        # =========================================================

        st.info(
            """
            Esta proyección no representa observaciones directas ni impactos causales.

            Muestra patrones espaciales esperables bajo condiciones ambientales
            similares, aprendidos a partir de datos globales.
            """
        )


