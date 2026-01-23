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
from src.ecology import (
    load_ecology_models,
    build_cell_dataframe,
    predict_ecological_impact
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
    return gpd.read_file("./data/grid_ocean/gdf_microplastics_with_env.gpkg")

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



# FUNCIONES AUXILIARES

def normalize(value, vmin, vmax):
    return max(0, min(1, (value - vmin) / (vmax - vmin)))

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

def add_click_marker(m, lat, lon):
    folium.CircleMarker(
        location=[lat, lon],
        radius=6,
        color="black",
        weight=2
    ).add_to(m)


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

@st.cache_resource
def get_ecology_models():
    return load_ecology_models()


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
    # INTERACCIÓN CAPA 1
    # Construir mapa inicial
    st.write(
    "Selecciona un punto del océano para estimar la concentración "
    "esperada de microplásticos a partir de las condiciones oceánicas reales."
    )
    m = build_map(gdf_continuo)

    if st.session_state.clicked_point is not None:
        add_click_marker(
            m,
            st.session_state.clicked_point["lat"],
            st.session_state.clicked_point["lon"]
        )

    map_data = st_folium(
        m,
        width=900,
        height=520
    )

    # Actualizar punto clicado. Aquí solo se guarda el click más reciente
    if map_data and map_data.get("last_clicked"):
        new_point = {
            "lat": map_data["last_clicked"]["lat"],
            "lon": map_data["last_clicked"]["lng"]
        }
        # Solo actualizar si es diferente
        if st.session_state.clicked_point != new_point:
            st.session_state.clicked_point = new_point
            st.session_state.hazard_index = None  # invalidar hazard al cambiar de punto
            st.rerun()
        
    if st.session_state.clicked_point is not None:
        lat = st.session_state.clicked_point["lat"]
        lon = st.session_state.clicked_point["lon"]

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

                hazard_pressure = compute_hazard_pressure(
                    mp_concentration=mp_real,
                    ref_max=MP_REF_P90
                )
                
                hazard_morphology = compute_hazard_morphology(proportions)

                st.session_state.hazard_index = compute_hazard_index(
                    hazard_pressure,
                    hazard_morphology
                )

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


    capa = st.radio(
        "",
        ["Microplásticos", "Hazard", "Ecología"],
        horizontal=True
    )

    if capa == "Microplásticos":
        st.header("Análisis de microplásticos")
        st.markdown(
            """
            Esta sección explor los datos observados de microplásticos en el océano.
            """
        )

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Distribución",
            "Costa",
            "Ambiente",
            "Mapa observado",
            "Clusters"
        ])

        # TAB 1: Distribución
        with tab1:
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
            
            data = mp_gdf["microplastics_measurement"].dropna()
            if scale == "Logarítmica":
                data = data[data > 0]
                plot_data = np.log10(data)
                xlabel = "Log₁₀(Microplásticos items/m³)"
            else:
                plot_data = data
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
                - Número de muestras: **{len(mp_gdf["microplastics_measurement"].dropna())}**
                - Concentración media: **{mp_gdf["microplastics_measurement"].mean():.2f} items/m³**
                - Concentración mediana: **{mp_gdf["microplastics_measurement"].median():.2f} items/m³**
                - Percentil 90: **{np.percentile(mp_gdf["microplastics_measurement"].dropna(), 90):.2f} items/m³**
                """
            )

        
            st.dataframe(mp_gdf.head(200))

        # TAB 2: Costa
        with tab2:
            st.subheader("Microplásticos y distancia a la costa")
            
            st.markdown(
                "Comparación de concentraciones según la proximidad a la costa."
            )

            bins = [0, 50, 200, np.inf]
            labels = ["0-50 km", "51-200 km", ">200 km"]

            mp_gdf["coastal_band"] = pd.cut(
                mp_gdf["distance_to_coast_km"],
                bins=bins,
                labels=labels,
            )
            # Transformación logarítmica para visualización
            mp_gdf = mp_gdf[mp_gdf["microplastics_measurement"] > 0]
            mp_gdf["log_microplastics"] = np.log10(mp_gdf["microplastics_measurement"])

            fig, ax = plt.subplots()
            
            mp_gdf.boxplot(
                column="log_microplastics",
                by="coastal_band",
                ax=ax,
                grid=False,
                showfliers=True
            )

            ax.set_xlabel("Distancia a la costa")
            ax.set_ylabel("Log₁₀(Microplásticos items/m³)")
            ax.set_title("Concentración de microplásticos según distancia a la costa")
            plt.suptitle("")

            st.pyplot(fig)

            with st.expander("¿Cómo interpretar este gráfico?"):
                st.markdown(
                    """
                    - Se utiliza una escala logarítmica para reducir la influencia de valores extremos.
                    - Las tres categorías muestran concentraciones bajas en la mayoría de observaciones
                    - Sin embargo, existen valores extremos en todas las distancias.
                    - Distribución asimétrica. Hotspots localizados.                   
                    """
                )

        # TAB 3: Ambiente
        with tab3:
            st.subheader("Microplásticos y variables ambientales")

            st.markdown(
                "Explora cómo varían las concentraciones de microplásticos "
                "en función de las variables ambientales medidas."
            )

            env_var = st.selectbox(
                "Selecciona una variable ambiental",
                FEATURES,
                index=0
            )

            st.caption(
                "Las relaciones mostradas son exploratorias y no implican causalidad."
            )

            mp_gdf = mp_gdf.copy()
            mp_df = mp_gdf[mp_gdf["microplastics_measurement"] > 0]
            mp_df["log_microplastics"] = np.log10(mp_df["microplastics_measurement"])

            fig, ax = plt.subplots()

            ax.scatter(
                mp_df[env_var],
                mp_df["log_microplastics"],
                alpha=0.5,
                s=20,
                color="#1f77b4"
            )
            ax.set_xlabel(f"{ENV_VARS_META[env_var]['label']}")
            ax.set_ylabel("Log₁₀(Microplásticos items/m³)")
            ax.set_title(
                f"Microplásticos vs {ENV_VARS_META[env_var]['label']}"
            )

            st.pyplot(fig)

            with st.expander("¿Cómo interpretar este gráfico?"):
                st.markdown(
                    """
                    - Las concentraciones de microplásticos no están controladas por una única 
                    variable ambiental. Su distribución refleja la superposición de 
                    múltiples procesos, fuentes, transporte, mezcla y acumulación, que generan 
                    patrones complejos y heterogéneos.
                    """
                )
        # TAB 4: Mapa observado
        with tab4:
            st.subheader("Distribución espacial observada")

            st.markdown(
                """
                Este mapa muestra la localización de las muestras de microplásticos utilizados en el análisis.
                Cada punto representa una observación puntual.
                """
            )

            # Preparar datos
            mp_gdf = mp_gdf.copy()
            mp_gdf = mp_gdf[mp_gdf["microplastics_measurement"] > 0]
            mp_gdf["log_microplastics"] = np.log10(mp_gdf["microplastics_measurement"])
            import pydeck as pdk
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=mp_gdf,
                get_position="[lon, lat]",
                get_radius=30000,
                get_fill_color="[log_microplastics * 30 + 100, 100, 160, 160]",
                pickable=True,
            )

            view_state = pdk.ViewState(
                latitude=mp_gdf["lat"].mean(),
                longitude=mp_gdf["lon"].mean(),
                zoom=2,
            )

            st.pydeck_chart(
                pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    tooltip={
                        "text": "Microplásticos (log₁₀): {log_microplastics}"
                    },
                )
            )
            with st.expander("Ver tabla de datos"):
                st.dataframe(
                    mp_gdf[[
                        "lat",
                        "lon",
                        "microplastics_measurement",
                        "log_microplastics"
                    ]].head(200)
                )

        # TAB 5: Clusters
        with tab5:
            st.subheader("Microplásticos y perfiles ambientales")

            st.markdown(
                """
                Explora cómo varían las concentraciones de microplásticos
                según los perfiles ambientales oceánicos definidos.
                """
            )

            mp_gdf = mp_gdf.copy()
            mp_gdf = mp_gdf[mp_gdf["microplastics_measurement"] > 0]
            mp_gdf["log_microplastics"] = np.log10(mp_gdf["microplastics_measurement"])

            fig, ax = plt.subplots()

            # Ordenar clusters para que el gráfico sea estable
            clusters = sorted(mp_gdf["profile_eco"].dropna().unique())

            ax.boxplot(
                [
                    mp_gdf.loc[mp_gdf["profile_eco"] == c, "log_microplastics"]
                    for c in clusters
                ],
                labels=[f'{PROFILE_LABELS.get(c, c)}' for c in clusters],
                showfliers=True
            )

            ax.set_xlabel("Perfil ambiental oceánico")
            ax.set_ylabel("Log₁₀(Microplásticos items/m³)")
            ax.set_title("Concentración de microplásticos según perfil ambiental")

            st.pyplot(fig)

            with st.expander("¿Cómo interpretar este gráfico?"):
                st.markdown(
                    """
                     - Cada caja representa la distribución de microplásticos dentro de un perfil ambiental.
                     - Los clusters agrupan puntos con condiciones ambientales similares.
                     - Algunos perfiles muestran concentraciones típicas más altas o mayor variabilidad.
                     - Estas diferencias no implican causalidad directa, sino asociaciones contextuales.
                    """
                )

            # Tabla resumen
            summary = (
                mp_gdf.groupby("profile_eco")["microplastics_measurement"]
                .agg(
                    n_observations="count",
                    median="median",
                    mean="mean",
                )
                .reset_index()
            )

            with st.expander("Ver resumen por cluster"):
                st.dataframe(summary)

            
        
            
    elif capa == "Hazard":
        st.header("Análisis del Hazard Index")

        fig, ax = plt.subplots()
        ax.hist(hazard_gdf["hazard_index"], bins=40)
        st.pyplot(fig)

        st.dataframe(hazard_gdf[["hazard_index"]].describe())

    elif capa == "Ecología":
        st.info("Capa ecológica en desarrollo.")

