# Esta capa se encarga de:
# - Cargar la grilla ambiental
# - Cargar el modelo de microplasticos
# - Dado un punto (lon, lat):
#   - encontrar la celda más cercana
#   - extraer las variables ambientales
#   - predecir la concentración de microplásticos (en log)
#   - devolver la concentración en unidades originales
# - Devolver un resultado estructurado.

import geopandas as gpd
import pandas as pd
import numpy as np
import joblib
from shapely.geometry import Point

# RUTAS
GRILLA_PATH = "./data/grid_ocean/grilla_agua.gpkg"
MODEL_PATH = "./models/Grilla_mp/rf_microplastics_final.joblib"


# Cargar recursos
def load_grilla():
    return gpd.read_file(GRILLA_PATH)

def load_model():
    return joblib.load(MODEL_PATH)


# Devolver la celda de la grilla más cercana al punto lon/lat del usuario
def get_nearest_cell(grilla: gpd.GeoDataFrame, lon: float, lat: float, max_distance_m=20000):
    point = Point(lon, lat)

    # Aseguramos mismo CRS
    if grilla.crs.is_geographic:
        grilla_proj = grilla.to_crs(epsg=3857)
        point_proj = gpd.GeoSeries([point], crs=4326).to_crs(epsg=3857).iloc[0]
    else:
        grilla_proj = grilla
        point_proj = point

    distances = grilla_proj.geometry.distance(point_proj)
    idx = distances.idxmin()
    min_dist=distances.loc[idx]

    if min_dist > max_distance_m:
        return None

    return grilla.loc[idx]


# Extraer variables ambientales de la celda
def extract_environmental_variables(cell: pd.Series, feature_names: list):
    missing = [v for v in feature_names if v not in cell.index]
    if missing:
        raise ValueError(f"Faltan variables en la celda: {missing}")
    return {var: cell[var] for var in feature_names}

# status:
# - "water": predicción válida
# - "land": punto fuera del dominio oceánico

# Predicción de microplásticos en un punto
def predict_microplastics_at_point(
    lon: float,
    lat: float,
    grilla: gpd.GeoDataFrame,
    model,
    feature_names: list
):
    # 1. Buscar celda más cercana
    cell = get_nearest_cell(grilla, lon, lat)
    if cell is None:
        return {
            "status": "land",
            "message": "El punto seleccionado está en tierra o fuera del dominio oceánico."
        }

    # 2. Extraer variables ambientales
    env_vars = extract_environmental_variables(cell, feature_names)

    X = pd.DataFrame([env_vars], columns=feature_names)

    # 3. Predicción en escala log
    mp_log = float(model.predict(X)[0])

    # 4. Conversión a escala real
    mp_real = float(np.expm1(mp_log))

    return {
        "status": "water",
        "microplastics_log": mp_log,
        "microplastics_real": mp_real,
        "environmental_variables": env_vars,
        "cell_index": cell.name
    }
