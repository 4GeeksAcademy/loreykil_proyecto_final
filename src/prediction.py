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
GRILLA_PATH = "./data/grid_ocean/grilla_agua_perfiles.gpkg"
MODEL_PATH = "./models/Grilla_mp/rf_microplastics_final.joblib"


# Cargar recursos
def load_grilla():
    gdf = gpd.read_file(GRILLA_PATH)
    if gdf.crs is None:
        raise ValueError("La grilla no tiene CRS definido")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf

def load_model():
    return joblib.load(MODEL_PATH)



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
    # 1. Asegurar CRS
    if grilla.crs is None:
        raise ValueError("La grilla no tiene CRS definido")
    
    if grilla.crs.is_geographic:
        grilla_proj = grilla.to_crs(epsg=3857)
    else:
        grilla_proj = grilla

    # Crear punto en CRS de la grilla
    point = gpd.GeoSeries(
        [Point(lon, lat)],
        crs=4326
    ).to_crs(grilla_proj.crs).iloc[0]

    # 2. Vecino más cercano 
    distances = grilla_proj.geometry.distance(point)
    idx = distances.idxmin()

    cell = grilla.loc[idx]
    if cell is None:
        return {
            "status": "land",
            "message": "El punto seleccionado está en tierra o fuera del dominio oceánico."
        }
    # 3. Features
    env_vars = extract_environmental_variables(cell, feature_names)

    X = pd.DataFrame([env_vars], columns=feature_names)
    
    

    # 4. Predicción
    y_log = float(model.predict(X)[0])
    y_real = float(np.expm1(y_log))

    return {
        "status": "water",
        "microplastics_log": y_log,
        "microplastics_real": y_real,
        "environmental_variables": {
            k: float(cell[k]) for k in feature_names
        },
        "profile_eco": cell["profile_eco"],
        "cell_index": idx
    }
