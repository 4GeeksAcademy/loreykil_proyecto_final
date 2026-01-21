# src/ecology.py

import joblib
import numpy as np
import pandas as pd


# ======================================================
# FEATURES (contrato explícito del modelo)
# ======================================================

FEATURES_RAW = [
    "noaa_mp_mean",
    "noaa_mp_max",
    "noaa_mp_count",
    "mp_pieces_m3",
    "FIBER",
    "FRAGMENT",
    "SPHERE",
    "OTHER",
    "eco_shape_richness",
    "log_dist_m",
    "eco_dist_m",
    "eco_count",
    "eco_mean_size",
    "eco_small_ratio",
    "temperature",
    "salinity",
    "chlorophyll",
    "nitrate",
    "phosphate",
    "oxygen_dissolved",
    "oxygen_utilization",
]

FEATURES_INDEX = [
    "hazard_index",
    "log_dist_m",
    "eco_dist_m",
    "eco_count",
    "eco_mean_size",
    "eco_small_ratio",
    "temperature",
    "salinity",
    "chlorophyll",
    "nitrate",
    "phosphate",
    "oxygen_dissolved",
    "oxygen_utilization",
]


# ======================================================
# CARGA DE MODELOS
# ======================================================

def load_ecology_models():
    """
    Carga los modelos ecológicos finales.
    """
    rf_risk = joblib.load("models/Grilla_species/rf_mean_risk_INDEX.joblib")["model"]
    rf_species = joblib.load("models/Grilla_species/rf_species_count_RAW.joblib")["model"]
    return rf_risk, rf_species


# ======================================================
# CONSTRUCCIÓN DE LA FILA DE ENTRADA
# ======================================================

def build_cell_dataframe(
    *,
    hazard_index: float,
    mp_real: float,
    morphology: dict,
    env_vars: dict,
    extra_ecology: dict | None = None,
):
    """
    Construye un DataFrame de una fila con todas las variables necesarias
    para los modelos ecológicos.
    """

    row = {
        # presión
        "hazard_index": hazard_index,
        "mp_pieces_m3": mp_real,

        # morfología
        "FIBER": morphology["fibers"],
        "FRAGMENT": morphology["fragments"],
        "SPHERE": morphology["spheres"],
        "OTHER": morphology["others"],
    }

    # variables ambientales
    row.update(env_vars)

    # contexto ecológico adicional (si existe)
    if extra_ecology:
        row.update(extra_ecology)

    return pd.DataFrame([row])


# ======================================================
# PREDICCIÓN ECOLÓGICA (CAPA 3)
# ======================================================

def predict_ecological_impact(
    cell_df: pd.DataFrame,
    rf_risk,
    rf_species,
):
    """
    Aplica los modelos ecológicos finales a una celda.
    """

    # Riesgo ecológico medio
    X_index = cell_df[FEATURES_INDEX]
    risk_pred = rf_risk.predict(X_index)[0]

    # Especies en riesgo (modelo entrenado en log-space)
    X_raw = cell_df[FEATURES_RAW]
    log_species_pred = rf_species.predict(X_raw)[0]
    species_pred = np.expm1(log_species_pred)

    return {
        "risk_mean": float(risk_pred),
        "species_at_risk": float(species_pred),
    }
