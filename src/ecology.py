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
# FEATURES — MODELOS ECOLOGÍA (ANÁLISIS POR CAPAS)
# ======================================================

FEATURES_ECO_RAW = [
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
]

FEATURES_ECO_INDEX = [
    "hazard_index",
    "log_dist_m",
    "eco_dist_m",
    "eco_count",
    "eco_mean_size",
    "eco_small_ratio",
]


# ======================================================
# CARGA DE MODELOS — FLUJO INTERACTIVO
# ======================================================

def load_ecology_models():
    """
    Modelos ecológicos usados en el flujo interactivo
    (dependen de variables ambientales).
    """
    rf_risk = joblib.load(
        "models/Grilla_species/rf_mean_risk_INDEX.joblib"
    )["model"]

    rf_species = joblib.load(
        "models/Grilla_species/rf_species_count_RAW.joblib"
    )["model"]

    return rf_risk, rf_species


# ======================================================
# CARGA DE MODELOS — ANÁLISIS POR CAPAS (INDEX)
# ======================================================

def load_ecology_models_index():
    """
    Modelos entrenados en modo INDEX
    (usados en Análisis por capas).
    """
    model_hazard = joblib.load(
        "models/Grilla_hazard/model_hazard.joblib"
    )

    model_risk = joblib.load(
        "models/Grilla_hazard/model_iucn_mean_risk_RF_RAW.joblib"
    )

    model_species = joblib.load(
        "models/Grilla_hazard/model_T2_iucn_species_count_XGB_Poisson_INDEX.joblib"
    )

    return model_hazard, model_risk, model_species


# ======================================================
# CONSTRUCCIÓN DE LA FILA DE ENTRADA (FLUJO)
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
    para los modelos ecológicos del flujo interactivo.
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
# PREDICCIÓN ECOLÓGICA — FLUJO INTERACTIVO
# ======================================================

def predict_ecological_impact(
    cell_df: pd.DataFrame,
    rf_risk,
    rf_species,
):
    """
    Aplica los modelos ecológicos finales a una celda
    (modo flujo interactivo).
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


# ======================================================
# PREDICCIÓN ECOLÓGICA — ANÁLISIS POR CAPAS (MODELO B)
# ======================================================

def predict_ecological_impact_index(
    *,
    hazard_index,
    mp_real=0,
    morphology=None,
    eco_count=0,
    eco_shape_richness=2,
    eco_mean_size=0,
    eco_small_ratio=0,
    log_dist_m=0,
    eco_dist_m=0,
):
    _, model_risk, model_species = load_ecology_models_index()

    if morphology is None:
        morphology = {
            "fibers": 0.25,
            "fragments": 0.25,
            "spheres": 0.25,
            "others": 0.25,
        }

    # -------------------------------
    # DataFrame completo (RAW)
    # -------------------------------
    X = pd.DataFrame([{
        # índice
        "hazard_index": hazard_index,

        # presión microplásticos
        "noaa_mp_mean": mp_real,
        "noaa_mp_max": mp_real,
        "noaa_mp_count": 1,
        "mp_pieces_m3": mp_real,

        # morfología
        "FIBER": morphology["fibers"],
        "FRAGMENT": morphology["fragments"],
        "SPHERE": morphology["spheres"],
        "OTHER": morphology["others"],
        "eco_shape_richness": eco_shape_richness,

        # proximidad
        "log_dist_m": log_dist_m,
        "eco_dist_m": eco_dist_m,

        # ecología
        "eco_count": eco_count,
        "eco_mean_size": eco_mean_size,
        "eco_small_ratio": eco_small_ratio,
    }])

    # -------------------------------
    # Predicciones
    # -------------------------------
    risk_pred = model_risk.predict(X[FEATURES_ECO_RAW])[0]
    species_pred = model_species.predict(X[FEATURES_ECO_INDEX])[0]

    return {
        "iucn_mean_risk": float(risk_pred),
        "species_count": float(species_pred),
    }


# ======================================================
# COHERENCIA ECOLÓGICA — MODELO A
# ======================================================

def predict_hazard_coherence(
    eco_shape_richness,
    ecotaxa_present,
    vuln,
):
    model_hazard, _, _ = load_ecology_models_index()

    X = pd.DataFrame([{
        "eco_shape_richness": eco_shape_richness,
        "ecotaxa_present": ecotaxa_present,
        "vuln": vuln,
    }])

    prob = model_hazard.predict_proba(X)[0, 1]
    return float(prob)
