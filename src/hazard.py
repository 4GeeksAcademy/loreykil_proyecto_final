import numpy as np


# Pesos morfológicos exploratorios basados en literatura general.
WEIGHTS = {
    "fibers": 1.0,
    "fragments": 0.7,
    "spheres": 0.4,
    "others": 0.2
}

def compute_hazard_pressure(mp_concentration: float, ref_max: float) -> float:
    # ref_min y ref_max definidos empíricamente (percentiles 5 y 95)
    if ref_max <= 0:
        return 0.0
    return float(np.clip(mp_concentration / ref_max, 0, 1))

def compute_hazard_morphology(proportions: dict) -> float:
    """
    Calcula el componente morfológico a partir de proporciones.
    proportions debe sumar 1.
    """
    total = sum(proportions.values())
    if not np.isclose(total, 1.0):
        raise ValueError("Las proporciones deben sumar 1.")

    score = sum(proportions[k] * WEIGHTS[k] for k in proportions)
    return float(np.clip(score, 0, 1))

def compute_hazard_index(hazard_pressure: float, hazard_morphology: float) -> float:
    """
    El índice solo se define cuando existe presión por microplásticos.
    """
    if hazard_pressure <= 0:
        return 0.0
    HI = 0.5 * hazard_pressure + 0.5 * hazard_morphology
    return float(np.clip(HI, 0, 1))

def hazard_label(HI: float) -> str:
    if HI == 0:
        return "Nulo"
    elif HI < 0.33:
        return "Bajo"
    elif HI < 0.66:
        return "Moderado"
    elif HI < 0.8:
        return "Alto"
    else:
        return "Muy Alto"
    
def compute_hazard_morphology_from_props(proportions, weights=WEIGHTS):
    return sum(
        proportions[k] * weights[k]
        for k in proportions
    )