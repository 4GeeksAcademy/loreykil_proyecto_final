from src.prediction import (
    load_grilla,
    load_model,
    predict_microplastics_at_point
)

FEATURES = ['temperature', 'salinity', 'chlorophyll', 'nitrate',
            'phosphate', 'oxygen_dissolved', 'oxygen_utilization']

grilla = load_grilla()
model = load_model()

result = predict_microplastics_at_point(
    lon=-30,
    lat=10,
    grilla=grilla,
    model=model,
    feature_names=FEATURES
)

print(result)
