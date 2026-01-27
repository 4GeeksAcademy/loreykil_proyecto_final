# Modelado espacial de microplásticos y análisis de presión ecológica
*Análisis y modelado espacial con enfoques predictivos y explicativos*

Este proyecto analiza la distribución espacial de microplásticos y su relación
con variables ambientales y de riesgo ecológico, combinando modelos predictivos
y análisis explicativos espaciales.

## Aplicación interactiva
🔗 **Render App:** [https://tu-app.streamlit.app](https://loreykil-proyecto-final.onrender.com/)

La aplicación permite explorar de forma interactiva los mapas de microplásticos,
comparar la presión por microplásticos con el riesgo ecológico y visualizar
resultados del modelado espacial.


## 1. Contexto y problema
La contaminación por microplásticos es una de las principales presiones antrópicas
sobre los ecosistemas marinos. Su disponibilidad espacial es altamente heterogénea y
depende de factores ambientales, dinámicos y geomorfológicos.

Comprender dónde se concentran los microplásticos y cómo se relacionan con indicadores
de riesgo ecológico es clave para identificar áreas potencialmente vulnerables y
orientar futuros estudios y estrategias de gestión.

## 2. Objetivos
Este proyecto tiene como objetivos principales:

- Modelar la concentración de microplásticos a partir de variables ambientales.
- Generar mapas espaciales continuos de microplásticos predichos.
- Comparar la presión por microplásticos con índices de riesgo ecológico.
- Explorar relaciones presión–impacto potencial desde un enfoque explicativo.

## 3. Tipo de proyecto
Este es un proyecto de ciencia de datos aplicada, que combina:

- Enfoque predictivo (modelado y proyección espacial).
- Enfoque explicativo (análisis de relaciones ecológicas).
- Análisis exploratorio espacial.

El proyecto no plantea inferencias causales, sino análisis de patrones y
asociaciones espaciales.

## 4. Datos
Se utilizan datos geoespaciales en formato vectorial, que incluyen:

- Mediciones puntuales de microplásticos.
- Variables ambientales asociadas (condiciones físicas y oceanográficas).
- Una grilla oceánica para proyección espacial.
- Un índice agregado de riesgo ecológico.
- Datos de especies marinas en distintos niveles de riesgo

Los datos han sido preprocesados para asegurar coherencia espacial

## 5. Metodología
El trabajo se estructura en dos workflows complementarios:

### Workflow A – Predictivo / Diagnóstico
1. Entrenamiento de modelos para predecir microplásticos a partir de variables ambientales.
2. Proyección del modelo sobre una grilla oceánica para obtener mapas continuos.
3. Uso de microplásticos modelados como métricas de presión.

### Workflow B – Explicativo / Ecológico
1. Construcción de indicadores agregados de presión y morfología.
2. Análisis estadístico y espacial de su relación con variables ecológicas agregadas.

## 6. Modelado y validación
- Modelos de regresión aplicados a datos espaciales.
- Transformaciones logarítmicas y estandarización cuando es necesario.
- Validación cruzada espacial basada en bloques.
- Métricas de evaluación: R², MAE y RMSE bajo validación espacial.

La validación espacial permite evaluar la capacidad de generalización del modelo
a nuevas regiones.

## 7. Resultados principales
- El modelo explica una fracción moderada de la variabilidad de microplásticos bajo
  validación espacial, consistente con la complejidad del sistema.
- Los mapas predichos muestran patrones espaciales claros y regiones de alta presión.

  ## 8. Limitaciones
- Alta heterogeneidad espacial y ruido en las mediciones.
- Escalas espaciales limitadas por la disponibilidad de datos.
- Interpretación no causal de las relaciones observadas.
- Resultados dependientes de la resolución de la grilla y del esquema de validación.

## 9. Reproducibilidad
El proyecto está organizado para facilitar su reproducción:

1. Crear un entorno de Python (>= 3.10).
2. Instalar dependencias: pip install -r requirements.txt
