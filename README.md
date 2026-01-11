# Práctica 2 - Tipología y Ciclo de Vida de los Datos

## Descripción

Este repositorio contiene la Práctica 2 de la asignatura **Tipología y Ciclo de Vida de los Datos** del Máster en Ciencia de Datos de la Universitat Oberta de Catalunya (UOC).

El proyecto consiste en el análisis de la **Base de Datos Nacional de Colisiones de Canadá (NCDB)** para el período 2015-2020, con el objetivo de identificar los factores que influyen en la severidad de los accidentes de tráfico.

## Integrantes del grupo

| Nombre | Correo |
|--------|--------|
| [abdallah tegguer] | [ategguer@uoc.edu] |
## Estructura del repositorio

```
practica2_final/project/tpcvd/
│
├── README.md                          # Este archivo
│
├── codigo/
│   └── practica2_analisis.py          # Código principal del análisis
│
├── datos/
│   ├── y_2015_en.csv                  # Datos originales 2015
│   ├── y_2016_en.csv                  # Datos originales 2016
│   ├── y_2017_en.csv                  # Datos originales 2017
│   ├── y_2018_en.csv                  # Datos originales 2018
│   ├── 2019_dataset_en.csv            # Datos originales 2019
│   ├── y_2020.csv                     # Datos originales 2020
│   └── dataset_limpio.csv             # Dataset después de la limpieza
│   └── * mas archivoes de 1999-2014 originamles
├── graficos/
│   ├── distribucion_severidad.png
│   ├── distribucion_variables_numericas.png
│   ├── evolucion_temporal.png
│   ├── analisis_hora.png
│   ├── analisis_edad.png
│   ├── analisis_clima.png
│   ├── matriz_confusion_rf.png
│   ├── importancia_caracteristicas.png
│   ├── clustering_metricas.png
│   ├── clusters_visualizacion.png
│   ├── matriz_correlacion.png
│   └── boxplots_comparativos.png
│
└── memoria/
    └── memoria_practica2.pdf          # Documento PDF con las respuestas
```

## Dataset

**Fuente:** National Collision Database (NCDB) - Transport Canada  
**URL:** https://open.canada.ca/data/en/dataset/1eb9ead3-3e3c-4acb-9d9d-4c8c9e7c7c1e

### Descripción de variables principales

| Variable | Descripción |
|----------|-------------|
| C_YEAR | Año de la colisión |
| C_MNTH | Mes de la colisión (1-12) |
| C_WDAY | Día de la semana (1=Lunes a 7=Domingo) |
| C_HOUR | Hora de la colisión (0-23) |
| C_SEV | Severidad (1=Fatal, 2=Con heridos) |
| C_VEHS | Número de vehículos involucrados |
| C_WTHR | Condiciones climáticas |
| C_RSUR | Condición de la superficie de la carretera |
| P_SEX | Sexo de la persona (M/F) |
| P_AGE | Edad de la persona |
| V_YEAR | Año del modelo del vehículo |
| V_TYPE | Tipo de vehículo |

## Requisitos

### Librerías de Python necesarias

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

### Versiones utilizadas

- Python 3.8+
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scipy >= 1.7.0
- scikit-learn >= 0.24.0

## Ejecución

1. Clonar el repositorio:
```bash
git clone https://github.com/abdallahtegguer/practica2_final.git
cd practica2-TCVD
```

2. Instalar las dependencias:
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```


3. Ejecutar el script principal:
```bash
cd project
cd tpcvd
python practica2_analisis.py
```

## Resultados principales

### Análisis realizado

1. **Limpieza de datos:** Tratamiento de valores faltantes, conversión de tipos, gestión de outliers
2. **Modelo supervisado:** Random Forest Classifier (Accuracy: 74.04%)
3. **Modelo no supervisado:** K-Means Clustering (K=4)
4. **Contraste de hipótesis:** Tests de Mann-Whitney U y Chi-cuadrado

### Conclusiones principales

- La **madrugada** (00:00-06:00) presenta la mayor tasa de fatalidad (3.42%)
- Los **fines de semana** tienen mayor tasa de fatalidad (1.94% vs 1.53%)
- Las **personas mayores** tienen mayor riesgo de fatalidad en accidentes
- El **número de vehículos** y la **hora del accidente** son los predictores más importantes

## Vídeo

Enlace al vídeo explicativo: 

## Licencia

Este proyecto se ha desarrollado como parte de una actividad académica de la UOC.

Los datos originales provienen de Transport Canada y están disponibles bajo la Open Government Licence - Canada.

## Referencias

- Transport Canada. National Collision Database (NCDB). https://open.canada.ca/
- Calvo M., Subirats L., Pérez D. (2019). Introducción a la limpieza y análisis de los datos. Editorial UOC.
- Scikit-learn: Machine Learning in Python. https://scikit-learn.org/
