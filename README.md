# TaxiDuration AI 🚕

Aplicación interactiva de Machine Learning para predecir la duración de viajes en taxi. El proyecto utiliza un modelo **SVR** seleccionado después de comparar cinco modelos clásicos y tres ensambles bajo la metodología CRISP-DM.

## Funcionalidades

- Predicción individual con distancia, pasajeros, clima, fecha, hora y coordenadas.
- Selector interactivo entre los ocho modelos evaluados; SVR aparece como recomendado.
- Conversión bidireccional entre kilómetros y millas.
- Procesamiento masivo de archivos CSV.
- Conversión automática de kilómetros a millas.
- Imputación segura de campos opcionales.
- Comparación con valores reales cuando el archivo incluye `trip_duration`.
- Descarga de las predicciones generadas.
- Panel exploratorio del dataset.
- Comparación de los ocho modelos evaluados.

## Ejecución local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue en Streamlit Community Cloud

1. Ingresar a [share.streamlit.io](https://share.streamlit.io/).
2. Seleccionar **Create app**.
3. Elegir el repositorio `StevenCu07/Taxi_Drip`, rama `main`.
4. Indicar `app.py` como archivo principal.
5. Presionar **Deploy**.

## Archivos principales

- `app.py`: aplicación Streamlit.
- `Taxi_Trip_Duration.csv`: dataset usado para entrenar el modelo.
- `Taxi_Trip_Duration_Info.txt`: descripción de variables y problemas de calidad intencionales del dataset.
- `Trabajo_final.ipynb`: cuaderno completo de preparación, modelamiento y evaluación.
- `requirements.txt`: dependencias para el despliegue.

## Resultados del modelo seleccionado

| Métrica | Resultado |
|---|---:|
| MAE | 148,44 segundos |
| RMSE | 192,54 segundos |
| R² | 0,8671 |

Proyecto académico desarrollado por Brandon Steven Calzada Urrea y Juan Felipe Giraldo.
