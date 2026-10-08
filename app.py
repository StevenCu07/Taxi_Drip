from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR


APP_DIR = Path(__file__).parent
DATA_PATH = APP_DIR / "Taxi_Trip_Duration.csv"
MODEL_FEATURES = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "passenger_count",
    "trip_distance_miles",
    "pickup_hour",
    "pickup_day_of_week",
    "pickup_is_weekend",
    "weather",
]
NUMERIC_FEATURES = MODEL_FEATURES[:-1]
WEATHER_OPTIONS = ["Clear", "Rain", "Snow", "Fog"]

MODEL_RESULTS = pd.DataFrame(
    [
        ["SVR", 148.44, 192.54, 0.8671],
        ["Regresión lineal", 149.38, 192.98, 0.8665],
        ["MLP", 151.67, 197.21, 0.8606],
        ["Bagging", 155.61, 200.34, 0.8561],
        ["Voting", 156.28, 201.19, 0.8549],
        ["Boosting", 159.62, 206.45, 0.8472],
        ["Árbol de decisión", 171.58, 218.44, 0.8290],
        ["KNN", 209.46, 271.02, 0.7367],
    ],
    columns=["Modelo", "MAE", "RMSE", "R²"],
)


st.set_page_config(
    page_title="TaxiDuration AI",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink:#132238; --muted:#607086; --blue:#2165d6; --cyan:#27b4d1; --gold:#ffb000; }
    .stApp { background: linear-gradient(180deg, #f6f9ff 0%, #ffffff 42%); }
    .block-container { max-width: 1180px; padding-top: 1.8rem; padding-bottom: 3rem; }
    [data-testid="stSidebar"] { background: #101b2f; }
    [data-testid="stSidebar"] * { color: #f8fbff; }
    .hero {
      padding: 2.1rem 2.2rem; border-radius: 24px; color: white; margin-bottom: 1.2rem;
      background: radial-gradient(circle at 85% 20%, rgba(39,180,209,.45), transparent 32%),
                  linear-gradient(125deg, #101b2f 5%, #17498d 58%, #2165d6 100%);
      box-shadow: 0 18px 45px rgba(24,73,141,.22);
    }
    .hero h1 { margin:0; font-size:2.45rem; letter-spacing:-.04em; }
    .hero p { max-width:720px; margin:.7rem 0 0; color:#deebff; font-size:1.04rem; }
    .eyebrow { color:#7de3f3; font-weight:700; letter-spacing:.14em; font-size:.76rem; text-transform:uppercase; }
    .result-card {
      background:linear-gradient(135deg,#132238,#2165d6); color:white; padding:1.4rem 1.6rem;
      border-radius:20px; box-shadow:0 12px 28px rgba(22,65,130,.2); text-align:center;
    }
    .result-card .value { font-size:2.4rem; font-weight:800; margin:.15rem 0; }
    .result-card .small { color:#d6e6ff; font-size:.9rem; }
    .soft-card { background:white; border:1px solid #e4ebf5; border-radius:18px; padding:1rem 1.15rem; }
    div[data-testid="stMetric"] { background:white; border:1px solid #e1e8f2; padding:14px 16px; border-radius:16px; }
    .stButton>button, .stDownloadButton>button { border-radius:12px; font-weight:700; }
    h2, h3 { color:var(--ink); letter-spacing:-.02em; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_and_prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    raw = pd.read_csv(DATA_PATH)
    selected = raw.drop(
        columns=[
            "RecordID", "FullName", "Phone", "ZodiacSign", "FavoriteColor", "Hobby",
            "total_fare", "VendorID", "trip_distance",
        ],
        errors="ignore",
    ).copy()
    valid = (
        selected["trip_duration"].between(60, 14_400)
        & selected["trip_distance_miles"].gt(0)
        & selected["trip_distance_miles"].le(100)
        & (
            selected["passenger_count"].isna()
            | (
                selected["passenger_count"].between(1, 6)
                & selected["passenger_count"].mod(1).eq(0)
            )
        )
    )
    clean = selected.loc[valid].copy()
    dt = pd.to_datetime(clean["pickup_datetime"], dayfirst=True, errors="coerce")
    clean["pickup_hour"] = dt.dt.hour
    clean["pickup_day_of_week"] = dt.dt.dayofweek
    clean["pickup_is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    clean = clean.drop(columns=["pickup_datetime", "hour", "day_of_week", "is_weekend"])
    return raw, clean[MODEL_FEATURES], clean["trip_duration"]


@st.cache_resource
def train_model() -> tuple[Pipeline, dict[str, float | str]]:
    _, x, y = load_and_prepare_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=42
    )
    numeric_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Desconocido")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [("num", numeric_pipe, NUMERIC_FEATURES), ("cat", categorical_pipe, ["weather"])]
    )
    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("model", SVR(C=50, epsilon=0.05, kernel="linear"))]
    )
    pipeline.fit(x_train, y_train)
    prediction = pipeline.predict(x_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, prediction))),
        "r2": float(r2_score(y_test, prediction)),
        "training_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "model": "SVR lineal",
    }
    return pipeline, metrics


def seconds_to_text(seconds: float) -> str:
    rounded = max(0, int(round(seconds)))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours} h {minutes} min"
    return f"{minutes} min {secs} s"


def prepare_uploaded_data(upload: pd.DataFrame, reference: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    data = upload.copy()
    notes: list[str] = []
    data.columns = data.columns.str.strip()

    if "trip_distance_miles" not in data and "trip_distance" in data:
        data["trip_distance_miles"] = pd.to_numeric(data["trip_distance"], errors="coerce") / 1.609344
        notes.append("Se convirtió `trip_distance` de kilómetros a millas.")
    if "trip_distance_miles" not in data:
        raise ValueError("El archivo debe incluir `trip_distance_miles` o `trip_distance`.")

    if "pickup_datetime" in data:
        dt = pd.to_datetime(data["pickup_datetime"], dayfirst=True, errors="coerce")
        data["pickup_hour"] = dt.dt.hour
        data["pickup_day_of_week"] = dt.dt.dayofweek
        data["pickup_is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(float)
        notes.append("Las variables temporales se generaron desde `pickup_datetime`.")

    defaults = {
        **{column: float(reference[column].median()) for column in NUMERIC_FEATURES},
        "weather": str(reference["weather"].mode().iloc[0]),
    }
    missing = [column for column in MODEL_FEATURES if column not in data]
    for column in missing:
        data[column] = defaults[column]
    if missing:
        notes.append("Campos ausentes completados con valores típicos: " + ", ".join(missing) + ".")

    for column in NUMERIC_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["weather"] = data["weather"].astype("string").fillna("Desconocido")
    return data[MODEL_FEATURES], notes


raw_data, model_x, model_y = load_and_prepare_data()
model, model_metrics = train_model()

with st.sidebar:
    st.markdown("## 🚕 TaxiDuration AI")
    st.caption("Predicción y análisis de duración de viajes")
    st.markdown("---")
    st.markdown("**Modelo activo**")
    st.success("SVR lineal · Validado")
    st.metric("MAE de referencia", f"{model_metrics['mae']:.1f} s")
    st.metric("R²", f"{model_metrics['r2']:.3f}")
    st.markdown("---")
    st.caption("Proyecto final · Metodología CRISP-DM")
    st.caption("Brandon Steven Calzada Urrea · Juan Felipe Giraldo")

st.markdown(
    """
    <section class="hero">
      <div class="eyebrow">Machine learning aplicado a movilidad</div>
      <h1>Predice cuánto durará tu viaje</h1>
      <p>Una experiencia interactiva para estimar trayectos, analizar nuevos archivos y comparar el desempeño del modelo con datos reales.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_batch, tab_explore, tab_model = st.tabs(
    ["✨ Predicción individual", "📂 Analizar archivo", "📊 Explorar datos", "🧠 Acerca del modelo"]
)

with tab_predict:
    st.subheader("Configura un viaje")
    st.caption("Completa las condiciones conocidas antes de iniciar el recorrido.")
    with st.form("single_prediction"):
        left, middle, right = st.columns(3)
        with left:
            distance = st.number_input("Distancia estimada (millas)", min_value=0.1, max_value=100.0, value=3.0, step=0.1)
            passengers = st.number_input("Número de pasajeros", min_value=1, max_value=6, value=1)
            weather = st.selectbox("Clima", WEATHER_OPTIONS)
        with middle:
            travel_date = st.date_input("Fecha del viaje", value=date(2023, 6, 15))
            travel_time = st.time_input("Hora de recogida", value=time(12, 0))
            st.caption("Las coordenadas permiten incorporar el contexto espacial del trayecto.")
        with right:
            pickup_lat = st.number_input("Latitud de origen", value=40.7500, format="%.6f")
            pickup_lon = st.number_input("Longitud de origen", value=-73.9800, format="%.6f")
            dropoff_lat = st.number_input("Latitud de destino", value=40.7700, format="%.6f")
            dropoff_lon = st.number_input("Longitud de destino", value=-73.9500, format="%.6f")
        submitted = st.form_submit_button("Estimar duración", type="primary", width="stretch")

    if submitted:
        trip_dt = datetime.combine(travel_date, travel_time)
        row = pd.DataFrame(
            [{
                "pickup_latitude": pickup_lat, "pickup_longitude": pickup_lon,
                "dropoff_latitude": dropoff_lat, "dropoff_longitude": dropoff_lon,
                "passenger_count": passengers, "trip_distance_miles": distance,
                "pickup_hour": trip_dt.hour, "pickup_day_of_week": trip_dt.weekday(),
                "pickup_is_weekend": int(trip_dt.weekday() >= 5), "weather": weather,
            }]
        )
        estimate = float(model.predict(row)[0])
        low = max(60.0, estimate - model_metrics["mae"])
        high = estimate + model_metrics["mae"]
        c1, c2 = st.columns([1.1, 1])
        with c1:
            st.markdown(
                f"""<div class="result-card"><div class="small">Duración estimada</div>
                <div class="value">{seconds_to_text(estimate)}</div><div>{estimate:,.0f} segundos</div>
                <div class="small" style="margin-top:.65rem">Rango orientativo: {seconds_to_text(low)} – {seconds_to_text(high)}</div></div>""",
                unsafe_allow_html=True,
            )
        with c2:
            st.info(
                "El rango usa el MAE del conjunto de prueba como referencia. No es un intervalo de confianza y puede variar por tráfico, incidentes o cierres viales."
            )

with tab_batch:
    st.subheader("Predicción masiva y comparación")
    st.write(
        "Carga un CSV con `trip_distance_miles` o `trip_distance`. Si también contiene `trip_duration`, la app calculará métricas contra los valores reales."
    )
    with st.expander("Ver columnas recomendadas"):
        st.code(
            "pickup_datetime, pickup_latitude, pickup_longitude, dropoff_latitude, "
            "dropoff_longitude, passenger_count, trip_distance_miles, weather, trip_duration",
            language="text",
        )
        sample = raw_data[[
            "pickup_datetime", "pickup_latitude", "pickup_longitude", "dropoff_latitude",
            "dropoff_longitude", "passenger_count", "trip_distance_miles", "weather", "trip_duration",
        ]].head(20)
        st.download_button(
            "Descargar CSV de ejemplo", sample.to_csv(index=False).encode("utf-8"),
            "ejemplo_viajes_taxi.csv", "text/csv",
        )

    uploaded = st.file_uploader("Arrastra un archivo CSV", type=["csv"])
    if uploaded is not None:
        try:
            uploaded_df = pd.read_csv(uploaded)
            prepared, preparation_notes = prepare_uploaded_data(uploaded_df, model_x)
            output = uploaded_df.copy()
            output["predicted_trip_duration"] = model.predict(prepared)
            output["predicted_duration_minutes"] = output["predicted_trip_duration"] / 60
            st.success(f"Archivo procesado correctamente: {len(output):,} registros.")
            for note in preparation_notes:
                st.caption("• " + note)

            if "trip_duration" in output:
                valid_target = pd.to_numeric(output["trip_duration"], errors="coerce").notna()
                actual = pd.to_numeric(output.loc[valid_target, "trip_duration"])
                predicted = output.loc[valid_target, "predicted_trip_duration"]
                if len(actual) >= 2:
                    mae = mean_absolute_error(actual, predicted)
                    rmse = np.sqrt(mean_squared_error(actual, predicted))
                    r2 = r2_score(actual, predicted)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("MAE del archivo", f"{mae:,.1f} s", delta=f"{model_metrics['mae']-mae:+.1f} vs. referencia")
                    m2.metric("RMSE", f"{rmse:,.1f} s")
                    m3.metric("R²", f"{r2:.3f}")
                    comparison = pd.DataFrame({"Real": actual, "Predicción": predicted})
                    fig = px.scatter(
                        comparison, x="Real", y="Predicción", opacity=0.55,
                        title="Duración real frente a predicción", template="plotly_white",
                    )
                    max_value = float(comparison.max().max())
                    fig.add_shape(type="line", x0=0, y0=0, x1=max_value, y1=max_value, line=dict(dash="dash", color="#e74c3c"))
                    st.plotly_chart(fig, width="stretch")
            st.dataframe(output.head(200), width="stretch", hide_index=True)
            st.download_button(
                "Descargar predicciones", output.to_csv(index=False).encode("utf-8"),
                "predicciones_taxi.csv", "text/csv", type="primary",
            )
        except Exception as exc:
            st.error(f"No fue posible procesar el archivo: {exc}")

with tab_explore:
    st.subheader("Radiografía del conjunto de datos")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Registros originales", f"{len(raw_data):,}")
    e2.metric("Registros modelados", f"{len(model_x):,}")
    e3.metric("Distancia mediana", f"{model_x.trip_distance_miles.median():.2f} mi")
    e4.metric("Duración mediana", seconds_to_text(model_y.median()))

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(
            pd.DataFrame({"Duración (min)": model_y / 60}), x="Duración (min)", nbins=45,
            title="Distribución de la duración", color_discrete_sequence=["#2165d6"], template="plotly_white",
        )
        st.plotly_chart(fig, width="stretch")
    with c2:
        relation = pd.DataFrame({"Distancia (mi)": model_x.trip_distance_miles, "Duración (min)": model_y / 60})
        fig = px.scatter(
            relation, x="Distancia (mi)", y="Duración (min)", opacity=0.4,
            title="Distancia frente a duración", color_discrete_sequence=["#27b4d1"], template="plotly_white",
        )
        st.plotly_chart(fig, width="stretch")

    weather_df = pd.DataFrame({"Clima": model_x.weather.fillna("Desconocido"), "Duración (min)": model_y / 60})
    fig = px.box(
        weather_df, x="Clima", y="Duración (min)", color="Clima",
        title="Duración según condición climática", template="plotly_white",
    )
    st.plotly_chart(fig, width="stretch")

with tab_model:
    st.subheader("Modelo, evaluación y transparencia")
    st.write(
        "El flujo aplica CRISP-DM: selección por conocimiento del negocio, limpieza de errores, ingeniería temporal, preprocesamiento sin fuga de información, validación cruzada y evaluación independiente."
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Modelo ganador", "SVR")
    m2.metric("MAE", f"{model_metrics['mae']:.2f} s")
    m3.metric("RMSE", f"{model_metrics['rmse']:.2f} s")
    m4.metric("R²", f"{model_metrics['r2']:.4f}")

    fig = px.bar(
        MODEL_RESULTS.sort_values("MAE", ascending=False), x="MAE", y="Modelo", orientation="h",
        color="MAE", color_continuous_scale=["#27b4d1", "#2165d6", "#132238"],
        title="MAE de los ocho modelos en el conjunto de prueba", template="plotly_white",
    )
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, width="stretch")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Variables utilizadas")
        st.markdown(
            "- Coordenadas de origen y destino\n- Pasajeros y distancia estimada\n"
            "- Hora, día de la semana y fin de semana\n- Condición climática"
        )
    with col_b:
        st.markdown("#### Salvaguardas metodológicas")
        st.markdown(
            "- `total_fare` excluida por fuga de información\n- Imputación aprendida solo con entrenamiento\n"
            "- One-Hot Encoding para el clima\n- Prueba independiente del 30 %"
        )
    st.warning(
        "Limitación: el modelo no recibe tráfico en tiempo real, obras, cierres ni incidentes. La estimación es académica y no sustituye un sistema de navegación."
    )
