"""
Servidor de inferencia Flask para SageMaker BYOC.
Endpoints requeridos:
  GET  /ping        → 200 si el modelo está listo
  POST /invocations → recibe CSV, devuelve predicciones CSV y persiste en S3
"""
 
import io
import os
import uuid
from datetime import datetime, timezone
 
import boto3
import flask
import joblib
import numpy as np
import polars as pl
import pandas as pd
 
from utils.features import KEY_COLS, NUM_COLS
 
MODEL_PATH = "/opt/ml/model"
 
# Bucket y prefix de destino para predicciones
PREDICTIONS_BUCKET = "1c-company-medallion"
PREDICTIONS_PREFIX = "gold/predictions"
 
# Intervalo de confianza: ±1 desviación estándar de los residuos de entrenamiento.
# Se estima como el RMSE del holdout guardado en el artefacto del modelo.
# Si el archivo no existe, se usa un fallback del 15 %.
FALLBACK_INTERVAL_RATIO = 0.15
 
app = flask.Flask(__name__)
model         = None
last_hist     = None
interval_std  = None   # desviación estándar estimada para upper/lower
 
 
def _load_interval_std() -> float:
    """
    Lee el RMSE del holdout guardado por train como estimación de la
    dispersión del error. Si no existe usa FALLBACK_INTERVAL_RATIO.
    """
    metrics_path = os.path.join(MODEL_PATH, "holdout_metrics.csv")
    if os.path.exists(metrics_path):
        metrics = pd.read_csv(metrics_path)
        rmse_row = metrics[metrics["metric"] == "RMSE"]
        if not rmse_row.empty:
            return float(rmse_row["value"].iloc[0])
    return None
 
 
def load_model():
    global model, last_hist, interval_std
    model     = joblib.load(os.path.join(MODEL_PATH, "xgb_monthly_forecast_model.joblib"))
    last_hist = pl.read_parquet(os.path.join(MODEL_PATH, "last_hist.parquet"))
    app.items_catalog            = pl.read_parquet(os.path.join(MODEL_PATH, "items_catalog.parquet"))
    app.items_categories_catalog = pl.read_parquet(os.path.join(MODEL_PATH, "items_categories_catalog.parquet"))
    app.shops_catalog            = pl.read_parquet(os.path.join(MODEL_PATH, "shops_catalog.parquet"))
    interval_std = _load_interval_std()
 
 
def _infer_prediction_date() -> str:
    """
    Infiere la fecha de predicción como el primer día del mes siguiente
    al último registro en last_hist (campo year_month_date).
    Devuelve un string ISO YYYY-MM-DD.
    """
    if last_hist is not None and "year_month_date" in last_hist.columns:
        last_date = last_hist["year_month_date"].max()
        # Avanzar un mes
        year  = last_date.year + (1 if last_date.month == 12 else 0)
        month = 1 if last_date.month == 12 else last_date.month + 1
        return f"{year:04d}-{month:02d}-01"
    # Fallback: mes siguiente al actual
    now = datetime.now(timezone.utc)
    year  = now.year + (1 if now.month == 12 else 0)
    month = 1 if now.month == 12 else now.month + 1
    return f"{year:04d}-{month:02d}-01"
 
 
def _save_to_s3(df: pd.DataFrame, prediction_date: str) -> str:
    """
    Persiste el DataFrame de predicciones en S3 como parquet particionado
    por fecha de predicción.
    Retorna la ruta S3 completa donde se escribió el archivo.
    """
    s3 = boto3.client("s3")
    run_id   = uuid.uuid4().hex[:8]
    s3_key   = f"{PREDICTIONS_PREFIX}/prediction_date={prediction_date}/predictions_{run_id}.parquet"
 
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, engine="pyarrow")
    buffer.seek(0)
 
    s3.put_object(
        Bucket=PREDICTIONS_BUCKET,
        Key=s3_key,
        Body=buffer.getvalue(),
        ContentType="application/octet-stream",
    )
    return f"s3://{PREDICTIONS_BUCKET}/{s3_key}"
 
 
@app.route("/ping", methods=["GET"])
def ping():
    status = 200 if model is not None else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")
 
 
@app.route("/invocations", methods=["POST"])
def invocations():
    """
    Recibe CSV con columnas [ID, shop_id, item_id].
    Devuelve CSV con columnas:
      id, item_id, shop_id, category_id, region, date, created_at,
      value, value_upper, value_lower
    y persiste el resultado como parquet en S3.
    """
    if flask.request.content_type != "text/csv":
        return flask.Response(
            response="Este predictor solo acepta text/csv",
            status=415,
            mimetype="text/plain",
        )
 
    data = flask.request.data.decode("utf-8")
    test_to_predict = pl.read_csv(io.StringIO(data))
 
    # ── Feature engineering ──────────────────────────
    num_cols_wo_avg = [c for c in NUM_COLS if c != "avg_price"]
 
    test_augmented = (
        test_to_predict
        .with_columns(
            pl.col("shop_id").cast(pl.String),
            pl.col("item_id").cast(pl.String),
            pl.lit(11).alias("month"),
        )
        .join(
            app.items_catalog.select("item_id", "item_category_id", "item_age").with_columns(
                pl.col("item_id").cast(pl.String),
                pl.col("item_category_id").cast(pl.String),
            ),
            on="item_id", how="left",
        )
        .join(app.items_categories_catalog.select("item_category_id", "general_category", "device"),
              on="item_category_id", how="left")
        .join(app.shops_catalog.select("shop_id", "macro_region", "segment"),
              on="shop_id", how="left")
        .join(last_hist.select(*KEY_COLS, *NUM_COLS), on=KEY_COLS, how="left")
        .with_columns(
            pl.col(num_cols_wo_avg).fill_null(0),
            pl.when(pl.col("avg_price").is_null())
              .then(pl.col("avg_price").mean().over("general_category"))
              .otherwise(pl.col("avg_price"))
              .alias("avg_price"),
        )
    )
 
    X = test_augmented.to_pandas()
    y_pred = np.expm1(model.predict(X))
 
    # ── Intervalos de confianza ──────────────────────────────────────────────
    if interval_std is not None:
        margin = interval_std
    else:
        margin = y_pred * FALLBACK_INTERVAL_RATIO
 
    # ── Construcción del output enriquecido ──────────────────────────────────
    prediction_date = _infer_prediction_date()
    created_at      = datetime.now(timezone.utc).strftime("%Y-%m-%d %Human:%M:%S")
 
    aug_pd = test_augmented.to_pandas()
 
    result = pd.DataFrame({
        "id":          [str(uuid.uuid4()) for _ in range(len(y_pred))],
        "item_id":     aug_pd["item_id"],
        "shop_id":     aug_pd["shop_id"],
        "category_id": aug_pd["item_category_id"],
        "region":      aug_pd["macro_region"],
        "date":        prediction_date,
        "created_at":  created_at,
        "value":       np.round(y_pred, 4),
        "value_upper": np.round(y_pred + margin, 4),
        "value_lower": np.round(np.maximum(y_pred - margin, 0), 4),  # nunca negativo
    })
 
    # ── Persistencia en S3 ───────────────────────────────────────────────────
    s3_path = _save_to_s3(result, prediction_date)
    app.logger.info("Predicciones guardadas en %s", s3_path)
 
    # ── Respuesta CSV al caller ──────────────────────────────────────────────
    out = io.StringIO()
    result.to_csv(out, index=False)
    return flask.Response(response=out.getvalue(), status=200, mimetype="text/csv")
 
 
load_model()