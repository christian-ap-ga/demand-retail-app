"""
Servidor de inferencia Flask para SageMaker BYOC.
Endpoints requeridos:
  GET  /ping        → 200 si el modelo está listo
  POST /invocations → recibe CSV, devuelve predicciones CSV
"""

import io
import os

import flask
import joblib
import numpy as np
import polars as pl
import pandas as pd

from utils.features import KEY_COLS, NUM_COLS

MODEL_PATH = "/opt/ml/model"

app = flask.Flask(__name__)
model     = None
last_hist = None


def load_model():
    global model, last_hist
    model     = joblib.load(os.path.join(MODEL_PATH, "xgb_monthly_forecast_model.joblib"))
    last_hist = pl.read_parquet(os.path.join(MODEL_PATH, "last_hist.parquet"))
    # Catálogos auxiliares que también se guardaron en MODEL_PATH durante train
    app.items_catalog            = pl.read_parquet(os.path.join(MODEL_PATH, "items_catalog.parquet"))
    app.items_categories_catalog = pl.read_parquet(os.path.join(MODEL_PATH, "items_categories_catalog.parquet"))
    app.shops_catalog            = pl.read_parquet(os.path.join(MODEL_PATH, "shops_catalog.parquet"))


@app.route("/ping", methods=["GET"])
def ping():
    """SageMaker llama a este endpoint para verificar que el container está sano."""
    status = 200 if model is not None else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def invocations():
    """
    Recibe un CSV con columnas [ID, shop_id, item_id] y devuelve predicciones CSV.
    """
    if flask.request.content_type != "text/csv":
        return flask.Response(
            response="Este predictor solo acepta text/csv",
            status=415,
            mimetype="text/plain",
        )

    data = flask.request.data.decode("utf-8")
    test_to_predict = pl.read_csv(io.StringIO(data))

    # Misma transformación que inference.py → data_transform()
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

    result = test_augmented.select("ID").with_columns(
        pl.Series("item_cnt_month", y_pred)
    ).to_pandas()

    out = io.StringIO()
    result.to_csv(out, index=False)
    return flask.Response(response=out.getvalue(), status=200, mimetype="text/csv")


# Carga el modelo al importar el módulo
load_model()