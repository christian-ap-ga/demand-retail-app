"""
Generación de predicciones mensuales a partir de un modelo entrenado.

Este módulo:
- Carga el modelo entrenado y catálogos auxiliares.
- Reconstruye el set de features requerido por el modelo.
- Aplica el pipeline de inferencia.
- Genera predicciones y guarda el archivo final.

Salida:
- data/predictions/predictions.csv
"""

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
from joblib import load

from src.utils.config import (
    ARTIFACTS_DIR,
    INFERENCE_DIR,
    PREDICTIONS_DIR,
    RAW_DIR,
    log_process_time,
    setup_logging,
)
from src.utils.features import KEY_COLS, NUM_COLS

logger = setup_logging("inference")

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the inference pipeline.
 
    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Batch inference pipeline for monthly sales forecasting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
 
    # Input filenames
    parser.add_argument(
        "--test-file",
        type=str,
        default="test.csv",
        help="Filename of the test data (inside RAW_DIR)",
    )
    parser.add_argument(
        "--model-file",
        type=str,
        default="xgb_monthly_forecast_model.joblib",
        help="Filename of the trained model (inside ARTIFACTS_DIR)",
    )
    parser.add_argument(
        "--last-hist-file",
        type=str,
        default="last_hist.parquet",
        help="Filename of the last historical record (inside INFERENCE_DIR)",
    )
 
    # Output filename (resolved against PREDICTIONS_DIR)
    parser.add_argument(
        "--predictions-file",
        type=str,
        default="predictions.csv",
        help="Output filename for predictions (inside PREDICTIONS_DIR)",
    )
 
    return parser.parse_args()


# Carga de objetos


def load_objects(raw_dir: Path, artifacts_dir: Path, inference_dir: Path, test_file:str, model_file:str, last_hist_file: str):
    """
    Carga el modelo entrenado y los artefactos necesarios para inferencia.

    Parameters
    ----------
    raw_dir : Path
        Directorio con el archivo test.csv.
    artifacts_dir : Path
        Directorio con el modelo entrenado.
    inference_dir : Path
        Directorio con catálogos y features históricos.

    Returns
    -------
    tuple
        clf_model : sklearn.pipeline.Pipeline
            Modelo entrenado.
        items_catalog : pl.DataFrame
            Catálogo de productos.
        items_categories_catalog : pl.DataFrame
            Catálogo de categorías.
        shops_catalog : pl.DataFrame
            Catálogo de tiendas.
        last_hist : pl.DataFrame
            Última observación histórica por shop-item.
        test : pl.DataFrame
            Dataset de entrada para predicciones.
    """

    clf_model = load(artifacts_dir / model_file)
    items_catalog = pl.read_parquet(inference_dir / "items_catalog.parquet")
    items_categories_catalog = pl.read_parquet(inference_dir / "items_categories_catalog.parquet")
    shops_catalog = pl.read_parquet(inference_dir / "shops_catalog.parquet")
    last_hist = pl.read_parquet(inference_dir / last_hist_file)
    test = pl.read_csv(raw_dir / test_file)

    return (
        clf_model,
        items_catalog,
        items_categories_catalog,
        shops_catalog,
        last_hist,
        test,
    )


def data_transform(
    test_to_predict: pl.DataFrame,
    items_catalog: pl.DataFrame,
    items_categories_catalog: pl.DataFrame,
    shops_catalog: pl.DataFrame,
    last_hist: pl.DataFrame,
) -> pl.DataFrame:
    """
    Reconstruye el set de features requerido para inferencia.

    Realiza:
    - Casting de IDs.
    - Unión con catálogos.
    - Incorporación de lags y variables temporales.
    - Imputación de valores faltantes.

    Parameters
    ----------
    df : pl.DataFrame
        Dataset base de predicción.
    items_catalog : pl.DataFrame
    items_categories_catalog : pl.DataFrame
    shops_catalog : pl.DataFrame
    last_hist : pl.DataFrame

    Returns
    -------
    pl.DataFrame
        Dataset listo para predicción.
    """

    test_augmented = (
        test_to_predict.with_columns(
            pl.col("shop_id").cast(pl.String),
            pl.col("item_id").cast(pl.String),
            pl.lit(11).alias("month"),
        )
        .join(
            items_catalog.select("item_id", "item_category_id", "item_age").with_columns(
                pl.col("item_id").cast(pl.String),
                pl.col("item_category_id").cast(pl.String),
            ),
            on="item_id",
            how="left",
        )
        .join(
            items_categories_catalog.select("item_category_id", "general_category", "device"),
            on="item_category_id",
            how="left",
        )
        .join(
            shops_catalog.select("shop_id", "macro_region", "segment"),
            on="shop_id",
            how="left",
        )
    )

    num_cols_wo_avg = [c for c in NUM_COLS if c != "avg_price"]

    test_augmented = test_augmented.join(
        last_hist.select(*KEY_COLS, *NUM_COLS),
        on=KEY_COLS,
        how="left",
    ).with_columns(
        pl.col(num_cols_wo_avg).fill_null(0),
        pl.when(pl.col("avg_price").is_null())
        .then(pl.col("avg_price").mean().over("general_category"))
        .otherwise(pl.col("avg_price"))
        .alias("avg_price"),
    )

    return test_augmented


def run_model(clf_model, test_augmented: pl.DataFrame) -> pl.DataFrame:
    """
    Genera predicciones a partir del modelo entrenado.

    Returns
    -------
    pl.DataFrame
        DataFrame con ID y predicción.
    """
    predictor = test_augmented.to_pandas()
    target_log = clf_model.predict(predictor)
    target = np.expm1(target_log)

    return test_augmented.select("ID").with_columns(pl.Series("item_cnt_month", target))


def main():
    """
    Ejecuta el flujo completo de inferencia.

    Pasos:
    - Carga de modelo y catálogos.
    - Transformación de datos de entrada.
    - Generación de predicciones.
    - Guardado del archivo final.
    """
    args = parse_arguments()

    logger.info("Pipeline Configuration:")
    logger.info(f"  Test data: {args.test_file}")
    logger.info(f"  Model: {args.model_file}")
    logger.info(f"  Predictions file: {args.predictions_file}")
    logger.info("")

    logger.info("Iniciando proceso de generación de predicciones")
    start_time = time.time()

    # Cargar de datos y modelo:
    (
        clf_model,
        items_catalog,
        items_categories_catalog,
        shops_catalog,
        last_hist,
        test_to_predict,
    ) = load_objects(
        RAW_DIR,
        ARTIFACTS_DIR,
        INFERENCE_DIR,
        test_file=args.test_file,
        model_file=args.model_file,
        last_hist_file=args.last_hist_file
        )

    logger.info("Carga de datos y modelo completa")
    logger.info("Realizando transformacion de datos...")

    # Transformación de datos de entrada para predicciones
    test_augmented = data_transform(
        test_to_predict,
        items_catalog,
        items_categories_catalog,
        shops_catalog,
        last_hist,
    )

    logger.info("Generando predicciones...")
    # Generación de predicciones
    predictions = run_model(clf_model, test_augmented)

    logger.info("Guardando predicciones...")
    predictions.write_csv(PREDICTIONS_DIR / args.predictions_file)

    logger.info("Predicciones guardadas")

    log_process_time(logger, start_time)


if __name__ == "__main__":
    main()
