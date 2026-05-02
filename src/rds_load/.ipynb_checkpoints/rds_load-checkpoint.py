"""
rds_load.py — ETL RDS Layer: carga de datos a PostgreSQL vía SQLAlchemy 2.0.

Este script forma parte del pipeline de demanda retail. Su responsabilidad es 
insertar los datos desde archivos parquet en S3 hacia una base de datos 
PostgreSQL en Amazon RDS, respetando el orden de dependencias FK definido 
en el modelo de datos:

  1. categories         — sin dependencias
  2. shops              — sin dependencias
  3. items              — depende de categories (category_id)
  4. predictions        — depende de items (item_id), shops (shop_id), categories (category_id)
  5. feedback           — depende de items (item_id), categories (category_id)
  6. model_evaluation   — depende de categories (category_id)
  7. system_events      — sin dependencias FK

Los archivos parquet se obtienen desde un bucket S3 configurado en S3_PATHS.
Las credenciales de RDS se obtienen exclusivamente desde AWS Secrets Manager.
El endpoint RDS y credenciales se definen en constantes de configuración.

Uso:
    python src/rds_load/rds_load.py

Las constantes de configuración deben actualizarse en el módulo:
    - S3_BUCKET: nombre del bucket S3
    - S3_PATHS: diccionario con rutas de archivos parquet
    - SECRET_ID: ARN o nombre del secreto en Secrets Manager
    - RDS_ENDPOINT: host del endpoint primario de RDS
    - REGION: región AWS
"""

import argparse
import json
import logging
import sys
from typing import Optional

import boto3
import pandas as pd
from sqlalchemy import (
    BigInteger,
    Boolean,
    Float,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    create_engine,
    insert,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


# Configuración de logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    """
    Configura y retorna el logger del módulo.

    Formato: timestamp — nivel — mensaje. Se usa logging estándar en lugar de
    print() para garantizar trazabilidad cuando el pipeline corre sin supervisión.

    Returns:
        logging.Logger: Logger configurado con nivel INFO y salida a stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


# Modelos SQLAlchemy 2.0
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """Base declarativa compartida por todos los modelos del pipeline."""
    pass


class Categories(Base):
    """Tabla de categorías de productos."""
    __tablename__ = "categories"
 
    category_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    general_category: Mapped[str] = mapped_column(String(100), nullable=False)
    main_category: Mapped[str] = mapped_column(String(100), nullable=False)
    sub_category: Mapped[str] = mapped_column(String(100), nullable=False)
    device: Mapped[str] = mapped_column(String(100), nullable=False)
 
 
class Items(Base):
    """Tabla de items/productos."""
    __tablename__ = "items"
 
    item_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    category_id: Mapped[str] = mapped_column(String(10), ForeignKey("categories.category_id"), nullable=False)
    item_name: Mapped[str] = mapped_column(String(100), nullable=False)
    item_age: Mapped[str] = mapped_column(String(100), nullable=False)
 
 
class Shops(Base):
    """Tabla de tiendas/sucursales."""
    __tablename__ = "shops"
 
    shop_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    shop_name: Mapped[str] = mapped_column(String(100), nullable=False)
    region: Mapped[str] = mapped_column(String(100), nullable=False)
    macro_region: Mapped[str] = mapped_column(String(100), nullable=False)
    segment: Mapped[str] = mapped_column(String(100), nullable=False)
 
 
class Predictions(Base):
    """Tabla de predicciones de demanda."""
    __tablename__ = "predictions"
 
    id: Mapped[str] = mapped_column(String(10), primary_key=True)
    item_id: Mapped[str] = mapped_column(String(10), ForeignKey("items.item_id"), nullable=False)
    shop_id: Mapped[str] = mapped_column(String(10), ForeignKey("shops.shop_id"), nullable=False)
    category_id: Mapped[Optional[str]] = mapped_column(String(10), ForeignKey("categories.category_id"))
    region: Mapped[str] = mapped_column(String(100), nullable=False)
    date: Mapped[Optional[str]] = mapped_column(String(50))
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    value: Mapped[Optional[float]] = mapped_column(Float)
    value_upper: Mapped[Optional[float]] = mapped_column(Float)
    value_lower: Mapped[Optional[float]] = mapped_column(Float)
 
 
class Feedback(Base):
    """Tabla de feedback/issues de usuarios."""
    __tablename__ = "feedback"
 
    event_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    issue_type: Mapped[str] = mapped_column(String(200), nullable=False)
    severity: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(500))
    region: Mapped[str] = mapped_column(String(100), nullable=False)
    item_id: Mapped[Optional[str]] = mapped_column(String(10), ForeignKey("items.item_id"))
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    category_id: Mapped[Optional[str]] = mapped_column(String(10), ForeignKey("categories.category_id"))
 
 
class ModelEvaluation(Base):
    """Tabla de evaluación de modelos."""
    __tablename__ = "model_evaluation"
 
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    category_id: Mapped[str] = mapped_column(String(10), ForeignKey("categories.category_id"), nullable=False)
    model_id: Mapped[str] = mapped_column(String(200), nullable=False)
    mape: Mapped[float] = mapped_column(Float, nullable=False)
    mse: Mapped[float] = mapped_column(Float, nullable=False)
    mae: Mapped[float] = mapped_column(Float, nullable=False)
    bias: Mapped[Optional[float]] = mapped_column(Float)
    samples: Mapped[Optional[int]] = mapped_column(Integer)
    training_start_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    training_end_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    test_start_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    test_end_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
    last_retrain_date: Mapped[Optional[datetime]] = mapped_column(DateTime)
 
 
class SystemEvents(Base):
    """Tabla de eventos del sistema."""
    __tablename__ = "system_events"
 
    sys_event_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    datetime: Mapped[str] = mapped_column(String(200), nullable=False)
    model_id: Mapped[str] = mapped_column(String(100), nullable=False)
    event_type: Mapped[Optional[str]] = mapped_column(String(50))
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    service: Mapped[Optional[str]] = mapped_column(String(100))

# Configuración — S3 y Secrets Manager
# ---------------------------------------------------------------------------

# Bucket S3
S3_BUCKET = "1c-company-medallion"

S3_PATHS = {
    "categories": "gold/inference/categories_fe/categories.parquet",
    "items": "gold/inference/items_fe/items.parquet",
    "shops": "gold/inference/shops_fe/shops.parquet",
    "predictions": "gold/predictions/predictions.parquet",
    "feedback": "gold/feedback/feedback.parquet",
    "model_evaluation": "gold/model_evaluation/model_evaluation.parquet",
    "system_events": "gold/system/system_events.parquet",
}

# Secrets Manager
SECRET_ID = "1c-company/rds/1ccompany/credentials"
REGION = "us-east-1"
RDS_ENDPOINT = "<host>.us-east-1.rds.amazonaws.com"


# Extract — S3
# ---------------------------------------------------------------------------

def get_dataframe_from_s3(s3_client, bucket: str, key: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Descarga un archivo parquet desde S3 y retorna un DataFrame.
    """
    logger.info(f"Descargando {key} desde S3://{bucket}")
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        df = pd.read_parquet(obj['Body'])
        logger.info(f"Descargado exitosamente: {len(df):,} filas")
        return df
    except Exception:
        logger.exception(f"Error al descargar {key} desde S3.")
        sys.exit(1)


# Transform y Load
# ---------------------------------------------------------------------------

def prepare_table_from_s3(
    s3_client,
    bucket: str,
    s3_key: str,
    logger: logging.Logger,
) -> list[dict]:
    """
    Descarga desde S3 y convierte a lista de dicts para bulk insert.
    """
    df = get_dataframe_from_s3(s3_client, bucket, s3_key, logger)
    records = df.to_dict(orient='records')
    return records


# Orquestador principal
# ---------------------------------------------------------------------------

def main(
    secret_id: str,
    rds_endpoint: str,
    s3_bucket: str,
    region: str,
    logger: logging.Logger,
) -> None:
    """
    Orquesta el pipeline: S3 a RDS.
    """
    # 1. Credenciales RDS
    creds = get_secret(secret_id, region, logger)

    # 2. Conexión RDS
    engine = build_engine(creds, rds_endpoint, logger)

    # 3. Cliente S3
    s3_client = boto3.client('s3', region_name=region)

    # 4. DDL
    logger.info("Recreando schema.")
    try:
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)
        logger.info("Schema creado correctamente.")
    except Exception:
        logger.exception("Error al crear el schema en RDS.")
        sys.exit(1)

    # 5. Descargar y preparar datos desde S3
    try:
        categories_records = prepare_table_from_s3(s3_client, s3_bucket, S3_PATHS["categories"], logger)
        items_records = prepare_table_from_s3(s3_client, s3_bucket, S3_PATHS["items"], logger)
        shops_records = prepare_table_from_s3(s3_client, s3_bucket, S3_PATHS["shops"], logger)
        predictions_records = prepare_table_from_s3(s3_client, s3_bucket, S3_PATHS["predictions"], logger)
        feedback_records = prepare_table_from_s3(s3_client, s3_bucket, S3_PATHS["feedback"], logger)
        model_eval_records = prepare_table_from_s3(s3_client, s3_bucket, S3_PATHS["model_evaluation"], logger)
        system_events_records = prepare_table_from_s3(s3_client, s3_bucket, S3_PATHS["system_events"], logger)
    except Exception:
        logger.exception("Error preparando datos.")
        sys.exit(1)

    # 6. Insert respetando FK
    summary = []
    with Session(engine) as session:
        bulk_insert(session, Categories, categories_records, "categories", logger)
        bulk_insert(session, Shops, shops_records, "shops", logger)
        bulk_insert(session, Items, items_records, "items", logger)
        bulk_insert(session, Predictions, predictions_records, "predictions", logger)
        bulk_insert(session, Feedback, feedback_records, "feedback", logger)
        bulk_insert(session, ModelEvaluation, model_eval_records, "model_evaluation", logger)
        bulk_insert(session, SystemEvents, system_events_records, "system_events", logger)

        logger.info("Ejecutando commit.")
        try:
            session.commit()
        except Exception:
            logger.exception("Error en commit.")
            sys.exit(1)

        summary = [
            {"table": "categories", "rows": len(categories_records)},
            {"table": "shops", "rows": len(shops_records)},
            {"table": "items", "rows": len(items_records)},
            {"table": "predictions", "rows": len(predictions_records)},
            {"table": "feedback", "rows": len(feedback_records)},
            {"table": "model_evaluation", "rows": len(model_eval_records)},
            {"table": "system_events", "rows": len(system_events_records)},
        ]

    # Resumen
    logger.info("=" * 60)
    logger.info("Resumen— RDS pipeline finalizado")
    logger.info("=" * 60)
    for entry in summary:
        logger.info(f"  {entry['table']:<20} {entry['rows']:>10,} filas insertadas")
    logger.info("=" * 60)


# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _logger = setup_logging()
    _logger.info("Iniciando pipeline S3 a RDS.")

    main(
        secret_id=SECRET_ID,
        rds_endpoint=RDS_ENDPOINT,
        s3_bucket=S3_BUCKET,
        region=REGION,
        logger=_logger,
    )