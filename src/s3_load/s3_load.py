"""
s3_load.py — S3 Layer: carga de archivos a SE vía boto3.

Este script forma parte del pipeline para 1C Company. Su responsabilidad 
es insertar los datos desde Jupyterlab en una estructura medallion en Amazon S3,
respetando el orden definido en la arquitectura:

  1. bronze  — catalogos, ventas y sample crudos
  2. silver  — catalogos y ventas preprocesadas
  3. gold   — inferencia, artifacts y predicciones

Uso:
    uv run s3_load.py
"""
# Dependencias 
# ----------------------------------------------------------------------------------
import boto3
import os
import logging

# Logging
# ----------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s')


# Conexión con el cliente de S3
s3 = boto3.client('s3')
bucket_name = '1c-company-medallion'


# Upload de datos a Bronze
# ----------------------------------------------------------------------------------

def load_bronze():
    # Catalogos
    s3.upload_file('../../data/bronze/categories.csv', bucket_name, 'bronze/categories/categories.csv')
    s3.upload_file('../../data/bronze/items.csv', bucket_name, 'bronze/items/items.csv')
    s3.upload_file('../../data/bronze/shops.csv', bucket_name, 'bronze/shops/shops.csv')

    #Ventas
    s3.upload_file('../../data/bronze/sales_historic.csv', bucket_name, 'bronze/sales/sales_historic.csv')
    
    # Samples
    s3.upload_file('../../data/bronze/sample_submission.csv', bucket_name, 'bronze/sample_submission/sample_submission.csv')
    s3.upload_file('../../data/bronze/test_keys.csv', bucket_name, 'bronze/test_keys/test_keys.csv')


# Upload de datos a Silver
# ----------------------------------------------------------------------------------

def load_silver():
    # Catalogos
    s3.upload_file('../../data/silver/items_catalog.parquet', bucket_name, 'silver/items_features/items_catalog.parquet')
    s3.upload_file('../../data/silver/categories_catalog.parquet', bucket_name, 'silver/categories_features/categories_catalog.parquet')
    s3.upload_file('../../data/silver/shops_catalog.parquet', bucket_name, 'silver/shops_features/shops_catalog.parquet')
    
    # Ventas
    s3.upload_file('../../data/silver/sales_preprocessed.parquet', bucket_name, 'silver/sales_prep/sales_preprocessed.parquet')
    s3.upload_file('../../data/silver/monthly_sales.parquet', bucket_name, 'silver/monthly_sales/monthly_sales.parquet')


# Upload de datos a Gold
# ----------------------------------------------------------------------------------

def load_gold():
    # Inference
    s3.upload_file('../../data/gold/items_catalog.parquet', bucket_name, 'gold/inference/items_fe/items_catalog.parquet')
    s3.upload_file('../../data/gold/categories_catalog.parquet', bucket_name, 'gold/inference/categories_fe/categories_catalog.parquet')
    s3.upload_file('../../data/gold/shops_catalog.parquet', bucket_name, 'gold/inference/shops_fe/shops_catalog.parquet')
    
    # Artifacts
    s3.upload_file('../../data/gold/xgb_monthly_forecast_model.joblib', bucket_name, 'gold/artifacts/xgb_monthly_forecast_model.joblib')
    
    # Predictions
    s3.upload_file('../../data/gold/predictions.csv', bucket_name, 'gold/predictions/predictions.csv')


def main():
    load_bronze()
    logging.info("Archivos insertados correctamente en S3 - bronze")
    load_silver()
    logging.info("Archivos insertados correctamente en S3 - silver")
    load_gold()
    logging.info("Archivos insertados correctamente en S3 - gold")

if __name__ == "__main__":
    logging.info("Comenzando proceso de upload a S3")
    main()
    logging.info("Proceso finalizado correctamente")