# Dependencias 
# ---------------------------------------------------------------------------
import json
import io
import boto3
import pandas as pd
from sqlalchemy import create_engine, text

# Parámetros 
# ---------------------------------------------------------------------------
SECRET_ID    = "x"
RDS_REPLICA  = "x"
REGION       = "us-east-1"

# Endpoint SageMaker y datos de entrada
SAGEMAKER_ENDPOINT = "demand-retail-endpoint"          # nombre del endpoint activo
TEST_KEYS_BUCKET   = "1c-company-medallion"
TEST_KEYS_KEY      = "bronze/test_keys/test_keys.csv"  # columnas: ID, shop_id, item_id

# Crear Engine
# ---------------------------------------------------------------------------
def build_engine():
    # Obtener credenciales desde SecretsManager
    client = boto3.client("secretsmanager", region_name=REGION)
    secret = client.get_secret_value(SecretId=SECRET_ID)
    creds  = json.loads(secret["SecretString"])

    # Engine SQLAlchemy apuntando a la Read Replica
    engine = create_engine(
        f"postgresql+psycopg2://{creds['username']}:{creds['password']}"
        f"@{RDS_REPLICA}:{creds['port']}/{creds['dbname']}",
        pool_pre_ping=True,
    )

    return engine

# Verificar conexión
# ---------------------------------------------------------------------------
engine = build_engine()
with engine.connect() as conn:
    conn.execute(text("SELECT 1"))

# Helpers internos — SageMaker
# ---------------------------------------------------------------------------
def _load_test_keys() -> pd.DataFrame:
    """
    Descarga el test set desde S3.
    Columnas esperadas: ID, shop_id, item_id
    """
    s3  = boto3.client("s3", region_name=REGION)
    obj = s3.get_object(Bucket=TEST_KEYS_BUCKET, Key=TEST_KEYS_KEY)
    return pd.read_csv(obj["Body"])
 
 
def _invoke_endpoint(test_keys: pd.DataFrame) -> pd.DataFrame:
    """
    Envía el test set al endpoint SageMaker en lotes de 500 filas.
    Devuelve un DataFrame con el schema completo de predicciones:
      id, item_id, shop_id, category_id, region, date, created_at,
      value, value_upper, value_lower
    """
    sm      = boto3.client("sagemaker-runtime", region_name=REGION)
    results = []
 
    for start in range(0, len(test_keys), 500):
        batch = test_keys.iloc[start:start + 500]
 
        buf = io.StringIO()
        batch.to_csv(buf, index=False)
 
        response = sm.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            ContentType="text/csv",
            Body=buf.getvalue().encode("utf-8"),
        )
        chunk = pd.read_csv(io.StringIO(response["Body"].read().decode("utf-8")))
        results.append(chunk)
 
    return pd.concat(results, ignore_index=True)

# Llamadas
# ---------------------------------------------------------------------------
def get_filter_options() -> dict:
    """
    Retorna las opciones únicas de filtro a partir de las predicciones
    obtenidas del endpoint. Compatible con el contrato original de la app.
    """
    df = _invoke_endpoint(_load_test_keys())
    return {
        "regions":    sorted(df["region"].dropna().unique().tolist()),
        "shops":      sorted(df["shop_id"].dropna().unique().tolist()),
        "categories": sorted(df["category_id"].dropna().unique().tolist()),
    }
 
 
def get_predictions(region_ids=None, shop_ids=None, category_ids=None) -> pd.DataFrame:
    """
    Llama al endpoint SageMaker con el test set completo y aplica
    los filtros opcionales de región, tienda y categoría en memoria.
 
    Columnas devueltas:
      id, item_id, shop_id, category_id, region, date, created_at,
      value, value_upper, value_lower
    """
    df = _invoke_endpoint(_load_test_keys())
 
    # Renombrar 'region' → 'region_id' para mantener compatibilidad con app.py
    df = df.rename(columns={"region": "region_id", "created_at": "run_date"})
 
    # Filtros opcionales
    if region_ids:
        df = df[df["region_id"].isin(region_ids)]
    if shop_ids:
        df = df[df["shop_id"].isin(shop_ids)]
    if category_ids:
        df = df[df["category_id"].isin(category_ids)]
 
    df["date"]     = pd.to_datetime(df["date"])
    df["run_date"] = pd.to_datetime(df["run_date"])
 
    return df.sort_values("date").reset_index(drop=True)

def get_model_evaluation():
    """
    Retorna métricas del modelo por categoría.
    """
    SQL = """
    SELECT
        category_id,
        mape, rmse, mae, bias, samples,
        model_id,
        training_start_date, training_end_date,
        test_start_date,     test_end_date,
        last_retrain_date
    FROM model_evaluation
    ORDER BY category_id;
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(SQL), conn)

    date_cols = ["training_start_date", "training_end_date",
                 "test_start_date", "test_end_date", "last_retrain_date"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    return df

def save_to_s3(data: bytes, filename: str, bucket: str) -> str:
    """Guarda el archivo en S3 y retorna la URL."""
    s3 = boto3.client("s3", region_name=REGION)
    s3.put_object(
        Bucket=bucket,
        Key=f"exports/{filename}",
        Body=data,
    )
    return f"s3://{bucket}/exports/{filename}"

def submit_feedback(item_id, issue_type, severity, region_id, category_id, description):
    """Inserta un reporte de feedback. Retorna True si tuvo éxito."""
    SQL = """
    INSERT INTO feedback
        (item_id, issue_type, severity,
         region_id, category_id, description, created_at)
    VALUES
        (:item_id, :issue_type, :severity,
         :region_id, :category_id, :description, NOW());
    """
    try:
        with engine.begin() as conn:
            conn.execute(text(SQL), {
                "item_id":     item_id,
                "issue_type":  issue_type,
                "severity":    severity,
                "region_id":   region_id,
                "category_id": category_id,
                "description": description,
            })
        return True
    except Exception as e:
        print(f"Error al insertar feedback: {e}")
        return False

def get_feedback_summary():
    """Conteo de issues por tipo y severidad para los KPIs de Tab 4."""
    SQL = """
    SELECT
        issue_type,
        severity_description,
        COUNT(*) AS total
    FROM feedback
    GROUP BY issue_type, severity_description
    ORDER BY total DESC;
    """
    with engine.connect() as conn:
        return pd.read_sql(text(SQL), conn)

def get_system_events(hours=24):
    """Eventos del sistema en las últimas N horas."""
    SQL = f"""
    SELECT
        event_id, datetime, model_id,
        event_type, status, duration_ms, message, service
    FROM system_events
    WHERE datetime >= NOW() - INTERVAL '{int(hours)} hours'
    ORDER BY datetime DESC;
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(SQL), conn)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

def get_api_health_metrics(hours=24):
    """
    Calcula los 3 KPIs para Tab 5:
    - availability
    - avg_latency_ms
    - error_count
    """
    SQL = f"""
    SELECT
        COUNT(*) AS total,
        SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS successes,
        AVG(duration_ms) AS avg_latency,
        SUM(CASE WHEN status = 'failed'  THEN 1 ELSE 0 END) AS errors
    FROM system_events
    WHERE datetime >= NOW() - INTERVAL '{int(hours)} hours';
    """
    with engine.connect() as conn:
        row = conn.execute(text(SQL)).fetchone()

    total = row.total or 1
    return {
        "availability":   round((row.successes / total) * 100, 1),
        "avg_latency_ms": round(row.avg_latency or 0, 1),
        "error_count":    int(row.errors or 0),
    }

def export_predictions(df, format="csv"):
    """Convierte el DataFrame filtrado a bytes descargables."""
    if format == "csv":
        return df.to_csv(index=False).encode("utf-8")
    elif format == "parquet":
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        return buffer.getvalue()
    raise ValueError(f"Formato no soportado: {format}")