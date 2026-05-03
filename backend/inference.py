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


# Llamadas
# ---------------------------------------------------------------------------
def get_filter_options():
    """Retorna las opciones únicas para los filtros de Tab 1"""
    SQL = """
    SELECT DISTINCT 
        region_id AS regions,
        shop_id AS shops,
        category_id AS categories
    FROM predictions;
    """
    with engine.connect() as conn:
        row = conn.execute(text(SQL)).fetchone()
    return {
        "regions": row.regions,
        "shops": row.shops,
        "categories": row.categories
    }



def get_predictions(region_ids=None, shop_ids=None, category_ids=None):
    """
    Retorna predicciones filtrando opcionalmente por región, tienda y categoría.
    Columnas: id, item_id, shop_id, category_id, region_id, date, created_at
    """
    filters = {}
    params = {}

    if region_ids:
        filters.append("region_id = ANY(:regions)")
        params["regions"] = region_ids

    if shop_ids:
        filters.append("shop_id = ANY(:shops)")
        params["shops"] = shop_ids
    
    if category_ids:
        filters.append("category_id = ANY(:category)")
        params["categories"] = category_ids
    
    where = ( "WHERE " + "AND ".join(filters)) if filters else ""

    SQL = f"""
    SELECT id, item_id, shop_id, category_id, region_id,
        date, run_date, value, value_lower, value_upper
    FROM predictions
    {where}
    ORDER BY date;
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(SQL), conn)
    
    df["date"] = pd.to_datetime(df["date"])
    df["run_date"] = pd.to_datetime(df["run_date"])
    return df

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