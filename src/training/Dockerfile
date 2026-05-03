# Dockerfile
# Container de entrenamiento e inferencia para SageMaker (BYOC).
#
# Stack de serving: nginx → gunicorn → Flask
# Modos de ejecución:
#   - Entrenamiento: docker run <image> train
#   - Inferencia:    docker run <image> serve

FROM python:3.11-slim

# Dependencias del sistema: nginx para el servidor de inferencia
RUN apt-get -y update && apt-get install -y --no-install-recommends \
        nginx \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Dependencias Python
RUN pip --no-cache-dir install \
        polars==1.37.1 \
        scikit-learn==1.8.0 \
        pyarrow==23.0.0 \
        joblib==1.5.0 \
        numpy==2.4.1 \
        pandas==2.3.3 \
        xgboost==3.1.3 \
        category-encoders==2.9.0 \
        flask \
        gunicorn

# Variables de entorno
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Código de la aplicación
# SageMaker invoca 'train' o 'serve' según el tipo de job
COPY src/ /opt/program
WORKDIR /opt/program
