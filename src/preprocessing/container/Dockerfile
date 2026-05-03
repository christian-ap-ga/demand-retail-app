# Dockerfile — SageMaker Processing BYOC
 
FROM python:3.11-slim
 
WORKDIR /opt/ml/processing
 
RUN pip install --no-cache-dir \
    numpy==2.2.3 \
    polars==1.27.1 \
    scikit-learn==1.6.1 \
    pyarrow==19.0.1
