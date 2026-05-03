#Dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir \
    polars==1.37.1 \
    pyarrow==23.0.0 \
    scikit-learn==1.8.0 \
    numpy==2.4.1 \
    joblib==1.5.0 \
    xgboost==3.1.3 \
    category-encoders==2.9.0
ENV PYTHONPATH=/app/src
ENTRYPOINT ["python", "-m", "inference"]
CMD []