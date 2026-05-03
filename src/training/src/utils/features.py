"""
Definición centralizada de columnas de features para entrenamiento e inferencia.

Este módulo actúa como *única fuente verdadera* para las variables utilizadas
en el pipeline de modelado. Permite garantizar que los conjuntos de columnas
sean consistentes entre las etapas de:

- Feature engineering
- Entrenamiento
- Inferencia

Contiene:
- HIGH_CARD_COLS : variables categóricas de alta cardinalidad.
- CAT_COLS       : variables categóricas de baja/media cardinalidad.
- NUM_COLS       : variables numéricas y temporales.
- KEY_COLS       : llaves primarias para joins (shop-item).
"""

HIGH_CARD_COLS = ["item_id", "shop_id"]

CAT_COLS = [
    "month",
    "segment",
    "macro_region",
    "general_category",
    "device",
    "item_age",
]

NUM_COLS = [
    "avg_price",
    "monthly_items_lag_1",
    "monthly_items_lag_12",
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_std_3",
    "rolling_std_6",
    "trend_3",
    "group_avg_region_category",
    "group_avg_category_age",
    "group_avg_segment_age",
]

KEY_COLS = ["shop_id", "item_id"]
