"""
preprocess.py — SageMaker Processing Job (BYOC)

Flujo de datos gestionado por SageMaker:
    S3 (raw) → /opt/ml/processing/input/ → [este script] → /opt/ml/processing/output/ → S3 (procesado)

Inputs esperados en /opt/ml/processing/input/:
    - sales_train.csv
    - items_en.csv
    - item_categories_en.csv
    - shops_en.csv

Outputs generados en /opt/ml/processing/output/:
    - augmented/augmented_sales.parquet
    - cleaned/augmented_sales_cleaned.parquet
    - monthly/monthly_sales.parquet
    - inference/shops_catalog.parquet
    - inference/items_catalog.parquet
    - inference/items_categories_catalog.parquet
"""

from __future__ import print_function, unicode_literals

import argparse
import logging
import os
import time
import warnings

import polars as pl
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("preprocess")

# ── Rutas SageMaker ───────────────────────────────────────────────────────────
INPUT_DIR  = "/opt/ml/processing/input"
OUTPUT_DIR = "/opt/ml/processing/output"


# ── Argumentos ────────────────────────────────────────────────────────────────
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocessing and feature engineering pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sales-file",      type=str,   default="sales_train.csv")
    parser.add_argument("--contamination",   type=float, default=0.02)
    parser.add_argument("--random-state",    type=int,   default=42)
    return parser.parse_args()


# ── Carga ─────────────────────────────────────────────────────────────────────
def load_raw(input_dir: str, sales_file: str):
    sales_data             = pl.read_csv(os.path.join(input_dir, sales_file))
    items_catalog          = pl.read_csv(os.path.join(input_dir, "items_en.csv"))
    item_categories_catalog = pl.read_csv(os.path.join(input_dir, "item_categories_en.csv"))
    shops_catalog          = pl.read_csv(os.path.join(input_dir, "shops_en.csv"))
    return sales_data, items_catalog, item_categories_catalog, shops_catalog


# ── Casting ───────────────────────────────────────────────────────────────────
def cast_types(sales_data, items_catalog, item_categories_catalog, shops_catalog):
    sales_data = sales_data.with_columns(
        pl.col("date").str.to_date(format="%d.%m.%Y"),
        pl.col("date_block_num").cast(pl.String),
        pl.col("shop_id").cast(pl.String),
        pl.col("item_id").cast(pl.String),
        pl.col("item_price").cast(pl.Float32),
        pl.col("item_cnt_day").cast(pl.Float32),
    )
    items_catalog = items_catalog.with_columns(
        pl.col("item_id").cast(pl.String),
        pl.col("item_category_id").cast(pl.String),
    )
    item_categories_catalog = item_categories_catalog.with_columns(
        pl.col("item_category_id").cast(pl.String)
    )
    shops_catalog = shops_catalog.with_columns(pl.col("shop_id").cast(pl.String))
    return sales_data, items_catalog, item_categories_catalog, shops_catalog


# ── Features de tiendas ───────────────────────────────────────────────────────
def add_shop_features(sales_data: pl.DataFrame, shops_catalog: pl.DataFrame) -> pl.DataFrame:
    shops_catalog = shops_catalog.with_columns(
        pl.col("shop_name").str.extract(r"[!\s]*((?:St\.\s)?[A-Z][a-z]+)", 1).alias("region")
    )

    regions_corrected = {"Nizhny": "Nizhny Novgorod", "Sergiyev": "Sergiyev Posad", "Czechs": "Chekhov"}
    shops_catalog = shops_catalog.with_columns(
        pl.col("region").replace(regions_corrected).alias("region")
    ).with_columns(
        pl.when(pl.col("region").is_in(["Itinerant", "Shop", "Digital"]))
        .then(pl.lit("Other"))
        .otherwise(pl.col("region"))
        .alias("region")
    )

    moscow_cluster    = ["Moscow", "Mytishchi", "Khimki", "Balashikha", "Zhukovsky", "Kolomna", "Sergiyev Posad", "Chekhov"]
    central_volga     = ["Kazan", "Nizhny Novgorod", "Samara", "Ufa", "Yaroslavl", "Kaluga", "Voronezh", "Kursk", "Vologda"]
    south_caucasus    = ["Rostov", "Adygea", "Volzhsky"]
    siberia_far_east  = ["Yakutsk", "Krasnoyarsk", "Novosibirsk", "Omsk", "Tomsk", "Surgut", "Tyumen"]

    shops_catalog = shops_catalog.with_columns(
        pl.when(pl.col("region").is_in(moscow_cluster))
        .then(pl.lit("Moscow Area"))
        .when(pl.col("region") == "St. Petersburg")
        .then(pl.lit("St. Pete"))
        .when(pl.col("region").is_in(central_volga))
        .then(pl.lit("Volga & Central Russia"))
        .when(pl.col("region").is_in(south_caucasus))
        .then(pl.lit("Caucasus"))
        .when(pl.col("region").is_in(siberia_far_east))
        .then(pl.lit("Siberia"))
        .otherwise(pl.lit("Other"))
        .alias("macro_region")
    )

    shops_category = (
        sales_data.group_by("shop_id")
        .agg(pl.col("item_cnt_day").sum().alias("total_items"))
        .with_columns(
            pl.col("total_items")
            .qcut([0.25, 0.5, 0.75, 0.90], labels=["Low", "Medium-low", "Medium", "High", "Premium"])
            .alias("segment")
        )
        .select(["shop_id", "segment"])
    )

    shops_catalog = shops_catalog.join(shops_category.select("shop_id", "segment"), on="shop_id", how="left")
    return shops_catalog


# ── Features de categorías ────────────────────────────────────────────────────
def add_items_categories_features(item_categories_catalog: pl.DataFrame) -> pl.DataFrame:
    item_categories_catalog = item_categories_catalog.with_columns(
        pl.col("item_category_name").str.split_exact(" - ", 1).struct.field("field_0").alias("main_category")
    )

    categories = {
        "PC": "PC Headsets", "Игры": "Games", "Games Android": "Games", "Games MAC": "Games",
        "Payment card (Movies, Music, Games)": "Payment cards", "Payment card": "Payment cards",
        "Movie": "Movies", "Cinema": "Movies",
    }
    category_mapping = {
        "Games PC": "Games", "Games": "Games", "Game consoles": "Game Consoles",
        "Movies": "Media", "Music": "Media", "Program": "Programs", "Programs": "Programs",
        "System Tools": "Programs", "Gifts": "Gifts", "Books": "Books",
        "Accessories": "Components", "PC Headsets": "Components",
        "Net carriers (piece)": "Components", "Net carriers (spire)": "Components",
        "batteries": "Components", "Tickets (digits)": "Services",
        "Utilities": "Services", "Payment cards": "Services", "Delivery of goods": "Services",
    }

    item_categories_catalog = item_categories_catalog.with_columns(
        pl.col("main_category").replace(categories).alias("main_category")
    ).with_columns(pl.col("main_category").replace(category_mapping).alias("general_category"))

    item_categories_catalog = item_categories_catalog.with_columns(
        pl.col("item_category_name").str.split_exact(" - ", 1).struct.field("field_1").alias("device")
    )

    valid_devices = ["PS2", "PS3", "PS4", "PSP", "PSVita", "XBOX ONE", "XBOX 360"]
    item_categories_catalog = item_categories_catalog.with_columns(
        pl.when(pl.col("device").is_in(valid_devices)).then(pl.col("device")).otherwise(pl.lit("other")).alias("device")
    )
    return item_categories_catalog


# ── Features de items ─────────────────────────────────────────────────────────
def add_items_features(sales_data: pl.DataFrame, items_catalog: pl.DataFrame) -> pl.DataFrame:
    items_catalog = items_catalog.group_by("item_name", "item_category_id").agg(
        pl.col("item_id").min().alias("item_id")
    )

    item_age  = sales_data.group_by("item_id").agg(pl.col("date").min().alias("first_sale"))
    ref_date  = sales_data.select(pl.col("date").max()).item()

    item_age = item_age.with_columns(
        ((pl.lit(ref_date) - pl.col("first_sale")).dt.total_days() / 365.25).alias("age_years")
    ).with_columns(
        pl.when(pl.col("age_years") >= 1.5).then(pl.lit("1.5+ years"))
        .when(pl.col("age_years") >= 1.0).then(pl.lit("1–1.5 years"))
        .when(pl.col("age_years") >= 0.5).then(pl.lit("0.5–1 year"))
        .otherwise(pl.lit("< 0.5 year"))
        .alias("item_age")
    )

    items_catalog = items_catalog.join(item_age.select(["item_id", "item_age"]), on="item_id", how="left")
    items_catalog = items_catalog.with_columns(pl.col("item_age").fill_null(pl.lit("no_sales")))
    return items_catalog


# ── Features de fecha ─────────────────────────────────────────────────────────
def add_date_features(sales_data: pl.DataFrame) -> pl.DataFrame:
    return sales_data.with_columns([pl.col("date").dt.truncate("1mo").alias("year_month_date")])


# ── Deduplicación de items ────────────────────────────────────────────────────
def clean_duplicated_items(sales_data: pl.DataFrame) -> pl.DataFrame:
    dicc_items = {
        "945": "946", "8622": "8623", "8370": "8371", "9767": "9802",
        "21426": "21432", "11563": "12321", "11688": "11689", "14044": "14045",
        "16495": "16509", "16622": "16623", "19465": "19475", "19579": "19581",
        "14537": "14539", "15700": "15709", "15698": "15708",
    }
    return sales_data.with_columns(pl.col("item_id").replace(dicc_items).alias("item_id"))


# ── Outliers ──────────────────────────────────────────────────────────────────
def label_outliers(sales_data: pl.DataFrame, contamination: float = 0.02, random_state: int = 42) -> pl.DataFrame:
    def detect(group: pl.DataFrame) -> pl.DataFrame:
        if group.height < 10:
            return group.with_columns(pl.lit(1).alias("outlier_label"))
        cols = group.select(["item_price", "item_cnt_day"]).to_numpy()
        iso  = IsolationForest(contamination=contamination, random_state=random_state)
        labels = iso.fit_predict(cols)
        return group.with_columns(pl.Series("outlier_label", labels))

    groups = sales_data.partition_by(["general_category", "macro_region"], include_key=True)
    return pl.concat([detect(g) for g in groups])


# ── Agregación mensual ────────────────────────────────────────────────────────
def agg_monthly_data(sales_data: pl.DataFrame) -> pl.DataFrame:
    monthly_sales = (
        sales_data.select(
            "year_month_date", "shop_id", "segment", "macro_region",
            "item_id", "general_category", "device", "item_age",
            "item_price", "item_cnt_day",
        )
        .with_columns(pl.col("year_month_date").dt.month().alias("month"))
        .group_by(
            "year_month_date", "month", "shop_id", "segment", "macro_region",
            "item_id", "general_category", "device", "item_age",
        )
        .agg(
            pl.col("item_price").mean().round(2).alias("avg_price"),
            pl.col("item_cnt_day").sum().round(2).alias("monthly_items"),
        )
    )
    return monthly_sales.sort(by=["shop_id", "item_id", "year_month_date"], descending=False)


# ── Features temporales ───────────────────────────────────────────────────────
def temporal_features(monthly_sales_data: pl.DataFrame) -> pl.DataFrame:
    monthly_sales = monthly_sales_data.with_columns([
        pl.col("monthly_items").shift(1).over(["shop_id", "item_id"]).alias("monthly_items_lag_1"),
        pl.col("monthly_items").shift(12).over(["shop_id", "item_id"]).alias("monthly_items_lag_12"),
        pl.col("monthly_items").shift(1).rolling_mean(window_size=3).over(["shop_id", "item_id"]).alias("rolling_mean_3"),
        pl.col("monthly_items").shift(1).rolling_mean(window_size=6).over(["shop_id", "item_id"]).alias("rolling_mean_6"),
    ])

    monthly_sales = monthly_sales.with_columns([
        pl.col("monthly_items").shift(1).rolling_std(window_size=3).over(["shop_id", "item_id"]).alias("rolling_std_3"),
        pl.col("monthly_items").shift(1).rolling_std(window_size=6).over(["shop_id", "item_id"]).alias("rolling_std_6"),
        (pl.col("monthly_items_lag_1") - pl.col("monthly_items").shift(3).over(["shop_id", "item_id"])).alias("trend_3"),
    ])

    monthly_sales = monthly_sales.with_columns([
        pl.col("monthly_items_lag_1").fill_null(0),
        pl.col("monthly_items_lag_12").fill_null(0),
        pl.col("rolling_mean_3").fill_null(0),
        pl.col("rolling_mean_6").fill_null(0),
        pl.col("rolling_std_3").fill_null(0),
        pl.col("rolling_std_6").fill_null(0),
        pl.col("trend_3").fill_null(0),
    ])

    monthly_sales = monthly_sales.with_columns([
        pl.col("monthly_items").mean().over(["general_category", "macro_region", "device"]).alias("group_avg_region_category"),
        pl.col("monthly_items").mean().over(["general_category", "item_age"]).alias("group_avg_category_age"),
        pl.col("monthly_items").mean().over(["segment", "item_age"]).alias("group_avg_segment_age"),
    ])

    return monthly_sales


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_arguments()
    start_time = time.time()

    logger.info("=== SageMaker Processing Job — Sales Preprocessing ===")
    logger.info(f"Input dir  : {INPUT_DIR}")
    logger.info(f"Output dir : {OUTPUT_DIR}")
    logger.info(f"Sales file : {args.sales_file}")
    logger.info(f"Contamination: {args.contamination} | Random state: {args.random_state}")

    # Crear subdirectorios de salida
    for subdir in ["augmented", "cleaned", "monthly", "inference"]:
        os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)

    # 1. Carga + casting
    logger.info("Cargando datos...")
    sales_data, items_catalog, item_categories_catalog, shops_catalog = load_raw(INPUT_DIR, args.sales_file)
    sales_data, items_catalog, item_categories_catalog, shops_catalog = cast_types(
        sales_data, items_catalog, item_categories_catalog, shops_catalog
    )
    logger.info(f"  {len(sales_data):,} registros cargados")

    # 2. Feature engineering
    logger.info("Creando features...")
    shops_catalog           = add_shop_features(sales_data, shops_catalog)
    item_categories_catalog = add_items_categories_features(item_categories_catalog)
    items_catalog           = add_items_features(sales_data, items_catalog)
    sales_data              = add_date_features(sales_data)
    sales_data              = clean_duplicated_items(sales_data)

    augmented_data = (
        sales_data
        .join(items_catalog.select(["item_id", "item_category_id", "item_age"]), on="item_id", how="left")
        .join(item_categories_catalog.select(["item_category_id", "main_category", "general_category", "device"]), on="item_category_id", how="left")
        .join(shops_catalog.select(["shop_id", "region", "macro_region", "segment"]), on="shop_id", how="left")
    )
    logger.info(f"  augmented_data shape: {augmented_data.shape}")

    # 3. Outliers
    logger.info("Detectando outliers...")
    labeled_data = label_outliers(augmented_data, contamination=args.contamination, random_state=args.random_state)
    cleaned_data = labeled_data.filter(pl.col("outlier_label") == 1).drop("outlier_label")
    removed = len(labeled_data) - len(cleaned_data)
    logger.info(f"  {removed:,} registros eliminados como outliers")

    # 4. Agregación mensual + features temporales
    logger.info("Agregando datos mensualmente...")
    monthly_data = agg_monthly_data(cleaned_data)
    monthly_data = temporal_features(monthly_data)
    monthly_data = monthly_data.filter(pl.col("monthly_items") >= 0)
    logger.info(f"  {len(monthly_data):,} registros mensuales generados")

    # 5. Guardar outputs
    logger.info("Exportando outputs...")

    augmented_data.write_parquet(os.path.join(OUTPUT_DIR, "augmented", "augmented_sales.parquet"))
    logger.info("  ✓ augmented/augmented_sales.parquet")

    cleaned_data.write_parquet(os.path.join(OUTPUT_DIR, "cleaned", "augmented_sales_cleaned.parquet"))
    logger.info("  ✓ cleaned/augmented_sales_cleaned.parquet")

    monthly_data.write_parquet(os.path.join(OUTPUT_DIR, "monthly", "monthly_sales.parquet"))
    logger.info("  ✓ monthly/monthly_sales.parquet")

    shops_catalog.write_parquet(os.path.join(OUTPUT_DIR, "inference", "shops_catalog.parquet"))
    item_categories_catalog.write_parquet(os.path.join(OUTPUT_DIR, "inference", "items_categories_catalog.parquet"))
    items_catalog.write_parquet(os.path.join(OUTPUT_DIR, "inference", "items_catalog.parquet"))
    logger.info("  ✓ inference/shops_catalog.parquet")
    logger.info("  ✓ inference/items_categories_catalog.parquet")
    logger.info("  ✓ inference/items_catalog.parquet")

    elapsed = time.time() - start_time
    logger.info(f"=== Preprocessing completado en {elapsed:.1f}s ===")


if __name__ == "__main__":
    main()
