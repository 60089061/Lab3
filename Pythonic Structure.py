# ==============================
# GOLD LAYER PIPELINE - Pythonic
# ==============================

from __future__ import annotations
from typing import Optional
from pathlib import PurePosixPath
import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, trim, length
from pyspark.sql.types import IntegerType

# -----------------------------
# Configuration & Constants
# -----------------------------
STORAGE_ACCOUNT = "60089061lab3"
CONTAINER = "lakehouse"
BASE_PATH = f"abfss://{CONTAINER}@{STORAGE_ACCOUNT}.dfs.core.windows.net"

# Paths
PROCESSED_PATH = PurePosixPath(BASE_PATH) / "processed"
GOLD_PATH = PurePosixPath(BASE_PATH) / "gold" / "curated_reviews"

# Unity Catalog
EXTERNAL_LOCATION_NAME = "curated_gold_reviews_loc"
STORAGE_CREDENTIAL_NAME = "60089061lab3"  # Must exist in Unity Catalog
TABLE_NAME = "curated_reviews"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------
# Helper Functions
# -----------------------------
def get_spark() -> SparkSession:
    """Get or create Spark session."""
    return SparkSession.builder.getOrCreate()


def read_parquet(spark: SparkSession, path: str) -> DataFrame:
    """Safely read Parquet with logging."""
    logger.info(f"Reading Parquet from: {path}")
    return spark.read.parquet(path)


def write_delta(df: DataFrame, path: str, mode: str = "overwrite") -> None:
    """Write DataFrame as Delta with overwrite schema."""
    logger.info(f"Writing Delta table to: {path} (mode={mode})")
    df.write \
        .format("delta") \
        .mode(mode) \
        .option("overwriteSchema", "true") \
        .save(path)


def create_external_location(spark: SparkSession, url: str, cred_name: str, loc_name: str) -> None:
    """Create external location in Unity Catalog if not exists."""
    sql = f"""
    CREATE EXTERNAL LOCATION IF NOT EXISTS `{loc_name}`
    URL '{url}'
    WITH (STORAGE CREDENTIAL `{cred_name}`)
    """
    logger.info(f"Creating external location: {loc_name}")
    try:
        spark.sql(sql)
        logger.info("External location created successfully.")
    except Exception as e:
        if "NO_SUCH_STORAGE_CREDENTIAL" in str(e):
            logger.error(f"Storage credential '{cred_name}' does not exist. Create it in Unity Catalog first.")
        else:
            logger.error(f"Failed to create external location: {e}")
        raise


def register_delta_table(spark: SparkSession, path: str, table_name: str) -> None:
    """Register Delta table in Unity Catalog."""
    sql = f"""
    CREATE TABLE IF NOT EXISTS `{table_name}`
    USING DELTA
    LOCATION '{path}'
    """
    logger.info(f"Registering Delta table: {table_name}")
    spark.sql(sql)
    logger.info(f"Table `{table_name}` registered at {path}")


# -----------------------------
# Data Loading
# -----------------------------
def load_silver_data(spark: SparkSession) -> dict[str, DataFrame]:
    """Load all silver-layer datasets."""
    paths = {
        "books": str(PROCESSED_PATH / "books"),
        "authors": str(PROCESSED_PATH / "authors"),
        "reviews": str(PROCESSED_PATH / "reviews")
    }

    data = {}
    for name, path in paths.items():
        data[name] = read_parquet(spark, path)
        logger.info(f"{name.capitalize()} loaded: {data[name].count()} rows")

    return data


# -----------------------------
# Data Cleaning
# -----------------------------
def clean_reviews(df: DataFrame) -> DataFrame:
    """Clean and validate reviews data."""
    logger.info("Starting reviews cleaning...")

    cleaned = (
        df
        # 1. Drop rows missing critical keys
        .filter(
            col("review_id").isNotNull() &
            col("book_id").isNotNull() &
            col("user_id").isNotNull()
        )
        # 2. Cast and validate rating
        .withColumn("rating_int", col("rating").cast(IntegerType()))
        .filter(
            col("rating_int").isNotNull() &
            (col("rating_int").between(1, 5))
        )
        # 3. Clean and filter review text
        .withColumn("review_text", trim(col("review_text")))
        .filter(
            col("review_text").isNotNull() &
            (length(col("review_text")) >= 10)
        )
        # 4. Deduplicate
        .dropDuplicates(["review_id"])
        # 5. Final selection
        .select(
            "review_id",
            "book_id",
            "user_id",
            col("rating_int").alias("rating"),
            "review_text"
        )
    )

    logger.info(f"Reviews cleaning complete: {cleaned.count()} valid rows")
    return cleaned


# -----------------------------
# Gold Table Construction
# -----------------------------
def build_gold_table(
    books: DataFrame,
    authors: DataFrame,
    reviews: DataFrame
) -> DataFrame:
    """Join books, authors, and cleaned reviews into gold table."""
    logger.info("Building gold table...")

    gold = (
        books
        .select("book_id", "title", "author_id", col("language_code").alias("language"))
        .join(
            authors.select("author_id", col("name").alias("author_name")),
            on="author_id",
            how="left"
        )
        .join(reviews, on="book_id", how="left")
        .select(
            "review_id",
            "book_id",
            "title",
            "author_id",
            "author_name",
            "user_id",
            "rating",
            "review_text",
            "language"
        )
    )

    logger.info(f"Gold table built: {gold.count()} rows")
    return gold


# -----------------------------
# Main Pipeline
# -----------------------------
def main() -> None:
    """Orchestrate the full gold layer pipeline."""
    spark = get_spark()

    # 1. Load data
    data = load_silver_data(spark)
    reviews_clean = clean_reviews(data["reviews"])

    # 2. Build gold table
    gold_table = build_gold_table(
        books=data["books"],
        authors=data["authors"],
        reviews=reviews_clean
    )

    # 3. Write to gold layer
    write_delta(gold_table, str(GOLD_PATH))

    # 4. Register in Unity Catalog
    try:
        create_external_location(
            spark=spark,
            url=str(PurePosixPath(BASE_PATH) / "gold"),
            cred_name=STORAGE_CREDENTIAL_NAME,
            loc_name=EXTERNAL_LOCATION_NAME
        )
        register_delta_table(spark, str(GOLD_PATH), TABLE_NAME)
    except Exception as e:
        logger.warning(f"Unity Catalog registration failed: {e}")
        logger.info("Data still written to storage. Register manually if needed.")

    # 5. Verification
    logger.info("Verification: Sample from gold table")
    gold_table.show(5, truncate=False)

    spark.sql(f"SELECT * FROM `{TABLE_NAME}` LIMIT 5").show(truncate=False)

    logger.info("Gold pipeline completed successfully!")


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    main()
