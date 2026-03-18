"""
╔══════════════════════════════════════════════════════════════════╗
║          Movie Rating Analysis System — PySpark Pipeline         ║
║          Designed for Google Colab | MovieLens 32M Dataset       ║
╚══════════════════════════════════════════════════════════════════╝

DATASET:  MovieLens 32M  (ml-32m)
FORMAT:   Standard CSV files with header row, comma-separated
FILES:    movies.csv   — movieId, title, genres
          ratings.csv  — userId, movieId, rating, timestamp
          tags.csv     — userId, movieId, tag, timestamp
          links.csv    — movieId, imdbId, tmdbId

NOTE: The 32M dataset does NOT include user demographic data
      (no users.csv). All user-level analytics use behavioural
      signals derived from rating patterns alone.

NOTEBOOK CELL BREAKDOWN:
  Cell 1  → Environment Setup
  Cell 2  → Upload Dataset from Local Machine → Colab
  Cell 3  → Initialize SparkSession (32M-tuned config)
  Cell 4  → Data Ingestion (CSV → Spark DataFrames)
  Cell 5  → Data Quality Checks & Cleaning
  Cell 6  → Core Analytics
  Cell 7  → Advanced Analytics
  Cell 8  → Parquet Write (performance optimisation)
  Cell 9  → Visualisations (inline Plotly charts)
  Cell 10 → Streamlit Dashboard
"""


# ─────────────────────────────────────────────────────────────────
# CELL 1 — Environment Setup
# ─────────────────────────────────────────────────────────────────
import os
import subprocess

def install_dependencies():
    """Install all required packages silently."""
    packages = ["pyspark==3.5.1", "streamlit", "plotly", "pyngrok"]
    for pkg in packages:
        result = subprocess.run(
            ["pip", "install", pkg, "--quiet"],
            capture_output=True,
        )
        if result.returncode != 0:
            print(f"⚠️  Warning: could not install {pkg}")
    print("✅ All dependencies installed.")

install_dependencies()


# ─────────────────────────────────────────────────────────────────
# CELL 2 — Upload Dataset from Local Machine → Colab
# ─────────────────────────────────────────────────────────────────
"""
You already have the ml-32m folder downloaded locally.
This cell provides two options to bring it into Colab.

─────────────────────────────────────────────────────────────────
OPTION A — Upload the ZIP directly (default, ~820 MB)
─────────────────────────────────────────────────────────────────
  1. Zip your ml-32m folder:
       Windows: right-click → Send to → Compressed (zipped) folder
       macOS  : right-click → Compress "ml-32m"
       This produces ml-32m.zip (~820 MB).

  2. Run this cell. A file picker appears. Select ml-32m.zip.
     Upload typically takes 2–10 min depending on your connection.

─────────────────────────────────────────────────────────────────
OPTION B — Mount Google Drive (faster if you have Drive space)
─────────────────────────────────────────────────────────────────
  1. Upload ml-32m.zip to your Google Drive manually.
  2. Comment out the OPTION A block below and uncomment OPTION B.
─────────────────────────────────────────────────────────────────
"""
import zipfile
import shutil
from pathlib import Path

DATA_DIR    = Path("/content/movielens")
RAW_DIR     = DATA_DIR / "raw"
PARQUET_DIR = DATA_DIR / "parquet"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

# ── OPTION A: Upload ZIP from your computer ──────────────────────
from google.colab import files as colab_files

def upload_and_extract():
    """Upload ml-32m.zip from local machine, extract to RAW_DIR."""
    if (RAW_DIR / "ratings.csv").exists():
        print("📦 Dataset already present at", RAW_DIR)
        return

    print("📂 Select your ml-32m.zip file in the upload dialog…")
    uploaded = colab_files.upload()   # opens browser file picker

    for filename, data in uploaded.items():
        zip_path = DATA_DIR / filename
        zip_path.write_bytes(data)
        print(f"✅ Uploaded: {filename}  ({len(data) / 1e6:.1f} MB)")

        print("📂 Extracting…")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)

        # The zip extracts to ml-32m/; flatten CSVs into raw/
        extracted = DATA_DIR / "ml-32m"
        if extracted.exists():
            for f in extracted.iterdir():
                shutil.move(str(f), RAW_DIR / f.name)
            extracted.rmdir()

        zip_path.unlink()
        print("✅ Dataset ready at", RAW_DIR)
        break  # only handle the first uploaded file

upload_and_extract()

# ── OPTION B: Mount Google Drive ─────────────────────────────────
# from google.colab import drive
# drive.mount("/content/drive")
# ZIP_ON_DRIVE = "/content/drive/MyDrive/ml-32m.zip"   # ← update path
# with zipfile.ZipFile(ZIP_ON_DRIVE, "r") as zf:
#     zf.extractall(DATA_DIR)
# extracted = DATA_DIR / "ml-32m"
# if extracted.exists():
#     for f in extracted.iterdir():
#         shutil.move(str(f), RAW_DIR / f.name)
#     extracted.rmdir()
# print("✅ Dataset extracted from Drive.")

# ── Verify required files are present ────────────────────────────
REQUIRED = ["movies.csv", "ratings.csv", "tags.csv"]
missing  = [f for f in REQUIRED if not (RAW_DIR / f).exists()]
if missing:
    raise FileNotFoundError(
        f"Missing files in {RAW_DIR}: {missing}\n"
        "Please re-run this cell and upload ml-32m.zip."
    )

print("\n📁 Dataset contents:")
for f in sorted(RAW_DIR.iterdir()):
    print(f"   {f.name:<20}  {f.stat().st_size / 1e6:>8.1f} MB")


# ─────────────────────────────────────────────────────────────────
# CELL 3 — Initialize SparkSession (32M-tuned)
# ─────────────────────────────────────────────────────────────────
"""
Key differences vs a 1M config:
  - Driver memory increased (4g → 6g) to hold larger aggregation results
  - Shuffle partitions raised (8 → 16) to avoid OOM on wide shuffles
  - autoBroadcastJoinThreshold raised to 50 MB so movies.csv and
    tags aggregates are auto-broadcast, avoiding expensive sort-merge joins
  - Skew join handling enabled for the ratings table (heavily skewed
    toward popular movies)
  - Kryo serialisation is faster than Java default for large shuffles
"""
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

from pyspark.sql import SparkSession
from pyspark import SparkConf

def create_spark_session() -> SparkSession:
    conf = (
        SparkConf()
        .setAppName("MovieRatingAnalysis_32M")
        .setMaster("local[*]")
        .set("spark.driver.memory", "6g")
        .set("spark.executor.memory", "6g")
        .set("spark.sql.shuffle.partitions", "16")
        .set("spark.sql.adaptive.enabled", "true")
        .set("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .set("spark.sql.adaptive.skewJoin.enabled", "true")
        .set("spark.sql.autoBroadcastJoinThreshold", str(50 * 1024 * 1024))  # 50 MB
        .set("spark.memory.offHeap.enabled", "true")
        .set("spark.memory.offHeap.size", "2g")
        .set("spark.sql.parquet.compression.codec", "snappy")
        .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .set("spark.ui.showConsoleProgress", "false")
    )

    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    print(f"✅ SparkSession active | Version: {spark.version}")
    print(f"   Cores available    : {spark.sparkContext.defaultParallelism}")
    print(f"   Shuffle partitions : 16  (tuned for 32M on Colab)")
    return spark

spark = create_spark_session()


# ─────────────────────────────────────────────────────────────────
# CELL 4 — Data Ingestion: CSV → Spark DataFrames
# ─────────────────────────────────────────────────────────────────
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    IntegerType, LongType, FloatType, StringType
)

# ── Explicit schemas ─────────────────────────────────────────────
# Providing schemas upfront avoids the "schema inference scan" that
# Spark would otherwise run (an extra full pass over every file).

RATINGS_SCHEMA = StructType([
    StructField("user_id",   IntegerType(), False),
    StructField("movie_id",  IntegerType(), False),
    StructField("rating",    FloatType(),   False),
    StructField("timestamp", LongType(),    False),
])

MOVIES_SCHEMA = StructType([
    StructField("movie_id", IntegerType(), False),
    StructField("title",    StringType(),  False),
    StructField("genres",   StringType(),  False),   # pipe-separated e.g. "Action|Drama"
])

TAGS_SCHEMA = StructType([
    StructField("user_id",   IntegerType(), False),
    StructField("movie_id",  IntegerType(), False),
    StructField("tag",       StringType(),  False),
    StructField("timestamp", LongType(),    False),
])

LINKS_SCHEMA = StructType([
    StructField("movie_id", IntegerType(), True),
    StructField("imdb_id",  StringType(),  True),
    StructField("tmdb_id",  IntegerType(), True),
])

def load_csv(filename: str, schema: StructType) -> DataFrame:
    """
    Load a ml-32m CSV into a Spark DataFrame.
    All ml-32m files are:  comma-separated, UTF-8, with a header row.
    """
    return (
        spark.read
        .option("header", "true")
        .option("encoding", "UTF-8")
        .option("nullValue", "")
        .csv(str(RAW_DIR / filename), schema=schema)
    )

print("📥 Loading CSV files into Spark DataFrames…")
df_ratings_raw = load_csv("ratings.csv", RATINGS_SCHEMA)
df_movies_raw  = load_csv("movies.csv",  MOVIES_SCHEMA)
df_tags_raw    = load_csv("tags.csv",    TAGS_SCHEMA)
df_links_raw   = load_csv("links.csv",   LINKS_SCHEMA)

# Cache raw frames — they are referenced by multiple transformations
df_ratings_raw.cache()
df_movies_raw.cache()
df_tags_raw.cache()
df_links_raw.cache()

print(f"\n   ratings.csv : {df_ratings_raw.count():>12,} rows   ← the 32M backbone")
print(f"   movies.csv  : {df_movies_raw.count():>12,} rows")
print(f"   tags.csv    : {df_tags_raw.count():>12,} rows")
print(f"   links.csv   : {df_links_raw.count():>12,} rows")


# ─────────────────────────────────────────────────────────────────
# CELL 5 — Data Quality Checks & Cleaning
# ─────────────────────────────────────────────────────────────────

def quality_report(df: DataFrame, name: str) -> None:
    """Print null counts, duplicate row count, and row total."""
    total = df.count()
    print(f"\n{'─'*55}")
    print(f"  Quality Report → {name}")
    print(f"{'─'*55}")
    print(f"  Rows       : {total:,}")

    null_counts = df.select(
        [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]
    ).collect()[0].asDict()
    non_zero = {k: v for k, v in null_counts.items() if v > 0}
    print(f"  Nulls      : {non_zero if non_zero else 'None ✅'}")

    dupes = total - df.dropDuplicates().count()
    print(f"  Duplicates : {dupes:,}")

quality_report(df_ratings_raw, "Ratings")
quality_report(df_movies_raw,  "Movies")
quality_report(df_tags_raw,    "Tags")

# ── Clean Ratings ────────────────────────────────────────────────
df_ratings = (
    df_ratings_raw
    .dropna(subset=["user_id", "movie_id", "rating"])
    .filter(F.col("rating").between(0.5, 5.0))
    .withColumn("rated_at", F.to_timestamp(F.from_unixtime("timestamp")))
    .withColumn("year",  F.year("rated_at"))
    .withColumn("month", F.month("rated_at"))
    .withColumn("hour",  F.hour("rated_at"))
    .drop("timestamp")
)

# ── Clean Movies ─────────────────────────────────────────────────
df_movies = (
    df_movies_raw
    .dropna(subset=["movie_id", "title"])
    # Extract 4-digit release year from "Title (YYYY)" pattern
    .withColumn(
        "release_year",
        F.regexp_extract("title", r"\((\d{4})\)\s*$", 1).cast(IntegerType())
    )
    # Strip "(YYYY)" suffix for display
    .withColumn(
        "clean_title",
        F.trim(F.regexp_replace("title", r"\s*\(\d{4}\)\s*$", ""))
    )
    # Convert pipe-separated genres string → array for explode analytics
    .withColumn("genre_array", F.split("genres", r"\|"))
)

# ── Clean Tags ───────────────────────────────────────────────────
df_tags = (
    df_tags_raw
    .dropna(subset=["user_id", "movie_id", "tag"])
    .filter(F.length(F.trim("tag")) > 0)
    .withColumn("tag_lower", F.lower(F.trim("tag")))   # normalise case
    .withColumn("tagged_at", F.to_timestamp(F.from_unixtime("timestamp")))
    .drop("timestamp")
)

# Cache cleaned frames
df_ratings.cache()
df_movies.cache()
df_tags.cache()

print("\n✅ All DataFrames cleaned and enriched.")
df_ratings.printSchema()
df_movies.printSchema()


# ─────────────────────────────────────────────────────────────────
# CELL 6 — Core Analytics
# ─────────────────────────────────────────────────────────────────
from pyspark.sql.window import Window

# ── 6.1  Per-movie statistics ────────────────────────────────────
# Threshold raised to 100 vs. 50 in 1M — with 32M ratings there is
# far more coverage so we can demand more evidence before ranking.
movie_stats = (
    df_ratings
    .groupBy("movie_id")
    .agg(
        F.count("rating").alias("total_ratings"),
        F.avg("rating").alias("avg_rating"),
        F.stddev("rating").alias("rating_stddev"),
        F.min("rating").alias("min_rating"),
        F.max("rating").alias("max_rating"),
        F.percentile_approx("rating", 0.5).alias("median_rating"),
    )
    .filter(F.col("total_ratings") >= 100)
    .withColumn("avg_rating",    F.round("avg_rating",    3))
    .withColumn("rating_stddev", F.round("rating_stddev", 3))
    .join(
        df_movies.select("movie_id", "clean_title", "genres",
                         "genre_array", "release_year"),
        "movie_id"
    )
    .orderBy(F.desc("avg_rating"), F.desc("total_ratings"))
)
movie_stats.cache()

print("🎬 Top 10 Highest-Rated Movies (≥100 ratings):")
movie_stats.select("clean_title", "avg_rating", "median_rating",
                   "total_ratings", "release_year") \
           .show(10, truncate=False)

# ── 6.2  Most active users ────────────────────────────────────────
# ml-32m has no demographics, so all signals are purely behavioural
user_activity = (
    df_ratings
    .groupBy("user_id")
    .agg(
        F.count("rating").alias("total_ratings"),
        F.avg("rating").alias("avg_rating_given"),
        F.stddev("rating").alias("rating_stddev"),
        F.countDistinct("movie_id").alias("unique_movies_rated"),
        F.min("rated_at").alias("first_rating_at"),
        F.max("rated_at").alias("last_rating_at"),
    )
    .withColumn("avg_rating_given", F.round("avg_rating_given", 3))
    .withColumn(
        "active_days",
        F.datediff("last_rating_at", "first_rating_at").cast(IntegerType())
    )
    .orderBy(F.desc("total_ratings"))
)
user_activity.cache()

print("\n👤 Top 10 Most Active Users:")
user_activity.select("user_id", "total_ratings", "unique_movies_rated",
                     "avg_rating_given", "active_days") \
             .show(10)

# ── 6.3  Genre popularity (exploded genres) ───────────────────────
genre_stats = (
    df_movies
    .select("movie_id", F.explode("genre_array").alias("genre"))
    .join(df_ratings.select("movie_id", "rating"), "movie_id")
    .filter(F.col("genre") != "(no genres listed)")
    .groupBy("genre")
    .agg(
        F.count("rating").alias("total_ratings"),
        F.avg("rating").alias("avg_rating"),
        F.countDistinct("movie_id").alias("num_movies"),
        F.stddev("rating").alias("rating_stddev"),
    )
    .withColumn("avg_rating",    F.round("avg_rating",    3))
    .withColumn("rating_stddev", F.round("rating_stddev", 3))
    .orderBy(F.desc("total_ratings"))
)

print("\n🎭 Genre Popularity:")
genre_stats.show(25, truncate=False)

# ── 6.4  Rating distribution ──────────────────────────────────────
rating_distribution = (
    df_ratings
    .groupBy("rating")
    .count()
    .orderBy("rating")
)
print("\n⭐ Rating Distribution:")
rating_distribution.show()

# ── 6.5  Yearly rating trends ─────────────────────────────────────
# ml-32m spans approximately 1995–2023
yearly_trends = (
    df_ratings
    .filter(F.col("year").between(1995, 2023))
    .groupBy("year")
    .agg(
        F.count("rating").alias("total_ratings"),
        F.avg("rating").alias("avg_rating"),
        F.countDistinct("user_id").alias("active_users"),
        F.countDistinct("movie_id").alias("movies_rated"),
    )
    .withColumn("avg_rating", F.round("avg_rating", 3))
    .orderBy("year")
)
print("\n📅 Yearly Trends:")
yearly_trends.show(30)


# ─────────────────────────────────────────────────────────────────
# CELL 7 — Advanced Analytics
# ─────────────────────────────────────────────────────────────────

# ── 7.1  Bayesian Average Rating ─────────────────────────────────
# Formula: (C × m + Σratings) / (C + n)
#   C = global mean rating across all movies
#   m = minimum votes threshold  (raised to 500 for 32M)
#   n = number of votes for this specific movie
# This is the same formula IMDb uses for its Top 250.
# A film with 1 rating of 5.0 will be ranked far below a film with
# 10,000 ratings averaging 4.4 — which is the correct behaviour.
C         = df_ratings.agg(F.avg("rating")).collect()[0][0]
MIN_VOTES = 500

movie_bayesian = (
    movie_stats
    .withColumn(
        "bayesian_avg",
        F.round(
            (F.lit(C) * F.lit(MIN_VOTES) + F.col("avg_rating") * F.col("total_ratings"))
            / (F.lit(MIN_VOTES) + F.col("total_ratings")),
            4
        )
    )
    .orderBy(F.desc("bayesian_avg"))
)

print(f"Global mean C = {C:.3f}")
print("\n🏆 Top 15 by Bayesian Average:")
movie_bayesian.select("clean_title", "bayesian_avg", "avg_rating",
                      "total_ratings", "release_year") \
              .show(15, truncate=False)

# ── 7.2  Genre trends over time ───────────────────────────────────
genre_year_trends = (
    df_movies
    .select("movie_id", F.explode("genre_array").alias("genre"))
    .join(df_ratings.select("movie_id", "rating", "year"), "movie_id")
    .filter(
        (F.col("genre") != "(no genres listed)") &
        F.col("year").between(1995, 2023)
    )
    .groupBy("genre", "year")
    .agg(
        F.count("rating").alias("rating_count"),
        F.avg("rating").alias("avg_rating"),
    )
    .withColumn("avg_rating", F.round("avg_rating", 3))
    .orderBy("genre", "year")
)
print("\n📈 Genre Year Trend (sample):")
genre_year_trends.show(20)

# ── 7.3  User segmentation ───────────────────────────────────────
user_segments = (
    user_activity
    .withColumn(
        "segment",
        F.when(F.col("avg_rating_given") >= 4.0, "Generous (≥4.0)")
         .when(F.col("avg_rating_given") >= 3.0, "Moderate (3–4)")
         .otherwise("Harsh (<3.0)")
    )
    .groupBy("segment")
    .agg(
        F.count("user_id").alias("user_count"),
        F.round(F.avg("total_ratings"), 1).alias("avg_ratings_per_user"),
        F.round(F.avg("unique_movies_rated"), 1).alias("avg_unique_movies"),
    )
    .orderBy(F.desc("user_count"))
)
print("\n🔖 User Segments:")
user_segments.show()

# ── 7.4  Tag analytics (unique advantage of ml-32m over ml-1m) ───
# The 1M dataset had no tags file — this is exclusive to 32M.

# Top tags across the whole catalogue
top_tags = (
    df_tags
    .groupBy("tag_lower")
    .agg(
        F.count("*").alias("tag_count"),
        F.countDistinct("movie_id").alias("movies_tagged"),
        F.countDistinct("user_id").alias("users_who_tagged"),
    )
    .filter(F.col("tag_count") >= 50)
    .orderBy(F.desc("tag_count"))
)
print("\n🏷️  Top 20 Tags:")
top_tags.show(20, truncate=False)

# Most tag-rich movies — deepest audience vocabulary
movie_tag_richness = (
    df_tags
    .groupBy("movie_id")
    .agg(
        F.count("tag_lower").alias("total_tags"),
        F.countDistinct("tag_lower").alias("unique_tags"),
    )
    .join(df_movies.select("movie_id", "clean_title", "genres"), "movie_id")
    .orderBy(F.desc("unique_tags"))
)
print("\n🔤 Most Tag-Rich Movies:")
movie_tag_richness.select("clean_title", "unique_tags", "total_tags", "genres") \
                  .show(15, truncate=False)

# ── 7.5  Hidden Gems ─────────────────────────────────────────────
# High-quality films with limited mainstream exposure.
# With 32M ratings the "low exposure" bar is raised to 500.
hidden_gems = (
    movie_stats
    .filter(
        (F.col("avg_rating") >= 4.0) &
        (F.col("total_ratings").between(100, 500))
    )
    .select("clean_title", "avg_rating", "median_rating",
            "total_ratings", "genres", "release_year")
    .orderBy(F.desc("avg_rating"))
)
print("\n💎 Hidden Gems (avg ≥ 4.0, 100–500 ratings):")
hidden_gems.show(15, truncate=False)

# ── 7.6  Peak rating hours ────────────────────────────────────────
hourly_activity = (
    df_ratings
    .groupBy("hour")
    .agg(
        F.count("rating").alias("rating_count"),
        F.round(F.avg("rating"), 3).alias("avg_rating"),
    )
    .orderBy("hour")
)
print("\n🕐 Hourly Activity:")
hourly_activity.show(24)

# ── 7.7  Quality by release decade ───────────────────────────────
# Do films from certain decades rate consistently higher or lower?
decade_quality = (
    movie_stats
    .filter(F.col("release_year").isNotNull())
    .withColumn("decade", (F.col("release_year") / 10).cast(IntegerType()) * 10)
    .groupBy("decade")
    .agg(
        F.count("movie_id").alias("num_movies"),
        F.round(F.avg("avg_rating"), 3).alias("avg_rating"),
        F.round(F.avg("total_ratings"), 0).alias("avg_ratings_received"),
    )
    .filter(F.col("decade").between(1900, 2020))
    .orderBy("decade")
)
print("\n📆 Quality by Decade:")
decade_quality.show(15)


# ─────────────────────────────────────────────────────────────────
# CELL 8 — Parquet Write (Performance Optimisation)
# ─────────────────────────────────────────────────────────────────

def write_parquet(df: DataFrame, name: str, partition_cols: list = None) -> None:
    """
    Persist a DataFrame as Snappy-compressed Parquet.
    partition_cols: columns to partition by (enables partition pruning
    on filtered reads — e.g. WHERE year = 2010 skips other partitions).
    """
    path = str(PARQUET_DIR / name)
    writer = df.write.mode("overwrite").option("compression", "snappy")
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)
    writer.parquet(path)

    # Report on-disk compressed size
    total_bytes = sum(
        f.stat().st_size for f in Path(path).rglob("*.parquet")
    )
    print(f"  ✅  {name:<28}  {total_bytes / 1e6:>8.1f} MB")

print("💾 Writing Parquet files (Snappy compression)…\n")

# ratings partitioned by year → time-range queries skip irrelevant partitions
write_parquet(df_ratings,        "ratings",        partition_cols=["year"])
write_parquet(df_movies,         "movies")
# tags partitioned by movie_id → per-movie tag lookups avoid full scans
write_parquet(df_tags,           "tags",           partition_cols=["movie_id"])
write_parquet(movie_stats,       "movie_stats")
write_parquet(user_activity,     "user_activity")
write_parquet(genre_stats,       "genre_stats")
write_parquet(movie_bayesian,    "movie_bayesian")
write_parquet(genre_year_trends, "genre_year_trends")
write_parquet(top_tags,          "top_tags")
write_parquet(hourly_activity,   "hourly_activity")
write_parquet(decade_quality,    "decade_quality")
write_parquet(movie_tag_richness,"movie_tag_richness")

print("\n📁 Parquet directory summary:")
for p in sorted(PARQUET_DIR.iterdir()):
    size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    print(f"   {p.name:<30} {size / 1e6:>7.1f} MB")


# ─────────────────────────────────────────────────────────────────
# CELL 9 — Visualisations (inline Plotly charts in Colab)
# ─────────────────────────────────────────────────────────────────
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TEMPLATE = "plotly_dark"

# Convert key results to Pandas for plotting
pd_genre      = genre_stats.toPandas()
pd_dist       = rating_distribution.toPandas()
pd_yearly     = yearly_trends.toPandas()
pd_top        = movie_bayesian.limit(20).toPandas()
pd_hourly     = hourly_activity.toPandas()
pd_genre_yr   = genre_year_trends.toPandas()
pd_decade     = decade_quality.toPandas()
pd_tags       = top_tags.limit(30).toPandas()
pd_user       = user_activity.limit(5000).toPandas()  # sample for histograms

# ── Chart 1: Genre Popularity ─────────────────────────────────────
fig1 = px.bar(
    pd_genre.sort_values("total_ratings"),
    x="total_ratings", y="genre", orientation="h",
    color="avg_rating", color_continuous_scale="Viridis",
    title="Genre Popularity — Total Ratings (32M dataset)",
    template=TEMPLATE, height=600,
    labels={"total_ratings": "Total Ratings", "avg_rating": "Avg Rating"},
)
fig1.show()

# ── Chart 2: Rating Distribution ─────────────────────────────────
fig2 = px.bar(
    pd_dist, x="rating", y="count",
    title="Rating Distribution (32M Ratings)",
    color="count", color_continuous_scale="Blues",
    template=TEMPLATE,
    labels={"count": "Number of Ratings"},
)
fig2.show()

# ── Chart 3: Yearly Trends (dual-axis) ───────────────────────────
fig3 = make_subplots(specs=[[{"secondary_y": True}]])
fig3.add_trace(go.Bar(
    x=pd_yearly["year"], y=pd_yearly["total_ratings"],
    name="Total Ratings", marker_color="#4ECDC4"), secondary_y=False)
fig3.add_trace(go.Scatter(
    x=pd_yearly["year"], y=pd_yearly["avg_rating"],
    name="Avg Rating", mode="lines+markers",
    line=dict(color="#FF6B6B", width=3)), secondary_y=True)
fig3.update_layout(title="Yearly Rating Volume & Average (1995–2023)",
                   template=TEMPLATE, height=450,
                   legend=dict(orientation="h"))
fig3.update_yaxes(title_text="Total Ratings", secondary_y=False)
fig3.update_yaxes(title_text="Average Rating", secondary_y=True)
fig3.show()

# ── Chart 4: Top 20 Movies by Bayesian Avg ───────────────────────
fig4 = px.bar(
    pd_top.sort_values("bayesian_avg"),
    x="bayesian_avg", y="clean_title", orientation="h",
    color="total_ratings", color_continuous_scale="Plasma",
    title="Top 20 Movies — Bayesian Average Rating",
    template=TEMPLATE, height=700,
)
fig4.show()

# ── Chart 5: Hourly Activity ──────────────────────────────────────
fig5 = px.area(
    pd_hourly, x="hour", y="rating_count",
    title="Rating Activity by Hour of Day",
    template=TEMPLATE, color_discrete_sequence=["#A8EDEA"],
)
fig5.update_layout(xaxis=dict(tickmode="linear", tick0=0, dtick=1))
fig5.show()

# ── Chart 6: Genre Trends Over Time (top 6 genres) ───────────────
top6 = pd_genre.nlargest(6, "total_ratings")["genre"].tolist()
fig6 = px.line(
    pd_genre_yr[pd_genre_yr["genre"].isin(top6)],
    x="year", y="rating_count", color="genre",
    title="Top Genre Rating Volume Over Time (1995–2023)",
    template=TEMPLATE,
)
fig6.show()

# ── Chart 7: Quality by Decade ────────────────────────────────────
fig7 = px.bar(
    pd_decade, x="decade", y="avg_rating",
    color="num_movies", color_continuous_scale="Teal",
    title="Average Movie Rating by Release Decade",
    template=TEMPLATE,
    labels={"avg_rating": "Avg Rating", "num_movies": "# Movies"},
)
fig7.show()

# ── Chart 8: Top Tags Bubble Chart ───────────────────────────────
fig8 = px.scatter(
    pd_tags.head(25), x="movies_tagged", y="tag_count",
    size="users_who_tagged", text="tag_lower", color="tag_count",
    color_continuous_scale="Magma",
    title="Top 25 Tags — Coverage vs Frequency",
    template=TEMPLATE,
)
fig8.update_traces(textposition="top center")
fig8.show()

print("✅ All 8 charts rendered.")


# ─────────────────────────────────────────────────────────────────
# CELL 10 — Streamlit Dashboard
# ─────────────────────────────────────────────────────────────────
STREAMLIT_APP = r'''
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyspark.sql import SparkSession
import os

st.set_page_config(
    page_title="Movie Rating Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 12px 20px;
    }
</style>
""", unsafe_allow_html=True)

PARQUET_DIR = "/content/movielens/parquet"
TEMPLATE    = "plotly_dark"

@st.cache_resource
def get_spark():
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
    spark = (
        SparkSession.builder
        .appName("MovieDashboard")
        .master("local[*]")
        .config("spark.driver.memory", "5g")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark

@st.cache_data
def load(name):
    return get_spark().read.parquet(f"{PARQUET_DIR}/{name}").toPandas()

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 MovieLens 32M")
    st.caption("32 million ratings · Apache Spark")
    st.markdown("---")
    section = st.radio("Navigate", [
        "📊 Overview",
        "🎭 Genre Deep Dive",
        "🏆 Top Movies",
        "💎 Hidden Gems",
        "🏷️ Tag Explorer",
        "👤 User Insights",
        "📆 Decade Trends",
    ])
    st.markdown("---")
    st.caption("Built with PySpark + Streamlit")

st.title("🎬 Movie Rating Analysis System")
st.caption("Scalable analytics on 32 million ratings · MovieLens 32M Dataset")

with st.spinner("Loading analytics from Parquet…"):
    df_movie    = load("movie_stats")
    df_genre    = load("genre_stats")
    df_user     = load("user_activity")
    df_bayesian = load("movie_bayesian")
    df_genre_yr = load("genre_year_trends")
    df_hourly   = load("hourly_activity")
    df_decade   = load("decade_quality")
    df_tags     = load("top_tags")

# ════════════════════════════════════════════════════════════════
if section == "📊 Overview":
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Ratings",   "~32 Million")
    col2.metric("Movies Analysed", f"{len(df_movie):,}")
    col3.metric("Unique Users",    f"{df_user['user_id'].nunique():,}")
    col4.metric("Global Mean",     f"{df_movie['avg_rating'].mean():.2f} ⭐")

    st.markdown("---")
    st.subheader("📅 Yearly Rating Volume & Average")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df_genre_yr.groupby("year")["rating_count"].sum().index,
                         y=df_genre_yr.groupby("year")["rating_count"].sum().values,
                         name="Total Ratings", marker_color="#4ECDC4"), secondary_y=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("⭐ Genre Overview")
    fig2 = px.bar(df_genre, x="genre", y="total_ratings",
                  color="avg_rating", color_continuous_scale="Viridis",
                  template=TEMPLATE, title="Total Ratings by Genre")
    st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════════════════════════
elif section == "🎭 Genre Deep Dive":
    st.subheader("🎭 Genre Popularity & Quality")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(df_genre.sort_values("total_ratings"),
                     x="total_ratings", y="genre", orientation="h",
                     color="avg_rating", color_continuous_scale="Viridis",
                     template=TEMPLATE, title="Volume by Genre")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.scatter(df_genre, x="num_movies", y="avg_rating",
                          size="total_ratings", color="genre",
                          template=TEMPLATE, title="Movies Count vs Avg Rating")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📈 Genre Trends Over Time")
    top_n  = df_genre.nlargest(10, "total_ratings")["genre"].tolist()
    chosen = st.multiselect("Select Genres", top_n, default=top_n[:5])
    if chosen:
        fig3 = px.line(df_genre_yr[df_genre_yr["genre"].isin(chosen)],
                       x="year", y="rating_count", color="genre",
                       template=TEMPLATE)
        st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════════════════════
elif section == "🏆 Top Movies":
    st.subheader("🏆 Top Movies by Bayesian Average Rating")
    n = st.slider("Show top N", 10, 50, 20)
    top = df_bayesian.nlargest(n, "bayesian_avg")
    fig = px.bar(top.sort_values("bayesian_avg"),
                 x="bayesian_avg", y="clean_title", orientation="h",
                 color="total_ratings", color_continuous_scale="Plasma",
                 template=TEMPLATE, height=max(400, n * 28))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        top[["clean_title","bayesian_avg","avg_rating","median_rating",
             "total_ratings","genres","release_year"]]
        .rename(columns={"clean_title":"Title","bayesian_avg":"Bayesian Avg",
                         "avg_rating":"Raw Avg","median_rating":"Median",
                         "total_ratings":"# Ratings","genres":"Genres",
                         "release_year":"Year"}),
        use_container_width=True,
    )

# ════════════════════════════════════════════════════════════════
elif section == "💎 Hidden Gems":
    st.subheader("💎 Hidden Gems — High Quality, Low Exposure")
    st.caption("Films with high average ratings but few total ratings")
    col1, col2 = st.columns(2)
    with col1:
        min_r, max_r = st.slider("Rating count range", 50, 2000, (100, 500))
    with col2:
        min_avg = st.slider("Min average rating", 3.5, 5.0, 4.0, step=0.05)

    gems = df_movie[
        (df_movie["avg_rating"] >= min_avg) &
        (df_movie["total_ratings"] >= min_r) &
        (df_movie["total_ratings"] <= max_r)
    ].sort_values("avg_rating", ascending=False)

    st.metric("Hidden Gems Found", len(gems))
    fig = px.scatter(gems, x="total_ratings", y="avg_rating",
                     hover_name="clean_title", color="genres",
                     template=TEMPLATE, title="Exposure vs Quality")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        gems[["clean_title","avg_rating","median_rating","total_ratings",
              "genres","release_year"]].head(50),
        use_container_width=True,
    )

# ════════════════════════════════════════════════════════════════
elif section == "🏷️ Tag Explorer":
    st.subheader("🏷️ Tag Analytics — exclusive to ml-32m")
    st.caption("User-generated tags reveal how audiences actually describe films")
    fig = px.scatter(df_tags.head(40),
                     x="movies_tagged", y="tag_count",
                     size="users_who_tagged", text="tag_lower",
                     color="tag_count", color_continuous_scale="Magma",
                     template=TEMPLATE, title="Tag Coverage vs Frequency")
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        df_tags[["tag_lower","tag_count","movies_tagged","users_who_tagged"]]
        .rename(columns={"tag_lower":"Tag","tag_count":"Uses",
                         "movies_tagged":"Movies","users_who_tagged":"Users"})
        .head(60),
        use_container_width=True,
    )

# ════════════════════════════════════════════════════════════════
elif section == "👤 User Insights":
    st.subheader("👤 User Behaviour (ml-32m has no demographics)")
    st.caption("All signals are purely behavioural — no age/gender/occupation data")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df_user, x="total_ratings", nbins=60,
                           template=TEMPLATE, title="Distribution: Ratings per User",
                           color_discrete_sequence=["#4ECDC4"])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.histogram(df_user, x="avg_rating_given", nbins=40,
                            template=TEMPLATE, title="Distribution: Avg Rating Given",
                            color_discrete_sequence=["#FF6B6B"])
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("⏰ Hourly Rating Activity")
    fig3 = px.area(df_hourly, x="hour", y="rating_count",
                   template=TEMPLATE, color_discrete_sequence=["#A8EDEA"])
    fig3.update_layout(xaxis=dict(tickmode="linear", tick0=0, dtick=1))
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("🔖 User Rating Segments")
    df_u = df_user.copy()
    df_u["segment"] = pd.cut(
        df_u["avg_rating_given"],
        bins=[0, 3.0, 4.0, 5.1],
        labels=["Harsh (<3)", "Moderate (3–4)", "Generous (≥4)"]
    )
    seg = df_u["segment"].value_counts().reset_index()
    seg.columns = ["segment", "count"]
    fig4 = px.pie(seg, names="segment", values="count",
                  template=TEMPLATE, title="User Segments by Avg Rating Given",
                  color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig4, use_container_width=True)

# ════════════════════════════════════════════════════════════════
elif section == "📆 Decade Trends":
    st.subheader("📆 Quality by Release Decade")
    fig = px.bar(df_decade, x="decade", y="avg_rating",
                 color="num_movies", color_continuous_scale="Teal",
                 template=TEMPLATE, title="Average Rating by Release Decade",
                 labels={"avg_rating": "Avg Rating", "num_movies": "# Movies"})
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(df_decade, x="decade", y="avg_ratings_received",
                      size="num_movies", color="avg_rating",
                      color_continuous_scale="Viridis", template=TEMPLATE,
                      title="Avg Ratings Received per Movie by Decade")
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(df_decade, use_container_width=True)
'''

app_path = Path("/content/movie_dashboard.py")
app_path.write_text(STREAMLIT_APP)
print(f"✅ Streamlit app written to {app_path}")

def launch_dashboard():
    import time
    from pyngrok import ngrok

    os.system("pkill -f streamlit 2>/dev/null || true")
    time.sleep(1)

    proc = subprocess.Popen(
        ["streamlit", "run", str(app_path),
         "--server.port=8501",
         "--server.headless=true",
         "--server.enableCORS=false",
         "--server.enableXsrfProtection=false"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(4)

    tunnel = ngrok.connect(8501)
    url    = tunnel.public_url

    print("\n" + "═" * 62)
    print(f"  🚀  Dashboard Live  →  {url}")
    print("═" * 62)
    print("  Open the link above in any browser.")
    print("  Kernel → Interrupt to stop the server.\n")
    return proc, tunnel

proc, tunnel = launch_dashboard()
