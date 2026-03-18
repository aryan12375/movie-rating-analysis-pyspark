#  Movie Rating Analysis System — ml-32m

A scalable Big Data analytics platform built with **Apache Spark (PySpark)** to analyse
32 million movie ratings from the [MovieLens 32M](https://grouplens.org/datasets/movielens/latest/)
dataset, with an interactive **Streamlit dashboard** for the live demo.

---

##  Dataset Notes (ml-32m vs ml-1m)

| Feature              | ml-1M           | ml-32M (yours)          |
|----------------------|-----------------|-------------------------|
| File format          | `::` separated `.dat` | `,` separated `.csv` |
| Has header row       |  No           |  Yes                  |
| User demographics    |  `users.dat`  |  **Not included**     |
| Tags file            |  Not included |  `tags.csv`           |
| Rating count         | ~1 million      | ~32 million             |
| Encoding             | ISO-8859-1      | UTF-8                   |

This codebase is written **exclusively for ml-32m**. Do not use it with the 1M dataset.

---

## 🚀 Quick Start (Google Colab)

### Step 1 — Prepare your ZIP
Zip your `ml-32m` folder locally:
- **Windows**: right-click → Send to → Compressed (zipped) folder → `ml-32m.zip`
- **macOS**: right-click → Compress "ml-32m" → `ml-32m.zip`

The zip is ~820 MB.

### Step 2 — Open in Colab
Upload `movie_rating_analysis.py` or paste it into a new Colab notebook.
Split the file at each `# CELL N` comment into separate notebook cells.

### Step 3 — Runtime Settings
**Runtime → Change runtime type**:
- Runtime type: `Python 3`
- Hardware accelerator: `None` (Spark does not use GPUs)

### Step 4 — Run cells in order

| Cell | What it does |
|------|-------------|
| 1 | Installs PySpark 3.5.1, Streamlit, Plotly, pyngrok |
| 2 | Uploads and extracts `ml-32m.zip` into Colab |
| 3 | Starts SparkSession with 32M-tuned config |
| 4 | Loads all CSV files into Spark DataFrames with explicit schemas |
| 5 | Data quality audit (nulls, duplicates) + enrichment |
| 6 | Core analytics — avg ratings, active users, genre stats, yearly trends |
| 7 | Advanced analytics — Bayesian avg, tag analytics, hidden gems, decade quality |
| 8 | Writes Parquet (Snappy, ratings partitioned by year) |
| 9 | Renders 8 inline Plotly charts inside Colab |
| 10 | Launches Streamlit + ngrok tunnel → prints public URL |

---

## 📊 Analytics Implemented

### Core
- **Average rating per movie** — with median, stddev, min/max; 100-rating threshold
- **Most active users** — total ratings, unique movies, active lifespan in days
- **Genre popularity** — volume + average quality per genre
- **Rating distribution** — histogram of all half-star increments
- **Yearly trends** — dual-axis: volume + average rating, 1995–2023

### Advanced
- **Bayesian Average Rating** — IMDb-style dampened ranking (m=500 for 32M scale)
- **Tag Analytics** — top tags by frequency + coverage; most tag-rich films (exclusive to ml-32m)
- **Genre trends over time** — per-genre rating volume per year, 1995–2023
- **User segmentation** — Generous / Moderate / Harsh raters by avg rating given
- ** Hidden Gems** — high-rated films (avg ≥ 4.0) with limited exposure (100–500 ratings)
- ** Hourly activity** — when users rate most during the day
- ** Quality by decade** — do older films consistently rate higher?

---

##  Tech Stack

| Component | Tool |
|-----------|------|
| Processing Engine | Apache Spark 3.5.1 (PySpark) |
| Language | Python 3.10+ |
| Storage Format | Parquet (Snappy compressed) |
| Visualisation | Plotly |
| Dashboard | Streamlit |
| Tunnelling | pyngrok |
| Dataset | MovieLens 32M |

---

##  Spark Optimisations (32M-specific)

| Optimisation | Setting | Why |
|---|---|---|
| Shuffle partitions | 16 (vs default 200) | Single-node Colab has no benefit from 200; 16 halves shuffle overhead |
| Broadcast join threshold | 50 MB | `movies.csv` (~4 MB) auto-broadcasts, avoiding sort-merge joins |
| Adaptive Query Execution | Enabled | Auto-coalesces partitions and picks join strategies at runtime |
| Skew join handling | Enabled | `ratings` is heavily skewed toward popular movies |
| Kryo serialiser | Enabled | ~2x faster than Java default serialiser for large shuffles |
| Off-heap memory | 2 GB | Offloads aggregation buffers from JVM heap, reducing GC pressure |
| Partition by year | ratings | Time-range queries skip irrelevant year partitions entirely |
| Partition by movie_id | tags | Per-movie tag lookups avoid full table scans |
| Schema inference disabled | All tables | Avoids an extra full scan just to detect column types |
| DataFrame caching | All cleaned frames | Prevents recomputation when a frame feeds multiple aggregations |

---

##  Output Structure

```
/content/movielens/
├── raw/
│   ├── ratings.csv        ← 856 MB
│   ├── movies.csv         ←   4 MB
│   ├── tags.csv           ←  71 MB
│   └── links.csv          ←   2 MB
└── parquet/               ← Snappy compressed
    ├── ratings/           ← partitioned by year=XXXX/
    ├── movies/
    ├── tags/              ← partitioned by movie_id
    ├── movie_stats/
    ├── user_activity/
    ├── genre_stats/
    ├── movie_bayesian/
    ├── genre_year_trends/
    ├── top_tags/
    ├── hourly_activity/
    ├── decade_quality/
    └── movie_tag_richness/
```

---

##  My Additions 

| Feature | Rationale |
|---------|-----------|
| **Bayesian Average** | Raw averages unfairly rank films with 1 rating at 5.0. Bayesian dampening is the industry-standard fix — same formula as IMDb Top 250. |
| **Tag Analytics** | ml-32m includes `tags.csv` that ml-1m lacks. Analysing tag frequency and coverage reveals how audiences actually describe films, not just how they score them. |
| **Hidden Gems** | Surfaces quality films that never got mainstream exposure — a genuinely useful discovery feature. |
| **Hourly Activity** | When users rate most during the day; useful for recommendation timing in a real product. |
| **Quality by Decade** | A natural question with this dataset: do older films rate higher because only the classics survive, while newer films include more noise? |
| **User Active Lifespan** | `active_days` tells you whether a user rated 500 films over 10 years or in a single weekend — very different user profiles. |

---

## 📈 Scaling Further

The Parquet files written in Cell 8 can be queried independently without re-running the full pipeline. To reload them in a fresh session:

```python
df_movie_stats = spark.read.parquet("/content/movielens/parquet/movie_stats")
df_ratings     = spark.read.parquet("/content/movielens/parquet/ratings")
# etc.
```
