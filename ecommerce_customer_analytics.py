#!/usr/bin/env python3
"""
E-commerce Customer Analytics: SQL + Python + RFM K-Means segmentation
======================================================================
Dataset : UCI "Online Retail" (Chen, Sain & Guo), CC BY 4.0
          https://archive.ics.uci.edu/dataset/352/online+retail

Run from the project folder (see README.md):
    python ecommerce_customer_analytics.py

What the script does (in order)
    1. Load the raw file and validate the expected columns
    2. Inspect data quality (missing values, duplicates, cancellations, odd values)
    3. Flag every row and assign ONE exclusion reason per row (nothing is silently dropped)
    4. Build a SQLite database and run the business queries in sql/03_business_queries.sql
    5. Compute KPIs and exploratory charts in pandas
    6. Build RFM features, log-transform + scale, choose k, fit K-Means, evaluate it
    7. Profile clusters, suggest labels from the actual numbers, write findings
    8. Run automatic validation checks (SQL vs pandas reconciliation, NaN checks, ...)

All numbers in outputs are calculated from the data you provide. Nothing is hard-coded.
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # no GUI needed; charts are saved to files
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
SQL_DIR = PROJECT_DIR / "sql"
DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "raw" / "Online Retail.xlsx"

EXPECTED_COLUMNS = ["InvoiceNo", "StockCode", "Description", "Quantity",
                    "InvoiceDate", "UnitPrice", "CustomerID", "Country"]

# A normal merchandise stock code = 5 digits optionally followed by 1-2 letters (e.g. 85123A).
# Anything else (POST, M, D, BANK CHARGES, gift vouchers, ...) is a non-merchandise code.
PRODUCT_CODE_PATTERN = re.compile(r"^\d{5}[A-Za-z]{0,2}$")

# Order matters: the FIRST matching rule becomes the row's exclusion reason.
EXCLUSION_RULES = [
    ("cancellation_invoice", "InvoiceNo starts with 'C' (cancellation / return)"),
    ("adjustment_invoice", "InvoiceNo starts with 'A' (accounting adjustment)"),
    ("non_product_code", "StockCode is not a merchandise code (postage, fees, manual, vouchers)"),
    ("nonpositive_quantity", "Quantity <= 0 on a non-cancellation invoice"),
    ("nonpositive_price", "UnitPrice <= 0 (free items / stock adjustments)"),
    ("exact_duplicate_row", "All 8 columns identical to an earlier row"),
]

RANDOM_SEED = 42
K_MIN, K_MAX = 2, 10
MIN_K_FOR_BUSINESS_USE = 3      # k=2 is usually too coarse for targeting (documented rule)
MIN_CLUSTER_SHARE = 0.02        # ignore k values that create a cluster smaller than 2% of customers

sns.set_theme(style="whitegrid")


def step(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# ----------------------------------------------------------------------------
# 2. LOADING + INSPECTION
# ----------------------------------------------------------------------------
def load_raw_data(path: Path) -> pd.DataFrame:
    """Load the Online Retail file (.xlsx or .csv) and validate its columns."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Download 'Online Retail.xlsx' from https://archive.ics.uci.edu/dataset/352/online+retail\n"
            "and place it in data/raw/ (or pass --data-path <file>)."
        )
    print(f"Reading {path} (the Excel file can take a minute or more) ...")
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, dtype={"InvoiceNo": str, "StockCode": str})
    elif suffix == ".csv":
        df = pd.read_csv(path, dtype={"InvoiceNo": str, "StockCode": str}, encoding="ISO-8859-1")
    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Use .xlsx or .csv")

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns {missing_cols}. Found: {list(df.columns)}")
    df = df[EXPECTED_COLUMNS].copy()

    # Type standardisation (no rows are dropped here)
    if not pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"]):
        try:
            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="mixed")
        except (TypeError, ValueError):
            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    if df["InvoiceDate"].isna().any():
        raise ValueError(f"{int(df['InvoiceDate'].isna().sum())} InvoiceDate values could not be parsed.")
    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["Description"] = df["Description"].astype("object").where(df["Description"].notna(), None)
    df["Description"] = df["Description"].map(lambda v: v.strip() if isinstance(v, str) and v.strip() else None)
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="raise").astype("int64")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="raise").astype("float64")
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce").astype("Int64")
    df["Country"] = df["Country"].fillna("Unknown").astype(str).str.strip()
    return df.reset_index(drop=True)


def inspect_data(df: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    """Print and save a first look at the raw data."""
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Date range: {df['InvoiceDate'].min()}  ->  {df['InvoiceDate'].max()}")
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())
    print("\nQuantity / UnitPrice summary:")
    print(df[["Quantity", "UnitPrice"]].describe().round(2).to_string())

    report = pd.DataFrame({
        "metric": [
            "raw_rows", "unique_invoices", "unique_stock_codes", "unique_customer_ids", "unique_countries",
            "first_invoice_date", "last_invoice_date", "missing_customer_id", "missing_description",
            "quantity_le_0", "unit_price_le_0", "cancellation_rows (InvoiceNo starts with C)",
            "exact_duplicate_rows",
        ],
        "value": [
            len(df), df["InvoiceNo"].nunique(), df["StockCode"].nunique(), df["CustomerID"].nunique(),
            df["Country"].nunique(), str(df["InvoiceDate"].min()), str(df["InvoiceDate"].max()),
            int(df["CustomerID"].isna().sum()), int(df["Description"].isna().sum()),
            int((df["Quantity"] <= 0).sum()), int((df["UnitPrice"] <= 0).sum()),
            int(df["InvoiceNo"].str.upper().str.startswith("C").sum()),
            int(df.duplicated(subset=EXPECTED_COLUMNS).sum()),
        ],
    })
    report.to_csv(tables_dir / "data_quality_report.csv", index=False)
    print("\nData-quality report:")
    print(report.to_string(index=False))
    return report


# ----------------------------------------------------------------------------
# 3. FLAGGING + CLEANING STRATEGY
# ----------------------------------------------------------------------------
def add_flags_and_exclusion_reason(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean quality flags and ONE exclusion_reason per row ('kept' = valid sale line).

    Strategy: flag first, decide later. The full flagged table is stored in SQLite, so
    cancellations, free items and missing-customer rows remain available for analysis.
    """
    d = df.copy()
    inv = d["InvoiceNo"].str.upper()
    d["line_revenue"] = d["Quantity"] * d["UnitPrice"]
    d["is_cancellation"] = inv.str.startswith("C")
    d["is_adjustment"] = inv.str.startswith("A")
    d["non_product_code"] = ~d["StockCode"].str.match(PRODUCT_CODE_PATTERN)
    d["nonpositive_quantity"] = d["Quantity"] <= 0
    d["nonpositive_price"] = d["UnitPrice"] <= 0
    d["is_duplicate"] = d.duplicated(subset=EXPECTED_COLUMNS, keep="first")
    d["missing_customer"] = d["CustomerID"].isna()

    conditions = [
        d["is_cancellation"],
        d["is_adjustment"],
        d["non_product_code"],
        d["nonpositive_quantity"],
        d["nonpositive_price"],
        d["is_duplicate"],
    ]
    d["exclusion_reason"] = np.select(conditions, [name for name, _ in EXCLUSION_RULES], default="kept")
    return d


def build_exclusion_summary(d: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    """Rows and value removed by each rule (sequential, so rows add up to the raw total)."""
    rule_text = dict(EXCLUSION_RULES)
    rule_text["kept"] = "Valid sale line used for revenue, product and customer analysis"
    summary = (d.groupby("exclusion_reason")
                 .agg(rows=("InvoiceNo", "size"),
                      invoices=("InvoiceNo", "nunique"),
                      value_gbp=("line_revenue", "sum"))
                 .reset_index())
    summary["pct_of_raw_rows"] = (100 * summary["rows"] / len(d)).round(2)
    summary["rule"] = summary["exclusion_reason"].map(rule_text)
    order = [n for n, _ in EXCLUSION_RULES] + ["kept"]
    summary["_o"] = summary["exclusion_reason"].map({n: i for i, n in enumerate(order)})
    summary = summary.sort_values("_o").drop(columns="_o")
    summary["value_gbp"] = summary["value_gbp"].round(2)
    summary.to_csv(tables_dir / "exclusion_summary.csv", index=False)

    # Customer-ID missing is NOT an exclusion for revenue KPIs, so report it separately.
    kept = d[d["exclusion_reason"] == "kept"]
    extra = pd.DataFrame({
        "metric": ["valid_sale_rows", "valid_sale_rows_without_customer_id",
                   "valid_sale_revenue_without_customer_id_gbp", "pct_valid_revenue_without_customer_id"],
        "value": [len(kept), int(kept["missing_customer"].sum()),
                  round(float(kept.loc[kept["missing_customer"], "line_revenue"].sum()), 2),
                  round(100 * float(kept.loc[kept["missing_customer"], "line_revenue"].sum())
                        / float(kept["line_revenue"].sum()), 2)],
    })
    extra.to_csv(tables_dir / "missing_customer_summary.csv", index=False)

    non_prod = (d[d["non_product_code"]].groupby("StockCode")
                  .agg(rows=("InvoiceNo", "size"), value_gbp=("line_revenue", "sum"),
                       example_description=("Description", "first"))
                  .sort_values("rows", ascending=False).head(25).reset_index())
    non_prod.to_csv(tables_dir / "non_product_codes.csv", index=False)

    print(summary[["exclusion_reason", "rows", "pct_of_raw_rows", "value_gbp"]].to_string(index=False))
    print("\n", extra.to_string(index=False))
    return summary


# ----------------------------------------------------------------------------
# 4. SQLITE DATABASE + SQL QUERIES
# ----------------------------------------------------------------------------
def _py(series: pd.Series, kind: str) -> list:
    """Convert a pandas column to plain Python values (None for missing) for sqlite3."""
    caster = {"int": int, "float": float, "str": str}[kind]
    return [None if pd.isna(v) else caster(v) for v in series]


def build_database(d: pd.DataFrame, db_path: Path) -> sqlite3.Connection:
    """Create the SQLite database, load all rows (with flags) and derive clean tables."""
    if sqlite3.sqlite_version_info < (3, 25, 0):
        raise RuntimeError(f"SQLite {sqlite3.sqlite_version} is too old for window functions (need 3.25+).")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript((SQL_DIR / "01_schema.sql").read_text(encoding="utf-8"))

    cols = [
        list(range(1, len(d) + 1)),
        _py(d["InvoiceNo"], "str"), _py(d["StockCode"], "str"), _py(d["Description"], "str"),
        _py(d["Quantity"], "int"), _py(d["InvoiceDate"].dt.strftime("%Y-%m-%d %H:%M:%S"), "str"),
        _py(d["UnitPrice"], "float"), _py(d["CustomerID"], "int"), _py(d["Country"], "str"),
        _py(d["line_revenue"], "float"),
        _py(d["is_cancellation"], "int"), _py(d["is_adjustment"], "int"), _py(d["non_product_code"], "int"),
        _py(d["nonpositive_quantity"], "int"), _py(d["nonpositive_price"], "int"),
        _py(d["is_duplicate"], "int"), _py(d["exclusion_reason"], "str"),
    ]
    conn.executemany("INSERT INTO raw_transactions VALUES (" + ",".join("?" * 17) + ")", zip(*cols))
    conn.commit()
    conn.executescript((SQL_DIR / "02_build_tables.sql").read_text(encoding="utf-8"))
    conn.commit()
    for tbl in ("raw_transactions", "products", "clean_sales"):
        n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"  table {tbl:<17} {n:>9,} rows")
    return conn


def parse_sql_file(path: Path) -> dict[str, str]:
    """Split sql/03_business_queries.sql into {query_name: sql_text}."""
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"^-- name:\s*", text, flags=re.M)[1:]
    queries = {}
    for block in blocks:
        name, _, body = block.partition("\n")
        queries[name.strip()] = body.strip()
    return queries


def run_sql_queries(conn: sqlite3.Connection, snapshot_date: str, tables_dir: Path) -> dict[str, pd.DataFrame]:
    """Run every query, save to outputs/tables/sql_<name>.csv, and return the results."""
    results = {}
    for name, sql in parse_sql_file(SQL_DIR / "03_business_queries.sql").items():
        try:
            res = pd.read_sql_query(sql, conn, params={"snapshot_date": snapshot_date})
        except Exception as exc:  # show which query failed
            raise RuntimeError(f"SQL query '{name}' failed: {exc}") from exc
        res.to_csv(tables_dir / f"sql_{name}.csv", index=False)
        results[name] = res
        print(f"  {name:<40} -> {len(res):>5} rows")
    return results


# ----------------------------------------------------------------------------
# 5. KPIs + EDA (pandas)
# ----------------------------------------------------------------------------
def compute_kpis(clean: pd.DataFrame, flagged: pd.DataFrame, tables_dir: Path) -> tuple[dict, pd.DataFrame]:
    """Business KPIs on valid sales, plus the order-level table."""
    orders = (clean.groupby("InvoiceNo")
                   .agg(order_value=("line_revenue", "sum"), units=("Quantity", "sum"),
                        lines=("StockCode", "size"), order_date=("InvoiceDate", "min"),
                        customer_id=("CustomerID", "first"), country=("Country", "first"))
                   .reset_index())
    total_rev = float(clean["line_revenue"].sum())
    ident_rev = float(clean.loc[clean["CustomerID"].notna(), "line_revenue"].sum())
    canc = flagged[flagged["is_cancellation"] & ~flagged["non_product_code"]]
    kpis = {
        "total_revenue_gbp": total_rev,
        "total_orders": int(len(orders)),
        "avg_order_value_gbp": float(orders["order_value"].mean()),
        "median_order_value_gbp": float(orders["order_value"].median()),
        "identified_customers": int(clean["CustomerID"].nunique()),
        "units_sold": int(clean["Quantity"].sum()),
        "avg_units_per_order": float(orders["units"].mean()),
        "avg_lines_per_order": float(orders["lines"].mean()),
        "pct_revenue_with_customer_id": 100 * ident_rev / total_rev,
        "cancelled_value_gbp": float(-canc["line_revenue"].sum()),
        "cancelled_pct_of_sales": 100 * float(-canc["line_revenue"].sum()) / total_rev,
        "cancellation_invoices": int(canc["InvoiceNo"].nunique()),
    }
    pd.Series(kpis, name="value").round(2).to_csv(tables_dir / "kpi_summary_pandas.csv", header=True, index_label="kpi")
    return kpis, orders


def add_partial_month_flag(monthly: pd.DataFrame, first_date: pd.Timestamp, last_date: pd.Timestamp) -> pd.DataFrame:
    """Mark months that are not fully covered by the observation window."""
    m = monthly.copy()
    flags = []
    for month in m["month"]:
        p = pd.Period(month, freq="M")
        partial = (month == first_date.strftime("%Y-%m") and first_date.day > 1) \
            or (month == last_date.strftime("%Y-%m") and last_date.normalize() < p.end_time.normalize())
        flags.append(bool(partial))
    m["is_partial_month"] = flags
    return m


def save_fig(fig, charts_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(charts_dir / name, dpi=150)
    plt.close(fig)


def plot_eda_charts(sql: dict[str, pd.DataFrame], orders: pd.DataFrame, cust_rev: pd.Series, charts_dir: Path) -> None:
    """EDA charts built from the SQL results and order/customer tables."""
    monthly = sql["q04_monthly_kpis"]
    full = monthly[~monthly["is_partial_month"]]
    part = monthly[monthly["is_partial_month"]]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(full["month"], full["revenue"], marker="o", label="Full month")
    if len(part):
        ax.scatter(part["month"], part["revenue"], marker="o", facecolors="none", edgecolors="red",
                   s=90, label="Partial month (incomplete data)", zorder=5)
    ax.set_title("Monthly revenue (GBP) - valid sales only")
    ax.set_ylabel("Revenue (GBP)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    save_fig(fig, charts_dir, "01_monthly_revenue.png")

    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax1.bar(monthly["month"], monthly["orders"], color="#4C72B0", alpha=0.8)
    ax1.set_ylabel("Orders")
    ax1.tick_params(axis="x", rotation=45)
    ax2 = ax1.twinx()
    ax2.plot(monthly["month"], monthly["avg_order_value"], color="#DD8452", marker="o")
    ax2.set_ylabel("Average order value (GBP)")
    ax2.grid(False)
    ax1.set_title("Monthly orders (bars) and average order value (line)")
    save_fig(fig, charts_dir, "02_monthly_orders_aov.png")

    prod = sql["q06_top_products_by_revenue"].head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(prod["description"].str.slice(0, 40), prod["revenue"], color="#55A868")
    ax.set_title("Top 10 products by revenue (GBP)")
    ax.set_xlabel("Revenue (GBP)")
    save_fig(fig, charts_dir, "03_top_products_revenue.png")

    ctry = sql["q08_country_performance"].head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(ctry["country"], ctry["pct_of_revenue"], color="#8172B2")
    ax.set_title("Top 10 countries by share of revenue (%)")
    ax.set_xlabel("% of total valid revenue")
    save_fig(fig, charts_dir, "04_top_countries_revenue_share.png")

    s = cust_rev.sort_values(ascending=False).reset_index(drop=True)
    cum = s.cumsum() / s.sum() * 100
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(100 * (np.arange(len(s)) + 1) / len(s), cum)
    ax.set_xlabel("% of identified customers (highest revenue first)")
    ax.set_ylabel("Cumulative % of identified-customer revenue")
    ax.set_title("Revenue concentration curve")
    ax.axvline(10, ls="--", color="grey")
    save_fig(fig, charts_dir, "05_revenue_concentration.png")

    cap = orders["order_value"].quantile(0.99)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.hist(orders.loc[orders["order_value"] <= cap, "order_value"], bins=50, color="#4C72B0")
    ax.axvline(orders["order_value"].mean(), color="red", ls="--", label=f"Mean {orders['order_value'].mean():,.0f}")
    ax.axvline(orders["order_value"].median(), color="green", ls="--", label=f"Median {orders['order_value'].median():,.0f}")
    ax.set_title("Order value distribution (orders above the 99th percentile hidden for readability)")
    ax.set_xlabel("Order value (GBP)")
    ax.legend()
    save_fig(fig, charts_dir, "06_order_value_distribution.png")


# ----------------------------------------------------------------------------
# 6. RFM FEATURES + K-MEANS
# ----------------------------------------------------------------------------
def build_rfm(clean: pd.DataFrame, flagged: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    """Customer-level RFM table.

    Population : valid sale lines that have a CustomerID.
    Recency    : whole days from the customer's last valid purchase to the snapshot date.
    Frequency  : number of distinct valid invoices (an invoice = one order).
    Monetary   : total GBP of valid sale lines (cancellations are NOT netted off; they are
                 reported separately as profile-only columns).
    """
    cust = clean[clean["CustomerID"].notna()]
    rfm = (cust.groupby("CustomerID")
               .agg(last_purchase=("InvoiceDate", "max"), first_purchase=("InvoiceDate", "min"),
                    Frequency=("InvoiceNo", "nunique"), Monetary=("line_revenue", "sum"))
               .reset_index())
    rfm["Recency"] = (snapshot - rfm["last_purchase"]).dt.days
    canc = flagged[flagged["is_cancellation"] & ~flagged["non_product_code"] & flagged["CustomerID"].notna()]
    canc_cust = (canc.groupby("CustomerID")
                     .agg(cancel_invoices=("InvoiceNo", "nunique"), cancel_value=("line_revenue", lambda s: -s.sum()))
                     .reset_index())
    rfm = rfm.merge(canc_cust, on="CustomerID", how="left")
    rfm[["cancel_invoices", "cancel_value"]] = rfm[["cancel_invoices", "cancel_value"]].fillna(0)
    rfm["CustomerID"] = rfm["CustomerID"].astype("int64")
    return rfm[["CustomerID", "Recency", "Frequency", "Monetary", "cancel_invoices", "cancel_value",
                "first_purchase", "last_purchase"]]


def prepare_features(rfm: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """log1p transform (fixes right-skew) then standardise (equal weight for R, F, M)."""
    log_feats = np.log1p(rfm[["Recency", "Frequency", "Monetary"]].astype(float))
    log_feats.columns = ["log_Recency", "log_Frequency", "log_Monetary"]
    X = StandardScaler().fit_transform(log_feats)
    return X, log_feats


def evaluate_k_range(X: np.ndarray, seed: int) -> pd.DataFrame:
    rows = []
    sample = min(10000, len(X))
    for k in range(K_MIN, K_MAX + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        lab = km.labels_
        rows.append({
            "k": k, "inertia": km.inertia_,
            "silhouette": silhouette_score(X, lab, sample_size=sample, random_state=seed),
            "davies_bouldin": davies_bouldin_score(X, lab),
            "min_cluster_share": np.bincount(lab).min() / len(lab),
        })
    return pd.DataFrame(rows)


def choose_k(scores: pd.DataFrame) -> tuple[int, str]:
    """Documented rule: highest silhouette among k >= 3 whose smallest cluster holds >= 2% of customers."""
    cand = scores[(scores["k"] >= MIN_K_FOR_BUSINESS_USE) & (scores["min_cluster_share"] >= MIN_CLUSTER_SHARE)]
    if cand.empty:
        cand = scores
        note = "no k met the minimum-size rule; fell back to best silhouette overall"
    else:
        note = (f"highest silhouette among k>={MIN_K_FOR_BUSINESS_USE} with smallest cluster >= "
                f"{MIN_CLUSTER_SHARE:.0%} of customers")
    best = int(cand.loc[cand["silhouette"].idxmax(), "k"])
    return best, note


def fit_final_model(X: np.ndarray, k: int, seed: int) -> tuple[KMeans, dict]:
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    aris = [adjusted_rand_score(km.labels_, KMeans(n_clusters=k, n_init=10, random_state=s).fit_predict(X))
            for s in range(1, 6)]
    metrics = {
        "k": k,
        "silhouette": float(silhouette_score(X, km.labels_, sample_size=min(10000, len(X)), random_state=seed)),
        "davies_bouldin": float(davies_bouldin_score(X, km.labels_)),
        "inertia": float(km.inertia_),
        "stability_ARI_mean": float(np.mean(aris)),
        "stability_ARI_min": float(np.min(aris)),
    }
    return km, metrics


def reorder_clusters(rfm: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    """Renumber clusters so cluster 0 has the highest median monetary value, 1 the next, ..."""
    tmp = rfm.assign(_c=labels)
    order = tmp.groupby("_c")["Monetary"].median().sort_values(ascending=False).index.tolist()
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[c] for c in labels])


# Friendly names + actions keyed by (recent?, frequent?, high-spend?) relative to overall medians.
SEGMENT_PLAYBOOK = {
    (True, True, True): ("Champions",
                         "Protect and reward: early access, loyalty tiers, service perks. Test non-discount rewards first; "
                         "avoid blanket discounts to customers who already buy often."),
    (False, True, True): ("At-risk high-value",
                          "Priority win-back: personal outreach or account-manager contact, ask for feedback, offer a "
                          "capped, time-limited incentive. Test against a holdout group."),
    (True, True, False): ("Engaged regulars (lower spend)",
                          "Grow basket size: bundles, cross-sell, free-delivery or volume thresholds."),
    (True, False, True): ("Big-basket recent buyers",
                          "Encourage the next order: replenishment reminders and wholesale-style repeat offers."),
    (True, False, False): ("New or occasional recent buyers",
                           "Onboarding and second-purchase journey: welcome series, recommendations based on first basket."),
    (False, False, True): ("Lapsed big spenders (few orders)",
                           "Win-back with relevance: remind of previous purchases, show new arrivals in the same categories."),
    (False, True, False): ("Lapsed regulars (lower spend)",
                           "Low-cost reactivation: reminder emails around their usual purchase cycle."),
    (False, False, False): ("Lapsed low-value",
                            "Low-cost automated reactivation only; suppress from paid campaigns; consider sunsetting inactive contacts."),
}


def profile_and_label_clusters(rfm: pd.DataFrame) -> pd.DataFrame:
    """Profile each cluster with ACTUAL RFM statistics, then suggest a label from those statistics.

    Labels compare each cluster's MEDIAN with the overall customer base: "recent" = median recency below the
    overall median; "frequent" / "high-spend" = median at or above the overall 75th percentile.
    They are suggestions to review, not facts: check the profile table before using them.
    """
    overall = rfm[["Recency", "Frequency", "Monetary"]].median()
    high_bar = rfm[["Frequency", "Monetary"]].quantile(0.75)   # "frequent" / "high-spend" = at or above the 75th percentile
    prof = (rfm.groupby("cluster")
               .agg(customers=("CustomerID", "count"),
                    median_recency_days=("Recency", "median"), mean_recency_days=("Recency", "mean"),
                    median_frequency=("Frequency", "median"), mean_frequency=("Frequency", "mean"),
                    median_monetary=("Monetary", "median"), mean_monetary=("Monetary", "mean"),
                    total_revenue=("Monetary", "sum"),
                    avg_cancel_invoices=("cancel_invoices", "mean"))
               .reset_index())
    prof["pct_customers"] = 100 * prof["customers"] / prof["customers"].sum()
    prof["pct_revenue"] = 100 * prof["total_revenue"] / prof["total_revenue"].sum()

    labels, actions = [], []
    seen: dict[str, int] = {}
    for _, r in prof.iterrows():
        key = (r["median_recency_days"] < overall["Recency"],
               r["median_frequency"] >= high_bar["Frequency"],
               r["median_monetary"] >= high_bar["Monetary"])
        name, action = SEGMENT_PLAYBOOK[key]
        seen[name] = seen.get(name, 0) + 1
        labels.append(name)
        actions.append(action)
    prof["suggested_label"] = labels
    dup = prof["suggested_label"].duplicated(keep=False)
    prof.loc[dup, "suggested_label"] = prof.loc[dup].apply(
        lambda r: f"{r['suggested_label']} (cluster {int(r['cluster'])})", axis=1)
    prof["suggested_action"] = actions
    return prof.round(2)


def plot_model_charts(scores: pd.DataFrame, chosen_k: int, X: np.ndarray, rfm: pd.DataFrame, log_feats: pd.DataFrame,
                      prof: pd.DataFrame, charts_dir: Path) -> None:
    # RFM distributions raw vs log
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for j, col in enumerate(["Recency", "Frequency", "Monetary"]):
        sns.histplot(rfm[col], bins=40, ax=axes[0, j], color="#4C72B0")
        axes[0, j].set_title(f"{col} (raw)")
        sns.histplot(np.log1p(rfm[col].astype(float)), bins=40, ax=axes[1, j], color="#55A868")
        axes[1, j].set_title(f"log(1 + {col})")
    save_fig(fig, charts_dir, "07_rfm_distributions_raw_vs_log.png")

    # Elbow + silhouette
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(scores["k"], scores["inertia"], marker="o")
    axes[0].axvline(chosen_k, color="red", ls="--")
    axes[0].set_title("Elbow plot (inertia)")
    axes[0].set_xlabel("k")
    axes[1].plot(scores["k"], scores["silhouette"], marker="o", color="#DD8452")
    axes[1].axvline(chosen_k, color="red", ls="--", label=f"chosen k = {chosen_k}")
    axes[1].set_title("Silhouette score")
    axes[1].set_xlabel("k")
    axes[1].legend()
    save_fig(fig, charts_dir, "08_cluster_selection_elbow_silhouette.png")

    # PCA + Frequency vs Monetary scatter
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(X)
    palette = sns.color_palette("tab10", n_colors=chosen_k)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=rfm["cluster"], palette=palette, s=14, ax=axes[0], legend="full")
    axes[0].set_title("Clusters in 2-D PCA space")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} of variance)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} of variance)")
    sns.scatterplot(x=log_feats["log_Frequency"], y=log_feats["log_Monetary"], hue=rfm["cluster"],
                    palette=palette, s=14, ax=axes[1], legend=False)
    axes[1].set_title("log Frequency vs log Monetary")
    save_fig(fig, charts_dir, "09_clusters_visualised.png")

    # Profile heatmap: colour = mean standardised feature, text = actual medians
    z = pd.DataFrame(X, columns=["Recency", "Frequency", "Monetary"]).groupby(rfm["cluster"].values).mean()
    annot = np.array([[f"{r.median_recency_days:,.0f} days", f"{r.median_frequency:,.0f} order(s)", f"GBP {r.median_monetary:,.0f}"]
                      for r in prof.itertuples()])
    ylabels = [f"C{int(r.cluster)}: {r.suggested_label}" for r in prof.itertuples()]
    fig, ax = plt.subplots(figsize=(10, 0.9 * chosen_k + 2.5))
    sns.heatmap(z.values, annot=annot, fmt="", cmap="vlag", center=0, ax=ax,
                xticklabels=["Recency", "Frequency", "Monetary"], yticklabels=ylabels, cbar_kws={"label": "mean standardised log value"})
    ax.set_title("Cluster profile: colour = standardised level, text = actual median")
    save_fig(fig, charts_dir, "10_cluster_profile_heatmap.png")

    # Share of customers vs share of revenue
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(prof))
    ax.bar(x - 0.2, prof["pct_customers"], 0.4, label="% of customers", color="#4C72B0")
    ax.bar(x + 0.2, prof["pct_revenue"], 0.4, label="% of revenue", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{int(c)}" for c in prof["cluster"]])
    ax.set_title("Segment size vs revenue contribution")
    ax.set_ylabel("%")
    ax.legend()
    save_fig(fig, charts_dir, "11_segment_customers_vs_revenue.png")


# ----------------------------------------------------------------------------
# 7. FINDINGS (numbers come from the computed tables)
# ----------------------------------------------------------------------------
def finding(title, question, metric, result, meaning, action, limitation) -> str:
    return (f"### {title}\n- **Question:** {question}\n- **Metric / analysis:** {metric}\n- **Result:** {result}\n"
            f"- **What it means:** {meaning}\n- **Proposed action:** {action}\n- **Limitation / uncertainty:** {limitation}\n")


def write_findings(kpis, sql, cust_rev, prof, metrics, k_note, snapshot, out_path: Path) -> None:
    monthly = sql["q04_monthly_kpis"]
    full = monthly[~monthly["is_partial_month"]]
    best = full.loc[full["revenue"].idxmax()]
    med_month = full["revenue"].median()
    deciles = sql["q10_revenue_deciles"]
    top10_share = float(deciles.loc[deciles["decile"] == 1, "pct_of_identified_revenue"].iloc[0])
    s = cust_rev.sort_values(ascending=False)
    n80 = int((s.cumsum() / s.sum() < 0.8).sum() + 1)
    country = sql["q08_country_performance"].iloc[0]
    rep = sql["q15_repeat_purchase_rate"].iloc[0]
    freq = sql["q09_customer_frequency_bands"]
    one_time = freq[freq["frequency_band"].str.startswith("1 order")]
    one_time_pct = 100 * float(one_time["customers"].sum()) / float(freq["customers"].sum()) if len(one_time) else float("nan")

    parts = ["# Business findings (auto-generated from the data you ran)\n",
             f"Observation window: valid sales through {snapshot.date() - pd.Timedelta(days=1)}; RFM snapshot date = {snapshot.date()}.\n",
             "Revenue is sales value in GBP (quantity x unit price). It is NOT profit: the dataset has no cost data.\n",
             "## Observed findings\n"]
    parts.append(finding(
        "1. Headline KPIs", "How large is the business and what does a typical order look like?",
        "Total revenue, orders, mean and median order value on valid sale lines",
        f"Revenue GBP {kpis['total_revenue_gbp']:,.0f}; {kpis['total_orders']:,} orders; mean order value GBP {kpis['avg_order_value_gbp']:,.0f} "
        f"vs median GBP {kpis['median_order_value_gbp']:,.0f}; {kpis['identified_customers']:,} identified customers.",
        (f"The mean is {kpis['avg_order_value_gbp']/kpis['median_order_value_gbp']:.2f}x the median, so " +
         ("a tail of very large orders pulls the average up; the median is the better 'typical' basket."
          if kpis['avg_order_value_gbp'] > 1.2 * kpis['median_order_value_gbp'] else
          "order values are fairly symmetric and the mean is a reasonable 'typical' basket.")),
        "Report both mean and median AOV; use medians when judging a 'typical' basket.",
        f"Only {kpis['pct_revenue_with_customer_id']:.1f}% of revenue carries a CustomerID, so customer analysis covers part of the business."))
    parts.append(finding(
        "2. Revenue concentration", "Which customers contribute the most revenue?",
        "Customer-level revenue ranked high to low (SQL q05, q10)",
        f"Top 10% of identified customers = {top10_share:.1f}% of identified-customer revenue; {n80:,} customers ({100*n80/len(s):.1f}%) "
        f"account for 80% of it.",
        ("Revenue is heavily concentrated: losing a few top customers could move results noticeably." if top10_share >= 60 else
         "Revenue is noticeably concentrated in the top customers." if top10_share >= 40 else
         "Revenue is only moderately concentrated in the top decile."),
        "Assign relationship owners to the top accounts and track their order cadence monthly.",
        "Based on one observed period and identified customers only."))
    parts.append(finding(
        "3. Monthly trend", "What are the monthly revenue patterns?",
        "Monthly revenue on valid sales, full months only (SQL q04)",
        f"Highest full month: {best['month']} (GBP {best['revenue']:,.0f}), {best['revenue']/med_month:.2f}x the median full-month revenue.",
        ("The peak month stands clearly above a typical month; one year of data cannot separate seasonality from growth."
         if best['revenue'] / med_month >= 1.3 else
         "No month stands far above the typical month; one year of data cannot separate seasonality from growth."),
        "If the peak is genuine seasonality, plan inventory and campaigns around it; verify against a second year if available.",
        "First/last months may be partial (flagged on the chart); only about 12-13 months exist."))
    parts.append(finding(
        "4. Geography", "Where does revenue come from?", "Revenue share by country (SQL q08)",
        f"{country['country']} contributes {country['pct_of_revenue']:.1f}% of valid revenue.",
        (f"The largest country accounts for most of revenue; other countries are a minority." if country['pct_of_revenue'] >= 60 else
         "Revenue is spread across several countries rather than dominated by one."),
        "Treat any international growth push as a separate experiment with its own targets.",
        "Country is the customer's country, not delivery cost or profitability by region."))
    parts.append(finding(
        "5. Repeat buying", "Do customers come back?", "Customers with 2+ orders (SQL q09, q15)",
        f"{rep['repeat_customer_pct']:.1f}% of identified customers placed 2+ orders; {one_time_pct:.1f}% bought once.",
        "The larger the one-time group, the bigger the potential gain from converting first-time buyers into repeat buyers.",
        "Test a second-purchase journey (post-purchase email series) against a control group.",
        "Customers who bought once near the end of the window have had little time to return."))
    parts.append(finding(
        "6. Cancellations", "How material are cancellations?", "Cancelled merchandise value vs valid sales (SQL q11)",
        f"{kpis['cancellation_invoices']:,} cancellation invoices worth GBP {kpis['cancelled_value_gbp']:,.0f} "
        f"= {kpis['cancelled_pct_of_sales']:.1f}% of valid sales value.",
        "This is a proxy for revenue leakage; its size relative to sales shows how much a cancellation-reduction project could matter.",
        "Capture cancellation reasons and link them to the original invoice.",
        "Cancellations cannot be matched to original orders, and some may refer to orders before the window."))
    parts.append("### 7. Customer segments (K-Means on RFM)\n"
                 f"- **Model:** k = {metrics['k']} ({k_note}); silhouette = {metrics['silhouette']:.3f}; Davies-Bouldin = "
                 f"{metrics['davies_bouldin']:.3f}; stability ARI mean = {metrics['stability_ARI_mean']:.3f} "
                 f"(min {metrics['stability_ARI_min']:.3f}).\n"
                 "- **Limitation:** RFM data usually forms a continuum, so clusters are partitions of a continuum rather than naturally "
                 "separated groups. Labels are suggestions derived from cluster medians.\n")
    parts.append(prof[["cluster", "suggested_label", "customers", "pct_customers", "median_recency_days", "median_frequency",
                       "median_monetary", "pct_revenue"]].to_markdown(index=False) if _has_tabulate() else
                 prof[["cluster", "suggested_label", "customers", "pct_customers", "median_recency_days", "median_frequency",
                       "median_monetary", "pct_revenue"]].to_string(index=False))
    parts.append("\n## Recommendations (hypotheses to test, not guaranteed outcomes)\n")
    n = 1
    for r in prof.itertuples():
        parts.append(f"{n}. **{r.suggested_label}** (cluster {int(r.cluster)}; {r.pct_customers:.1f}% of customers, "
                     f"{r.pct_revenue:.1f}% of revenue): {r.suggested_action}")
        n += 1
    parts.append(f"{n}. **Reduce dependence on a few accounts:** top-decile customers hold {top10_share:.1f}% of identified revenue; "
                 "set up account reviews and early-warning triggers when a top customer's gap between orders grows.")
    n += 1
    parts.append(f"{n}. **Convert one-time buyers:** run an A/B test of a second-purchase email journey; measure repeat rate versus a holdout.")
    n += 1
    parts.append(f"{n}. **Capture cancellation reasons and link them to orders** before attempting any cancellation-reduction project.")
    n += 1
    parts.append(f"{n}. **Add data before optimising spend:** product costs/margins, marketing channel, campaign history, customer IDs for all orders.")
    parts.append("\n## Trade-offs to discuss\n"
                 "- Discounts reduce margin and may train customers to wait for offers; test and cap them.\n"
                 "- Over-targeting high-value customers can cause fatigue; set contact-frequency limits.\n"
                 "- Win-back effort on low-value lapsed customers may cost more than it returns.\n"
                 "- Segments describe past behaviour; they do not prove a campaign will change behaviour. Use control groups.\n")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


# ----------------------------------------------------------------------------
# 8. VALIDATION
# ----------------------------------------------------------------------------
def run_validation(flagged, clean, sql, rfm, X, metrics, kpis, out_dir: Path, charts_dir: Path, tables_dir: Path) -> bool:
    checks = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    check("Exclusion reasons cover every raw row", flagged["exclusion_reason"].notna().all(), f"{len(flagged):,} rows")
    sql_total = float(sql["q03_kpi_summary"]["total_revenue_gbp"].iloc[0])
    check("SQL total revenue equals pandas total revenue", abs(sql_total - kpis["total_revenue_gbp"]) < 0.5,
          f"SQL {sql_total:,.2f} vs pandas {kpis['total_revenue_gbp']:,.2f}")
    check("SQL order count equals pandas order count",
          int(sql["q03_kpi_summary"]["total_orders"].iloc[0]) == kpis["total_orders"])
    s = sql["q13_rfm_customer_table"].rename(columns={"customer_id": "CustomerID"}).set_index("CustomerID").sort_index()
    p = rfm.set_index("CustomerID").sort_index()
    same_ids = s.index.equals(p.index)
    check("SQL and pandas RFM cover the same customers", same_ids, f"{len(p):,} customers")
    if same_ids:
        check("SQL and pandas Recency match", (s["recency_days"].values == p["Recency"].values).all())
        check("SQL and pandas Frequency match", (s["frequency"].values == p["Frequency"].values).all())
        check("SQL and pandas Monetary match (+-0.01)", np.allclose(s["monetary"].values, p["Monetary"].values, atol=0.011))
    check("Feature matrix has no NaN / inf", np.isfinite(X).all(), f"shape {X.shape}")
    check("Recency >= 0, Frequency >= 1, Monetary > 0",
          (rfm["Recency"] >= 0).all() and (rfm["Frequency"] >= 1).all() and (rfm["Monetary"] > 0).all())
    check("Every customer has a cluster", rfm["cluster"].notna().all())
    check("No empty clusters", rfm["cluster"].nunique() == metrics["k"], f"{rfm['cluster'].nunique()} clusters")
    check("Silhouette within [-1, 1]", -1 <= metrics["silhouette"] <= 1, f"{metrics['silhouette']:.3f}")
    check("Cluster sizes sum to number of customers", int(rfm.groupby("cluster").size().sum()) == len(rfm))
    for f in ["rfm_customer_segments.csv", "cluster_profile.csv", "business_findings.md"]:
        check(f"Output exists: {f}", (out_dir / f).exists() or (tables_dir / f).exists())
    check("At least 10 charts saved", len(list(charts_dir.glob("*.png"))) >= 10, f"{len(list(charts_dir.glob('*.png')))} charts")

    lines = ["VALIDATION REPORT (automatic checks run by the script)", "-" * 60]
    for name, ok, detail in checks:
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    all_ok = all(ok for _, ok, _ in checks)
    lines.append("-" * 60)
    lines.append("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED - review above")
    (out_dir / "validation_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return all_ok


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="E-commerce customer analytics (SQL + Python + RFM K-Means)")
    ap.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Path to Online Retail .xlsx or .csv")
    ap.add_argument("--workdir", type=Path, default=PROJECT_DIR, help="Folder where database/, outputs/, data/processed are written")
    ap.add_argument("--k", type=int, default=None, help="Force a number of clusters instead of the automatic rule")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    work = args.workdir
    out_dir, charts_dir, tables_dir = work / "outputs", work / "outputs" / "charts", work / "outputs" / "tables"
    processed_dir, db_path = work / "data" / "processed", work / "database" / "online_retail.db"
    for dname in (out_dir, charts_dir, tables_dir, processed_dir):
        dname.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    step("STEP 1 - Load raw data")
    raw = load_raw_data(args.data_path)

    step("STEP 2 - Inspect data quality")
    inspect_data(raw, tables_dir)

    step("STEP 3 - Flag rows and apply cleaning rules")
    flagged = add_flags_and_exclusion_reason(raw)
    build_exclusion_summary(flagged, tables_dir)
    clean = flagged[flagged["exclusion_reason"] == "kept"].copy()
    clean.drop(columns=["exclusion_reason"]).to_csv(processed_dir / "clean_sales.csv", index=False)
    if clean.empty:
        raise RuntimeError("No valid sale lines remain after cleaning - check the input file.")

    snapshot = clean["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    print(f"\nRFM snapshot (reference) date: {snapshot.date()}  (day after the last valid sale)")

    step("STEP 4 - Build SQLite database and run SQL queries")
    conn = build_database(flagged, db_path)
    sql = run_sql_queries(conn, snapshot.strftime("%Y-%m-%d"), tables_dir)
    conn.close()
    sql["q04_monthly_kpis"] = add_partial_month_flag(sql["q04_monthly_kpis"], clean["InvoiceDate"].min(), clean["InvoiceDate"].max())
    sql["q04_monthly_kpis"].to_csv(tables_dir / "sql_q04_monthly_kpis.csv", index=False)

    step("STEP 5 - KPIs and exploratory charts")
    kpis, orders = compute_kpis(clean, flagged, tables_dir)
    for key, val in kpis.items():
        print(f"  {key:<32} {val:>16,.2f}")
    cust_rev = clean[clean["CustomerID"].notna()].groupby("CustomerID")["line_revenue"].sum()
    plot_eda_charts(sql, orders, cust_rev, charts_dir)

    step("STEP 6 - RFM features and K-Means")
    rfm = build_rfm(clean, flagged, snapshot)
    print(rfm[["Recency", "Frequency", "Monetary"]].describe().round(2).to_string())
    X, log_feats = prepare_features(rfm)
    extreme = int((np.abs(X) > 4).any(axis=1).sum())
    print(f"\nCustomers with any |standardised log feature| > 4: {extreme} (kept; legitimate extreme customers)")
    scores = evaluate_k_range(X, args.seed)
    scores.to_csv(tables_dir / "kmeans_k_selection.csv", index=False)
    print(scores.round(3).to_string(index=False))
    if args.k:
        chosen_k, k_note = args.k, "k forced by --k"
    else:
        chosen_k, k_note = choose_k(scores)
    print(f"\nChosen k = {chosen_k} ({k_note})")
    km, metrics = fit_final_model(X, chosen_k, args.seed)
    rfm["cluster"] = reorder_clusters(rfm, km.labels_)
    pd.Series(metrics, name="value").to_csv(tables_dir / "model_metrics.csv", index_label="metric")
    print({k: round(v, 4) for k, v in metrics.items()})

    step("STEP 7 - Profile clusters, charts, findings")
    prof = profile_and_label_clusters(rfm)
    rfm = rfm.merge(prof[["cluster", "suggested_label"]], on="cluster", how="left")
    rfm.to_csv(out_dir / "rfm_customer_segments.csv", index=False)
    prof.to_csv(out_dir / "cluster_profile.csv", index=False)
    print(prof[["cluster", "suggested_label", "customers", "pct_customers", "median_recency_days",
                "median_frequency", "median_monetary", "pct_revenue"]].to_string(index=False))
    plot_model_charts(scores, chosen_k, X, rfm, log_feats, prof, charts_dir)
    write_findings(kpis, sql, cust_rev, prof, metrics, k_note, snapshot, out_dir / "business_findings.md")

    step("STEP 8 - Validation checks")
    ok = run_validation(flagged, clean, sql, rfm, X, metrics, kpis, out_dir, charts_dir, tables_dir)
    print(f"\nDone. Outputs are in: {out_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as err:
        print(f"\nERROR: {err}", file=sys.stderr)
        sys.exit(2)
