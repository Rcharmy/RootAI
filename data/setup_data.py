"""
data/setup_data.py

One-shot data loader. Reads Olist CSVs from data/raw/, loads each into a
raw_* table in DuckDB, then builds a denormalized order_items_denorm view
at order_item grain.

Usage:
    python data/setup_data.py

The script is idempotent: existing tables/views are dropped and recreated.
Run once at project setup. Re-run only if you change the schema or want a
fresh DB file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb

# Make rootai package importable regardless of where the script is run from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rootai.config import config


# CSV filename -> raw table name mapping
CSV_TO_TABLE = {
    "olist_customers_dataset.csv": "raw_customers",
    "olist_geolocation_dataset.csv": "raw_geolocation",
    "olist_orders_dataset.csv": "raw_orders",
    "olist_order_items_dataset.csv": "raw_order_items",
    "olist_order_payments_dataset.csv": "raw_order_payments",
    "olist_order_reviews_dataset.csv": "raw_order_reviews",
    "olist_products_dataset.csv": "raw_products",
    "olist_sellers_dataset.csv": "raw_sellers",
    "product_category_name_translation.csv": "raw_category_translation",
}


def load_raw_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Load all 9 CSVs into raw_* tables. Drop existing tables first."""
    for csv_name, table_name in CSV_TO_TABLE.items():
        csv_path = config.raw_data_dir / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV: {csv_path}")

        print(f"Loading {csv_name} -> {table_name}")
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        # read_csv_auto handles type inference and header detection
        con.execute(
            f"CREATE TABLE {table_name} AS "
            f"SELECT * FROM read_csv_auto('{csv_path.as_posix()}', header=true)"
        )


def build_denorm_view(con: duckdb.DuckDBPyConnection) -> None:
    """
    Create order_items_denorm view at order_item grain.

    Grain: one row per (order_id, order_item_id).
    Joins: order_items -> orders -> customers -> products -> payments (aggregated) -> reviews (aggregated).

    Payments and reviews are aggregated to order level to preserve order_item grain
    (a single order can have multiple payment records or multiple reviews).
    """
    print("Building order_items_denorm view")
    con.execute("DROP VIEW IF EXISTS order_items_denorm")
    con.execute(
        """
        CREATE VIEW order_items_denorm AS
        WITH payment_agg AS (
            SELECT
                order_id,
                SUM(payment_value) AS payment_value_total,
                MAX(payment_installments) AS payment_installments_max,
                -- Primary payment type: the one with highest value on the order
                ARG_MAX(payment_type, payment_value) AS payment_type
            FROM raw_order_payments
            GROUP BY order_id
        ),
        review_agg AS (
            SELECT
                order_id,
                AVG(review_score) AS review_score_avg,
                COUNT(*) AS review_count
            FROM raw_order_reviews
            GROUP BY order_id
        )
        SELECT
            -- Order item identity
            oi.order_id,
            oi.order_item_id,
            oi.product_id,
            oi.seller_id,

            -- Pricing (line-level: this is why order_item grain matters)
            oi.price,
            oi.freight_value,
            oi.price + oi.freight_value AS line_value,

            -- Order-level attributes
            o.customer_id,
            o.order_status,
            o.order_purchase_timestamp,
            o.order_approved_at,
            o.order_delivered_carrier_date,
            o.order_delivered_customer_date,
            o.order_estimated_delivery_date,

            -- Delivery timing (days)
            DATEDIFF('day', o.order_purchase_timestamp, o.order_delivered_customer_date)
                AS delivery_days,
            DATEDIFF('day', o.order_purchase_timestamp, o.order_estimated_delivery_date)
                AS estimated_delivery_days,

            -- Customer geography
            c.customer_unique_id,
            c.customer_city,
            c.customer_state,
            c.customer_zip_code_prefix,

            -- Product attributes
            p.product_category_name,
            pt.product_category_name_english AS product_category_english,
            p.product_weight_g,
            p.product_length_cm,
            p.product_height_cm,
            p.product_width_cm,

            -- Payment (from aggregated payment_agg)
            pay.payment_type,
            pay.payment_value_total,
            pay.payment_installments_max,

            -- Review (from aggregated review_agg)
            rev.review_score_avg AS review_score,
            rev.review_count

        FROM raw_order_items oi
        LEFT JOIN raw_orders o ON oi.order_id = o.order_id
        LEFT JOIN raw_customers c ON o.customer_id = c.customer_id
        LEFT JOIN raw_products p ON oi.product_id = p.product_id
        LEFT JOIN raw_category_translation pt
            ON p.product_category_name = pt.product_category_name
        LEFT JOIN payment_agg pay ON oi.order_id = pay.order_id
        LEFT JOIN review_agg rev ON oi.order_id = rev.order_id
        """
    )


def verify(con: duckdb.DuckDBPyConnection) -> None:
    """Print sanity checks so you can spot problems immediately."""
    print("\n=== Verification ===")

    # Row counts on raw tables
    print("\nRaw table row counts:")
    for table_name in CSV_TO_TABLE.values():
        count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"  {table_name}: {count:,}")

    # Denorm view stats
    denorm_count = con.execute("SELECT COUNT(*) FROM order_items_denorm").fetchone()[0]
    print(f"\norder_items_denorm rows: {denorm_count:,}")

    date_range = con.execute(
        "SELECT MIN(order_purchase_timestamp), MAX(order_purchase_timestamp) "
        "FROM order_items_denorm"
    ).fetchone()
    print(f"order_purchase_timestamp range: {date_range[0]} to {date_range[1]}")

    # Distinct dimensions we care about
    states = con.execute(
        "SELECT COUNT(DISTINCT customer_state) FROM order_items_denorm"
    ).fetchone()[0]
    print(f"distinct customer_state values: {states}")

    categories = con.execute(
        "SELECT COUNT(DISTINCT product_category_english) FROM order_items_denorm "
        "WHERE product_category_english IS NOT NULL"
    ).fetchone()[0]
    print(f"distinct product_category_english values: {categories}")

    sellers = con.execute(
        "SELECT COUNT(DISTINCT seller_id) FROM order_items_denorm"
    ).fetchone()[0]
    print(f"distinct seller_id values: {sellers:,}")

    print("\n=== Setup complete ===")


def main() -> None:
    # Ensure processed dir exists
    config.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"DuckDB file: {config.duckdb_path}")

    con = duckdb.connect(str(config.duckdb_path))
    try:
        load_raw_tables(con)
        build_denorm_view(con)
        verify(con)
    finally:
        con.close()


if __name__ == "__main__":
    main()