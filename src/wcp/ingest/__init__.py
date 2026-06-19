"""Data ingestion subpackage. One module per source.

Each ingester exposes a top-level ``ingest()`` function returning the path of the
materialised parquet, plus optionally a ``DataFrame``-returning helper.
"""
