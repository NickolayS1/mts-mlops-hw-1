"""Build the preprocessing lookup tables from the competition train.csv.

The scoring service needs three things that are derived from the historical
dataset: the category groupings, the target mean-encoding tables and the
imputer statistics.  Computing them from train.csv on every container start
costs a full parse of a 138 MB file, so they are precomputed here once and
stored in a small JSON file that the service loads at boot.

Usage:
    python tools/build_encoders.py [--train PATH] [--out PATH]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# Columns that identify a customer/merchant location and are replaced by distance
# Columns dropped by the original preprocessing pipeline
DROP_COLS = ["name_1", "name_2", "street", "post_code"]

CATEGORICAL_COLS = ["gender", "merch", "cat_id", "one_city", "us_state", "jobs"]
TIME_COLS = ["hour", "year", "month", "day_of_month", "day_of_week"]
CONTINUOUS_COLS = ["amount", "population_city"]
TARGET_COL = "target"

# Categories outside the top N are grouped into a single bucket
N_TOP_CATEGORIES = 50

# Mean Earth radius in km, as used by geopy's great_circle distance
EARTH_RADIUS_KM = 6371.009


def great_circle_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km on a sphere of mean Earth radius."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def add_time_features(df):
    """Expand the timestamp into the calendar features used by the model."""
    dt = pd.to_datetime(df["transaction_time"]).dt
    df["hour"] = dt.hour.astype("int64")
    df["year"] = dt.year.astype("int64")
    df["month"] = dt.month.astype("int64")
    df["day_of_month"] = dt.day.astype("int64")
    df["day_of_week"] = dt.dayofweek.astype("int64")
    return df.drop(columns="transaction_time")


def label_of(rank, is_nan, n_top=N_TOP_CATEGORIES):
    """Map a frequency rank to its category label."""
    if is_nan:
        return "cat_NAN"
    if rank < n_top:
        return "cat_" + str(rank)
    return "cat_%d+" % n_top


def build_category_maps(train):
    """Build raw value -> group label maps, ordered by descending frequency."""
    maps = {}
    for col in CATEGORICAL_COLS:
        counts = train[col].value_counts(dropna=False)
        values, labels = [], []
        for rank, (value, _) in enumerate(counts.items()):
            if pd.isna(value):
                continue
            values.append(str(value))
            labels.append(label_of(rank, False))
        maps[col] = dict(zip(values, labels))
    return maps


def apply_category_maps(train, maps):
    """Attach the grouped category columns to the training frame."""
    for col in CATEGORICAL_COLS:
        train[col + "_cat"] = (
            train[col].astype("string").map(maps[col]).fillna("cat_NAN")
        )
    return train


def build_mean_encodings(train):
    """Average target value per category and per calendar feature."""
    tables = {}
    for col in CATEGORICAL_COLS + TIME_COLS:
        key = col if col in TIME_COLS else col + "_cat"
        grouped = train.groupby(key, dropna=False)[TARGET_COL].mean()
        tables[key] = {
            ("" if pd.isna(k) else str(k)): float(v) for k, v in grouped.items()
        }
    return tables


def build_imputer_stats(df):
    """Column means used to fill gaps in the continuous features."""
    stats = {col: float(df[col].mean()) for col in CONTINUOUS_COLS}
    stats["distance"] = float(df["_distance"].mean())
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument(
        "--train",
        default=os.path.join(here, "teta-ml-1-2025", "train.csv"),
        help="path to the competition train.csv",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(here, "fraud_detector", "models", "encoders.json"),
        help="where to write the lookup tables",
    )
    args = parser.parse_args()

    started = time.time()
    print("Reading %s" % args.train)
    train = pd.read_csv(args.train)
    train = train.drop(columns=[c for c in DROP_COLS if c in train.columns])
    train = add_time_features(train)
    print("  %d rows, %d columns (%.1fs)" % (len(train), train.shape[1], time.time() - started))
    print("  base fraud rate: %.6f" % train[TARGET_COL].mean())

    print("Building category maps (top %d kept)" % N_TOP_CATEGORIES)
    maps = build_category_maps(train)
    for col in CATEGORICAL_COLS:
        print("  %-10s %4d distinct values" % (col, len(maps[col])))

    train = apply_category_maps(train, maps)

    print("Building mean encodings")
    mean_enc = build_mean_encodings(train)
    for key in sorted(mean_enc):
        print("  %-20s %4d levels" % (key, len(mean_enc[key])))

    print("Computing distances for imputer stats")
    train["_distance"] = great_circle_km(
        train["lat"].to_numpy(),
        train["lon"].to_numpy(),
        train["merchant_lat"].to_numpy(),
        train["merchant_lon"].to_numpy(),
    )
    print("  distance mean %.4f km, min %.4f, max %.4f"
          % (train["_distance"].mean(), train["_distance"].min(), train["_distance"].max()))

    imputer = build_imputer_stats(train)

    payload = {
        "meta": {
            "source_rows": int(len(train)),
            "base_fraud_rate": float(train[TARGET_COL].mean()),
            "n_top_categories": N_TOP_CATEGORIES,
            "earth_radius_km": EARTH_RADIUS_KM,
        },
        "category_maps": maps,
        "mean_encodings": mean_enc,
        "imputer_stats": imputer,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1, sort_keys=True)

    size_kb = os.path.getsize(args.out) / 1024.0
    print("\nWrote %s (%.1f KB) in %.1fs" % (args.out, size_kb, time.time() - started))


if __name__ == "__main__":
    sys.exit(main())
