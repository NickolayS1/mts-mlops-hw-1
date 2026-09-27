"""Обучение модели обнаружения фродовых транзакций.

Скрипт оформлен ячейками (# %%) — его можно запускать целиком
(`python training/train.py`) или выполнять по ячейкам в VS Code / PyCharm.

Что делает:
  1. читает train.csv,
  2. строит out-of-fold кодирование категориальных признаков (без утечки),
  3. обучает CatBoost и считает метрики на отложенной выборке,
  4. переобучает модель на всех данных,
  5. сохраняет модель, таблицы кодирования и submission.csv для Kaggle.
"""

# %% [markdown]
# ## 1. Импорты и настройки

# %%
import json
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TRAIN_PATH = os.path.join(ROOT, "teta-ml-1-2025", "train.csv")
TEST_PATH = os.path.join(ROOT, "teta-ml-1-2025", "test.csv")
MODELS_DIR = os.path.join(ROOT, "fraud_detector", "models")
OUT_MODEL = os.path.join(MODELS_DIR, "my_catboost.cbm")
OUT_ENCODERS = os.path.join(MODELS_DIR, "encoders.json")
OUT_SUBMISSION = os.path.join(HERE, "submission.csv")

TARGET = "target"
CATEGORICAL_COLS = ["gender", "merch", "cat_id", "one_city", "us_state", "jobs"]
DROP_COLS = ["name_1", "name_2", "street", "post_code"]
CONTINUOUS_COLS = ["amount", "population_city", "distance"]
N_TOP_CATEGORIES = 50
N_FOLDS = 5
EARTH_RADIUS_KM = 6371.009

# Порог перевода вероятности в метку для сабмита.
# Метрика соревнования принимает бинарные метки, а не вероятности.
# Значение выбрано по валидации: максимум F1 (см. вывод в ячейке 6.5).
SUBMISSION_THRESHOLD = 0.3

# %% [markdown]
# ## 2. Загрузка данных


# %%
def load_raw(path):
    """Читает датасет и убирает неинформативные колонки."""
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    return df


# %% [markdown]
# ## 3. Признаки


# %%
def add_time_features(df):
    """Календарные признаки из отметки времени."""
    dt = pd.to_datetime(df["transaction_time"]).dt
    df["hour"] = dt.hour.astype("int64")
    df["year"] = dt.year.astype("int64")
    df["month"] = dt.month.astype("int64")
    df["day_of_month"] = dt.day.astype("int64")
    df["day_of_week"] = dt.dayofweek.astype("int64")
    return df.drop(columns="transaction_time")


def add_distance_features(df):
    """Расстояние между клиентом и мерчантом, км."""
    lat1 = np.radians(df["lat"].to_numpy())
    lon1 = np.radians(df["lon"].to_numpy())
    lat2 = np.radians(df["merchant_lat"].to_numpy())
    lon2 = np.radians(df["merchant_lon"].to_numpy())
    a = (np.sin((lat2 - lat1) / 2.0) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2)
    df["distance"] = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return df.drop(columns=["lat", "lon", "merchant_lat", "merchant_lon"])


def build_base_features(df):
    """Общие для train и test преобразования: время и расстояние."""
    df = add_time_features(df)
    df = add_distance_features(df)
    return df


# %% [markdown]
# ## 4. Кодирование категориальных признаков
#
# Частые категории нумеруются (`cat_0`, `cat_1`, ...), редкие попадают в `cat_50+`.
# Среднее целевой переменной считается **только по обучающим фолдам**
# (out-of-fold), иначе модель увидит целевую переменную через кодирование.


# %%
def fit_category_maps(df, cols, n_top=N_TOP_CATEGORIES):
    """Частотное разбиение категорий: частые — по отдельности, редкие — в общий бакет."""
    maps = {}
    for col in cols:
        counts = df[col].value_counts(dropna=False)
        mapping = {}
        for rank, (value, _) in enumerate(counts.items()):
            if pd.isna(value):
                continue
            mapping[str(value)] = f"cat_{rank}" if rank < n_top else f"cat_{n_top}+"
        maps[col] = mapping
    return maps


def apply_category_maps(df, maps):
    """Заменяет исходные категории на их номера."""
    df = df.copy()
    for col, mapping in maps.items():
        df[col + "_cat"] = df[col].astype("string").map(mapping).fillna("cat_NAN")
        df = df.drop(columns=col)
    return df


def compute_mean_encodings(df, keys, target=TARGET):
    """Среднее целевой переменной по каждому значению ключа."""
    tables = {}
    for key in keys:
        grouped = df.groupby(key, dropna=False)[target].mean()
        tables[key] = {("" if pd.isna(k) else str(k)): float(v) for k, v in grouped.items()}
    return tables


def apply_mean_encodings(df, tables):
    """Джойнит таблицы средних по ключам."""
    df = df.copy()
    for key, table in tables.items():
        df[f"{key}_mean_enc"] = df[key].astype(str).map(table)
    return df


# %% [markdown]
# ## 5. Сборка матрицы признаков


# %%
def build_train_frame(raw):
    """Полный пайплайн признаков для обучающей выборки (без кодирования средними)."""
    df = build_base_features(raw)
    maps = fit_category_maps(df, CATEGORICAL_COLS)
    df = apply_category_maps(df, maps)
    return df, maps


def build_test_frame(raw, maps):
    """Тот же пайплайн для тестовой выборки, с готовыми картами категорий."""
    df = build_base_features(raw)
    df = apply_category_maps(df, maps)
    return df


def finalize(df, mean_tables, cat_keys, time_keys, continuous_cols):
    """Добавляет mean-encoding и логарифмы непрерывных признаков."""
    df = apply_mean_encodings(df, mean_tables)
    for col in continuous_cols:
        df[col + "_log"] = np.log1p(df[col].clip(lower=0))
        df = df.drop(columns=col)
    return df


def feature_columns(cat_keys, time_keys, continuous_cols):
    """Порядок признаков, который ожидает модель."""
    cols = time_keys + cat_keys
    cols += [f"{k}_mean_enc" for k in cat_keys]
    cols += [f"{k}_mean_enc" for k in time_keys]
    cols += [c + "_log" for c in continuous_cols]
    return cols


# %% [markdown]
# ## 6. Основной сценарий


# %%
def main():
    started = time.time()

    # --- 6.1. Данные ---
    raw_train = load_raw(TRAIN_PATH)
    print(f"train: {raw_train.shape}, fraud rate {raw_train[TARGET].mean():.4%}")

    df, maps = build_train_frame(raw_train)
    cat_keys = [c + "_cat" for c in CATEGORICAL_COLS]
    time_keys = ["hour", "year", "month", "day_of_month", "day_of_week"]

    # --- 6.2. Отложенная выборка ---
    # Стратификация важна: фродов всего ~0.57%.
    idx_train, idx_valid = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df[TARGET],
    )
    print(f"train {len(idx_train)}, valid {len(idx_valid)}")

    # --- 6.3. Out-of-fold кодирование ---
    # Таблицы средних считаются на train-фолдах и применяются к valid,
    # чтобы кодирование не подглядывало в целевую переменную.
    keys = cat_keys + time_keys
    train_part = df.iloc[idx_train].copy()
    valid_part = df.iloc[idx_valid].copy()

    oof_tables = compute_mean_encodings(train_part, keys)
    train_enc = finalize(train_part, oof_tables, cat_keys, time_keys, CONTINUOUS_COLS)
    valid_enc = finalize(valid_part, oof_tables, cat_keys, time_keys, CONTINUOUS_COLS)

    cols = feature_columns(cat_keys, time_keys, CONTINUOUS_COLS)
    X_train = train_enc[cols].copy()
    X_valid = valid_enc[cols].copy()
    y_train = train_enc[TARGET].to_numpy()
    y_valid = valid_enc[TARGET].to_numpy()

    # Категориальные признаки отдаём CatBoost как строки.
    for c in time_keys + cat_keys:
        X_train[c] = X_train[c].astype(str)
        X_valid[c] = X_valid[c].astype(str)

    cat_idx = list(range(len(time_keys) + len(cat_keys)))

    # --- 6.4. Обучение с early stopping ---
    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        od_type="Iter",
        od_wait=100,
        verbose=100,
        thread_count=-1,
    )
    model.fit(
        Pool(X_train, y_train, cat_features=cat_idx),
        eval_set=Pool(X_valid, y_valid, cat_features=cat_idx),
        use_best_model=True,
    )

    # --- 6.5. Метрики ---
    proba = model.predict_proba(X_valid)[:, 1]
    roc = roc_auc_score(y_valid, proba)
    pr = average_precision_score(y_valid, proba)
    print(f"\nvalid ROC-AUC: {roc:.5f}")
    print(f"valid PR-AUC : {pr:.5f}")

    for th in (0.5, 0.9, 0.98):
        pred = (proba > th).astype(int)
        tp = int(((pred == 1) & (y_valid == 1)).sum())
        fp = int(((pred == 1) & (y_valid == 0)).sum())
        fn = int(((pred == 0) & (y_valid == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        print(f"  th={th:<5} precision={prec:.4f} recall={rec:.4f} tp={tp} fp={fp}")

    # --- 6.6. Финальная модель на всех данных ---
    # Кодирование для продакшена строится по всей обучающей выборке.
    full_tables = compute_mean_encodings(df, keys)
    full_enc = finalize(df, full_tables, cat_keys, time_keys, CONTINUOUS_COLS)
    X_full = full_enc[cols].copy()
    for c in time_keys + cat_keys:
        X_full[c] = X_full[c].astype(str)
    y_full = full_enc[TARGET].to_numpy()

    final_model = CatBoostClassifier(
        iterations=model.get_best_iteration() or 500,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        random_seed=RANDOM_STATE,
        verbose=100,
        thread_count=-1,
    )
    final_model.fit(Pool(X_full, y_full, cat_features=cat_idx))
    print(f"\nfinal model trees: {final_model.tree_count_}")

    # --- 6.7. Сохранение артефактов ---
    os.makedirs(MODELS_DIR, exist_ok=True)
    final_model.save_model(OUT_MODEL)
    print(f"saved model -> {OUT_MODEL}")

    payload = {
        "meta": {
            "source_rows": int(len(df)),
            "base_fraud_rate": float(df[TARGET].mean()),
            "n_top_categories": N_TOP_CATEGORIES,
            "earth_radius_km": EARTH_RADIUS_KM,
            "valid_roc_auc": float(roc),
            "valid_pr_auc": float(pr),
        },
        "category_maps": maps,
        "mean_encodings": full_tables,
        "imputer_stats": {c: float(df[c].mean()) for c in CONTINUOUS_COLS + ["distance"]},
    }
    with open(OUT_ENCODERS, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"saved encoders -> {OUT_ENCODERS}")

    # --- 6.8. Сабмит для Kaggle ---
    # Метрика соревнования работает с бинарными метками, поэтому вероятности
    # переводим в 0/1 по порогу, подобранному на валидации (см. ячейку 6.5).
    if os.path.exists(TEST_PATH):
        raw_test = load_raw(TEST_PATH)
        test_df = build_test_frame(raw_test, maps)
        test_enc = finalize(test_df, full_tables, cat_keys, time_keys, CONTINUOUS_COLS)
        X_test = test_enc[cols].copy()
        for c in time_keys + cat_keys:
            X_test[c] = X_test[c].astype(str)
        test_proba = final_model.predict_proba(X_test)[:, 1]
        submission = pd.DataFrame({
            "index": np.arange(len(test_proba), dtype="int64"),
            "prediction": (test_proba > SUBMISSION_THRESHOLD).astype("int64"),
        })
        # Разделитель строк — LF и никакого \r: так же, как в sample_submition.csv
        submission.to_csv(OUT_SUBMISSION, index=False, lineterminator="\n")
        print(f"saved submission -> {OUT_SUBMISSION} ({len(submission)} rows, "
              f"{int(submission['prediction'].sum())} positives)")

    print(f"\ndone in {time.time() - started:.1f}s")


# %%
if __name__ == "__main__":
    main()
