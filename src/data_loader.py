import pandas as pd
import numpy as np


def load_and_process_data(filepath="average_metrics_uap_5min.csv"):
    """
    Loads CSV, cleans missing values, and standardizes the data.
    Returns:
        X: The standardized Numpy array (for clustering)
        df_clean: The cleaned Pandas DataFrame (for results)
        feature_cols: List of column names used
    """
    print("Loading and Preprocessing Data...")
    df = pd.read_csv(filepath)

    # Drop rows with missing 'Core Metrics'
    df_clean = df.dropna(
        subset=["avg_client_count", "avg_cpu_usage_ratio", "avg_memory_usage_ratio"]
    ).copy()

    # Impute missing 'Signal Metrics' with Noise Floor (-95 dBm)
    df_clean["avg_signal_5g_dbm"] = df_clean["avg_signal_5g_dbm"].fillna(-95)
    df_clean["avg_signal_24g_dbm"] = df_clean["avg_signal_24g_dbm"].fillna(-95)

    feature_cols = [
        "avg_client_count",
        "avg_cpu_usage_ratio",
        "avg_memory_usage_ratio",
        "avg_signal_5g_dbm",
        "avg_signal_24g_dbm",
    ]

    # Convert to Numpy and Standardize (Z-Score)
    X_raw = df_clean[feature_cols].to_numpy(dtype=float)
    X_mean = np.mean(X_raw, axis=0)
    X_std = np.std(X_raw, axis=0)
    X = (X_raw - X_mean) / X_std

    print(f"Data Ready. Shape: {X.shape}")
    return X, df_clean, feature_cols
