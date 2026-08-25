from pathlib import Path

import pandas as pd


def read_data(path: Path) -> pd.DataFrame:
    """Reads a Parquet file into a Pandas DataFrame."""
    return pd.read_parquet(path)


def split_data(
    df: pd.DataFrame, split_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the dataset sequentially based on the provided ratio."""
    train_size = int(split_ratio * len(df))
    df_train = df.iloc[:train_size].copy()
    df_val = df.iloc[train_size:].copy()

    return df_train, df_val
