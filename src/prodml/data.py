import pandas as pd
from . import config as cfg

def read_data(path: str) -> pd.DataFrame:
  df = pd.read_parquet(path)
  return df

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
  df = df.copy()
  df['duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60
  df = df[(df['duration'] >= cfg.MIN_TRIP_DURATION) & (df['duration'] <= cfg.MAX_TRIP_DURATION)].copy()
  df['PULocationID'] = df['PULocationID'].astype(str)
  df['DOLocationID'] = df['DOLocationID'].astype(str)
  df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
  return df