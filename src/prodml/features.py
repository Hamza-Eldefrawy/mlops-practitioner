import pandas as pd
from scipy import sparse
from . import config as cfg
from sklearn.feature_extraction import DictVectorizer

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
  df = df.copy()
  df['duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60
  df = df[(df['duration'] >= cfg.MIN_TRIP_DURATION) & (df['duration'] <= cfg.MAX_TRIP_DURATION)].copy()
  df['PULocationID'] = df['PULocationID'].astype(str)
  df['DOLocationID'] = df['DOLocationID'].astype(str)
  df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
  return df

def vectorize_train(df: pd.DataFrame, dv: DictVectorizer) -> tuple[sparse.csr_matrix, DictVectorizer]:
  dicts = df[cfg.CATEGORICAL_FEATURES + cfg.NUMERICAL_FEATURES].to_dict(orient='records')
  X = dv.fit_transform(dicts)
  return (X, dv)

def vectorize(df: pd.DataFrame, dv: DictVectorizer) -> sparse.csr_matrix:
  dicts = df[cfg.CATEGORICAL_FEATURES + cfg.NUMERICAL_FEATURES].to_dict(orient='records')
  X = dv.transform(dicts)
  return X
