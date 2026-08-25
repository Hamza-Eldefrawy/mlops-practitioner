import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.feature_extraction import DictVectorizer

from . import config as cfg , data, features

def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
  train_size = int(cfg.TRAIN_TEST_SPLIT * len(df))
  df_train = df.iloc[:train_size].copy()
  df_val = df.iloc[train_size:].copy()
  return (df_train, df_val)

def train(df_train: pd.DataFrame, df_val: pd.DataFrame) -> tuple:
  dv = DictVectorizer()
  
  x_train, dv = features.vectorize_train(df_train, dv)
  y_train = df_train['duration'].values

  x_val = features.vectorize(df_val, dv)
  y_val = df_val['duration'].values

  model = LinearRegression()
  model.fit(x_train, y_train)

  y_pred = model.predict(x_val)
  mae = mean_absolute_error(y_val, y_pred)
  rmse = root_mean_squared_error(y_val, y_pred)
  return (model, dv, mae, rmse)

def save_model(model, dv):
  cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
  with open(cfg.OUTPUT_FILE_NAME, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

def run():
  df = data.read_data(cfg.DATA_PATH)
  df = data.prepare_features(df)
  df_train, df_val = split(df)
  model, dv, mae, rmse = train(df_train, df_val)
  print(f"MAE: {mae:.2f}")
  print(f"RMSE: {rmse:.2f}")
  save_model(model, dv)

if __name__ == "__main__":
  run()
