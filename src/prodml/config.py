from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"

OUTPUT_FILE_NAME = MODELS_DIR / "baseline.pkl"
DATA_PATH = DATA_DIR / "green_tripdata_2023-01.parquet"

# Data Constants
MIN_TRIP_DURATION = 1
MAX_TRIP_DURATION = 60
TRAIN_TEST_SPLIT = 0.8

# Features names
CATEGORICAL_FEATURES = ['PU_DO']
NUMERICAL_FEATURES = ['trip_distance']
