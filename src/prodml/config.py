from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Calculate the absolute path to the repository root (mlops-practitioner/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # Paths
    data_dir: Path = BASE_DIR / "data"
    models_dir: Path = BASE_DIR / "models"
    output_model_path: Path = BASE_DIR / "models" / "model.pkl"
    data_path: Path = BASE_DIR / "data" / "green_tripdata_2023-01.parquet"

    # Data Constants
    min_trip_duration: float = 1.0
    max_trip_duration: float = 60.0
    train_test_split: float = Field(
        default=0.8, ge=0.0, le=1.0
    )  # Ensures it stays a valid ratio

    # Features
    categorical_features: list[str] = ["PU_DO"]
    numerical_features: list[str] = ["trip_distance"]

    # Instruct Pydantic to read overrides from a .env file if it exists
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# Instantiate the settings once to be imported across your application
config = Settings()
