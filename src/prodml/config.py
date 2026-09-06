from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Fallback to current working directory (e.g., /app in Docker or project root locally)
CWD = Path.cwd()


class Settings(BaseSettings):
    # Paths - prioritize CWD so Docker (/app) and local execution both map correctly
    data_dir: Path = CWD / "data"
    models_dir: Path = CWD / "models"
    output_model_path: Path = CWD / "models" / "model.pkl"
    output_onnx_path: Path = CWD / "models" / "model.onnx"
    data_path: Path = CWD / "data" / "green_tripdata_2023-01.parquet"

    # Data Constants
    min_trip_duration: float = 1.0
    max_trip_duration: float = 60.0
    train_test_split: float = Field(default=0.8, ge=0.0, le=1.0)

    # Features
    categorical_features: list[str] = ["PU_DO"]
    numerical_features: list[str] = ["trip_distance"]

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


config = Settings()
