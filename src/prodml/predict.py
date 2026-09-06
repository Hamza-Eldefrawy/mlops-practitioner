import pickle
import time

import numpy as np
import onnxruntime as rt

from functools import wraps
from pathlib import Path
from typing import Any

from prodml.config import config
from prodml.logger import get_logger

logger = get_logger(__name__)


# 1. The Decorator: Measures how long a function takes to execute
def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        logger.info(
            "Function executed",
            extra={
                "function_name": func.__name__,
                "latency_seconds": round(end_time - start_time, 4),
            },
        )

        return result

    return wrapper


# 2. The Abstraction Layer: Wraps ML logic in a clean OOP interface
class DurationPredictor:
    def __init__(self, model_path: Path | None = None):
        """Initializes the predictor. Defaults to the path in config.py if none provided."""
        self.model_path = config.output_model_path
        self.onnx_path = config.output_onnx_path
        self.dv = None
        self.session = None

    def load(self):
        """Loads the vectorizer from pickle, and the model into ONNX Runtime."""
        with open(self.model_path, "rb") as f:
            self.dv, _ = pickle.load(f)

        self.session = rt.InferenceSession(
            str(self.onnx_path), providers=["CPUExecutionProvider"]
        )

    @timed
    def predict_one(self, ride: dict[str, Any]) -> float:
        """Makes a prediction for a single ride dictionary."""
        # Check if the model is loaded in memory
        if self.session is None or self.dv is None:
            self.load()

        # The vectorizer validates dictionaries
        X = self.dv.transform([ride]).astype(np.float32).toarray()

        input_name = self.session.get_inputs()[0].name

        # Extract the single float prediction from the Numpy array
        prediction = self.session.run(None, {input_name: X})[0][0]
        return float(prediction.item())

    @timed
    def predict_batch(self, rides: list[dict[str, Any]]) -> list[float]:
        """Makes predictions for a batch of rides."""
        if self.session is None or self.dv is None:
            self.load()

        X = self.dv.transform(rides).astype(np.float32).toarray()
        input_name = self.session.get_inputs()[0].name

        predictions = self.session.run(None, {input_name: X})[0]
        return predictions.flatten().tolist()
