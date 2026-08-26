import pickle
import numpy as np
from skl2onnx import to_onnx

from prodml.config import config
from prodml.logger import get_logger

logger = get_logger(__name__)


def export_onnx():
    logger.info("Loading pickled model for ONNX export...")
    with open(config.output_model_path, "rb") as f:
        dv, model = pickle.load(f)

    # dummy array with the number of features.
    # skl2onnx automatically leaves the first dimension (batch size) dynamic.
    num_features = len(dv.feature_names_)
    X_sample = np.zeros((1, num_features), dtype=np.float32)

    logger.info("Compiling graph...")
    onx = to_onnx(model, X_sample)

    with open(config.output_onnx_path, "wb") as f_out:
        f_out.write(onx.SerializeToString())

    logger.info(
        "ONNX export complete", extra={"onnx_path": str(config.output_onnx_path)}
    )


if __name__ == "__main__":
    export_onnx()
