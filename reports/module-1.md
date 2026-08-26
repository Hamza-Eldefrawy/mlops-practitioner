# Module 1 Report: Packaging, APIs, and Containerization

## 1. Baseline Model Metrics (00-baseline.ipynb)
* **Dataset:** NYC TLC Green Taxi (January 2023)
* **Validation MAE:** 5.89 minutes
* **Validation RMSE:** 68.59 minutes <span style="color:red;">(very high) may be due to data skewness</span>
* **Artifact:** `models/baseline.pkl`

## 2. Serialization & Benchmarking

**Parity Benchmark (500 validation rows):**
* Pickle Mean Latency: 0.0064 ms
* ONNX Mean Latency: 0.0043 ms
* Pickle Mean/row: 0.0128 ms
* ONNX Mean/row  : 0.0087 ms
### Serialization Formats Comparison

| Format | Human-Readable | Cross-Language | Schema-Enforced | Safe for Untrusted Sources |
| :--- | :--- | :--- | :--- | :--- |
| **JSON** | Yes | Yes | No (requires external like JSONSchema) | Yes |
| **Protobuf** | No (Binary) | Yes | Yes | Yes |
| **Pickle** | No (Binary) | No (Python only) | No | **No** |
| **ONNX** | No (Binary) | Yes | Yes | Yes |

**Format Selection:**
This service serves predictions using the ONNX format because it completely decouples our inference runtime from the Python Scikit-Learn training environment, allowing for safer and faster cross-platform execution.

**SECURITY WARNING:**
Pickle executes arbitrary code on load. Never load a `.pkl` file you did not produce yourself.
