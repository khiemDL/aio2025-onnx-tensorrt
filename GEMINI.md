# GEMINI.md

## Project Overview
This project is a high-performance object detection API built with **FastAPI**, serving **YOLO** (specifically YOLO26l) models. It supports multiple inference backends to demonstrate performance gains:
- **PyTorch**: The native format.
- **ONNX**: Optimized cross-platform format.
- **TensorRT**: Maximum performance on NVIDIA GPUs.

The system is designed to benchmark these formats, comparing inference latency, throughput (FPS), and accuracy (detection counts) using the COCO val2014 dataset.

### Key Technologies
- **Backend**: FastAPI, Uvicorn
- **Machine Learning**: Ultralytics YOLO, PyTorch, ONNX, TensorRT
- **Data Processing**: NumPy, Pillow
- **Benchmarking/Visualization**: Requests, Matplotlib
- **Infrastructure**: CUDA 13.0, NVIDIA RTX 3060

## Building and Running

### Environment Setup
The project uses **Conda** for environment management.
```bash
# Create environment
conda create -n aio2025-onnx-tensorrt-env python=3.11 -y
conda activate aio2025-onnx-tensorrt-env

# Install dependencies via conda/pip as defined in conda.yml
# Note: Requires CUDA 13.0 for the specified torch/tensorrt versions
```

### Model Preparation
1. **Download Weights**:
   ```bash
   python download_yolo.py
   ```
2. **Convert Models**:
   ```bash
   python convert_2_onnx.py
   python convert_2_tensorrt.py
   ```

### Running the API
Start the server using the provided shell script or uvicorn:
```bash
# Using script
./run_app.sh --host 0.0.0.0 --port 8000

# Or directly via uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Benchmarking
Run the benchmark suite against the running API:
```bash
# Standard benchmark
python benchmark.py --image images/val2014 --max-images 100
```

## Development Conventions

### Project Structure
- `app/`: Contains the FastAPI application logic.
  - `api/`: Route definitions and endpoints (`yolo.py`).
  - `utils/`: Model loading logic (`load_model.py`) and image processing (`process.py`).
- `models/`: (Local) Directory for `.pt`, `.onnx`, and `.engine` files.
- `notebooks/`: Jupyter notebooks for research and experimentation with ONNX/TensorRT.
- `benchmark_results/`: (Generated) Stores JSON results and performance plots.

### Coding Standards
- **Model Loading**: Use `app.utils.load_model.ModelLoader` to manage lifecycle and caching of models.
- **Inference**: Inference is performed via the `ultralytics` library even for ONNX and TensorRT engines for consistent API usage.
- **Async Endpoints**: FastAPI endpoints use `async def` for I/O bound operations (like reading uploaded files).

### Testing & Validation
- **Benchmarking as Validation**: The `benchmark.py` script serves as the primary tool for verifying performance and functional correctness across different backends.
- **Notebooks**: Use the provided notebooks in `notebooks/` to validate conversion steps and compare outputs manually.
