# ONNX and TensorRT

## Overview

This project provides a FastAPI-based REST API for serving YOLO26 object detection models in multiple formats:

- **PyTorch** (.pt) - Native YOLO format for maximum flexibility
- **ONNX** (.onnx) - Optimized cross-platform inference
- **TensorRT** (.engine) - High-performance inference on NVIDIA GPUs

## Setup

### Device information

- OS: Ubuntu 24.04.3 LTS x86_64
- CPU: i7-12700 (12 cores / 20 threads)
- RAM 32GB
- GPU: RTX 3060 12GB
- NVIDIA Driver 580.95.05
- CUDA 13.0

### TensorRT

Setup TensorRT by running:

```bash
bash setup_tensorrt.sh
```

### Environment

- Python 3.11.9
- Conda

```bash
conda create -n onnx_tensorrt python=3.11.9 --y
conda activate onnx_tensorrt
```

### Dependencies

```bash
pip install -r requirements.txt
```

### Download and Prepare Models

```bash
python setup_models.py
```

This will:
1. Download YOLO26 PyTorch model
2. Convert to ONNX format
3. Convert to TensorRT format (requires GPU)

### Prepare Test Images

For benchmarking, download and extract the COCO val2014 dataset:

```bash
# Download COCO val2014 images
wget http://images.cocodataset.org/zips/val2014.zip

# Extract to images directory
unzip val2014.zip -d ./images/
```

This provides 40,504 validation images for comprehensive benchmarking.

## Usage

### Start API Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Benchmarking

The benchmark script tests model performance across multiple images and provides comprehensive statistics.

#### Basic Usage

Test all images in the val2014 directory:

```bash
python benchmark.py
```

#### Advanced Options

```bash
# Test with specific number of images
python benchmark.py --max-images 50

# Test specific image file
python benchmark.py --image images/val2014/COCO_val2014_000000000042.jpg

# Test with specific models only
python benchmark.py --models onnx tensorrt --max-images 20

# Custom number of iterations
python benchmark.py --iterations 20 --warmup 5

# Specify API server URL
python benchmark.py --url http://localhost:8001
```

#### Benchmark Options

- `--image`: Path to test image or directory (default: `images/val2014`)
- `--max-images`: Maximum number of images to test (default: all images in directory)
- `--models`: Model formats to benchmark (choices: pytorch, onnx, tensorrt; default: all)
- `--iterations`: Number of benchmark iterations per image (default: 10)
- `--warmup`: Number of warmup iterations (default: 3)
- `--output-dir`: Output directory for results (default: `benchmark_results`)
- `--url`: API server URL (default: `http://localhost:8001`)
- `--no-plot`: Skip generating performance plots

#### Output

The benchmark provides:
- Mean inference time with standard deviation
- Throughput (FPS)
- Min/Max/Median inference times
- P95/P99 percentiles
- Average detections per image
- Performance comparison across model formats
- JSON results and performance plots saved to `benchmark_results/`

Example output:
```
BENCHMARK SUMMARY
============================================================
Total Images Tested: 50
Total Tests per Model: 500
============================================================

PYTORCH:
  Images Processed: 50
  Total Iterations: 500
  Inference Time: 45.23 ms ± 2.15 ms
  Throughput: 22.11 FPS
  Min/Max: 42.10 / 52.34 ms
  Median: 45.01 ms
  P95/P99: 48.76 / 50.21 ms
  Avg Detections: 8.3

ONNX:
  Images Processed: 50
  Total Iterations: 500
  Inference Time: 28.45 ms ± 1.32 ms
  Throughput: 35.15 FPS
  Min/Max: 26.78 / 32.11 ms
  Median: 28.32 ms
  P95/P99: 30.54 / 31.23 ms
  Avg Detections: 8.3

TENSORRT:
  Images Processed: 50
  Total Iterations: 500
  Inference Time: 15.67 ms ± 0.89 ms
  Throughput: 63.82 FPS
  Min/Max: 14.23 / 18.45 ms
  Median: 15.54 ms
  P95/P99: 17.12 / 17.89 ms
  Avg Detections: 8.3

⚡ SPEEDUP vs PyTorch:
  onnx: 1.59×
  tensorrt: 2.89×
```
