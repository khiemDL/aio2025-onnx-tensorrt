"""
Script to download and prepare YOLO26 models in different formats
"""
from pathlib import Path
from ultralytics import YOLO

# Create models directory
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def download_and_convert_yolo26():
    """Download YOLO26 and convert to different formats"""
    
    print("=" * 60)
    print("YOLO26 Model Setup")
    print("=" * 60)
    
    # Model paths
    pytorch_path = MODELS_DIR / "yolo26n.pt"
    onnx_path = MODELS_DIR / "yolo26n.onnx"
    tensorrt_path = MODELS_DIR / "yolo26n.engine"
    
    # Step 1: Download PyTorch model
    print("\n[1/3] Downloading YOLO26 PyTorch model...")
    try:
        model = YOLO(str(pytorch_path))
        print(f"✓ PyTorch model saved to: {pytorch_path}")
    except Exception as e:
        print(f"✗ Failed to download PyTorch model: {e}")
        return
    
    # Step 2: Export to ONNX
    # print("\n[2/3] Converting to ONNX format...")
    # try:
    #     model = YOLO(str(pytorch_path))
    #     model.export(format="onnx", dynamic=True, simplify=True)
    #     print(f"✓ ONNX model saved to: {onnx_path}")
    # except Exception as e:
    #     print(f"✗ Failed to export to ONNX: {e}")

    # Skip ONNX export because when converting to TensorRT, it will automatically create ONNX first.
    
    # Step 3: Export to TensorRT
    print("\n[3/3] Converting to TensorRT format...")
    print("Note: TensorRT conversion requires NVIDIA GPU and TensorRT installed")
    try:
        model = YOLO(str(pytorch_path))
        model.export(format="engine", device=0, half=True)
        print(f"✓ TensorRT model saved to: {tensorrt_path}")
    except Exception as e:
        print(f"✗ Failed to export to TensorRT: {e}")
        print("  This is normal if TensorRT is not installed or GPU is not available")
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("\nAvailable models:")
    for model_file in MODELS_DIR.glob("yolo26n.*"):
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"  - {model_file.name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    download_and_convert_yolo26()
