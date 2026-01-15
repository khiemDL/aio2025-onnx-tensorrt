"""
Benchmark script to compare performance across different model formats (PyTorch, ONNX, TensorRT).
This script sends requests to the API and measures response times, then generates performance reports.
"""

import requests
import time
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from datetime import datetime
import argparse


class APIBenchmark:
    """Benchmark API performance across different model formats"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize benchmark configuration
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url
        self.api_url = f"{base_url}/api/v1/yolo/predict"
        self.models_url = f"{base_url}/api/v1/yolo/models"
        self.health_url = f"{base_url}/api/v1/yolo/health"
        
        self.results = {
            "pytorch": [],
            "onnx": [],
            "tensorrt": []
        }
        
    def check_server_health(self) -> bool:
        """Check if the API server is running and healthy"""
        try:
            print(self.health_url)
            response = requests.get(self.health_url, timeout=5)
            if response.status_code == 200:
                print("✓ Server is healthy")
                print(json.dumps(response.json(), indent=2))
                return True
            else:
                print(f"✗ Server returned status code: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"✗ Cannot connect to server at {self.base_url}")
            print("  Please ensure the API server is running with: ./run_app.sh")
            return False
        except Exception as e:
            print(f"✗ Error checking server health: {e}")
            return False
    
    def check_models_availability(self) -> Dict[str, bool]:
        """Check which models are available"""
        try:
            response = requests.get(self.models_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                models_status = {}
                print("\n📦 Models Status:")
                for format_name, info in data.get("models", {}).items():
                    available = info.get("available", False)
                    loaded = info.get("loaded", False)
                    status = "✓" if available else "✗"
                    print(f"  {status} {format_name}: available={available}, loaded={loaded}")
                    models_status[format_name] = available
                return models_status
            return {}
        except Exception as e:
            print(f"Error checking models: {e}")
            return {}
    
    def get_image_paths(self, image_path: str = "images/val2014", max_images: int = None) -> List[str]:
        """
        Get list of image paths for testing
        
        Args:
            image_path: Path to the image file or directory (default uses val2014 directory)
            max_images: Maximum number of images to process (None for all)
            
        Returns:
            List of image file paths
        """
        image_paths = []
        
        # If a specific image file path is provided and exists, use it
        if os.path.isfile(image_path):
            print(f"✓ Using single image: {image_path}")
            return [image_path]
        
        # If a directory is provided, collect all images from it
        if os.path.isdir(image_path):
            images = sorted([f for f in os.listdir(image_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            if images:
                if max_images:
                    images = images[:max_images]
                image_paths = [os.path.join(image_path, img) for img in images]
                print(f"✓ Found {len(image_paths)} images in directory: {image_path}")
                return image_paths
        
        # Try to use images from val2014 directory
        val2014_dir = "images/val2014"
        if os.path.exists(val2014_dir):
            images = sorted([f for f in os.listdir(val2014_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            if images:
                if max_images:
                    images = images[:max_images]
                image_paths = [os.path.join(val2014_dir, img) for img in images]
                print(f"✓ Found {len(image_paths)} images in val2014: {val2014_dir}")
                return image_paths
        
        raise FileNotFoundError(
            f"Could not find image at '{image_path}' or in '{val2014_dir}' directory. "
            f"Please ensure images are available in the val2014 directory."
        )
    
    def run_inference(
        self,
        image_path: str,
        model_format: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        img_size: int = 640
    ) -> Dict:
        """
        Run a single inference request
        
        Args:
            image_path: Path to the image file
            model_format: Model format to use (pytorch, onnx, tensorrt)
            conf_threshold: Confidence threshold
            iou_threshold: IOU threshold
            img_size: Input image size
            
        Returns:
            Dictionary containing inference results and timing
        """
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            data = {
                'model_format': model_format,
                'conf_threshold': conf_threshold,
                'iou_threshold': iou_threshold,
                'img_size': img_size
            }
            
            start_time = time.time()
            response = requests.post(self.api_url, files=files, data=data)
            end_time = time.time()
            
            total_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "total_time": total_time,
                    "inference_time": result.get("inference_time", 0),
                    "detections_count": result.get("detections_count", 0),
                    "model_format": model_format,
                    "response": result
                }
            else:
                return {
                    "success": False,
                    "total_time": total_time,
                    "error": response.text,
                    "model_format": model_format
                }
    
    def benchmark_model(
        self,
        image_path: str,
        model_format: str,
        num_iterations: int = 10,
        warmup_iterations: int = 3
    ) -> Dict:
        """
        Benchmark a specific model format
        
        Args:
            image_path: Path to the image file
            model_format: Model format to benchmark
            num_iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations (not counted)
            
        Returns:
            Dictionary containing benchmark statistics
        """
        print(f"\n🔥 Warming up {model_format} model ({warmup_iterations} iterations)...")
        for i in range(warmup_iterations):
            result = self.run_inference(image_path, model_format)
            if not result["success"]:
                print(f"  ✗ Warmup iteration {i+1} failed: {result.get('error', 'Unknown error')}")
                return None
            print(f"  Warmup {i+1}/{warmup_iterations}: {result['total_time']:.4f}s")
        
        print(f"\n⚡ Benchmarking {model_format} model ({num_iterations} iterations)...")
        
        inference_times = []
        total_times = []
        detections_counts = []
        
        for i in range(num_iterations):
            result = self.run_inference(image_path, model_format)
            
            if result["success"]:
                inference_times.append(result["inference_time"])
                total_times.append(result["total_time"])
                detections_counts.append(result["detections_count"])
                print(f"  Iteration {i+1}/{num_iterations}: "
                      f"inference={result['inference_time']:.4f}s, "
                      f"total={result['total_time']:.4f}s, "
                      f"detections={result['detections_count']}")
            else:
                print(f"  ✗ Iteration {i+1} failed: {result.get('error', 'Unknown error')}")
        
        if not inference_times:
            print(f"  ✗ No successful inferences for {model_format}")
            return None
        
        # Calculate statistics
        stats = {
            "model_format": model_format,
            "num_iterations": len(inference_times),
            "inference_time": {
                "mean": np.mean(inference_times),
                "std": np.std(inference_times),
                "min": np.min(inference_times),
                "max": np.max(inference_times),
                "median": np.median(inference_times),
                "p95": np.percentile(inference_times, 95),
                "p99": np.percentile(inference_times, 99),
                "all": inference_times
            },
            "total_time": {
                "mean": np.mean(total_times),
                "std": np.std(total_times),
                "min": np.min(total_times),
                "max": np.max(total_times),
                "median": np.median(total_times),
                "all": total_times
            },
            "detections": {
                "mean": np.mean(detections_counts),
                "all": detections_counts
            },
            "throughput_fps": 1.0 / np.mean(inference_times) if np.mean(inference_times) > 0 else 0
        }
        
        self.results[model_format] = stats
        
        print(f"\n📊 {model_format} Statistics:")
        print(f"  Inference Time: {stats['inference_time']['mean']:.4f}s ± {stats['inference_time']['std']:.4f}s")
        print(f"  Min/Max: {stats['inference_time']['min']:.4f}s / {stats['inference_time']['max']:.4f}s")
        print(f"  Median: {stats['inference_time']['median']:.4f}s")
        print(f"  P95/P99: {stats['inference_time']['p95']:.4f}s / {stats['inference_time']['p99']:.4f}s")
        print(f"  Throughput: {stats['throughput_fps']:.2f} FPS")
        print(f"  Detections: {stats['detections']['mean']:.1f} objects")
        
        return stats
    
    def run_full_benchmark(
        self,
        image_paths: List[str],
        model_formats: List[str] = None,
        num_iterations: int = 10,
        warmup_iterations: int = 3
    ) -> Dict[str, Dict]:
        """
        Run benchmark for all specified model formats across multiple images
        
        Args:
            image_paths: List of image file paths to test
            model_formats: List of model formats to benchmark (default: all)
            num_iterations: Number of benchmark iterations per model per image
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Dictionary containing all benchmark results
        """
        if model_formats is None:
            model_formats = ["pytorch", "onnx", "tensorrt"]
        
        print(f"\n{'='*60}")
        print(f"🚀 Starting Benchmark")
        print(f"{'='*60}")
        print(f"Images: {len(image_paths)} files")
        print(f"Models: {', '.join(model_formats)}")
        print(f"Iterations per image: {num_iterations} (+ {warmup_iterations} warmup)")
        print(f"Total tests: {len(image_paths)} images × {len(model_formats)} models × {num_iterations} iterations")
        print(f"{'='*60}")
        
        # Aggregate results across all images
        aggregated_results = {fmt: {
            "inference_times": [],
            "total_times": [],
            "detections_counts": [],
            "num_images": 0,
            "num_iterations": 0
        } for fmt in model_formats}
        
        for idx, image_path in enumerate(image_paths, 1):
            print(f"\n{'─'*60}")
            print(f"📸 Processing image {idx}/{len(image_paths)}: {os.path.basename(image_path)}")
            print(f"{'─'*60}")
            
            for model_format in model_formats:
                # Warmup only on first image
                warmup = warmup_iterations if idx == 1 else 0
                
                if warmup > 0:
                    print(f"\n🔥 Warming up {model_format} model ({warmup} iterations)...")
                    for i in range(warmup):
                        result = self.run_inference(image_path, model_format)
                        if not result["success"]:
                            print(f"  ✗ Warmup iteration {i+1} failed: {result.get('error', 'Unknown error')}")
                        else:
                            print(f"  Warmup {i+1}/{warmup}: {result['total_time']:.4f}s")
                
                print(f"\n⚡ Testing {model_format} on image {idx} ({num_iterations} iterations)...")
                
                for i in range(num_iterations):
                    result = self.run_inference(image_path, model_format)
                    
                    if result["success"]:
                        aggregated_results[model_format]["inference_times"].append(result["inference_time"])
                        aggregated_results[model_format]["total_times"].append(result["total_time"])
                        aggregated_results[model_format]["detections_counts"].append(result["detections_count"])
                        
                        if (i + 1) % 5 == 0 or i == 0 or i == num_iterations - 1:
                            print(f"  Iteration {i+1}/{num_iterations}: "
                                  f"inference={result['inference_time']:.4f}s, "
                                  f"detections={result['detections_count']}")
                    else:
                        print(f"  ✗ Iteration {i+1} failed: {result.get('error', 'Unknown error')}")
                
                aggregated_results[model_format]["num_images"] = idx
                aggregated_results[model_format]["num_iterations"] = len(aggregated_results[model_format]["inference_times"])
        
        # Calculate final statistics
        benchmark_results = {}
        for model_format in model_formats:
            data = aggregated_results[model_format]
            if data["inference_times"]:
                inference_times = data["inference_times"]
                total_times = data["total_times"]
                detections_counts = data["detections_counts"]
                
                stats = {
                    "model_format": model_format,
                    "num_images": data["num_images"],
                    "num_iterations": data["num_iterations"],
                    "inference_time": {
                        "mean": np.mean(inference_times),
                        "std": np.std(inference_times),
                        "min": np.min(inference_times),
                        "max": np.max(inference_times),
                        "median": np.median(inference_times),
                        "p95": np.percentile(inference_times, 95),
                        "p99": np.percentile(inference_times, 99),
                        "all": inference_times
                    },
                    "total_time": {
                        "mean": np.mean(total_times),
                        "std": np.std(total_times),
                        "min": np.min(total_times),
                        "max": np.max(total_times),
                        "median": np.median(total_times),
                        "all": total_times
                    },
                    "detections": {
                        "mean": np.mean(detections_counts),
                        "all": detections_counts
                    },
                    "throughput_fps": 1.0 / np.mean(inference_times) if np.mean(inference_times) > 0 else 0
                }
                
                benchmark_results[model_format] = stats
                self.results[model_format] = stats
        
        return benchmark_results
    
    def save_results(self, output_dir: str = "benchmark_results"):
        """Save benchmark results to JSON file"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"benchmark_{timestamp}.json")
        
        # Prepare results for JSON serialization
        json_results = {}
        for model_format, stats in self.results.items():
            if stats:
                json_results[model_format] = {
                    "model_format": stats["model_format"],
                    "num_iterations": stats["num_iterations"],
                    "inference_time_mean": stats["inference_time"]["mean"],
                    "inference_time_std": stats["inference_time"]["std"],
                    "inference_time_min": stats["inference_time"]["min"],
                    "inference_time_max": stats["inference_time"]["max"],
                    "inference_time_median": stats["inference_time"]["median"],
                    "inference_time_p95": stats["inference_time"]["p95"],
                    "inference_time_p99": stats["inference_time"]["p99"],
                    "throughput_fps": stats["throughput_fps"],
                    "detections_mean": stats["detections"]["mean"]
                }
        
        with open(output_file, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "results": json_results
            }, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
        return output_file
    
    def plot_results(self, output_dir: str = "benchmark_results"):
        """Generate performance visualization plots"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter out empty results
        valid_results = {k: v for k, v in self.results.items() if v}
        
        if not valid_results:
            print("No results to plot")
            return
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Mean Inference Time Comparison (Bar Chart)
        ax1 = plt.subplot(2, 3, 1)
        models = list(valid_results.keys())
        mean_times = [valid_results[m]["inference_time"]["mean"] * 1000 for m in models]  # Convert to ms
        std_times = [valid_results[m]["inference_time"]["std"] * 1000 for m in models]
        
        colors = {'pytorch': '#EE4C2C', 'onnx': '#5E9ED6', 'tensorrt': '#76B900'}
        bar_colors = [colors.get(m, '#888888') for m in models]
        
        bars = ax1.bar(models, mean_times, yerr=std_times, capsize=5, color=bar_colors, alpha=0.8)
        ax1.set_ylabel('Inference Time (ms)', fontsize=11)
        ax1.set_title('Mean Inference Time Comparison', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, mean_time in zip(bars, mean_times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean_time:.2f}ms',
                    ha='center', va='bottom', fontsize=9)
        
        # 2. Throughput Comparison (FPS)
        ax2 = plt.subplot(2, 3, 2)
        fps_values = [valid_results[m]["throughput_fps"] for m in models]
        bars = ax2.bar(models, fps_values, color=bar_colors, alpha=0.8)
        ax2.set_ylabel('Throughput (FPS)', fontsize=11)
        ax2.set_title('Throughput Comparison', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, fps in zip(bars, fps_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{fps:.1f}',
                    ha='center', va='bottom', fontsize=9)
        
        # 3. Inference Time Distribution (Box Plot)
        ax3 = plt.subplot(2, 3, 3)
        data_to_plot = [np.array(valid_results[m]["inference_time"]["all"]) * 1000 for m in models]
        bp = ax3.boxplot(data_to_plot, labels=models, patch_artist=True)
        
        # Color the box plots
        for patch, model in zip(bp['boxes'], models):
            patch.set_facecolor(colors.get(model, '#888888'))
            patch.set_alpha(0.6)
        
        ax3.set_ylabel('Inference Time (ms)', fontsize=11)
        ax3.set_title('Inference Time Distribution', fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Latency Percentiles
        ax4 = plt.subplot(2, 3, 4)
        percentiles = ['mean', 'median', 'p95', 'p99']
        x = np.arange(len(percentiles))
        width = 0.25
        
        for i, model in enumerate(models):
            values = [
                valid_results[model]["inference_time"]["mean"] * 1000,
                valid_results[model]["inference_time"]["median"] * 1000,
                valid_results[model]["inference_time"]["p95"] * 1000,
                valid_results[model]["inference_time"]["p99"] * 1000
            ]
            offset = (i - len(models)/2 + 0.5) * width
            ax4.bar(x + offset, values, width, label=model, 
                   color=colors.get(model, '#888888'), alpha=0.8)
        
        ax4.set_ylabel('Inference Time (ms)', fontsize=11)
        ax4.set_title('Latency Percentiles', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(['Mean', 'Median', 'P95', 'P99'])
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. Inference Time Over Iterations
        ax5 = plt.subplot(2, 3, 5)
        for model in models:
            times = np.array(valid_results[model]["inference_time"]["all"]) * 1000
            iterations = range(1, len(times) + 1)
            ax5.plot(iterations, times, marker='o', label=model, 
                    color=colors.get(model, '#888888'), linewidth=2, markersize=4)
        
        ax5.set_xlabel('Iteration', fontsize=11)
        ax5.set_ylabel('Inference Time (ms)', fontsize=11)
        ax5.set_title('Inference Time Over Iterations', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Speedup Comparison (relative to PyTorch baseline)
        ax6 = plt.subplot(2, 3, 6)
        if 'pytorch' in valid_results:
            baseline_time = valid_results['pytorch']['inference_time']['mean']
            speedups = []
            speedup_models = []
            
            for model in models:
                if model != 'pytorch':
                    speedup = baseline_time / valid_results[model]['inference_time']['mean']
                    speedups.append(speedup)
                    speedup_models.append(model)
            
            if speedups:
                bars = ax6.bar(speedup_models, speedups, 
                             color=[colors.get(m, '#888888') for m in speedup_models], 
                             alpha=0.8)
                ax6.axhline(y=1.0, color='red', linestyle='--', label='PyTorch baseline', linewidth=2)
                ax6.set_ylabel('Speedup (×)', fontsize=11)
                ax6.set_title('Speedup vs PyTorch Baseline', fontsize=12, fontweight='bold')
                ax6.legend()
                ax6.grid(axis='y', alpha=0.3)
                
                # Add value labels
                for bar, speedup in zip(bars, speedups):
                    height = bar.get_height()
                    ax6.text(bar.get_x() + bar.get_width()/2., height,
                            f'{speedup:.2f}×',
                            ha='center', va='bottom', fontsize=9)
        else:
            ax6.text(0.5, 0.5, 'PyTorch baseline not available', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Speedup vs PyTorch Baseline', fontsize=12, fontweight='bold')
        
        plt.suptitle('YOLO Model Format Performance Benchmark', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"benchmark_plot_{timestamp}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {output_file}")
        
        # Also save as PDF for high quality
        output_file_pdf = os.path.join(output_dir, f"benchmark_plot_{timestamp}.pdf")
        plt.savefig(output_file_pdf, bbox_inches='tight')
        print(f"✓ Plot saved to: {output_file_pdf}")
        
        plt.show()
        
        return output_file


def main():
    """Main entry point for the benchmark script"""
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO API performance across different model formats"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the API server (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="images/val2014",
        help="Path to test image or directory (default: images/val2014)"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to test (default: all images in directory)"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["pytorch", "onnx", "tensorrt"],
        choices=["pytorch", "onnx", "tensorrt"],
        help="Model formats to benchmark (default: all)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of benchmark iterations (default: 10)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations (default: 3)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Output directory for results (default: benchmark_results)"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plots"
    )
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = APIBenchmark(base_url=args.url)
    
    # Check server health
    print("🔍 Checking server health...")
    if not benchmark.check_server_health():
        print("\n❌ Server is not available. Please start it first with: ./run_app.sh")
        return 1
    
    # Check models availability
    models_status = benchmark.check_models_availability()
    
    # Filter requested models by availability
    available_models = [m for m in args.models if models_status.get(m, False)]
    
    if not available_models:
        print(f"\n❌ None of the requested models are available: {args.models}")
        print("Please ensure models are downloaded. Run: python setup_models.py")
        return 1
    
    if len(available_models) < len(args.models):
        missing = set(args.models) - set(available_models)
        print(f"\n⚠️  Warning: Some models are not available: {missing}")
        print(f"Will benchmark only: {available_models}")
    
    # Get test images
    try:
        image_paths = benchmark.get_image_paths(args.image, max_images=args.max_images)
        if not image_paths:
            print(f"\n❌ No images found at: {args.image}")
            return 1
    except Exception as e:
        print(f"\n❌ Error loading images: {e}")
        return 1
    
    # Run benchmark
    results = benchmark.run_full_benchmark(
        image_paths=image_paths,
        model_formats=available_models,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup
    )
    
    if not results:
        print("\n❌ Benchmark failed - no results collected")
        return 1
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Total Images Tested: {len(image_paths)}")
    print(f"Total Tests per Model: {len(image_paths) * args.iterations}")
    print(f"{'='*60}")
    
    for model_format, stats in results.items():
        print(f"\n{model_format.upper()}:")
        print(f"  Images Processed: {stats['num_images']}")
        print(f"  Total Iterations: {stats['num_iterations']}")
        print(f"  Inference Time: {stats['inference_time']['mean']*1000:.2f} ms ± {stats['inference_time']['std']*1000:.2f} ms")
        print(f"  Throughput: {stats['throughput_fps']:.2f} FPS")
        print(f"  Min/Max: {stats['inference_time']['min']*1000:.2f} / {stats['inference_time']['max']*1000:.2f} ms")
        print(f"  Median: {stats['inference_time']['median']*1000:.2f} ms")
        print(f"  P95/P99: {stats['inference_time']['p95']*1000:.2f} / {stats['inference_time']['p99']*1000:.2f} ms")
        print(f"  Avg Detections: {stats['detections']['mean']:.1f}")
    
    # Calculate and print speedups
    if 'pytorch' in results:
        baseline = results['pytorch']['inference_time']['mean']
        print(f"\n⚡ SPEEDUP vs PyTorch:")
        for model_format, stats in results.items():
            if model_format != 'pytorch':
                speedup = baseline / stats['inference_time']['mean']
                print(f"  {model_format}: {speedup:.2f}×")
    
    # Save results
    benchmark.save_results(args.output_dir)
    
    # Generate plots
    if not args.no_plot:
        print("\n📈 Generating performance plots...")
        try:
            benchmark.plot_results(args.output_dir)
        except Exception as e:
            print(f"⚠️  Warning: Could not generate plots: {e}")
    
    print(f"\n{'='*60}")
    print("✅ Benchmark completed successfully!")
    print(f"{'='*60}\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
