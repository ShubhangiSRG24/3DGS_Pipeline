"""
Adaptive Batch Size Optimizer for maximum GPU utilization
Dynamically adjusts batch sizes based on memory availability and workload characteristics
"""

import torch
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import psutil

logger = logging.getLogger(__name__)


@dataclass
class BatchPerformanceMetric:
    """Performance metrics for a specific batch configuration"""
    batch_size: int
    processing_time: float
    memory_usage: int  # bytes
    throughput: float  # items per second
    memory_efficiency: float  # throughput per MB
    timestamp: float


class AdaptiveBatchOptimizer:
    """
    Advanced adaptive batch size optimizer with learning capabilities
    Automatically finds optimal batch sizes for different workload patterns
    """
    
    def __init__(self, device: torch.device, initial_batch_size: int = 8):
        self.device = device
        self.initial_batch_size = initial_batch_size
        self.use_gpu = device.type == 'cuda' and torch.cuda.is_available()
        
        # Learning history
        self.performance_history: deque = deque(maxlen=100)
        self.optimal_batch_sizes: Dict[str, int] = {}  # workload_pattern -> batch_size
        
        # Adaptive parameters
        self.min_batch_size = 1
        self.max_batch_size = 64
        self.safety_margin = 0.8  # Use 80% of available memory
        self.learning_rate = 0.1
        
        # Performance tracking
        self.current_batch_size = initial_batch_size
        self.consecutive_oom_count = 0
        self.consecutive_success_count = 0
        
        # System monitoring
        self.memory_overhead_estimate = 1.5  # Estimate 50% overhead for operations
        
        logger.info(f"Adaptive Batch Optimizer initialized on {device}")
    
    def get_workload_pattern(self, avg_image_size: Tuple[int, int], 
                           num_images: int, feature_type: str) -> str:
        """Generate a workload pattern key for caching optimal batch sizes"""
        h, w = avg_image_size
        size_category = self._categorize_image_size(h, w)
        count_category = self._categorize_image_count(num_images)
        return f"{feature_type}_{size_category}_{count_category}"
    
    def _categorize_image_size(self, height: int, width: int) -> str:
        """Categorize image size for pattern matching"""
        pixels = height * width
        if pixels < 640 * 480:
            return "small"
        elif pixels < 1280 * 720:
            return "medium"
        elif pixels < 1920 * 1080:
            return "large"
        else:
            return "xlarge"
    
    def _categorize_image_count(self, count: int) -> str:
        """Categorize dataset size for pattern matching"""
        if count < 50:
            return "tiny"
        elif count < 200:
            return "small"
        elif count < 1000:
            return "medium"
        else:
            return "large"
    
    def calculate_optimal_batch_size(self, avg_image_size: Tuple[int, int], 
                                   num_images: int, feature_type: str = "superpoint") -> int:
        """
        Calculate optimal batch size using multiple strategies
        """
        # Check cached optimal size for this workload pattern
        pattern = self.get_workload_pattern(avg_image_size, num_images, feature_type)
        if pattern in self.optimal_batch_sizes:
            cached_size = self.optimal_batch_sizes[pattern]
            logger.info(f"Using cached optimal batch size: {cached_size} for pattern {pattern}")
            return cached_size
        
        # Calculate base batch size using multiple methods
        memory_based = self._calculate_memory_based_batch_size(avg_image_size, feature_type)
        performance_based = self._calculate_performance_based_batch_size()
        conservative = self._calculate_conservative_batch_size(avg_image_size)
        
        # Choose the most conservative estimate
        candidates = [memory_based, performance_based, conservative]
        optimal_size = min(filter(lambda x: x > 0, candidates))
        
        # Ensure within bounds
        optimal_size = max(self.min_batch_size, min(optimal_size, self.max_batch_size))
        
        logger.info(f"Calculated optimal batch size: {optimal_size} "
                   f"(memory: {memory_based}, perf: {performance_based}, conservative: {conservative})")
        
        return optimal_size
    
    def _calculate_memory_based_batch_size(self, avg_image_size: Tuple[int, int], 
                                         feature_type: str) -> int:
        """Calculate batch size based on available GPU memory"""
        if not self.use_gpu:
            return 4  # Conservative for CPU
        
        try:
            # Get GPU memory info
            gpu_properties = torch.cuda.get_device_properties(0)
            total_memory = gpu_properties.total_memory
            allocated_memory = torch.cuda.memory_allocated()
            reserved_memory = torch.cuda.memory_reserved()
            
            # Calculate available memory
            available_memory = total_memory - max(allocated_memory, reserved_memory)
            usable_memory = available_memory * self.safety_margin
            
            # Estimate memory per image
            h, w = avg_image_size
            
            # Base memory for image tensor (RGB, float32)
            image_memory = h * w * 3 * 4  # 3 channels, 4 bytes per float
            
            # Feature extraction overhead (varies by extractor)
            if feature_type == "superpoint":
                # SuperPoint processes at multiple scales internally
                extraction_overhead = 3.0
            elif feature_type == "aliked":
                extraction_overhead = 2.5
            elif feature_type == "disk":
                extraction_overhead = 3.5
            else:
                extraction_overhead = 3.0
            
            # Total memory per image including overhead
            memory_per_image = image_memory * extraction_overhead * self.memory_overhead_estimate
            
            # Calculate batch size
            max_batch_size = int(usable_memory // memory_per_image)
            
            return max(1, min(max_batch_size, self.max_batch_size))
            
        except Exception as e:
            logger.warning(f"Failed to calculate memory-based batch size: {e}")
            return self.initial_batch_size
    
    def _calculate_performance_based_batch_size(self) -> int:
        """Calculate batch size based on historical performance"""
        if len(self.performance_history) < 3:
            return self.initial_batch_size
        
        # Find the batch size with best memory efficiency from recent history
        recent_metrics = list(self.performance_history)[-10:]  # Last 10 measurements
        
        if not recent_metrics:
            return self.initial_batch_size
        
        # Sort by memory efficiency (throughput per MB)
        best_metric = max(recent_metrics, key=lambda x: x.memory_efficiency)
        
        # Slightly increase from the best known size to explore better options
        exploration_factor = 1.2
        suggested_size = int(best_metric.batch_size * exploration_factor)
        
        return max(1, min(suggested_size, self.max_batch_size))
    
    def _calculate_conservative_batch_size(self, avg_image_size: Tuple[int, int]) -> int:
        """Calculate conservative batch size based on image characteristics"""
        h, w = avg_image_size
        pixels = h * w
        
        # Conservative estimates based on image size
        if pixels < 640 * 480:  # Small images
            return min(32, self.max_batch_size)
        elif pixels < 1280 * 720:  # Medium images
            return min(16, self.max_batch_size)
        elif pixels < 1920 * 1080:  # Large images
            return min(8, self.max_batch_size)
        else:  # Very large images
            return min(4, self.max_batch_size)
    
    def adapt_batch_size(self, current_performance: BatchPerformanceMetric, 
                        oom_occurred: bool = False) -> int:
        """
        Adapt batch size based on current performance and memory events
        """
        # Record performance
        self.performance_history.append(current_performance)
        
        if oom_occurred:
            self.consecutive_oom_count += 1
            self.consecutive_success_count = 0
            
            # Aggressively reduce batch size on OOM
            reduction_factor = 0.5 if self.consecutive_oom_count == 1 else 0.7
            new_batch_size = max(1, int(self.current_batch_size * reduction_factor))
            
            logger.warning(f"OOM detected, reducing batch size: {self.current_batch_size} -> {new_batch_size}")
        else:
            self.consecutive_success_count += 1
            self.consecutive_oom_count = 0
            
            # Gradually increase batch size if consistently successful
            if self.consecutive_success_count >= 5:
                # Check if we can safely increase
                if current_performance.memory_efficiency > 0:
                    increase_factor = 1.25  # 25% increase
                    new_batch_size = min(self.max_batch_size, 
                                       int(self.current_batch_size * increase_factor))
                    
                    if new_batch_size > self.current_batch_size:
                        logger.info(f"Performance stable, increasing batch size: "
                                   f"{self.current_batch_size} -> {new_batch_size}")
                    else:
                        new_batch_size = self.current_batch_size
                else:
                    new_batch_size = self.current_batch_size
            else:
                new_batch_size = self.current_batch_size
        
        self.current_batch_size = new_batch_size
        return new_batch_size
    
    def record_optimal_batch_size(self, pattern: str, batch_size: int) -> None:
        """Record optimal batch size for a specific workload pattern"""
        self.optimal_batch_sizes[pattern] = batch_size
        logger.info(f"Recorded optimal batch size {batch_size} for pattern {pattern}")
    
    def benchmark_batch_sizes(self, test_function, test_data: List[Any], 
                            image_size: Tuple[int, int], feature_type: str) -> int:
        """
        Benchmark different batch sizes to find the optimal one
        """
        logger.info("Benchmarking batch sizes for optimal performance...")
        
        # Test different batch sizes
        test_sizes = [1, 2, 4, 8, 16, 32]
        test_sizes = [size for size in test_sizes if size <= len(test_data)]
        
        benchmark_results = []
        
        for batch_size in test_sizes:
            try:
                # Prepare test batch
                test_batch = test_data[:batch_size]
                
                # Warm up
                if self.use_gpu:
                    torch.cuda.synchronize()
                
                # Measure performance
                start_time = time.time()
                start_memory = torch.cuda.memory_allocated() if self.use_gpu else 0
                
                # Run test function
                result = test_function(test_batch)
                
                if self.use_gpu:
                    torch.cuda.synchronize()
                
                end_time = time.time()
                end_memory = torch.cuda.memory_allocated() if self.use_gpu else 0
                
                # Calculate metrics
                processing_time = end_time - start_time
                memory_used = end_memory - start_memory
                throughput = batch_size / processing_time
                memory_efficiency = throughput / max(1, memory_used / 1024**2)  # throughput per MB
                
                metric = BatchPerformanceMetric(
                    batch_size=batch_size,
                    processing_time=processing_time,
                    memory_usage=memory_used,
                    throughput=throughput,
                    memory_efficiency=memory_efficiency,
                    timestamp=time.time()
                )
                
                benchmark_results.append(metric)
                
                logger.info(f"Batch size {batch_size}: {throughput:.2f} items/sec, "
                           f"Memory efficiency: {memory_efficiency:.4f}")
                
                # Clean up
                if self.use_gpu:
                    torch.cuda.empty_cache()
                
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"OOM at batch size {batch_size}")
                break
            except Exception as e:
                logger.warning(f"Error testing batch size {batch_size}: {e}")
                continue
        
        if not benchmark_results:
            logger.warning("No successful benchmark results, using default batch size")
            return self.initial_batch_size
        
        # Find optimal batch size (best memory efficiency)
        optimal_metric = max(benchmark_results, key=lambda x: x.memory_efficiency)
        optimal_batch_size = optimal_metric.batch_size
        
        # Cache this result
        pattern = self.get_workload_pattern(image_size, len(test_data), feature_type)
        self.record_optimal_batch_size(pattern, optimal_batch_size)
        
        logger.info(f"Optimal batch size determined: {optimal_batch_size} "
                   f"(efficiency: {optimal_metric.memory_efficiency:.4f})")
        
        return optimal_batch_size
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory and performance statistics"""
        stats = {
            'current_batch_size': self.current_batch_size,
            'consecutive_success_count': self.consecutive_success_count,
            'consecutive_oom_count': self.consecutive_oom_count,
            'performance_history_length': len(self.performance_history),
            'cached_patterns': len(self.optimal_batch_sizes),
            'optimal_batch_sizes': dict(self.optimal_batch_sizes)
        }
        
        if self.use_gpu:
            stats.update({
                'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
                'gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**2,   # MB
                'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
            })
        
        # Add recent performance metrics
        if self.performance_history:
            recent_metrics = list(self.performance_history)[-5:]
            avg_throughput = np.mean([m.throughput for m in recent_metrics])
            avg_memory_efficiency = np.mean([m.memory_efficiency for m in recent_metrics])
            
            stats.update({
                'avg_throughput': avg_throughput,
                'avg_memory_efficiency': avg_memory_efficiency
            })
        
        return stats
    
    def reset(self):
        """Reset the optimizer state"""
        self.performance_history.clear()
        self.current_batch_size = self.initial_batch_size
        self.consecutive_oom_count = 0
        self.consecutive_success_count = 0
        logger.info("Adaptive batch optimizer reset")


# Global optimizer instance
_global_batch_optimizer: Optional[AdaptiveBatchOptimizer] = None


def get_global_batch_optimizer(device: torch.device = None, 
                              initial_batch_size: int = 8) -> AdaptiveBatchOptimizer:
    """Get or create global batch optimizer instance"""
    global _global_batch_optimizer
    
    if _global_batch_optimizer is None:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _global_batch_optimizer = AdaptiveBatchOptimizer(device, initial_batch_size)
    
    return _global_batch_optimizer