"""
Feature extractors for SfM pipeline
Supports SuperPoint, ALIKED, and DISK
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc
import time
from typing import Dict, List, Tuple, Any, Optional
from abc import ABC, abstractmethod
import cv2
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Don't import LightGlue at module level - use lazy imports
try:
    from .gpu_memory_pool import GPUMemoryPool, GPUMemoryContext
    GPU_MEMORY_POOL_AVAILABLE = True
except ImportError:
    GPU_MEMORY_POOL_AVAILABLE = False
    GPUMemoryPool = GPUMemoryContext = None

try:
    from .adaptive_batch_optimizer import AdaptiveBatchOptimizer, BatchPerformanceMetric
    ADAPTIVE_BATCH_AVAILABLE = True
except ImportError:
    ADAPTIVE_BATCH_AVAILABLE = False
    AdaptiveBatchOptimizer = BatchPerformanceMetric = None

logger = logging.getLogger(__name__)


class BaseFeatureExtractor(ABC):
    """Base class for feature extractors"""
    
    def __init__(self, device: torch.device, config: Dict[str, Any] = None):
        self.device = device
        self.config = config or {}
        self.model = None
        
        # Performance enhancements
        self.use_multi_scale = config.get('use_multi_scale', True)
        self.scale_factors = config.get('scale_factors', [1.0, 0.8, 1.2])  # Multi-scale processing
        self.adaptive_batch_size = config.get('adaptive_batch_size', True)
        self.memory_efficient = config.get('memory_efficient', True)
        
        # GPU memory management
        self.memory_pool_size = config.get('memory_pool_size', 512 * 1024 * 1024)  # 512MB pool
        self.max_batch_memory = config.get('max_batch_memory', 2 * 1024 * 1024 * 1024)  # 2GB limit
        
        # Initialize GPU memory pool
        self.memory_pool = None
        if GPU_MEMORY_POOL_AVAILABLE and self.device.type == 'cuda':
            try:
                self.memory_pool = GPUMemoryPool(self.device, pool_size_mb=512)
                logger.info("GPU Memory Pool initialized for feature extraction")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU memory pool: {e}")
        
        # Initialize adaptive batch optimizer
        self.batch_optimizer = None
        if ADAPTIVE_BATCH_AVAILABLE:
            try:
                self.batch_optimizer = AdaptiveBatchOptimizer(self.device, initial_batch_size=8)
                logger.info("Adaptive Batch Optimizer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize adaptive batch optimizer: {e}")
        
        self._setup_model()
    
    @abstractmethod
    def _setup_model(self):
        """Setup the feature extraction model"""
        pass
    
    @abstractmethod
    def extract_features(self, images: List[Dict], batch_size: int = 8) -> Dict[str, Any]:
        """Extract features from images"""
        pass
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for feature extraction"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return image_tensor.to(self.device)
    
    def _create_image_pyramid(self, image: np.ndarray) -> List[np.ndarray]:
        """Create multi-scale image pyramid for robust feature extraction"""
        if not self.use_multi_scale:
            return [image]
        
        pyramid = []
        h, w = image.shape[:2]
        
        for scale in self.scale_factors:
            if scale == 1.0:
                pyramid.append(image)
            else:
                new_h, new_w = int(h * scale), int(w * scale)
                if new_h > 32 and new_w > 32:  # Minimum size check
                    scaled_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    pyramid.append(scaled_img)
        
        return pyramid
    
    def _get_adaptive_batch_size(self, total_images: int, avg_image_size: Tuple[int, int], 
                                feature_type: str = "superpoint") -> int:
        """Calculate optimal batch size using advanced optimization"""
        if not self.adaptive_batch_size:
            return 8  # Default batch size
        
        # Use advanced batch optimizer if available
        if self.batch_optimizer:
            try:
                return self.batch_optimizer.calculate_optimal_batch_size(
                    avg_image_size, total_images, feature_type
                )
            except Exception as e:
                logger.warning(f"Failed to use advanced batch optimizer: {e}")
        
        # Fallback to original method
        try:
            if torch.cuda.is_available():
                # Get available GPU memory
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated()
                available_memory = gpu_memory - allocated_memory
                
                # Estimate memory per image (improved calculation)
                h, w = avg_image_size
                memory_per_image = h * w * 4 * 3  # float32 * 3 scales rough estimate
                
                # Conservative batch size calculation (use 70% of available memory)
                safe_memory = available_memory * 0.7
                optimal_batch_size = max(1, int(safe_memory // memory_per_image))
                
                # Clamp to reasonable range
                return min(max(optimal_batch_size, 1), 32)
            else:
                return 4  # Conservative for CPU
        except Exception as e:
            logger.warning(f"Failed to calculate adaptive batch size: {e}")
            return 8
    
    def _merge_multi_scale_features(self, feature_sets: List[Dict], original_size: Tuple[int, int]) -> Dict:
        """Merge features from different scales into a single robust set"""
        if len(feature_sets) == 1:
            return feature_sets[0]
        
        # Combine keypoints and descriptors from all scales
        all_keypoints = []
        all_descriptors = []
        all_scores = []
        
        h_orig, w_orig = original_size
        
        for i, (features, scale) in enumerate(zip(feature_sets, self.scale_factors)):
            keypoints = features['keypoints']
            descriptors = features['descriptors']
            scores = features['scores']
            
            # Scale keypoints back to original image coordinates
            if scale != 1.0:
                scaled_keypoints = keypoints.copy()
                scaled_keypoints[:, 0] /= scale  # x coordinates
                scaled_keypoints[:, 1] /= scale  # y coordinates
            else:
                scaled_keypoints = keypoints
            
            # Filter keypoints within original image bounds
            valid_mask = (
                (scaled_keypoints[:, 0] >= 0) & (scaled_keypoints[:, 0] < w_orig) &
                (scaled_keypoints[:, 1] >= 0) & (scaled_keypoints[:, 1] < h_orig)
            )
            
            if valid_mask.any():
                all_keypoints.append(scaled_keypoints[valid_mask])
                all_descriptors.append(descriptors[valid_mask])
                all_scores.append(scores[valid_mask])
        
        if not all_keypoints:
            # Fallback to empty features
            return {
                'keypoints': np.array([]).reshape(0, 2),
                'descriptors': np.array([]).reshape(0, self.config.get('descriptor_dim', 256)),
                'scores': np.array([]),
                'image_shape': original_size
            }
        
        # Concatenate all features
        merged_keypoints = np.vstack(all_keypoints)
        merged_descriptors = np.vstack(all_descriptors)
        merged_scores = np.concatenate(all_scores)
        
        # Apply Non-Maximum Suppression to remove duplicate keypoints
        final_keypoints, final_descriptors, final_scores = self._apply_nms(
            merged_keypoints, merged_descriptors, merged_scores
        )
        
        return {
            'keypoints': final_keypoints,
            'descriptors': final_descriptors,
            'scores': final_scores,
            'image_shape': original_size
        }
    
    def _apply_nms(self, keypoints: np.ndarray, descriptors: np.ndarray, 
                   scores: np.ndarray, radius: float = 4.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply Non-Maximum Suppression to remove duplicate keypoints"""
        if len(keypoints) == 0:
            return keypoints, descriptors, scores
        
        # Sort by score (descending)
        sort_indices = np.argsort(scores)[::-1]
        sorted_keypoints = keypoints[sort_indices]
        sorted_descriptors = descriptors[sort_indices]
        sorted_scores = scores[sort_indices]
        
        # Apply NMS
        keep_indices = []
        for i in range(len(sorted_keypoints)):
            kp = sorted_keypoints[i]
            
            # Check if this keypoint is too close to any already kept keypoint
            keep = True
            for kept_idx in keep_indices:
                kept_kp = sorted_keypoints[kept_idx]
                distance = np.linalg.norm(kp - kept_kp)
                if distance < radius:
                    keep = False
                    break
            
            if keep:
                keep_indices.append(i)
        
        # Apply max keypoints limit
        max_keypoints = self.config.get('max_keypoints', 4096)
        keep_indices = keep_indices[:max_keypoints]
        
        return (
            sorted_keypoints[keep_indices],
            sorted_descriptors[keep_indices],
            sorted_scores[keep_indices]
        )

    def clear_memory(self):
        """Clear GPU memory used by the feature extractor"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info(f"{self.__class__.__name__} memory cleared")
        except Exception as e:
            logger.warning(f"Error clearing {self.__class__.__name__} memory: {e}")


class SuperPointExtractor(BaseFeatureExtractor):
    """SuperPoint feature extractor"""
    
    def _setup_model(self):
        """Setup SuperPoint model"""
        try:
            # Import only SuperPoint, avoid SIFT module that needs pycolmap
            import lightglue
            from lightglue.superpoint import SuperPoint
            
            # SuperPoint configuration matching the provided format
            superpoint_conf = {
                "nms_radius": self.config.get('nms_radius', 4),
                "keypoint_threshold": self.config.get('keypoint_threshold', 0.005),  
                "max_keypoints": self.config.get('max_keypoints', 4096),  
                "remove_borders": self.config.get('remove_borders', 4),
                "fix_sampling": self.config.get('fix_sampling', False),
            }
            
            self.model = SuperPoint(
                nms_radius=superpoint_conf["nms_radius"],
                keypoint_threshold=superpoint_conf["keypoint_threshold"],
                max_keypoints=superpoint_conf["max_keypoints"],
                remove_borders=superpoint_conf["remove_borders"]
            ).eval().to(self.device)
        except ImportError as e:
            raise ImportError(f"LightGlue SuperPoint not available: {e}. Try: pip install lightglue @ git+https://github.com/cvg/LightGlue.git")
    
    def extract_features(self, images: List[Dict], batch_size: int = 8) -> Dict[str, Any]:
        """Extract SuperPoint features from images with multi-scale optimization"""
        if not images:
            return {}
        
        # Calculate adaptive batch size with feature type
        avg_size = np.mean([img['image'].shape[:2] for img in images], axis=0).astype(int)
        feature_type = self.__class__.__name__.lower().replace('extractor', '')
        adaptive_batch_size = self._get_adaptive_batch_size(len(images), tuple(avg_size), feature_type)
        actual_batch_size = min(batch_size, adaptive_batch_size)
        
        logger.info(f"Using adaptive batch size: {actual_batch_size} (requested: {batch_size}) for {feature_type}")
        
        features = {}
        processing_times = []
        
        for i in tqdm(range(0, len(images), actual_batch_size), desc="Extracting features"):
            batch_start = time.time()
            batch = images[i:i + actual_batch_size]
            
            # Process batch with multi-scale extraction
            batch_features = self._extract_batch_multiscale(batch)
            features.update(batch_features)
            
            batch_time = time.time() - batch_start
            processing_times.append(batch_time)
            
            # Memory cleanup after each batch
            if self.memory_efficient:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Log performance statistics
        avg_time = np.mean(processing_times) if processing_times else 0
        total_time = sum(processing_times)
        logger.info(f"Feature extraction completed: {len(features)} images in {total_time:.2f}s ")
        logger.info(f"Average time per batch: {avg_time:.3f}s, Images per second: {len(images)/total_time:.1f}")
        
        return features
    
    def _extract_batch_multiscale(self, batch: List[Dict]) -> Dict[str, Any]:
        """Extract features from a batch with multi-scale processing"""
        batch_features = {}
        
        for img_data in batch:
            image = img_data['image']
            image_path = img_data['path']
            original_size = image.shape[:2]
            
            # Create image pyramid
            image_pyramid = self._create_image_pyramid(image)
            
            # Extract features at each scale
            scale_features = []
            for scale_img, scale_factor in zip(image_pyramid, self.scale_factors[:len(image_pyramid)]):
                # Preprocess image
                image_tensor = self._preprocess_image(scale_img)
                
                # Extract features
                with torch.no_grad():
                    input_data = {'image': image_tensor}
                    pred = self.model(input_data)
                
                # Convert to numpy arrays
                keypoints = pred['keypoints'][0].cpu().numpy()
                descriptors = pred['descriptors'][0].cpu().numpy()
                scores = pred['keypoint_scores'][0].cpu().numpy()
                
                scale_features.append({
                    'keypoints': keypoints,
                    'descriptors': descriptors,
                    'scores': scores,
                    'image_shape': scale_img.shape[:2]
                })
                
                # Clean up intermediate tensors
                del image_tensor, pred
            
            # Merge multi-scale features
            merged_features = self._merge_multi_scale_features(scale_features, original_size)
            batch_features[image_path] = merged_features
        
        return batch_features


class ALIKEDExtractor(BaseFeatureExtractor):
    """ALIKED feature extractor"""
    
    def _setup_model(self):
        """Setup ALIKED model"""
        try:
            from lightglue.aliked import ALIKED
            
            # ALIKED configuration matching the provided format
            aliked_conf = {
                "model_name": self.config.get('model_name', 'aliked-n16'),
                "max_num_keypoints": self.config.get('max_num_keypoints', 4096),  # Set reasonable max limit
                "detection_threshold": self.config.get('detection_threshold', 0.1),  # Lower threshold for more keypoints
                "nms_radius": self.config.get('nms_radius', 2),
            }
            
            self.model = ALIKED(
                model_name=aliked_conf["model_name"],
                max_num_keypoints=aliked_conf["max_num_keypoints"],
                detection_threshold=aliked_conf["detection_threshold"],
                nms_radius=aliked_conf["nms_radius"]
            ).eval().to(self.device)
        except ImportError as e:
            raise ImportError(f"LightGlue ALIKED not available: {e}. Try: pip install lightglue @ git+https://github.com/cvg/LightGlue.git")
    
    def extract_features(self, images: List[Dict], batch_size: int = 8) -> Dict[str, Any]:
        """Extract ALIKED features from images"""
        features = {}
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            for img_data in batch:
                image = img_data['image']
                image_path = img_data['path']
                
                # Preprocess image
                image_tensor = self._preprocess_image(image)
                
                # Extract features
                with torch.no_grad():
                    # ALIKED expects a dictionary with 'image' key
                    input_data = {'image': image_tensor}
                    pred = self.model(input_data)
                
                # Convert to numpy arrays
                keypoints = pred['keypoints'][0].cpu().numpy()
                descriptors = pred['descriptors'][0].cpu().numpy()
                scores = pred['keypoint_scores'][0].cpu().numpy()
                
                features[image_path] = {
                    'keypoints': keypoints,
                    'descriptors': descriptors,
                    'scores': scores,
                    'image_shape': image.shape[:2]
                }
                
                # Clean up intermediate tensors
                del image_tensor, pred
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return features


class DISKExtractor(BaseFeatureExtractor):
    """DISK feature extractor"""
    
    def _setup_model(self):
        """Setup DISK model"""
        try:
            from lightglue.disk import DISK
            
            # DISK configuration matching the provided format
            disk_conf = {
                "weights": self.config.get('weights', 'depth'),
                "max_keypoints": self.config.get('max_keypoints', None),
                "nms_window_size": self.config.get('nms_window_size', 5),
                "detection_threshold": self.config.get('detection_threshold', 0.0),
                "pad_if_not_divisible": self.config.get('pad_if_not_divisible', True),
            }
            
            self.model = DISK(
                weights=disk_conf["weights"],
                max_keypoints=disk_conf["max_keypoints"],
                nms_window_size=disk_conf["nms_window_size"],
                detection_threshold=disk_conf["detection_threshold"],
                pad_if_not_divisible=disk_conf["pad_if_not_divisible"]
            ).eval().to(self.device)
        except ImportError as e:
            raise ImportError(f"LightGlue DISK not available: {e}. Try: pip install lightglue @ git+https://github.com/cvg/LightGlue.git")
    
    def extract_features(self, images: List[Dict], batch_size: int = 8) -> Dict[str, Any]:
        """Extract DISK features from images"""
        features = {}
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            for img_data in batch:
                image = img_data['image']
                image_path = img_data['path']
                
                # Preprocess image
                image_tensor = self._preprocess_image(image)
                
                # Extract features
                with torch.no_grad():
                    # DISK expects a dictionary with 'image' key
                    input_data = {'image': image_tensor}
                    pred = self.model(input_data)
                
                # Convert to numpy arrays
                keypoints = pred['keypoints'][0].cpu().numpy()
                descriptors = pred['descriptors'][0].cpu().numpy()
                scores = pred['keypoint_scores'][0].cpu().numpy()
                
                features[image_path] = {
                    'keypoints': keypoints,
                    'descriptors': descriptors,
                    'scores': scores,
                    'image_shape': image.shape[:2]
                }
                
                # Clean up intermediate tensors
                del image_tensor, pred
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return features


class FeatureExtractorFactory:
    """Factory for creating feature extractors"""
    
    _extractors = {
        'superpoint': SuperPointExtractor,
        'aliked': ALIKEDExtractor,
        'disk': DISKExtractor
    }
    
    @classmethod
    def create(cls, extractor_type: str, device: torch.device, config: Dict[str, Any] = None) -> BaseFeatureExtractor:
        """Create a feature extractor instance"""
        if extractor_type not in cls._extractors:
            raise ValueError(f"Unknown extractor type: {extractor_type}. Available: {list(cls._extractors.keys())}")
        
        return cls._extractors[extractor_type](device, config)
    
    @classmethod
    def get_supported_extractors(cls) -> List[str]:
        """Get list of supported extractor types"""
        return list(cls._extractors.keys())


class FeatureExtractor:
    """Main feature extractor class that wraps the factory"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.extractor_type = config.get('feature_extractor', 'superpoint')
        self.max_keypoints = config.get('max_keypoints', 4096)
        self.max_image_size = config.get('max_image_size', 1600)
        
        # Create the actual extractor
        self.extractor = FeatureExtractorFactory.create(self.extractor_type, self.device, self.config)
    
    def extract(self, image_paths: List[str]) -> Dict[str, Any]:
        """Extract features from images"""
        # Load and preprocess images
        images = []
        for path in image_paths:
            image = cv2.imread(str(path))
            if image is not None:
                # Resize if needed
                if max(image.shape[:2]) > self.max_image_size:
                    scale = self.max_image_size / max(image.shape[:2])
                    new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
                    image = cv2.resize(image, new_size)
                
                images.append({
                    'image': image,
                    'path': path
                })
        
        # Extract features
        features = self.extractor.extract_features(images)
        
        return features 