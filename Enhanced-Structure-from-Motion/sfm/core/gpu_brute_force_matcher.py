import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import cv2
from tqdm import tqdm
import logging
import time
import gc
from concurrent.futures import ThreadPoolExecutor
import math

try:
    from lightglue import LightGlue, SuperPoint, ALIKED, DISK
    LIGHTGLUE_AVAILABLE = True
except ImportError:
    LIGHTGLUE_AVAILABLE = False
    LightGlue = SuperPoint = ALIKED = DISK = None

logger = logging.getLogger(__name__)


class GPUTensorFeatureStorage:
    """
    GPU tensor-based feature storage system
    Keeps all features in GPU memory as tensors for immediate access
    """
    
    def __init__(self, device: torch.device, max_features_per_image: int = 4096):
        self.device = device
        self.max_features_per_image = max_features_per_image
        
        # GPU tensor storage
        self.image_names: List[str] = []
        self.keypoints_tensor: Optional[torch.Tensor] = None  # [N_images, max_features, 2]
        self.descriptors_tensor: Optional[torch.Tensor] = None  # [N_images, max_features, descriptor_dim]
        self.scores_tensor: Optional[torch.Tensor] = None  # [N_images, max_features]
        self.valid_mask: Optional[torch.Tensor] = None  # [N_images, max_features] - which features are valid
        self.feature_counts: Optional[torch.Tensor] = None  # [N_images] - actual number of features per image
        self.image_shapes: List[Tuple[int, int]] = []  # [(height, width), ...]
        
        # Mapping
        self.name_to_idx: Dict[str, int] = {}
        
        # Memory stats
        self.memory_allocated = 0
        
    def add_features(self, features: Dict[str, Any]) -> None:
        """
        Add features to GPU tensor storage
        """
        logger.info(f"Loading {len(features)} image features to GPU memory...")
        
        if not features:
            raise ValueError("No features provided")
        
        # Get first feature to determine descriptor dimension
        first_feat = next(iter(features.values()))
        descriptor_dim = first_feat['descriptors'].shape[-1] if len(first_feat['descriptors']) > 0 else 256
        
        n_images = len(features)
        self.image_names = list(features.keys())
        self.name_to_idx = {name: idx for idx, name in enumerate(self.image_names)}
        
        # Initialize tensors on GPU
        self.keypoints_tensor = torch.zeros(
            (n_images, self.max_features_per_image, 2), 
            dtype=torch.float32, 
            device=self.device
        )
        self.descriptors_tensor = torch.zeros(
            (n_images, self.max_features_per_image, descriptor_dim), 
            dtype=torch.float32, 
            device=self.device
        )
        self.scores_tensor = torch.zeros(
            (n_images, self.max_features_per_image), 
            dtype=torch.float32, 
            device=self.device
        )
        self.valid_mask = torch.zeros(
            (n_images, self.max_features_per_image), 
            dtype=torch.bool, 
            device=self.device
        )
        self.feature_counts = torch.zeros(
            n_images, 
            dtype=torch.int32, 
            device=self.device
        )
        
        # Fill tensors with feature data
        for idx, (img_name, feat_data) in enumerate(tqdm(features.items(), desc="Loading to GPU")):
            # Convert to numpy if needed
            keypoints = feat_data['keypoints']
            descriptors = feat_data['descriptors']
            scores = feat_data.get('scores', np.ones(len(keypoints)))
            
            if isinstance(keypoints, torch.Tensor):
                keypoints = keypoints.cpu().numpy()
            if isinstance(descriptors, torch.Tensor):
                descriptors = descriptors.cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            
            # Handle descriptor dimension mismatch
            if len(descriptors.shape) == 2 and descriptors.shape[0] != len(keypoints):
                descriptors = descriptors.T  # Transpose if needed
            
            n_features = min(len(keypoints), self.max_features_per_image)
            if n_features > 0:
                # Convert to tensors and move to GPU
                kpts_tensor = torch.from_numpy(keypoints[:n_features]).float().to(self.device)
                desc_tensor = torch.from_numpy(descriptors[:n_features]).float().to(self.device)
                scores_tensor = torch.from_numpy(scores[:n_features]).float().to(self.device)
                
                # Store in batch tensors
                self.keypoints_tensor[idx, :n_features] = kpts_tensor
                self.descriptors_tensor[idx, :n_features] = desc_tensor
                self.scores_tensor[idx, :n_features] = scores_tensor
                self.valid_mask[idx, :n_features] = True
                self.feature_counts[idx] = n_features
                
                # Store image shape
                self.image_shapes.append(feat_data['image_shape'])
            else:
                self.image_shapes.append(feat_data['image_shape'])
                logger.warning(f"No features found for {img_name}")
        
        # Calculate memory usage
        self.memory_allocated = (
            self.keypoints_tensor.numel() * 4 + 
            self.descriptors_tensor.numel() * 4 +
            self.scores_tensor.numel() * 4 +
            self.valid_mask.numel() * 1 +
            self.feature_counts.numel() * 4
        )
        
        logger.info(f"Loaded features to GPU: {self.memory_allocated / 1024**2:.1f} MB")
        logger.info(f"Max features per image: {self.max_features_per_image}")
        logger.info(f"Descriptor dimension: {descriptor_dim}")
    
    def get_features_tensor(self, image_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get features for a specific image as tensors (already on GPU)
        Returns: (keypoints, descriptors, scores) - all tensors are on GPU
        """
        if image_idx >= len(self.image_names):
            raise IndexError(f"Image index {image_idx} out of range")
        
        n_features = self.feature_counts[image_idx].item()
        if n_features == 0:
            # Return empty tensors
            return (
                torch.empty(0, 2, device=self.device),
                torch.empty(0, self.descriptors_tensor.shape[-1], device=self.device),
                torch.empty(0, device=self.device)
            )
        
        return (
            self.keypoints_tensor[image_idx, :n_features],
            self.descriptors_tensor[image_idx, :n_features],
            self.scores_tensor[image_idx, :n_features]
        )
    
    def get_features_by_name(self, image_name: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get features by image name"""
        if image_name not in self.name_to_idx:
            raise KeyError(f"Image {image_name} not found")
        
        return self.get_features_tensor(self.name_to_idx[image_name])
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        return {
            'allocated_mb': self.memory_allocated / 1024**2,
            'gpu_memory_mb': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
            'num_images': len(self.image_names),
            'max_features_per_image': self.max_features_per_image
        }


class GPUBruteForceMatcher:
    """
    GPU-accelerated brute force feature matcher
    Uses tensor operations for maximum performance
    """
    
    def __init__(self, device: torch.device, feature_type: str = 'superpoint', config: Dict[str, Any] = None):
        self.device = device
        self.feature_type = feature_type
        self.config = config or {}
        
        # Setup LightGlue matcher
        self.matcher = None
        self._setup_matcher()
        
        # Feature storage with memory pool integration
        self.feature_storage = GPUTensorFeatureStorage(
            device, 
            max_features_per_image=self.config.get('max_features_per_image', 4096)
        )
        
        # GPU memory pool for tensor operations
        try:
            from .gpu_memory_pool import GPUMemoryPool
            self.memory_pool = GPUMemoryPool(device, pool_size_mb=1024)
            self.use_memory_pool = True
            logger.info("GPU Memory Pool initialized for brute force matcher")
        except ImportError:
            self.memory_pool = None
            self.use_memory_pool = False
        
        # Matching parameters with dynamic adjustment
        default_batch_size = self.config.get('batch_size', 32)
        
        # Reduce batch size for large datasets to avoid GPU memory issues
        if 'max_features_per_image' in self.config:
            max_features = self.config['max_features_per_image']
            if max_features > 2000:
                default_batch_size = min(default_batch_size, 8)  # Large features -> smaller batch
            elif max_features > 1000:
                default_batch_size = min(default_batch_size, 16)  # Medium features -> medium batch
        
        self.batch_size = default_batch_size
        self.confidence_threshold = self.config.get('confidence_threshold', 0.2)
        self.min_matches = self.config.get('min_matches', 8)
        
        logger.info(f"Using batch_size={self.batch_size} for GPU processing")
        
        # Performance stats
        self.timing_stats = {
            'loading': [],
            'matching': [],
            'total': []
        }
    
    def _setup_matcher(self):
        """Setup LightGlue matcher"""
        if not LIGHTGLUE_AVAILABLE:
            raise ImportError("LightGlue is not available. Please install it.")
        
        # Map feature types to their correct LightGlue identifiers and configurations
        feature_mapping = {
            'superpoint': 'superpoint',
            'aliked': 'aliked',
            'disk': 'disk'
        }
        
        if self.feature_type not in feature_mapping:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")
        
        lightglue_features = feature_mapping[self.feature_type]
        
        lightglue_conf = {
            "features": lightglue_features,
            "depth_confidence": self.config.get('depth_confidence', 0.95),
            "width_confidence": self.config.get('width_confidence', 0.99),
            "compile": self.config.get('compile', False),
        }
        
        self.matcher = LightGlue(
            features=lightglue_conf["features"],
            depth_confidence=lightglue_conf["depth_confidence"],
            width_confidence=lightglue_conf["width_confidence"]
        ).eval().to(self.device)
        
        if lightglue_conf["compile"]:
            self.matcher.compile()
    
    def load_features(self, features: Dict[str, Any]) -> None:
        """
        Load features into GPU tensor storage
        """
        start_time = time.time()
        
        self.feature_storage.add_features(features)
        
        load_time = time.time() - start_time
        self.timing_stats['loading'].append(load_time)
        
        logger.info(f"Features loaded to GPU in {load_time:.2f}s")
    
    def match_all_pairs(self, max_pairs: Optional[int] = None) -> Dict[Tuple[str, str], Any]:
        """
        Perform brute force matching between all image pairs using GPU tensors
        Much faster than vocabulary tree approach for direct comparisons
        """
        start_time = time.time()
        
        if self.feature_storage.keypoints_tensor is None:
            raise ValueError("No features loaded. Call load_features() first.")
        
        image_names = self.feature_storage.image_names
        n_images = len(image_names)
        total_pairs = n_images * (n_images - 1) // 2
        
        if max_pairs is not None and total_pairs > max_pairs:
            logger.warning(f"Total pairs ({total_pairs}) exceeds max_pairs ({max_pairs}). Using subset.")
            # Use a subset of pairs
            pair_indices = np.random.choice(total_pairs, max_pairs, replace=False)
        else:
            pair_indices = None
        
        logger.info(f"GPU brute force matching for {n_images} images ({total_pairs} pairs)...")
        
        matches = {}
        pairs_processed = 0
        
        # Generate all pairs
        pairs = []
        for i in range(n_images):
            for j in range(i + 1, n_images):
                pairs.append((i, j))
        
        # Subset pairs if needed
        if pair_indices is not None:
            pairs = [pairs[idx] for idx in pair_indices]
        
        # Process pairs in batches for memory efficiency
        with tqdm(total=len(pairs), desc="Matching pairs", unit="pairs") as pbar:
            for batch_start in range(0, len(pairs), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(pairs))
                batch_pairs = pairs[batch_start:batch_end]
                
                batch_matches = self._match_batch_gpu(batch_pairs)
                matches.update(batch_matches)
                
                pairs_processed += len(batch_pairs)
                pbar.update(len(batch_pairs))
        
        total_time = time.time() - start_time
        self.timing_stats['matching'].append(total_time)
        self.timing_stats['total'].append(total_time)
        
        logger.info(f"GPU brute force matching completed in {total_time:.2f}s")
        logger.info(f"Successfully matched {len(matches)}/{len(pairs)} pairs")
        logger.info(f"Average matching time per pair: {total_time/len(pairs)*1000:.2f}ms")
        
        return matches
    
    def _match_batch_gpu(self, batch_pairs: List[Tuple[int, int]]) -> Dict[Tuple[str, str], Any]:
        """
        Match a batch of image pairs using GPU tensors with memory management
        """
        import traceback
        batch_matches = {}
        
        # Check GPU memory before processing
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_allocated = torch.cuda.memory_allocated()
            gpu_free = gpu_memory - gpu_allocated
            
            if gpu_free < 1024 * 1024 * 1024:  # Less than 1GB free
                logger.warning(f"Low GPU memory: {gpu_free / (1024**3):.1f}GB free, clearing cache...")
                torch.cuda.empty_cache()
                
        # Progress tracking handled by parent methods with tqdm
        # Optional GPU profiling
        enable_profiling = getattr(self, 'enable_profiling', False)
        if enable_profiling and torch.cuda.is_available():
            batch_start_event = torch.cuda.Event(enable_timing=True)
            batch_end_event = torch.cuda.Event(enable_timing=True)
            batch_start_event.record()
        for img1_idx, img2_idx in batch_pairs:
            try:
                # Get feature tensors (already on GPU)
                kpts1, desc1, scores1 = self.feature_storage.get_features_tensor(img1_idx)
                kpts2, desc2, scores2 = self.feature_storage.get_features_tensor(img2_idx)
                # Skip if no features
                if len(kpts1) == 0 or len(kpts2) == 0:
                    continue
                # Prepare data for LightGlue (tensors already on GPU)
                data = {
                    "image0": {
                        "keypoints": kpts1.unsqueeze(0),  # [1, N, 2]
                        "descriptors": desc1.unsqueeze(0),  # [1, N, D]
                    },
                    "image1": {
                        "keypoints": kpts2.unsqueeze(0),  # [1, N, 2]
                        "descriptors": desc2.unsqueeze(0),  # [1, N, D]
                    }
                }
                # Match features using LightGlue
                with torch.no_grad():
                    try:
                        pred = self.matcher(data)
                    except AssertionError as e:
                        # Debug dimension mismatch
                        desc1_shape = data['image0']['descriptors'].shape
                        desc2_shape = data['image1']['descriptors'].shape
                        expected_dim = self.matcher.conf.input_dim
                        raise AssertionError(
                            f"LightGlue dimension mismatch: "
                            f"desc1={desc1_shape}, desc2={desc2_shape}, "
                            f"expected_dim={expected_dim}, "
                            f"feature_type={self.feature_type}"
                        ) from e
                # Process matches
                if 'matches' in pred and len(pred['matches']) > 0:
                    matches_tensor = pred['matches']
                    scores_tensor = pred.get('scores', None)
                    # Robust conversion to tensor if needed
                    def robust_to_tensor(x):
                        if isinstance(x, torch.Tensor):
                            return x.to(self.device)
                        if isinstance(x, np.ndarray):
                            return torch.as_tensor(x, device=self.device)
                        if isinstance(x, list):
                            if len(x) > 0 and isinstance(x[0], (torch.Tensor, np.ndarray, list)):
                                # List of tensors/arrays/lists
                                return torch.stack([robust_to_tensor(xx) for xx in x])
                            else:
                                return torch.as_tensor(x, device=self.device)
                        return torch.as_tensor(x, device=self.device)
                    matches_tensor = robust_to_tensor(matches_tensor)
                    if scores_tensor is not None:
                        scores_tensor = robust_to_tensor(scores_tensor)
                    # Remove batch dimensions
                    while hasattr(matches_tensor, 'dim') and matches_tensor.dim() > 2:
                        matches_tensor = matches_tensor.squeeze(0)
                    if scores_tensor is not None:
                        while hasattr(scores_tensor, 'dim') and scores_tensor.dim() > 1:
                            scores_tensor = scores_tensor.squeeze(0)
                    if matches_tensor.shape[0] > 0:
                        matches0_indices = matches_tensor[:, 0].cpu().numpy()
                        matches1_indices = matches_tensor[:, 1].cpu().numpy()
                        match_scores = scores_tensor.cpu().numpy() if scores_tensor is not None else np.ones(len(matches0_indices))
                        # Filter by confidence
                        if len(match_scores) > 0:
                            confident_matches = match_scores > self.confidence_threshold
                            matches0_indices = matches0_indices[confident_matches]
                            matches1_indices = matches1_indices[confident_matches]
                            match_scores = match_scores[confident_matches]
                        # Only keep if enough matches
                        if len(matches0_indices) >= self.min_matches:
                            img1_name = self.feature_storage.image_names[img1_idx]
                            img2_name = self.feature_storage.image_names[img2_idx]
                            batch_matches[(img1_name, img2_name)] = {
                                'keypoints0': kpts1.cpu().numpy(),
                                'keypoints1': kpts2.cpu().numpy(),
                                'matches0': matches0_indices,
                                'matches1': matches1_indices,
                                'mscores0': match_scores,
                                'mscores1': match_scores,
                                'image_shape0': self.feature_storage.image_shapes[img1_idx],
                                'image_shape1': self.feature_storage.image_shapes[img2_idx]
                            }
            except Exception as e:
                img1_name = self.feature_storage.image_names[img1_idx]
                img2_name = self.feature_storage.image_names[img2_idx]
                logger.warning(f"Failed to match {img1_name} and {img2_name}: {e}\n{traceback.format_exc()}")
                continue
        if enable_profiling and torch.cuda.is_available():
            batch_end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = batch_start_event.elapsed_time(batch_end_event)
            logger.info(f"[Profiling] Batch of {len(batch_pairs)} pairs matched in {elapsed_ms:.2f} ms on GPU.")
        return batch_matches
    
    def match_specific_pairs(self, pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], Any]:
        """
        Match specific image pairs
        """
        start_time = time.time()
        
        logger.info(f"GPU matching {len(pairs)} specific pairs...")
        
        # Convert names to indices
        idx_pairs = []
        for img1_name, img2_name in pairs:
            if img1_name in self.feature_storage.name_to_idx and img2_name in self.feature_storage.name_to_idx:
                idx_pairs.append((
                    self.feature_storage.name_to_idx[img1_name],
                    self.feature_storage.name_to_idx[img2_name]
                ))
        
        matches = {}
        
        # Process in batches
        with tqdm(total=len(idx_pairs), desc="Matching pairs", unit="pairs") as pbar:
            for batch_start in range(0, len(idx_pairs), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(idx_pairs))
                batch_pairs = idx_pairs[batch_start:batch_end]
                
                batch_matches = self._match_batch_gpu(batch_pairs)
                matches.update(batch_matches)
                pbar.update(len(batch_pairs))
        
        total_time = time.time() - start_time
        self.timing_stats['matching'].append(total_time)
        
        logger.info(f"Specific pair matching completed in {total_time:.2f}s")
        logger.info(f"Successfully matched {len(matches)}/{len(pairs)} pairs")
        
        return matches
    
    def get_matched_keypoints(self, matches: Dict, img1_path: str, img2_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get matched keypoints for a specific image pair"""
        pair_key = (img1_path, img2_path)
        if pair_key not in matches:
            return np.array([]), np.array([])
        
        match_data = matches[pair_key]
        kpts0 = match_data['keypoints0']
        kpts1 = match_data['keypoints1']
        matches0 = match_data['matches0']
        matches1 = match_data['matches1']
        
        # Get matched keypoints
        matched_kpts0 = kpts0[matches0]
        matched_kpts1 = kpts1[matches1]
        
        return matched_kpts0, matched_kpts1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        memory_stats = self.feature_storage.get_memory_usage()
        
        stats = {
            'avg_loading_time': np.mean(self.timing_stats['loading']) if self.timing_stats['loading'] else 0.0,
            'avg_matching_time': np.mean(self.timing_stats['matching']) if self.timing_stats['matching'] else 0.0,
            'avg_total_time': np.mean(self.timing_stats['total']) if self.timing_stats['total'] else 0.0,
            'num_matching_sessions': len(self.timing_stats['total']),
            'batch_size': self.batch_size,
            'confidence_threshold': self.confidence_threshold,
            'min_matches': self.min_matches,
            **memory_stats
        }
        
        return stats
    
    def clear_memory(self):
        """Clear GPU memory"""
        if hasattr(self.feature_storage, 'keypoints_tensor') and self.feature_storage.keypoints_tensor is not None:
            del self.feature_storage.keypoints_tensor
            del self.feature_storage.descriptors_tensor
            del self.feature_storage.scores_tensor
            del self.feature_storage.valid_mask
            del self.feature_storage.feature_counts
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("GPU memory cleared")


# Main matcher class for compatibility
class FeatureMatcher:
    """
    Main feature matcher class using GPU brute force approach
    Drop-in replacement for vocabulary tree matcher
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        feature_type = config.get('feature_type', 'superpoint')
        self.matcher = GPUBruteForceMatcher(
            device=self.device,
            feature_type=feature_type,
            config=config
        )
    
    def match(self, features: Dict[str, Any]) -> Dict[Tuple[str, str], Any]:
        """Match features using GPU brute force approach"""
        # Load features to GPU
        self.matcher.load_features(features)
        
        # Perform brute force matching
        max_pairs = self.config.get('max_total_pairs', None)
        matches = self.matcher.match_all_pairs(max_pairs=max_pairs)
        
        return matches