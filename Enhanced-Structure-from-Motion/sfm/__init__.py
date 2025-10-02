"""
Enhanced Structure-from-Motion Package
GPU-accelerated SfM pipeline with modern feature extractors and matchers
"""

__version__ = "0.1.0"

# Lazy import function to avoid circular imports
def _get_sfm_pipeline():
    """Import sfm_pipeline when needed"""
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    _parent_dir = str(Path(__file__).parent.parent)
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    
    from sfm_pipeline import sfm_pipeline
    return sfm_pipeline

# Lazy imports for heavy dependencies - only import when actually used
def __getattr__(name):
    """Lazy import for module attributes"""
    
    # Handle sfm_pipeline specially
    if name == "sfm_pipeline":
        return _get_sfm_pipeline()
    
    # For heavy components, import only when needed
    if name == "FeatureExtractorFactory":
        from .core.feature_extractor import FeatureExtractorFactory
        return FeatureExtractorFactory
    elif name == "EnhancedLightGlueMatcher":
        from .core.feature_matcher import EnhancedLightGlueMatcher
        return EnhancedLightGlueMatcher
    elif name == "GeometricVerification":
        from .core.geometric_verification import GeometricVerification
        return GeometricVerification
    elif name == "RANSACMethod":
        from .core.geometric_verification import RANSACMethod
        return RANSACMethod
    elif name == "GPUVocabularyTree":
        from .core.gpu_vocabulary_tree import GPUVocabularyTree
        return GPUVocabularyTree
    elif name == "ScaleRecovery":
        from .core.scale_recovery import ScaleRecovery
        return ScaleRecovery
    # Utilities (lighter imports)
    elif name == "save_colmap_format":
        from .utils.io_utils import save_colmap_format
        return save_colmap_format
    elif name == "load_images":
        from .utils.io_utils import load_images
        return load_images
    elif name == "save_features":
        from .utils.io_utils import save_features
        return save_features
    elif name == "save_matches":
        from .utils.io_utils import save_matches
        return save_matches
    elif name == "resize_image":
        from .utils.image_utils import resize_image
        return resize_image
    elif name == "QualityMetrics":
        from .utils.quality_metrics import QualityMetrics
        return QualityMetrics
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # Main pipeline
    "sfm_pipeline",
    
    # Core components
    "FeatureExtractorFactory",
    "EnhancedLightGlueMatcher", 
    "GeometricVerification",
    "RANSACMethod",
    "GPUVocabularyTree",
    "ScaleRecovery",
    
    # Utilities
    "save_colmap_format",
    "load_images", 
    "save_features",
    "save_matches",
    "resize_image",
    "QualityMetrics"
] 