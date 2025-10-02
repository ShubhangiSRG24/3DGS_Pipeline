"""
Core SfM components
"""

# Import core functions - some conditionally
from .feature_extractor import FeatureExtractor
from .feature_matcher import FeatureMatcher
from .geometric_verification import GeometricVerification

# GPU modules - optional imports


try:
    from .gpu_vocabulary_tree import GPUVocabularyTree
    GPU_VOCAB_AVAILABLE = True
except ImportError:
    GPU_VOCAB_AVAILABLE = False
    GPUVocabularyTree = None

# Convenience functions for direct usage
def extract_features(images, config=None):
    """Extract features from images"""
    extractor = FeatureExtractor(config or {})
    return extractor.extract(images)

def match_features(features, config=None):
    """Match features between images"""
    matcher = FeatureMatcher(config or {})
    return matcher.match(features)

def verify_geometry(matches, config=None):
    """Verify geometric consistency"""
    verifier = GeometricVerification(config or {})
    return verifier.verify(matches)




def recover_scale(points3d, depth_maps, config=None):
    """Recover scale from depth maps"""
    scale_recovery = ScaleRecovery(config or {})
    return scale_recovery.recover(points3d, depth_maps)

# Build __all__ list dynamically based on available modules
__all__ = [
    # Core classes (always available)
    "FeatureExtractor",
    "FeatureMatcher", 
    "GeometricVerification",
    "ScaleRecovery",
    # Convenience functions
    "extract_features",
    "match_features", 
    "verify_geometry",
    "recover_scale"
]

# Add optional modules if available

# Add GPU classes if available
if GPU_VOCAB_AVAILABLE:
    __all__.append("GPUVocabularyTree") 