# Enhanced SfM Pipeline for 3D Gaussian Splatting

### Quick Install
```bash
# Clone repository
git clone https://github.com/jmhi9999/Enhanced-Structure-from-Motion
cd Enhanced-Structure-from-Motion

# Install dependencies
pip install -r requirements.txt

# For GPU support (recommended)
pip install cupy-cuda12x faiss-gpu 

## Quick Start

### Basic SfM Reconstruction
```bash
python sfm_pipeline.py  --input_dir path/to/images  --output_dir path/to/output  --feature_extractor aliked --use_vocab_tree
```

## Pipeline Architecture

```
Enhanced SfM Pipeline
├── Image Preprocessing → sfm/utils/image_utils.py
├── Feature Extraction → sfm/core/feature_extractor.py (SuperPoint/ALIKED)
├── Pair Selection → sfm/core/gpu_vocabulary_tree.py (FAISS-powered O(n log n))
├── Feature Matching → sfm/core/feature_matcher.py (LightGlue + GPU brute force fallback)
├── Geometric Verification → sfm/core/geometric_verification.py (cv2.USAC_MAGSAC)
├── SfM Reconstruction → sfm/core/colmap_binary.py (COLMAP binary execution)
└── COLMAP Output → 3DGS Ready
```

## Configuration Options

### Feature Extraction
```bash
--feature_extractor superpoint     # SuperPoint, ALIKED, or DISK
--max_image_size 1600              # Maximum image size for processing
--max_keypoints 4096               # Maximum keypoints per image
```

### Pair Selection and Matching
```bash
--use_vocab_tree                   
--use_brute_force                  # GPU brute force matching (default)
--max_pairs_per_image 20           # Maximum pairs per image for vocab tree
--max_total_pairs 10000            # Maximum total pairs for brute force
```
## Requirements

### Core Dependencies
- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.8+
- NumPy 1.24+ -> 1.26.4 recommended
- COLMAP binary (must be in PATH)

