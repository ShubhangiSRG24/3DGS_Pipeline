
# FloaterGaussian Pipeline — Extract → ESFM → Depth → Train → Post processing

Build a 3D Gaussian Splatting scene from a video in four stages:
1) **Extract** frames  
2) **ESFM (Enhanced Structure-from-Motion)**  
3) **Depth extraction & scaling (Depth-Anything-V2)**  
4) **Train** the Gaussian scene
5) **Post process** apply rotation and convex hull to the training outputs

---

## 📚 Documentation

### Quick Links
- **[AWS Batch Guide](docs/aws/)** - Run on AWS with GPU instances
- **[Docker Build Guide](docs/docker/BUILD.md)** - Build and deploy Docker image
- **[Scripts Reference](scripts/README.md)** - Utility scripts documentation

### AWS Batch Integration
- [AWS Batch Setup](docs/aws/SETUP.md) - Initial configuration
- [Quick Reference](docs/aws/QUICK_REFERENCE.md) - Commands and examples
- [Cost Guide](docs/aws/COSTS.md) - Pricing and optimization
- [Troubleshooting](docs/aws/TROUBLESHOOTING.md) - Common issues

### Docker
- [Build Guide](docs/docker/BUILD.md) - Build Docker image
- [Troubleshooting](docs/docker/TROUBLESHOOTING.md) - Build issues
- [Dockerfile Reference](docs/docker/DOCKERFILE_REFERENCE.md) - Technical details

---

## Prerequisites
- Ubuntu/Linux recommended
- Python **3.9.23**
- NVIDIA GPU with **CUDA 12.1 or >** (recommended)

---

## Steps

1. **Clone your repo and enter it.**  
2. **Create/activate a fresh Python environment** (conda).  
3. **Install dependencies.**  
   - If your repo has `Enhanced-Structure-from-Motion/requirements.txt`, install it.  
   - Install CuPy + FAISS (GPU build by default), rasterizers (`diff-gaussian-rasterization`, `fused-ssim`, `simple-knn`) and extras (`plyfile`, `lpips`).  
4. **Set handy environment variables** (`BASE_DIR`, `PROJ`, `VIDEO_URL`).  
5. **Extract frames** from the input video with `extractImg.py -s` (answer prompts).  
6. **Run ESFM** to produce the COLMAP model in `data/<PROJ>/colmap/`.  
7. **Download the Depth-Anything-V2 Large weight** (`depth_anything_v2_vitl.pth`) into `Depth-Anything-V2/checkpoints/`.  
8. **Run depth scaling** to produce depth maps in `data/<PROJ>/depthmap/`.  
9. **Train** the Gaussian scene using the COLMAP model and the depth maps.  
10. **Check outputs** (renders/checkpoints/logs) and iterate as needed.

---

## Commands

```bash
# 0) Get this repo (replace with YOUR URL/NAME)
git clone <REPO_URL>.git
cd <REPO_NAME>

# 1) Fresh Python environment (choose one)
# --- conda ---
# conda create -n gaussian_splatting python=3.9.23 -y
# conda activate gaussian_splatting

# 2) Install dependencies
# ESFM deps (if present)
[ -f Enhanced-Structure-from-Motion/requirements.txt ] && \
  python -m pip install -r Enhanced-Structure-from-Motion/requirements.txt

# Core numeric + GPU acceleration (default: GPU build)
python -m pip install --upgrade pip
python -m pip install cupy-cuda12x faiss-gpu numpy==1.26.4
# If you have NO CUDA, use CPU FAISS instead (comment GPU line above, then):
# python -m pip install faiss-cpu numpy==1.26.4

# Rasterizers & extras (local submodules)
python -m pip install submodules/diff-gaussian-rasterization/
python -m pip install submodules/fused-ssim/
python -m pip install submodules/simple-knn/
python -m pip install plyfile lpips

# 3) Handy env vars
export BASE_DIR="$(pwd)"                   # repo root
export PROJ="meet"                         # project name (folder under data/)
export VIDEO_URL="https://static-dev.meic-ai.com/test/meetingroom.mp4"

# 4) Extract frames (interactive prompts)
python extractImg.py -s
# Example answers:
#   Where is your video?  -> ${VIDEO_URL}
#   How many images?      -> 100
#   Where to save?        -> data/${PROJ}/images

# 5) ESFM → produces COLMAP model in data/<PROJ>/colmap
python Enhanced-Structure-from-Motion/sfm_pipeline.py \
  --input_dir  "${BASE_DIR}/data/${PROJ}/images" \
  --output_dir "${BASE_DIR}/data/${PROJ}" \
  --feature_extractor aliked \
  --use_vocab_tree

# 6) Depth-Anything-V2 weights (Large ViT-L) → put under checkpoints/
mkdir -p Depth-Anything-V2/checkpoints
# Option A: download manually then copy to checkpoints as "depth_anything_v2_vitl.pth"
# Option B: wget (may require cookies on some environments)
# wget -O Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth \
#   "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true"

# 7) Depth scaling → produces data/<PROJ>/depthmap
python Depth-Anything-V2/depth_scale.py --base_dir "${BASE_DIR}/data/${PROJ}"

# 8) Train the Gaussian scene
python trainFloaters.py \
  -s "${BASE_DIR}/data/${PROJ}/colmap" \
  -d "${BASE_DIR}/data/${PROJ}/depthmap"
