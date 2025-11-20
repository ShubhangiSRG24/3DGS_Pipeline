# Dockerfile Reference

Detailed explanation of the multi-stage Dockerfile for the 3DGS Pipeline.

## Overview

The Dockerfile uses a **two-stage build** to optimize image size:
1. **Builder stage**: Compiles CUDA extensions and builds packages (~20GB)
2. **Runtime stage**: Contains only runtime dependencies and compiled artifacts (~8GB)

This reduces the final image size by 60% while maintaining all functionality.

## Stage 1: Builder

### Base Image

```dockerfile
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder
```

**Why this image?**
- `nvidia/cuda`: Official NVIDIA CUDA images
- `12.1.1`: CUDA version compatible with PyTorch 2.5.1
- `devel`: Includes development tools (nvcc, headers) for compiling CUDA extensions
- `ubuntu22.04`: Long-term support, good package availability

### Environment Setup

```dockerfile
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC
```

**Purpose**: Prevent interactive prompts during package installation

### System Packages

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential cmake pkg-config \
    ffmpeg colmap \
    libgl1 libglib2.0-0 libxext6 libsm6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*
```

**Package breakdown:**
- `git, curl, ca-certificates`: Download and authentication
- `build-essential, cmake, pkg-config`: C/C++ compilation tools
- `ffmpeg`: Video frame extraction
- `colmap`: Structure-from-Motion (SfM) processing
- `libgl1, libglib2.0-0, libxext6, libsm6, libxrender1`: OpenCV dependencies

**Optimization**: `--no-install-recommends` and cleanup reduce layer size

### Micromamba Installation

```dockerfile
ARG MAMBA_ROOT_PREFIX=/opt/conda
ENV MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX}
RUN mkdir -p ${MAMBA_ROOT_PREFIX} /root/.conda && \
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba
```

**Why Micromamba?**
- Lightweight conda alternative (8MB vs 400MB)
- Fast dependency resolution
- Compatible with conda environments
- No Python bootstrap needed

### Shell Configuration

```dockerfile
SHELL ["/bin/bash", "-lc"]
ARG MAMBA_DOCKERFILE_ACTIVATE=1
```

**Purpose**: Auto-activate conda environment in build steps

### Python Environment

```dockerfile
RUN micromamba create -y -n gs python=3.9.23 && micromamba clean -a -y
ENV PATH=${MAMBA_ROOT_PREFIX}/envs/gs/bin:$PATH
RUN echo "micromamba activate gs" >> ~/.bashrc
```

**Why Python 3.9.23?**
- Compatible with PyTorch 2.5.1
- Stable, well-tested version
- Good package compatibility

### Copy Source Code

```dockerfile
WORKDIR /workspace/app
COPY . /workspace/app
RUN chmod +x /workspace/app/entrypoint.sh
```

**Structure**:
- `/workspace/app`: Main application directory
- All source files copied
- Entrypoint script made executable

### CUDA Setup

```dockerfile
RUN set -eux; \
    CUDADIR="$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -n1 || true)"; \
    if [ -n "$CUDADIR" ]; then ln -sfn "$CUDADIR" /usr/local/cuda; fi

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
```

**Purpose**: 
- Find latest CUDA installation
- Create `/usr/local/cuda` symlink
- Set environment for CUDA compilation

### CUDA Architecture Configuration

```dockerfile
ARG ARCHS="7.0;7.5;8.0;8.6;8.9"
ENV TORCH_CUDA_ARCH_LIST="${ARCHS}" FORCE_CUDA=1
```

**Supported GPUs:**
| Compute Capability | GPU Models | AWS Instance |
|-------------------|------------|--------------|
| 7.0 | V100 | p3 |
| 7.5 | T4 | g4dn |
| 8.0 | A100 | p4d |
| 8.6 | A10G | g5 |
| 8.9 | H100 | (future) |

**Trade-off**: More architectures = longer build time but broader compatibility

### Python Environment Variables

```dockerfile
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace/app
```

**Optimizations:**
- `PYTHONUNBUFFERED=1`: Immediate log output (important for debugging)
- `PIP_NO_CACHE_DIR=1`: Don't cache downloads (saves ~2GB)
- `PYTHONDONTWRITEBYTECODE=1`: No `.pyc` files (saves space)
- `PYTHONPATH=/workspace/app`: Import modules from app directory

### PyTorch Installation

```dockerfile
RUN micromamba run -n gs python -m pip install --upgrade pip && \
    micromamba run -n gs pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cu121 \
      torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121
```

**Why separate step?**
- PyTorch is large (~2GB)
- Early failure if incompatible
- Cache this layer for faster rebuilds

**Version lock**: Ensures CUDA 12.1 compatibility

### GPU Libraries

```dockerfile
RUN micromamba run -n gs pip install --no-cache-dir \
      faiss-gpu cupy-cuda12x==12.1.0 numpy==1.26.4 \
      opencv-python-headless==4.11.0.86 \
      timm lpips plyfile joblib boto3
```

**Package purposes:**
- `faiss-gpu`: Fast nearest neighbor search (GPU-accelerated)
- `cupy-cuda12x`: NumPy-like array library on GPU
- `numpy==1.26.4`: Version pinned for compatibility
- `opencv-python-headless`: Image processing without GUI
- `timm`: Vision transformer models (for Depth-Anything-V2)
- `lpips`: Perceptual loss metric
- `plyfile`: 3D point cloud I/O
- `joblib`: Parallel processing
- `boto3`: AWS SDK (for S3)

### ESFM Dependencies

```dockerfile
RUN if [ -f Enhanced-Structure-from-Motion/requirements.txt ]; then \
      sed -i 's/^opencv-python.*/opencv-python-headless==4.11.0.86/' ... && \
      sed -i '/^[[:space:]]*matplotlib/d;...' ... && \
      micromamba run -n gs pip install --no-cache-dir -r ... && \
      micromamba run -n gs pip install --no-deps "lightglue @ git+..."; \
    fi
```

**Modifications:**
- Replace `opencv-python` with `opencv-python-headless` (no X11)
- Remove dev dependencies (matplotlib, pytest, black, flake8, mypy)
- Install LightGlue from specific commit (stability)

### CUDA Extensions Build

```dockerfile
RUN micromamba run -n gs pip install --no-cache-dir -U pip setuptools wheel ninja cmake packaging

RUN micromamba run -n gs pip install --no-cache-dir --no-build-isolation \
      ./submodules/diff-gaussian-rasterization \
      ./submodules/fused-ssim \
      ./submodules/simple-knn
```

**Build tools:**
- `ninja`: Fast build system
- `cmake`: Cross-platform build
- `packaging`: Version handling

**Extensions:**
- `diff-gaussian-rasterization`: Core 3DGS rendering
- `fused-ssim`: SSIM loss calculation
- `simple-knn`: K-nearest neighbors

**Flag**: `--no-build-isolation` uses existing PyTorch installation

### Cleanup

```dockerfile
RUN find ${MAMBA_ROOT_PREFIX}/envs/gs -type d -name "__pycache__" -prune -exec rm -rf {} + && \
    find ${MAMBA_ROOT_PREFIX}/envs/gs -name "*.a" -delete || true && \
    find ${MAMBA_ROOT_PREFIX}/envs/gs -name "*.pyc" -delete || true && \
    (find ${MAMBA_ROOT_PREFIX}/envs/gs -name "*.so" -exec strip -s {} + 2>/dev/null || true) && \
    micromamba clean -a -y
```

**Optimizations:**
- Remove `__pycache__` directories
- Delete static libraries (`.a` files)
- Delete Python bytecode (`.pyc`)
- Strip debug symbols from shared libraries (`.so`)
- Clean micromamba cache

**Space saved**: ~2-3GB

## Stage 2: Runtime

### Base Image

```dockerfile
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS runtime
```

**Key difference**: `runtime` instead of `devel`
- No compilation tools
- Smaller base image
- Only CUDA runtime libraries

### Environment

```dockerfile
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONUNBUFFERED=1
```

**NVIDIA variables:**
- `NVIDIA_VISIBLE_DEVICES=all`: Use all available GPUs
- `NVIDIA_DRIVER_CAPABILITIES=compute,utility`: Enable GPU compute

### Runtime Packages

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ffmpeg colmap \
    libgl1 libglib2.0-0 libxext6 libsm6 libxrender1 \
    awscli \
 && rm -rf /var/lib/apt/lists/*
```

**Minimal set:**
- Runtime tools only (no compilers)
- `awscli`: For S3 operations in AWS Batch

### Copy Python Environment

```dockerfile
ARG MAMBA_ROOT_PREFIX=/opt/conda
ENV MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX}
COPY --from=builder ${MAMBA_ROOT_PREFIX}/envs/gs ${MAMBA_ROOT_PREFIX}/envs/gs
ENV PATH=${MAMBA_ROOT_PREFIX}/envs/gs/bin:$PATH
```

**Multi-stage copy:**
- Copies entire conda environment from builder
- Includes all installed packages and compiled extensions
- No need to rebuild

### CUDA Runtime Setup

```dockerfile
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
```

**Purpose**: Ensure CUDA runtime is in path

### Copy Application

```dockerfile
WORKDIR /workspace/app
COPY --from=builder /workspace/app /workspace/app
RUN chmod +x /workspace/app/entrypoint.sh
ENV PYTHONPATH=/workspace/app
```

**Final structure:**
```
/workspace/app/
├── entrypoint.sh (executable)
├── train.py
├── trainFloaters.py
├── render.py
├── ...
└── submodules/
```

### Entrypoint

```dockerfile
ENTRYPOINT ["/workspace/app/entrypoint.sh"]
```

**Benefits of entrypoint script:**
- Handles S3 sync
- Manages pipeline stages
- Timing/logging
- Flexible configuration via environment variables

## Size Comparison

| Stage | Size | Contents |
|-------|------|----------|
| Builder (intermediate) | ~20GB | Dev tools, source, build artifacts |
| Runtime (final) | ~8GB | Runtime only, compiled extensions |
| **Savings** | **60%** | Removed ~12GB of build dependencies |

## Build Arguments Reference

| Argument | Default | Purpose |
|----------|---------|---------|
| `ARCHS` | 7.0;7.5;8.0;8.6;8.9 | CUDA compute capabilities to build for |
| `MAMBA_ROOT_PREFIX` | /opt/conda | Micromamba installation path |
| `MAMBA_DOCKERFILE_ACTIVATE` | 1 | Auto-activate conda environment |

## Environment Variables Reference

### Build-time

| Variable | Value | Purpose |
|----------|-------|---------|
| `DEBIAN_FRONTEND` | noninteractive | Prevent apt prompts |
| `TORCH_CUDA_ARCH_LIST` | ${ARCHS} | PyTorch CUDA architectures |
| `FORCE_CUDA` | 1 | Force CUDA build |
| `CUDA_HOME` | /usr/local/cuda | CUDA toolkit location |
| `PYTHONUNBUFFERED` | 1 | Unbuffered Python output |
| `PIP_NO_CACHE_DIR` | 1 | Don't cache pip downloads |

### Runtime

| Variable | Value | Purpose |
|----------|-------|---------|
| `NVIDIA_VISIBLE_DEVICES` | all | Use all GPUs |
| `NVIDIA_DRIVER_CAPABILITIES` | compute,utility | GPU capabilities |
| `PYTHONPATH` | /workspace/app | Python import path |

## Related Documentation

- [Build Guide](./BUILD.md)
- [Troubleshooting](./TROUBLESHOOTING.md)
- [AWS Batch Setup](../aws/SETUP.md)
