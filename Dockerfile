# build
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

# Build-time OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential cmake pkg-config \
    ffmpeg colmap \
    libgl1 libglib2.0-0 libxext6 libsm6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# Micromamba - build the env
ARG MAMBA_ROOT_PREFIX=/opt/conda
ENV MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX}
RUN mkdir -p ${MAMBA_ROOT_PREFIX} /root/.conda && \
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba

SHELL ["/bin/bash", "-lc"]
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Create env
RUN micromamba create -y -n gs python=3.9.23 && micromamba clean -a -y
ENV PATH=${MAMBA_ROOT_PREFIX}/envs/gs/bin:$PATH
RUN echo "micromamba activate gs" >> ~/.bashrc

# Code
WORKDIR /workspace/app
COPY . /workspace/app
RUN chmod +x /workspace/app/entrypoint.sh

# CUDA link for builds
RUN set -eux; \
    CUDADIR="$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -n1 || true)"; \
    if [ -n "$CUDADIR" ]; then ln -sfn "$CUDADIR" /usr/local/cuda; fi

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Build
ARG ARCHS="8.0;8.6;8.9"
ENV TORCH_CUDA_ARCH_LIST="${ARCHS}" FORCE_CUDA=1
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/workspace/app:${PYTHONPATH}

# GPU Python deps
RUN micromamba run -n gs python -m pip install --upgrade pip && \
    micromamba run -n gs pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cu121 \
      torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 && \
    micromamba run -n gs pip install --no-cache-dir \
      faiss-gpu cupy-cuda12x==12.1.0 numpy==1.26.4 \
      opencv-python-headless==4.11.0.86 \
      timm lpips plyfile joblib concave_hull scipy

# ESfM deps
RUN if [ -f Enhanced-Structure-from-Motion/requirements.txt ]; then \
      sed -i 's/^opencv-python.*/opencv-python-headless==4.11.0.86/' Enhanced-Structure-from-Motion/requirements.txt && \
      sed -i '/^[[:space:]]*matplotlib/d;/^[[:space:]]*pytest/d;/^[[:space:]]*black/d;/^[[:space:]]*flake8/d;/^[[:space:]]*mypy/d' Enhanced-Structure-from-Motion/requirements.txt && \
      sed -i '/[Ll]ightglue/d' Enhanced-Structure-from-Motion/requirements.txt && \
      micromamba run -n gs pip install --no-cache-dir -r Enhanced-Structure-from-Motion/requirements.txt && \
      micromamba run -n gs pip install --no-deps "lightglue @ git+https://github.com/cvg/LightGlue.git@746fac2c042e05d1865315b1413419f1c1e7ba55"; \
    fi

# Build reqs - CUDA extensions
RUN micromamba run -n gs pip install --no-cache-dir -U pip setuptools wheel ninja cmake packaging

# Build + install local submodules
RUN micromamba run -n gs pip install --no-cache-dir --no-build-isolation \
      ./submodules/diff-gaussian-rasterization \
      ./submodules/fused-ssim \
      ./submodules/simple-knn

# Clean the env
RUN find ${MAMBA_ROOT_PREFIX}/envs/gs -type d -name "__pycache__" -prune -exec rm -rf {} + && \
    find ${MAMBA_ROOT_PREFIX}/envs/gs -name "*.a" -delete || true && \
    find ${MAMBA_ROOT_PREFIX}/envs/gs -name "*.pyc" -delete || true && \
    (find ${MAMBA_ROOT_PREFIX}/envs/gs -name "*.so" -exec strip -s {} + 2>/dev/null || true) && \
    micromamba clean -a -y

# runtime
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONUNBUFFERED=1

# runtime-only packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ffmpeg colmap \
    libgl1 libglib2.0-0 libxext6 libsm6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# Final Python env
ARG MAMBA_ROOT_PREFIX=/opt/conda
ENV MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX}
COPY --from=builder ${MAMBA_ROOT_PREFIX}/envs/gs ${MAMBA_ROOT_PREFIX}/envs/gs
ENV PATH=${MAMBA_ROOT_PREFIX}/envs/gs/bin:$PATH

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Code
WORKDIR /workspace/app
COPY --from=builder /workspace/app /workspace/app
RUN chmod +x /workspace/app/entrypoint.sh

ENV PYTHONPATH=/workspace/app:${PYTHONPATH}

ENTRYPOINT ["/workspace/app/entrypoint.sh"]
