# CUDA + Ubuntu 22.04 (devel has compiler toolchain for building CUDA extensions)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONUNBUFFERED=1

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential cmake pkg-config \
    ffmpeg colmap \
    libgl1 libglib2.0-0 libxext6 libsm6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# Micromamba
ARG MAMBA_ROOT_PREFIX=/opt/conda
ENV MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX}
RUN mkdir -p ${MAMBA_ROOT_PREFIX} /root/.conda
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba

SHELL ["/bin/bash", "-lc"]
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Python env
RUN micromamba create -y -n gs python=3.9.23 && micromamba clean -a -y
ENV PATH=${MAMBA_ROOT_PREFIX}/envs/gs/bin:$PATH
RUN echo "micromamba activate gs" >> ~/.bashrc

# App
WORKDIR /workspace/app
COPY . /workspace/app
COPY entrypoint.sh /workspace/app/entrypoint.sh
RUN chmod +x /workspace/app/entrypoint.sh

# Symlink /usr/local/cuda -> latest installed CUDA (some base images already do this)
RUN set -eux; \
  CUDADIR="$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -n1 || true)"; \
  if [ -n "$CUDADIR" ]; then ln -sfn "$CUDADIR" /usr/local/cuda; fi

# CUDA env + torch arch list (override at build with --build-arg ARCHS="7.5;8.6;8.9")
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ARG ARCHS="8.6"
ENV TORCH_CUDA_ARCH_LIST="${ARCHS}"
ENV FORCE_CUDA=1

# PyTorch (CU121) + deps
RUN micromamba run -n gs python -m pip install --upgrade pip && \
    micromamba run -n gs pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cu121 \
      torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 && \
    micromamba run -n gs pip install --no-cache-dir \
      cupy-cuda12x==12.1.0 faiss-gpu faiss-cpu numpy==1.26.4 timm lpips plyfile opencv-python-headless joblib
# ESfM requirements (path at repo root)
RUN if [ -f "Enhanced-Structure-from-Motion/requirements.txt" ]; then \
      micromamba run -n gs pip install --no-cache-dir -r Enhanced-Structure-from-Motion/requirements.txt ; \
    fi

# Build deps for CUDA extensions
RUN micromamba run -n gs pip install --no-cache-dir -U pip setuptools wheel ninja cmake packaging

# Build & install local CUDA submodules (non-editable, no build isolation so torch is visible)
RUN micromamba run -n gs pip install --no-cache-dir --no-build-isolation \
    ./submodules/diff-gaussian-rasterization \
    ./submodules/fused-ssim \
    ./submodules/simple-knn

# Run the pipeline by default
ENTRYPOINT ["/workspace/app/entrypoint.sh"]






# # CUDA + cuDNN + Ubuntu 22.04
# FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# ENV DEBIAN_FRONTEND=noninteractive \
#     TZ=Etc/UTC \
#     NVIDIA_VISIBLE_DEVICES=all \
#     NVIDIA_DRIVER_CAPABILITIES=compute,utility

# # System packages
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     git curl ca-certificates build-essential cmake pkg-config \
#     ffmpeg \
#     colmap \
#     libgl1 libglib2.0-0 libxext6 libsm6 libxrender1 \
#     && rm -rf /var/lib/apt/lists/*

# # Micromamba (conda) + Python 3.9.23
# ARG MAMBA_ROOT_PREFIX=/opt/conda
# ENV MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX}
# RUN mkdir -p ${MAMBA_ROOT_PREFIX} /root/.conda

# # Install micromamba
# RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
#   | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba

# SHELL ["/bin/bash", "-lc"]
# ARG MAMBA_DOCKERFILE_ACTIVATE=1

# # env "gs" + Python 3.9.23
# RUN micromamba create -y -n gs python=3.9.23 && micromamba clean -a -y
# ENV PATH=${MAMBA_ROOT_PREFIX}/envs/gs/bin:$PATH
# RUN echo "micromamba activate gs" >> ~/.bashrc

# # COPY whole local tree into the image
# WORKDIR /workspace/app

# COPY . /workspace/app
# # Copy and set entrypoint
# COPY entrypoint.sh /workspace/app/entrypoint.sh
# ENTRYPOINT ["/workspace/app/entrypoint.sh"]

# # 10) CUDA env + arch
# RUN set -eux; \
#   CUDADIR="$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -n1 || true)"; \
#   if [ -n "$CUDADIR" ]; then \
#     ln -sfn "$CUDADIR" /usr/local/cuda; \
#   fi; \
#   ls -l /usr/local | sed -n 's/^/[/p' | sed 's/$/]/' || true

# ENV CUDA_HOME=/usr/local/cuda
# ENV PATH=${CUDA_HOME}/bin:${PATH}
# ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
# ARG ARCHS="8.6"
# ENV TORCH_CUDA_ARCH_LIST="${ARCHS}"
# ENV FORCE_CUDA=1

# RUN micromamba run -n gs python -m pip install --upgrade pip && \
#     micromamba run -n gs pip install --no-cache-dir \
#       --extra-index-url https://download.pytorch.org/whl/cu121 \
#       torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 && \
#     micromamba run -n gs pip install --no-cache-dir \
#       cupy-cuda12x==13.6.0 faiss-gpu numpy==1.26.4 timm lpips plyfile opencv-python-headless

# # Enhanced-Structure-from-Motion requirements
# RUN if [ -f "Enhanced-Structure-from-Motion/requirements.txt" ]; then \
#       micromamba run -n gs pip install --no-cache-dir -r Enhanced-Structure-from-Motion/requirements.txt ; \
#     fi

# # ensure build deps are present in the env that sees torch
# RUN micromamba run -n gs pip install --no-cache-dir -U pip setuptools wheel ninja cmake packaging

# # Build & install local CUDA submodules
# # (diff-gaussian-rasterization, fused-ssim, simple-knn)
# RUN micromamba run -n gs pip install --no-cache-dir --no-build-isolation \
#     ./submodules/diff-gaussian-rasterization \
#     ./submodules/fused-ssim \
#     ./submodules/simple-knn

# # Final touches
# ENV PYTHONUNBUFFERED=1
# CMD ["/bin/bash"]
