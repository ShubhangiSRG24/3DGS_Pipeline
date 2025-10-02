#!/usr/bin/env bash
set -euo pipefail

# Inputs
VIDEO_URL="${VIDEO_URL:-}"          # set to a URL to auto-extract frames
NUM_IMAGES="${NUM_IMAGES:-120}"     # frames to sample from video
PROJECT="${PROJECT:-}"              # if empty, auto-derive
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
SKIP_SFM="${SKIP_SFM:-0}"
SKIP_DEPTH="${SKIP_DEPTH:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

# Optional: provide a URL to auto-download the depth checkpoint
DEPTH_CKPT_URL="${DEPTH_CKPT_URL:-}"
DEPTH_CKPT="Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth"

RUNPY="micromamba run -n gs python"

# Derive default project name
if [[ -z "${PROJECT}" ]]; then
  if [[ -n "${VIDEO_URL}" ]]; then
    bn="$(basename "${VIDEO_URL}")"
    PROJECT="${bn%%.*}"
  else
    PROJECT="proj-$(date +%Y%m%d-%H%M%S)"
  fi
fi

BASE="data/${PROJECT}"
IMAGES="${BASE}/images"
mkdir -p "${IMAGES}"

echo "[pipeline] project=${PROJECT}"
echo "[paths]   base=${BASE} images=${IMAGES}"

# 1) Extract frames (optional if VIDEO_URL provided)
if [[ "${SKIP_EXTRACT}" != "1" ]]; then
  if [[ -n "${VIDEO_URL}" ]]; then
    echo "[extract] url=${VIDEO_URL} num=${NUM_IMAGES}"
    printf '%s\n%s\n%s\n' "${VIDEO_URL}" "${NUM_IMAGES}" "${IMAGES}" \
    | ${RUNPY} extractImg.py -s
  else
    echo "[extract] VIDEO_URL not set; assuming images already exist in ${IMAGES}"
  fi
else
  echo "[extract] skipped"
fi

# 2) SfM (COLMAP)
if [[ "${SKIP_SFM}" != "1" ]]; then
  echo "[sfm] running COLMAP on ${IMAGES}"
  ${RUNPY} Enhanced-Structure-from-Motion/sfm_pipeline.py \
    --input_dir "${IMAGES}" \
    --output_dir "${BASE}" \
    --feature_extractor aliked \
    --use_vocab_tree
else
  echo "[sfm] skipped"
fi

# # 3) Depth-Anything V2 scaling (optional)
# if [[ "${SKIP_DEPTH}" != "1" ]]; then
#   if [[ ! -f "${DEPTH_CKPT}" ]]; then
#     if [[ -n "${DEPTH_CKPT_URL}" ]]; then
#       echo "[depth] downloading checkpoint..."
#       mkdir -p "$(dirname "${DEPTH_CKPT}")"
#       curl -L --fail --progress-bar -o "${DEPTH_CKPT}" "${DEPTH_CKPT_URL}"
#     else
#       echo "[depth] checkpoint missing at ${DEPTH_CKPT} and DEPTH_CKPT_URL not provided; skipping depth."
#       SKIP_DEPTH=1
#     fi
#   fi

#   if [[ "${SKIP_DEPTH}" != "1" ]]; then
#     echo "[depth] scaling depths at base_dir=${BASE}"
#     ${RUNPY} Depth-Anything-V2/depth_scale.py --base_dir "${BASE}"
#   fi
# else
#   echo "[depth] skipped"
# fi

# 3) Depth-Anything V2 scaling (optional)
if [[ "${SKIP_DEPTH}" != "1" ]]; then
  mkdir -p "Depth-Anything-V2/checkpoints" "checkpoints"

  CANON="Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth"
  if [[ ! -f "${CANON}" ]]; then
    if [[ -n "${DEPTH_CKPT_URL}" ]]; then
      echo "[depth] downloading checkpoint..."
      curl -L --fail --progress-bar -o "${CANON}" "${DEPTH_CKPT_URL}"
      SZ=$(stat -c%s "${CANON}" || echo 0)
      if (( SZ < 10*1024*1024 )); then
        echo "[depth] ERROR: downloaded file too small (${SZ} bytes)"
        exit 1
      fi
    else
      echo "[depth] checkpoint missing and DEPTH_CKPT_URL not provided; skipping depth."
      SKIP_DEPTH=1
    fi
  fi

  if [[ "${SKIP_DEPTH}" != "1" ]]; then
    # symlink for any code that expects ./checkpoints/...
    ln -sfn "../Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth" \
            "checkpoints/depth_anything_v2_vitl.pth"

    # ✅ make paths absolute and stay in repo root
    ABS_BASE="$(readlink -f "${BASE}")"
    ABS_IMAGES="${ABS_BASE}/images"

    echo "[depth] using checkpoint:"
    ls -l "${CANON}" || true
    ls -l "checkpoints/depth_anything_v2_vitl.pth" || true
    echo "[depth] ABS_BASE=${ABS_BASE}"

    if [[ ! -d "${ABS_IMAGES}" ]]; then
      echo "[depth] ERROR: expected '${ABS_IMAGES}' but it doesn't exist."
      echo "        PWD=$(pwd)"
      echo "        Tree under 'data/':"
      find data -maxdepth 3 -type d | sed 's/^/[data] /'
      exit 1
    fi

    # 🚀 run without changing directories; pass SCENE ROOT (not images/)
    ${RUNPY} Depth-Anything-V2/depth_scale.py --base_dir "${ABS_BASE}"
  fi
else
  echo "[depth] skipped"
fi



# 4) Train (floaters)
if [[ "${SKIP_TRAIN}" != "1" ]]; then
  echo "[train] source=${BASE}"
  ${RUNPY} trainFloaters.py -s "${BASE}"
else
  echo "[train] skipped"
fi

echo "[done]"
