#!/usr/bin/env bash
set -euo pipefail


ts_now() { date -Iseconds; }                              
epoch() { date +%s; }                                     # seconds since epoch
fmt_hms() {                                               
  local s=${1:-0}; printf "%02d:%02d:%02d" $((s/3600)) $(((s%3600)/60)) $((s%60));
}
log() { echo -e "$@"; }

STAGES=(extract sfm depth train)
declare -A ST_START ST_END ST_DUR ST_STATUS
for s in "${STAGES[@]}"; do ST_START[$s]=""; ST_END[$s]=""; ST_DUR[$s]=0; ST_STATUS[$s]="skipped"; done
CURRENT_STAGE=""

START_TS=$(epoch)
START_HUMAN=$(ts_now)

PROJECT="${PROJECT:-}"
BASE="data/${PROJECT:-unknown}"

append_csv_row() {
  local f="$1" project="$2" stage="$3" s_start="$4" s_end="$5" s_dur="$6" s_hms="$7" s_status="$8"
  if [[ ! -f "$f" ]]; then
    echo "project,stage,start_time,end_time,duration_sec,duration_hms,status" > "$f"
  fi
  echo "${project},${stage},${s_start},${s_end},${s_dur},${s_hms},${s_status}" >> "$f"
}

finalize() {
  local exit_code=$?
  local end_ts end_human total file_csv file_txt
  end_ts=$(epoch); end_human=$(ts_now)
  total=$(( end_ts - START_TS ))

  if [[ -n "${CURRENT_STAGE}" ]]; then
    local s="$CURRENT_STAGE"
    if [[ -n "${ST_START[$s]}" && -z "${ST_END[$s]}" ]]; then
      ST_END[$s]="${end_human}"
      
      local _start_epoch="${__EPOCH_START:-$START_TS}"
      ST_DUR[$s]=$(( end_ts - _start_epoch ))
      ST_STATUS[$s]="error"
    fi
  fi

  mkdir -p "${BASE}"
  file_csv="${BASE}/pipeline_times.csv"
  file_txt="${BASE}/pipeline_times.txt"

  # CSV summary
  for s in "${STAGES[@]}"; do
    append_csv_row "$file_csv" "${PROJECT}" "${s}" \
      "${ST_START[$s]}" "${ST_END[$s]}" "${ST_DUR[$s]}" "$(fmt_hms ${ST_DUR[$s]})" "${ST_STATUS[$s]}"
  done
  append_csv_row "$file_csv" "${PROJECT}" "total" \
    "${START_HUMAN}" "${end_human}" "${total}" "$(fmt_hms ${total})" "completed"

  # Text summary
  {
    echo "=================== TIMING SUMMARY ==================="
    echo "Project:   ${PROJECT}"
    echo "Started:   ${START_HUMAN}"
    echo "Finished:  ${end_human}"
    echo "------------------------------------------------------"
    for s in "${STAGES[@]}"; do
      printf "%-8s  %s  (%s)\n" \
        "$(tr '[:lower:]' '[:upper:]' <<< "${s:0:1}")${s:1}" \
        "$(fmt_hms ${ST_DUR[$s]})" \
        "${ST_STATUS[$s]}"
    done
    echo "------------------------------------------------------"
    echo "Total:     $(fmt_hms ${total})"
    echo "CSV:       ${file_csv}"
    echo "======================================================"
  } > "${file_txt}"

  # Summary to console
  cat "${file_txt}"

  exit $exit_code
}
trap finalize EXIT

stage_start() {
  local s="$1"
  CURRENT_STAGE="$s"
  ST_STATUS[$s]="running"
  ST_START[$s]="$(ts_now)"
  __EPOCH_START=$(epoch)  
  log "[$s][${ST_START[$s]}] START"
}
stage_end() {
  local s="$1"
  local now="$(ts_now)"
  local end_epoch="$(epoch)"
  ST_END[$s]="$now"
  ST_DUR[$s]=$(( end_epoch - __EPOCH_START ))
  ST_STATUS[$s]="ok"
  log "[$s][${now}] DONE in $(fmt_hms ${ST_DUR[$s]})"
  CURRENT_STAGE=""
  __EPOCH_START=""
}

VIDEO_URL="${VIDEO_URL:-}"          # auto-extract frames if provided
NUM_IMAGES="${NUM_IMAGES:-120}"
PROJECT="${PROJECT:-}"              # if empty, auto-derive
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
SKIP_SFM="${SKIP_SFM:-0}"
SKIP_DEPTH="${SKIP_DEPTH:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

# Depth checkpoint
DEPTH_CKPT_URL="${DEPTH_CKPT_URL:-}"
CANON_CKPT="Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth"

#RUNPY="micromamba run -n gs python"
RUNPY="python"

if [[ -z "${PROJECT}" ]]; then
  if [[ -n "${VIDEO_URL}" ]]; then
    bn="$(basename "${VIDEO_URL}")"
    PROJECT="${bn%%.*}"
  else
    PROJECT="proj-$(date +%Y%m%d-%H%M%S)"
  fi
fi

BASE="data/${PROJECT}"                  # scene root
IMAGES="${BASE}/images"
mkdir -p "${IMAGES}"

# Absolute paths
ABS_BASE="$(readlink -f "${BASE}")"
ABS_IMAGES="${ABS_BASE}/images"

log "[pipeline] start=${START_HUMAN}"
log "[pipeline] project=${PROJECT}"
log "[paths]    base=${BASE}  (abs=${ABS_BASE})"
log "[paths]    images=${IMAGES} (abs=${ABS_IMAGES})"


# 1) Extract frames 
if [[ "${SKIP_EXTRACT}" != "1" ]]; then
  stage_start "extract"
  if [[ -n "${VIDEO_URL}" ]]; then
    log "[extract] url=${VIDEO_URL} num=${NUM_IMAGES}"
    printf '%s\n%s\n%s\n' "${VIDEO_URL}" "${NUM_IMAGES}" "${IMAGES}" \
      | ${RUNPY} extractImg.py -s
  else
    log "[extract] VIDEO_URL not set; expecting images pre-existing in ${IMAGES}"
  fi
  stage_end "extract"
else
  log "[extract] skipped"
fi

# 2) SfM (COLMAP)

if [[ "${SKIP_SFM}" != "1" ]]; then
  if [[ ! -d "${ABS_IMAGES}" ]]; then
    log "[sfm] ERROR: expected '${ABS_IMAGES}' but it doesn't exist."; exit 1
  fi
  stage_start "sfm"
  ${RUNPY} Enhanced-Structure-from-Motion/sfm_pipeline.py \
    --input_dir "${ABS_IMAGES}" \
    --output_dir "${ABS_BASE}" \
    --feature_extractor aliked \
    --use_vocab_tree
  stage_end "sfm"
else
  log "[sfm] skipped"
fi


# 3) Depth-Anything V2 scaling 

if [[ "${SKIP_DEPTH}" != "1" ]]; then
  mkdir -p "Depth-Anything-V2/checkpoints" "checkpoints"
  if [[ ! -f "${CANON_CKPT}" ]]; then
    if [[ -n "${DEPTH_CKPT_URL}" ]]; then
      log "[depth] downloading checkpoint..."
      curl -L --fail --progress-bar -o "${CANON_CKPT}" "${DEPTH_CKPT_URL}"
      SZ=$(stat -c%s "${CANON_CKPT}" || echo 0)
      if (( SZ < 10*1024*1024 )); then
        log "[depth] ERROR: downloaded file too small (${SZ} bytes) — check DEPTH_CKPT_URL/network"; exit 1
      fi
    else
      log "[depth] checkpoint missing and DEPTH_CKPT_URL not provided; skipping depth."
      SKIP_DEPTH=1
    fi
  fi

  if [[ "${SKIP_DEPTH}" != "1" ]]; then
    ln -sfn "../Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth" \
            "checkpoints/depth_anything_v2_vitl.pth"

    if [[ ! -d "${ABS_IMAGES}" ]]; then
      log "[depth] ERROR: expected '${ABS_IMAGES}' but it doesn't exist."; exit 1
    fi

    stage_start "depth"
    ${RUNPY} Depth-Anything-V2/depth_scale.py --base_dir "${ABS_BASE}"
    stage_end "depth"
  fi
else
  log "[depth] skipped"
fi

# 4) Train

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  if [[ ! -d "${ABS_BASE}" ]]; then
    log "[train] ERROR: scene root '${ABS_BASE}' not found."; exit 1
  fi
  stage_start "train"
  ${RUNPY} trainFloaters.py -s "${ABS_BASE}"
  stage_end "train"
else
  log "[train] skipped"
fi


