#!/bin/sh
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT}"

WEIGHTS_DIR="${WEIGHTS_DIR:-${ROOT}/weights}"
INPUT_DIR="${INPUT_DIR:-${ROOT}/source_video}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs}"

PRE_TRAINED_PATH="${WEIGHTS_DIR}/stable-video-diffusion-img2vid-xt-1-1"
VIDEO_DEPTH_CHECKPOINT="${VIDEO_DEPTH_CHECKPOINT:-${WEIGHTS_DIR}/metric_video_depth_anything_vitl.pth}"
STEREOCRAFTER_UNET="${WEIGHTS_DIR}/StereoCrafter"

TILE_NUM="${TILE_NUM:-2}"
HFOV_DEG="${HFOV_DEG:-55}"
IPD_MM="${IPD_MM:-63}"
MAX_DISP_PX="${MAX_DISP_PX:-500}"
VIDEO_DEPTH_ENCODER="${VIDEO_DEPTH_ENCODER:-vitl}"

mkdir -p "${OUTPUT_DIR}"

for video in "${INPUT_DIR}"/*.mp4; do
    [ -f "${video}" ] || continue
    name="$(basename "${video}" .mp4)"
    splatting_output="${OUTPUT_DIR}/${name}_splatting_results.mp4"

    echo "==> Processing: ${video}"

    python "${ROOT}/depth_splatting_inference.py" \
        --input_video_path="${video}" \
        --output_video_path="${splatting_output}" \
        --video_depth_checkpoint="${VIDEO_DEPTH_CHECKPOINT}" \
        --encoder="${VIDEO_DEPTH_ENCODER}" \
        --horizontal_fov_deg="${HFOV_DEG}" \
        --ipd_mm="${IPD_MM}" \
        --max_disp_px="${MAX_DISP_PX}"

    python "${ROOT}/inpainting_inference.py" \
        --pre_trained_path "${PRE_TRAINED_PATH}" \
        --unet_path "${STEREOCRAFTER_UNET}" \
        --input_video_path "${splatting_output}" \
        --save_dir "${OUTPUT_DIR}" \
        --tile_num "${TILE_NUM}"

    echo "==> Done: ${name}"
done
