#!/bin/sh
set -e

WEIGHTS_DIR="${WEIGHTS_DIR:-./weights}"
INPUT_DIR="${INPUT_DIR:-./source_video}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"

PRE_TRAINED_PATH="${WEIGHTS_DIR}/stable-video-diffusion-img2vid-xt-1-1"
DEPTHCRAFTER_UNET="${WEIGHTS_DIR}/DepthCrafter"
STEREOCRAFTER_UNET="${WEIGHTS_DIR}/StereoCrafter"

TILE_NUM="${TILE_NUM:-2}"
MAX_DISP="${MAX_DISP:-20.0}"

mkdir -p "${OUTPUT_DIR}"

for video in "${INPUT_DIR}"/*.mp4; do
    [ -f "${video}" ] || continue
    name="$(basename "${video}" .mp4)"
    splatting_output="${OUTPUT_DIR}/${name}_splatting_results.mp4"

    echo "==> Processing: ${video}"

    python /app/depth_splatting_inference.py \
        --pre_trained_path "${PRE_TRAINED_PATH}" \
        --unet_path "${DEPTHCRAFTER_UNET}" \
        --input_video_path "${video}" \
        --output_video_path "${splatting_output}" \
        --max_disp "${MAX_DISP}"

    python /app/inpainting_inference.py \
        --pre_trained_path "${PRE_TRAINED_PATH}" \
        --unet_path "${STEREOCRAFTER_UNET}" \
        --input_video_path "${splatting_output}" \
        --save_dir "${OUTPUT_DIR}" \
        --tile_num "${TILE_NUM}"

    echo "==> Done: ${name}"
done
