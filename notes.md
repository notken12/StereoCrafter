# Local setup (skip if using HPC)
make sure to install git lfs before cloning weights (brew install git-lfs)

create huggingface account and use personal access token with read perms to clone weights

use python 3.10
requirements.txt is not working, you should install each of the packages manually

decord doesn't work for M4 macs, use eva-decord instead

# Check the installed version (e.g., gcc-13)
`brew list gcc` 

# Install xformers using the specific GCC version
CC=gcc-[version] CXX=g++-[version] pip install xformers
# Example: CC=gcc-13 CXX=g++-13 pip install xformers

# HPC Setup

Clone the repo into /scratch/your_computing_id/stereocrafter and cd into that directory.

## Run interactive job
Start an interactive job to allocate GPUs:

```
ijob -A shrew-crew -p gpu --gres=gpu:a100:1 --mem=64G -c 8
```
```
```

## Build apptainers

```
apptainer build stereocrafter-base.sif stereocrafter-base.def
apptainer build stereocrafter.sif stereocrafter.def
```

## Run stereocrafter by itself (for testing)

If you want to run it with the web UI, just skip to Run Frontend.

```
module load apptainer
apptainer exec --nv --env MAX_DISP=3.80 stereocrafter.sif sh run_inference.sh 
```

## Download results

```
rsync -avz --progress jjq7qj@login.hpc.virginia.edu:/scratch/jjq7qj/stereocrafter/outputs/ ~/Downloads/stereocrafter-outputs/
```

## Run frontend
Start inference ijob (see above)
run `hostname`, record down what you get (it should look like "udc-an34-31")

```
module load apptainer
apptainer exec --nv stereocrafter.sif python frontend/main.py
```

From your computer:

```
ssh -NL 6767:HOSTNAME_HERE:6767 -Y jjq7qj@login.hpc.virginia.edu
```

Now go to `localhost:6767` in your browser and the UI will be there.

---

# Codebase / agent notes (non-obvious)

## Architecture 

StereoCrafter here is executed **on a university HPC GPU node** inside **Apptainer**: a **two-layer image** builds **CUDA 11.8 + Python + PyTorch (cu118) + Forward-Warp** in `stereocrafter-base.sif`, then a thinner **`stereocrafter.sif`** layers in the app (code, Video-Depth-Anything dependency, optional metric checkpoint fetch in `%post`). **Weights** for Stable Video Diffusion and StereoCrafter typically live under **`/scratch/.../weights`** on the cluster (or bind-mounted over `/app`); **metric depth** defaults to **`dependency/Video-Depth-Anything/checkpoints`**.

The **processing pipeline** is monocular video → **streaming Metric Video Depth Anything** (per-frame, temporal cache, depths written to a **memory-mapped float32 file**) → **forward-warp splatting** with **`stereo_scale * f * B / Z`** (baseline from IPD, focal from HFOV or override) → either **fast preview** (SBS/anaglyph stitched from the splatting grid, **no diffusion**) or **chunked Stable Video Diffusion inpainting** that reads the splatting MP4 in windows and writes SBS/anaglyph incrementally.

**Interactive use:** an **ijob** allocates **GPU + RAM**; **`apptainer exec --nv stereocrafter.sif python frontend/main.py`** starts **Gradio** on the node. From a laptop, **SSH local port forwarding** exposes **`localhost:6767`** to the browser. That pattern keeps heavy dependencies inside the image while data and large model dirs stay on **scratch**.

---


## Depth + splatting (metric, streaming)

- **Depth model:** **Metric Video Depth Anything** via **`video_depth_anything.video_depth_stream`** (streaming), not the offline `video_depth.py` path. Wrapper: **`VideoDepthAnythingStreamingDemo`** in [`depth_splatting_inference.py`](depth_splatting_inference.py). **`reset_streaming()`** state must be fresh per video (handled inside `infer_streaming`).
- **Depth on disk:** Streaming writes **`float32` depths** to a temporary **`.depth.mmap`** (`(T, H, W)`), then **`DepthSplatting`** reads slices. File is **unlinked** after splatting in the frontend/CLI unless you comment that out for debugging.
- **Frame alignment:** `read_video_frames` (batch path) uses **FPS stride**; streaming **`infer_streaming`** builds a strided index list (`range(0, len(vid), stride)`). Depth row `t` corresponds to **source frame index** `depth_frame_indices[t]`, **not** always `t`. **`DepthSplatting`** must get **`depth_frame_indices`** from `infer_streaming` or splatting reads the wrong video frames vs depth. **`speedup_rate > 1`** in **`infer_streaming`** sets **`stride = max(1, int(round(speedup_rate)))`** (overrides **`target_fps`** for stride); **`speedup_rate == 1`** keeps the **`target_fps`**-based stride. CLI **`depth_splatting_inference.py`** exposes **`--speedup_rate`**.
- **Disparity:** `disp = stereo_scale * f_x * B / clamp(Z, min_depth)` (meters for `B` and `Z`; `f_x` in pixels). **`stereo_scale`** is a UI/CLI knob to compensate for bad metric scale without retraining.
- **HFOV → f:** `f_x = (width/2) / tan(HFOV/2)` using **full-frame pixel width** from the **source** video (first frame of the batch path). Wrong HFOV directly scales disparity.

## Inpainting vs fast preview

- **Full pipeline:** Chunked **diffusion inpainting** in [`frontend/main.py`](frontend/main.py) reads the **splatting MP4** in **temporal chunks** (does **not** load all frames at once).
- **Fast preview:** Checkbox skips inpainting; **`write_preview_sbs_from_splatting`** stitches **top-left** (left eye) and **bottom-right** (warped right) of the **2×2 splatting grid**. Holes/disocclusions are **expected**; diffusion normally fills them.
- **Preview speedup (fast preview only):** Gradio **`Preview speedup`** passes **`speedup_rate`** into **`infer_streaming`**. Values **> 1** subsample **every N-th source frame in order** with **`N = max(1, int(round(speedup_rate)))`**, so with **Process length = -1** the **whole timeline** is covered with fewer frames. Splatting still writes at the **source FPS**, so the preview MP4 is **shorter** and plays **faster** (~N×). Full runs (fast preview off) use **`speedup_rate = 1`**. **Process length** still limits how many **strided** frames depth+splat process (same semantics as before; **-1** = all strided frames through the clip).

## Weights paths

- **Metric VDA:** Default checkpoint under **`dependency/Video-Depth-Anything/checkpoints/`** (e.g. `metric_video_depth_anything_vitl.pth`). Env: **`VIDEO_DEPTH_CHECKPOINT`**, **`VIDEO_DEPTH_ENCODER`**.
- **SVD + StereoCrafter:** Still under **`./weights/`** (or env **`PRE_TRAINED_PATH`**, **`STEREOCRAFTER_PATH`**). **Not** auto-downloaded in `stereocrafter.def` for metric depth only; base image may bake metric weights into that checkpoints dir via `%post` wget.
- **DepthCrafter** was removed from the active pipeline and **`.gitmodules`**; old submodule dir may still exist locally.

## Containers / PyTorch CUDA

- **Base image** is **CUDA 11.8**. **PyTorch must be cu118** or **`torch.utils.cpp_extension` fails** when building Forward-Warp (`CUDA_MISMATCH_MESSAGE`). See [`stereocrafter-base.def`](stereocrafter-base.def): install **torch/torchvision from `https://download.pytorch.org/whl/cu118`** before [`requirements.txt`](requirements.txt) (torch is **not** in the root requirements file—installed in two steps in the def).

## Apptainer bind mounts

- **`-B /scratch/.../stereocrafter:/app`** replaces **entire** `/app` in the image. Baked-in files under `/app/dependency/.../checkpoints` **disappear** unless those paths exist on the host. **Either** download weights on the host into the same paths **or** bind **only** e.g. **`frontend/`** → `/app/frontend` so the image keeps its checkpoints.

## CLI

- **Splatting + depth:** `python depth_splatting_inference.py` (Fire). Flags include **`--stereo_scale`**, **`--speedup_rate`** (default 1). Uses streaming + memmap internally.
- **Batch stereo:** [`run_inference.sh`](run_inference.sh) — depth checkpoint default path uses **`$ROOT/dependency/Video-Depth-Anything/checkpoints/...`**.

## Memory / complexity gotchas

- Streaming depth avoids holding **all RGB** in RAM; **memmap** is **O(T)** on **disk**, not peak RAM for full depth tensor + full vis. **`depth_sequence_to_vis`** full-sequence path is avoided in splatting when using **`depth_vis_bounds`** + per-batch inferno.
- **Inpainting** path is chunked; **spatial_tiled_process** memory scales with **tile_num²** per **temporal** chunk, not `T` for tiles.
- **“Killed”** on Linux/HPC: usually **OOM** or Slurm memory limit—check `dmesg` / job `MaxRSS`.

## VDA streaming vs offline

- Streaming is **faster / lower RAM** but upstream documents **possible quality drop** vs offline `infer_video_depth`. Metric checkpoints still load into the **streaming** module (`run_streaming.py` pattern).

## Submodule / PYTHONPATH

- VDA imports expect **`dependency/Video-Depth-Anything`** on **`sys.path`** (inserted at top of [`depth_splatting_inference.py`](depth_splatting_inference.py)). Container sets **`PYTHONPATH=/app`**; code also prepends **`_VDA_ROOT`** for host runs from repo root.
