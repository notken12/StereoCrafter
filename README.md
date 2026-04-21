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
- **Processing speedup (Gradio):** **`Processing speedup`** passes **`speedup_rate`** into **`infer_streaming`** for **every** run (fast preview **or** full inpainting). Values **> 1** subsample **every N-th source frame in order** with **`N = max(1, int(round(speedup_rate)))`**, so with **Process length = -1** the **whole timeline** is covered with fewer frames. Splatting still writes at the **source FPS**, so the splatting MP4 is **shorter** and plays **faster** (~N×). **Process length** still limits how many **strided** frames depth+splat process (**-1** = all strided frames through the clip). With speedup **> 1**, inpainting runs on that **shorter** splatting video (fewer chunks, faster, not frame-aligned to the original frame count).

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
mkdir weights
cd ./weights
git lfs install
git clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1
```

#### 2. Download **Metric Video Depth Anything** weights for depth (meters-scale, used for disparity `f·B/Z`).

StereoCrafter expects checkpoints under **`dependency/Video-Depth-Anything/checkpoints/`** (same layout as upstream VDA). From the project root:

```bash
(cd dependency/Video-Depth-Anything && bash get_weights.sh)
```

Or fetch only the large metric model (default encoder `vitl`):

```bash
mkdir -p dependency/Video-Depth-Anything/checkpoints
wget -O dependency/Video-Depth-Anything/checkpoints/metric_video_depth_anything_vitl.pth \
  https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_vitl.pth
```

Override the path with env **`VIDEO_DEPTH_CHECKPOINT`** if needed. Other encoders: [Video Depth Anything README](dependency/Video-Depth-Anything/README.md).

**Camera geometry:** splatting uses baseline `B = IPD_mm / 1000` and focal length from horizontal FOV (`f = (W/2) / tan(HFOV/2)`) or an explicit `focal_length_px` override — see the Gradio settings or `depth_splatting_inference.py` / `run_inference.sh` env vars (`HFOV_DEG`, `IPD_MM`, `MAX_DISP_PX`).

#### 3. Download the [StereoCrafter model](https://huggingface.co/TencentARC/StereoCrafter) for the stereo video generation.
```bash
git clone https://huggingface.co/TencentARC/StereoCrafter
```


## 🔄 Inference

Script:

```bash
# in StereoCrafter project root directory
sh run_inference.sh
```

There are two main steps in this script for generating stereo video.

#### 1. Depth-Based Video Splatting (Metric Video Depth Anything)
Execute the following command (see `python depth_splatting_inference.py --help` for all flags):
```bash
python depth_splatting_inference.py \
  --input_video_path [PATH] \
  --output_video_path [PATH] \
  --video_depth_checkpoint [PATH_TO_metric_video_depth_anything_vitl.pth]
```
Arguments (selected):
- `--video_depth_checkpoint`: Metric VDA weights (default: `dependency/Video-Depth-Anything/checkpoints/metric_video_depth_anything_vitl.pth` or env `VIDEO_DEPTH_CHECKPOINT`).
- `--encoder`: `vits`, `vitb`, or `vitl` (must match the checkpoint).
- `--horizontal_fov_deg` / `--focal_length_px`: Camera focal length for `disp = f·B/Z` (if `focal_length_px` > 0, HFOV is ignored).
- `--ipd_mm`: Interpupillary distance in millimeters (baseline `B` in meters).
- `--max_disp_px`: Optional upper clamp on disparity in pixels (stability).
- `--input_video_path`: Input video (e.g., `./source_video/camel.mp4`).
- `--output_video_path`: Splatting grid output (e.g., `./outputs/camel_splatting_results.mp4`).

The first step generates a video grid with input video, visualized depth map, occlusion mask, and splatting right video, as shown below:

<img src="assets/camel_splatting_results.jpg" alt="camel_splatting_results" width="800"/> 

#### 2. Stereo Video Inpainting of the Splatting Video
Execute the following command:
```bash
python inpainting_inference.py --pre_trained_path [PATH] --unet_path [PATH]
                               --input_video_path [PATH] --save_dir [PATH]
```
Arguments:
- `--pre_trained_path`: Path to the SVD img2vid model weights (e.g., `./weights/stable-video-diffusion-img2vid-xt-1-1`).
- `--unet_path`: Path to the StereoCrafter model weights (e.g., `./weights/StereoCrafter`).
- `--input_video_path`: Path to the splatting video result generated by the first stage (e.g., `./outputs/camel_splatting_results.mp4`).
- `--save_dir`: Directory for the output stereo video (e.g., `./outputs`).
- `--tile_num`: The number of tiles in width and height dimensions for tiled processing, which allows for handling high resolution input without requiring more GPU memory. The default value is `1` (1 $\times$ 1 tile). For input videos with a resolution of 2K or higher, you could use more tiles to avoid running out of memory.

The stereo video inpainting generates the stereo video result in side-by-side format and anaglyph 3D format, as shown below:

<img src="assets/camel_sbs.jpg" alt="camel_sbs" width="800"/> 

<img src="assets/camel_anaglyph.jpg" alt="camel_anaglyph" width="400"/>

## 🤝 Acknowledgements

We would like to express our gratitude to the following open-source projects:
- [Stable Video Diffusion](https://github.com/Stability-AI/generative-models): A latent diffusion model trained to generate video clips from an image or text conditioning.
- [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything): Consistent video depth; **Metric** checkpoints enable physically motivated disparity for splatting.


## 📚 Citation

```bibtex
@article{zhao2024stereocrafter,
  title={Stereocrafter: Diffusion-based generation of long and high-fidelity stereoscopic 3d from monocular videos},
  author={Zhao, Sijie and Hu, Wenbo and Cun, Xiaodong and Zhang, Yong and Li, Xiaoyu and Kong, Zhe and Gao, Xiangjun and Niu, Muyao and Shan, Ying},
  journal={arXiv preprint arXiv:2409.07447},
  year={2024}
}
```
