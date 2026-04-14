
<div align="center">
<h2>StereoCrafter: Diffusion-based Generation of Long and High-fidelity Stereoscopic 3D from Monocular Videos</h2>

Sijie Zhao*&emsp;
Wenbo Hu*&emsp;
Xiaodong Cun*&emsp;
Yong Zhang&dagger;&emsp;
Xiaoyu Li&dagger;&emsp;<br>
Zhe Kong&emsp;
Xiangjun Gao&emsp;
Muyao Niu&emsp;
Ying Shan

&emsp;* equal contribution &emsp; &dagger; corresponding author 

<h3>Tencent AI Lab&emsp;&emsp;ARC Lab, Tencent PCG</h3>

<a href='https://arxiv.org/abs/2409.07447'><img src='https://img.shields.io/badge/arXiv-PDF-a92225'></a> &emsp;
<a href='https://stereocrafter.github.io/'><img src='https://img.shields.io/badge/Project_Page-Page-64fefe' alt='Project Page'></a> &emsp;
<a href='https://huggingface.co/TencentARC/StereoCrafter'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Weights-yellow'></a>
</div>

## 💡 Abstract

We propose a novel framework to convert any 2D videos to immersive stereoscopic 3D ones that can be viewed on different display devices, like 3D Glasses, Apple Vision Pro and 3D Display. It can be applied to various video sources, such as movies, vlogs, 3D cartoons, and AIGC videos.

![teaser](assets/teaser.jpg)

## 📣 News
- `2024/12/27` We released our inference code and model weights.
- `2024/09/11` We submitted our technical report on arXiv and released our project page.

## 🎞️ Showcases
Here we show some examples of input videos and their corresponding stereo outputs in Anaglyph 3D format.
<div align="center">
    <img src="assets/demo.gif">
</div>


## 🛠️ Installation

#### 1. Set up the environment
We run our code on Python 3.8 and Cuda 11.8.
You can use Anaconda or Docker to build this basic environment.

#### 2. Clone the repo
```bash
# use --recursive to clone dependent submodules (Forward-Warp, Video-Depth-Anything)
git clone --recursive https://github.com/TencentARC/StereoCrafter
cd StereoCrafter
```

#### 3. Install the requirements

PyTorch must match your CUDA version. The Apptainer base image uses **CUDA 11.8**; install the `cu118` wheels first, then the rest:

```bash
pip install torch==2.1.1 torchvision==0.16.1 xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

For **CUDA 12.1**, use `--index-url https://download.pytorch.org/whl/cu121` instead. See comments at the top of `requirements.txt`.


#### 4. Install customized 'Forward-Warp' package for forward splatting
```
cd ./dependency/Forward-Warp
chmod a+x install.sh
./install.sh
```


## 📦 Model Weights

#### 1. Download the [SVD img2vid model](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1) for the image encoder and VAE.

```bash
# in StereoCrafter project root directory
mkdir weights
cd ./weights
git lfs install
git clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1
```

#### 2. Download **Metric Video Depth Anything** weights for depth (meters-scale, used for disparity `f·B/Z`).

From the project root, either copy checkpoints into `./weights/` (recommended for StereoCrafter defaults) or use `dependency/Video-Depth-Anything/checkpoints/` and set `VIDEO_DEPTH_CHECKPOINT`.

```bash
mkdir -p weights
# Large metric model (default encoder vitl)
wget -O weights/metric_video_depth_anything_vitl.pth \
  https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_vitl.pth
```

Smaller / other encoders: see [Video Depth Anything README](dependency/Video-Depth-Anything/README.md) or run `dependency/Video-Depth-Anything/get_weights.sh` (downloads into that repo’s `checkpoints/` folder).

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
- `--video_depth_checkpoint`: Metric VDA weights (default: `./weights/metric_video_depth_anything_vitl.pth` or env `VIDEO_DEPTH_CHECKPOINT`).
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
