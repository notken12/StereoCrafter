import gc
import os
import uuid

import cv2
import numpy as np
import torch
import gradio as gr
from decord import VideoReader, cpu
from transformers import CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel

from depth_splatting_inference import (
    VideoDepthAnythingStreamingDemo,
    DepthSplatting,
    write_preview_sbs_from_splatting,
)
from pipelines.stereo_video_inpainting import (
    StableVideoDiffusionInpaintingPipeline,
    tensor2vid,
)
from inpainting_inference import spatial_tiled_process

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VDA_CHECKPOINTS = os.path.join(
    _REPO_ROOT, "dependency", "Video-Depth-Anything", "checkpoints"
)

PRE_TRAINED_PATH = os.environ.get(
    "PRE_TRAINED_PATH", "./weights/stable-video-diffusion-img2vid-xt-1-1"
)
STEREOCRAFTER_PATH = os.environ.get("STEREOCRAFTER_PATH", "./weights/StereoCrafter")
VIDEO_DEPTH_CHECKPOINT = os.environ.get(
    "VIDEO_DEPTH_CHECKPOINT",
    os.path.join(_VDA_CHECKPOINTS, "metric_video_depth_anything_vitl.pth"),
)
VIDEO_DEPTH_ENCODER = os.environ.get("VIDEO_DEPTH_ENCODER", "vitl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_models():
    print("Loading Metric Video Depth Anything (streaming)...")
    if not os.path.isfile(VIDEO_DEPTH_CHECKPOINT):
        raise FileNotFoundError(
            f"Missing depth weights: {VIDEO_DEPTH_CHECKPOINT}\n"
            "Download the metric checkpoint into dependency/Video-Depth-Anything/checkpoints (same as upstream VDA), e.g.:\n"
            "  (cd dependency/Video-Depth-Anything && bash get_weights.sh)\n"
            " # or: wget -O dependency/Video-Depth-Anything/checkpoints/metric_video_depth_anything_vitl.pth \\\n"
            "  # https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_vitl.pth\n"
            "Or set VIDEO_DEPTH_CHECKPOINT to an existing .pth path. "
            "The Apptainer app image downloads into that checkpoints directory during build."
        )
    depth_model = VideoDepthAnythingStreamingDemo(
        checkpoint_path=VIDEO_DEPTH_CHECKPOINT,
        encoder=VIDEO_DEPTH_ENCODER,
    )

    print("Loading StereoCrafter pipeline...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        PRE_TRAINED_PATH,
        subfolder="image_encoder",
        variant="fp16",
        torch_dtype=torch.float16,
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        PRE_TRAINED_PATH, subfolder="vae", variant="fp16", torch_dtype=torch.float16
    )
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        STEREOCRAFTER_PATH,
        subfolder="unet_diffusers",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    inpainting_pipeline = StableVideoDiffusionInpaintingPipeline.from_pretrained(
        PRE_TRAINED_PATH,
        image_encoder=image_encoder,
        vae=vae,
        unet=unet,
        torch_dtype=torch.float16,
    ).to("cuda")

    print("All models loaded.")
    return depth_model, inpainting_pipeline


depth_model, inpainting_pipeline = load_models()


def process_single_video(
    input_video: str,
    ipd_mm: float,
    horizontal_fov_deg: float,
    focal_length_px: float,
    max_disp_px: float,
    process_length: int,
    tile_num: int,
    max_res: int,
    fast_preview: bool,
    stereo_scale: float,
    preview_speedup: float,
    progress_offset: float,
    progress_scale: float,
    progress,
) -> tuple[str, str, str]:
    stem = os.path.splitext(os.path.basename(input_video))[0]
    job_id = uuid.uuid4().hex[:8]
    splatting_path = os.path.join(OUTPUT_DIR, f"{stem}_{job_id}_splatting_results.mp4")
    sbs_path = os.path.join(OUTPUT_DIR, f"{stem}_{job_id}_sbs.mp4")
    anaglyph_path = os.path.join(OUTPUT_DIR, f"{stem}_{job_id}_anaglyph.mp4")

    def p(frac, desc):
        progress(progress_offset + frac * progress_scale, desc=desc)

    baseline_m = float(ipd_mm) * 1e-3
    f_px = float(focal_length_px) if focal_length_px > 0 else None
    hfov = float(horizontal_fov_deg) if f_px is None else None
    clamp_px = float(max_disp_px) if max_disp_px > 0 else None
    pl = int(process_length)
    su = max(1.0, float(preview_speedup)) if fast_preview else 1.0

    mmap_path = os.path.splitext(splatting_path)[0] + ".depth.mmap"
    p(0.0, f"[{stem}] Estimating metric depth (streaming VDA)...")
    depth_mm, depth_bounds, depth_fidx = depth_model.infer_streaming(
        input_video_path=input_video,
        memmap_path=mmap_path,
        process_length=int(pl),
        max_res=int(max_res),
        speedup_rate=float(su),
    )

    try:
        p(0.35, f"[{stem}] Splatting (disp = stereo_scale * f*B/Z)...")
        DepthSplatting(
            input_video_path=input_video,
            output_video_path=splatting_path,
            video_depth=depth_mm,
            depth_vis=None,
            process_length=int(pl),
            batch_size=10,
            use_metric_depth=True,
            focal_length_px=f_px,
            horizontal_fov_deg=hfov,
            baseline_m=baseline_m,
            min_depth_m=1e-3,
            max_disp_px=clamp_px,
            depth_vis_bounds=depth_bounds,
            depth_frame_indices=depth_fidx,
            stereo_scale=float(stereo_scale),
        )
    finally:
        del depth_mm
        gc.collect()
        try:
            os.unlink(mmap_path)
        except OSError:
            pass

    if fast_preview:
        p(0.5, f"[{stem}] Fast preview: SBS/anaglyph from splatting (skip diffusion)...")
        write_preview_sbs_from_splatting(splatting_path, sbs_path, anaglyph_path)
        gc.collect()
        torch.cuda.empty_cache()
        return sbs_path, anaglyph_path, splatting_path

    p(0.5, f"[{stem}] Running stereo inpainting (chunked)...")
    frames_chunk = 23
    overlap = 3

    video_reader = VideoReader(splatting_path, ctx=cpu(0))
    fps = video_reader.get_avg_fps()
    num_frames = len(video_reader)

    f0 = video_reader[0].asnumpy()
    height_full, width_full = f0.shape[0] // 2, f0.shape[1] // 2
    h128 = height_full // 128 * 128
    w128 = width_full // 128 * 128

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_sbs = cv2.VideoWriter(sbs_path, fourcc, fps, (w128 * 2, h128))
    out_ana = cv2.VideoWriter(anaglyph_path, fourcc, fps, (w128, h128))

    generated = None
    for i in range(0, num_frames, frames_chunk - overlap):
        if i + overlap >= num_frames:
            break

        if generated is not None and i + frames_chunk > num_frames:
            cur_i = max(num_frames + overlap - frames_chunk, 0)
            cur_overlap = i - cur_i + overlap
        else:
            cur_i = i
            cur_overlap = overlap

        end_i = min(cur_i + frames_chunk, num_frames)
        idx_list = list(range(cur_i, end_i))
        chunk = video_reader.get_batch(idx_list).asnumpy()
        frames = torch.from_numpy(chunk).permute(0, 3, 1, 2).float()

        height, width = frames.shape[2] // 2, frames.shape[3] // 2
        frames_left = frames[:, :, :height, :width]
        frames_mask = frames[:, :, height:, :width]
        frames_warpped = frames[:, :, height:, width:]
        frames = torch.cat([frames_warpped, frames_left, frames_mask], dim=0)

        frames = frames[:, :, :h128, :w128] / 255.0
        frames_warpped, frames_left, frames_mask = torch.chunk(frames, chunks=3, dim=0)
        frames_mask = frames_mask.mean(dim=1, keepdim=True)

        input_frames_i = frames_warpped.clone()
        mask_frames_i = frames_mask

        if generated is not None:
            try:
                input_frames_i[:cur_overlap] = generated[-cur_overlap:]
            except Exception as e:
                print(e)

        video_latents = spatial_tiled_process(
            input_frames_i,
            mask_frames_i,
            inpainting_pipeline,
            int(tile_num),
            spatial_n_compress=8,
            min_guidance_scale=1.01,
            max_guidance_scale=1.01,
            decode_chunk_size=8,
            fps=7,
            motion_bucket_id=127,
            noise_aug_strength=0.0,
            num_inference_steps=8,
        )

        video_latents = video_latents.unsqueeze(0)
        video_frames = inpainting_pipeline.decode_latents(
            video_latents, num_frames=video_latents.shape[1], decode_chunk_size=2
        )
        video_frames = tensor2vid(
            video_frames, inpainting_pipeline.image_processor, output_type="pil"
        )[0]
        video_frames = [
            torch.tensor(np.array(f)).permute(2, 0, 1).float() / 255.0
            for f in video_frames
        ]

        generated = torch.stack(video_frames)
        start_left = cur_overlap if i != 0 else 0
        if i != 0:
            generated = generated[cur_overlap:]
        left_write = frames_left[start_left : start_left + generated.shape[0]]

        for k in range(generated.shape[0]):
            fl = (left_write[k] * 255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            fr = (generated[k] * 255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            sbs = np.concatenate([fl, fr], axis=1)
            out_sbs.write(cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR))
            al = fl.copy()
            al[:, :, 1:] = 0
            ar = fr.copy()
            ar[:, :, 0] = 0
            out_ana.write(cv2.cvtColor(al + ar, cv2.COLOR_RGB2BGR))

        p(0.5 + 0.45 * (i / num_frames), f"[{stem}] Running stereo inpainting...")

    out_sbs.release()
    out_ana.release()

    gc.collect()
    torch.cuda.empty_cache()

    return sbs_path, anaglyph_path, splatting_path


def run_inference(
    input_files: list,
    ipd_mm: float,
    horizontal_fov_deg: float,
    focal_length_px: float,
    max_disp_px: float,
    process_length: int,
    tile_num: int,
    max_res: int,
    fast_preview: bool,
    stereo_scale: float,
    preview_speedup: float,
    progress=gr.Progress(track_tqdm=True),
):
    if not input_files:
        raise gr.Error("Please upload at least one video.")

    n = len(input_files)
    all_outputs = []

    for idx, file in enumerate(input_files):
        video_path = file if isinstance(file, str) else file.name
        sbs, anaglyph, splatting = process_single_video(
            input_video=video_path,
            ipd_mm=float(ipd_mm),
            horizontal_fov_deg=float(horizontal_fov_deg),
            focal_length_px=float(focal_length_px),
            max_disp_px=float(max_disp_px),
            process_length=int(process_length),
            tile_num=int(tile_num),
            max_res=int(max_res),
            fast_preview=bool(fast_preview),
            stereo_scale=float(stereo_scale),
            preview_speedup=float(preview_speedup),
            progress_offset=idx / n,
            progress_scale=1.0 / n,
            progress=progress,
        )
        all_outputs.append((sbs, anaglyph, splatting))
        progress((idx + 1) / n, desc=f"Finished {idx + 1}/{n} videos.")

    last_sbs, last_anaglyph, last_splatting = all_outputs[-1]
    all_files = [
        path
        for sbs, anaglyph, splatting in all_outputs
        for path in (sbs, anaglyph, splatting)
    ]

    return last_sbs, last_anaglyph, last_splatting, all_files


def list_past_generations() -> list[str]:
    if not os.path.isdir(OUTPUT_DIR):
        return []
    files = sorted(
        (
            os.path.join(OUTPUT_DIR, f)
            for f in os.listdir(OUTPUT_DIR)
            if f.endswith(".mp4")
        ),
        key=os.path.getmtime,
        reverse=True,
    )
    return files


with gr.Blocks(title="StereoCrafter") as demo:
    gr.Markdown(
        """
        # StereoCrafter
        Convert monocular video to immersive stereoscopic 3D.
        Depth uses **Metric Video Depth Anything (streaming)**; splatting uses **stereo_scale × f·B / Z**. Use **Fast preview** to skip diffusion and subsample frames in order across the whole **Process length** (see **Preview speedup**); tune **Stereo scale** if metric depth is off.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_files = gr.File(
                label="Input Videos",
                file_count="multiple",
                file_types=["video"],
            )
            with gr.Accordion("Settings", open=True):
                ipd_mm = gr.Slider(
                    label="Interpupillary distance (mm)",
                    minimum=5.0,
                    maximum=80.0,
                    value=12.5,
                    step=0.1,
                    info="Real-world eye separation; sets stereo baseline B = IPD / 1000 in meters.",
                )
                horizontal_fov_deg = gr.Slider(
                    label="Horizontal field of view (°)",
                    minimum=20.0,
                    maximum=120.0,
                    value=55.0,
                    step=0.5,
                    info="Used to estimate focal length f from frame width: f = (W/2) / tan(HFOV/2). Ignored if focal length > 0.",
                )
                focal_length_px = gr.Slider(
                    label="Focal length override (px, 0 = use HFOV)",
                    minimum=0.0,
                    maximum=200000.0,
                    value=0.0,
                    step=1.0,
                    info="If > 0, use this fx in pixels instead of HFOV.",
                )
                max_disp_px = gr.Slider(
                    label="Max disparity clamp (px)",
                    minimum=0.0,
                    maximum=2000.0,
                    value=500.0,
                    step=1.0,
                    info="Caps splatting disparity for stability (0 disables clamp).",
                )
                process_length = gr.Slider(
                    label="Process Length (frames)",
                    minimum=-1,
                    maximum=200,
                    value=-1,
                    step=1,
                    info="-1 processes the full video.",
                )
                tile_num = gr.Slider(
                    label="Tile Number",
                    minimum=1,
                    maximum=4,
                    value=1,
                    step=1,
                    info="Increase for high-resolution video to reduce VRAM usage.",
                )
                max_res = gr.Slider(
                    label="Max Resolution",
                    minimum=256,
                    maximum=1024,
                    value=1024,
                    step=64,
                    info="Cap the longer edge for depth inference. Output splatting uses full-resolution video.",
                )
                fast_preview = gr.Checkbox(
                    label="Fast preview (no inpainting)",
                    value=False,
                    info="Skips StereoCrafter diffusion; SBS/anaglyph are raw splat-only (fast). Use to tune stereo_scale and camera settings.",
                )
                preview_speedup = gr.Slider(
                    label="Preview speedup (fast preview only)",
                    minimum=1.0,
                    maximum=16.0,
                    value=4.0,
                    step=0.5,
                    info="1 = every frame (slow). Higher = every ~N-th source frame in order, full timeline when Process length is -1; output keeps source FPS so playback is N× shorter/faster.",
                )
                stereo_scale = gr.Slider(
                    label="Stereo scale",
                    minimum=0.25,
                    maximum=4.0,
                    value=1.0,
                    step=0.05,
                    info="Multiplies disparity after f·B/Z (>1 = stronger 3D, same as assuming predicted depth is too large).",
                )
            run_btn = gr.Button("Generate Stereo Video", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("#### Preview (last video)")
            output_sbs = gr.Video(label="Side-by-Side", autoplay=True, loop=True)
            output_anaglyph = gr.Video(
                label="Anaglyph 3D (Red-Cyan glasses)", autoplay=True, loop=True
            )
            with gr.Accordion("Intermediate Results", open=False):
                output_splatting = gr.Video(
                    label="Depth Splatting Grid", autoplay=True, loop=True
                )
            gr.Markdown("#### Download All Outputs")
            output_files = gr.Files(
                label="All output videos (SBS, anaglyph, splatting)"
            )

    with gr.Accordion("Past Generations", open=False):
        refresh_btn = gr.Button("Refresh", size="sm")
        past_files = gr.Files(
            label="All output videos",
            value=list_past_generations,
        )
        refresh_btn.click(fn=list_past_generations, outputs=[past_files])

    run_btn.click(
        fn=run_inference,
        inputs=[
            input_files,
            ipd_mm,
            horizontal_fov_deg,
            focal_length_px,
            max_disp_px,
            process_length,
            tile_num,
            max_res,
            fast_preview,
            stereo_scale,
            preview_speedup,
        ],
        outputs=[output_sbs, output_anaglyph, output_splatting, output_files],
    ).then(
        fn=list_past_generations,
        outputs=[past_files],
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=6767)
