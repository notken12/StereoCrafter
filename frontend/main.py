import gc
import os
import uuid

import numpy as np
import torch
import gradio as gr
from decord import VideoReader, cpu
from transformers import CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel

from depth_splatting_inference import DepthCrafterDemo, DepthSplatting
from pipelines.stereo_video_inpainting import (
    StableVideoDiffusionInpaintingPipeline,
    tensor2vid,
)
from inpainting_inference import spatial_tiled_process, write_video_opencv

HUMAN_MAX_DISP = 20.0
HUMAN_IPD = 63.0

PRE_TRAINED_PATH = os.environ.get(
    "PRE_TRAINED_PATH", "./weights/stable-video-diffusion-img2vid-xt-1-1"
)
DEPTHCRAFTER_PATH = os.environ.get("DEPTHCRAFTER_PATH", "./weights/DepthCrafter")
STEREOCRAFTER_PATH = os.environ.get("STEREOCRAFTER_PATH", "./weights/StereoCrafter")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_models():
    print("Loading DepthCrafter pipeline...")
    depthcrafter = DepthCrafterDemo(
        unet_path=DEPTHCRAFTER_PATH,
        pre_trained_path=PRE_TRAINED_PATH,
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
    return depthcrafter, inpainting_pipeline


depthcrafter, inpainting_pipeline = load_models()


def process_single_video(
    input_video: str,
    max_disp: float,
    process_length: int,
    tile_num: int,
    max_res: int,
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

    # Stage 1: depth estimation
    p(0.0, f"[{stem}] Estimating depth...")
    video_depth, depth_vis = depthcrafter.infer(
        input_video_path=input_video,
        output_video_path=splatting_path,
        process_length=int(process_length),
        max_res=int(max_res),
    )

    # Stage 1: forward splatting
    p(0.35, f"[{stem}] Splatting to right view...")
    DepthSplatting(
        input_video_path=input_video,
        output_video_path=splatting_path,
        video_depth=video_depth,
        depth_vis=depth_vis,
        max_disp=float(max_disp),
        process_length=int(process_length),
        batch_size=10,
    )

    # Stage 2: stereo inpainting
    p(0.5, f"[{stem}] Running stereo inpainting...")
    frames_chunk = 23
    overlap = 3

    video_reader = VideoReader(splatting_path, ctx=cpu(0))
    fps = video_reader.get_avg_fps()
    frames = video_reader.get_batch(list(range(len(video_reader))))
    num_frames = len(video_reader)

    frames = torch.tensor(frames.asnumpy()).permute(0, 3, 1, 2).float()

    height, width = frames.shape[2] // 2, frames.shape[3] // 2
    frames_left = frames[:, :, :height, :width]
    frames_mask = frames[:, :, height:, :width]
    frames_warpped = frames[:, :, height:, width:]
    frames = torch.cat([frames_warpped, frames_left, frames_mask], dim=0)

    height = height // 128 * 128
    width = width // 128 * 128
    frames = frames[:, :, :height, :width] / 255.0
    frames_warpped, frames_left, frames_mask = torch.chunk(frames, chunks=3, dim=0)
    frames_mask = frames_mask.mean(dim=1, keepdim=True)

    results = []
    generated = None
    for i in range(0, num_frames, frames_chunk - overlap):
        if i + overlap >= frames_warpped.shape[0]:
            break

        if generated is not None and i + frames_chunk > frames_warpped.shape[0]:
            cur_i = max(frames_warpped.shape[0] + overlap - frames_chunk, 0)
            cur_overlap = i - cur_i + overlap
        else:
            cur_i = i
            cur_overlap = overlap

        input_frames_i = frames_warpped[cur_i : cur_i + frames_chunk].clone()
        mask_frames_i = frames_mask[cur_i : cur_i + frames_chunk]

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
        if i != 0:
            generated = generated[cur_overlap:]
        results.append(generated)

        p(0.5 + 0.45 * (i / num_frames), f"[{stem}] Running stereo inpainting...")

    frames_output = torch.cat(results, dim=0).cpu()

    # Side-by-side output
    frames_sbs = torch.cat([frames_left, frames_output], dim=3)
    frames_sbs = (frames_sbs * 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).numpy()
    write_video_opencv(frames_sbs, fps, sbs_path)

    # Anaglyph output
    vid_left = (frames_left * 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).numpy()
    vid_right = (frames_output * 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).numpy()
    vid_left[:, :, :, 1] = 0
    vid_left[:, :, :, 2] = 0
    vid_right[:, :, :, 0] = 0
    write_video_opencv(vid_left + vid_right, fps, anaglyph_path)

    gc.collect()
    torch.cuda.empty_cache()

    return sbs_path, anaglyph_path, splatting_path


def run_inference(
    input_files: list,
    ipd: float,
    process_length: int,
    tile_num: int,
    max_res: int,
    progress=gr.Progress(track_tqdm=True),
):
    if not input_files:
        raise gr.Error("Please upload at least one video.")

    max_disp = HUMAN_MAX_DISP * ipd / HUMAN_IPD
    n = len(input_files)
    all_outputs = []

    for idx, file in enumerate(input_files):
        video_path = file if isinstance(file, str) else file.name
        sbs, anaglyph, splatting = process_single_video(
            input_video=video_path,
            max_disp=max_disp,
            process_length=int(process_length),
            tile_num=int(tile_num),
            max_res=int(max_res),
            progress_offset=idx / n,
            progress_scale=1.0 / n,
            progress=progress,
        )
        all_outputs.append((sbs, anaglyph, splatting))
        progress((idx + 1) / n, desc=f"Finished {idx + 1}/{n} videos.")

    last_sbs, last_anaglyph, last_splatting = all_outputs[-1]
    all_files = [path for sbs, anaglyph, splatting in all_outputs for path in (sbs, anaglyph, splatting)]

    return last_sbs, last_anaglyph, last_splatting, all_files


def list_past_generations() -> list[str]:
    if not os.path.isdir(OUTPUT_DIR):
        return []
    files = sorted(
        (os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR) if f.endswith(".mp4")),
        key=os.path.getmtime,
        reverse=True,
    )
    return files


with gr.Blocks(title="StereoCrafter") as demo:
    gr.Markdown(
        """
        # StereoCrafter
        Convert monocular video to immersive stereoscopic 3D.
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
                ipd = gr.Slider(
                    label="Interpupillary Distance (mm)",
                    minimum=1,
                    maximum=100,
                    value=12,
                    step=0.1,
                    info="Distance between the eyes. Higher = stronger 3D effect.",
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
                    info="Cap the longer edge of the video. Lower values use less RAM and run faster.",
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
            output_files = gr.Files(label="All output videos (SBS, anaglyph, splatting)")

    with gr.Accordion("Past Generations", open=False):
        refresh_btn = gr.Button("Refresh", size="sm")
        past_files = gr.Files(
            label="All output videos",
            value=list_past_generations,
        )
        refresh_btn.click(fn=list_past_generations, outputs=[past_files])

    run_btn.click(
        fn=run_inference,
        inputs=[input_files, ipd, process_length, tile_num, max_res],
        outputs=[output_sbs, output_anaglyph, output_splatting, output_files],
    ).then(
        fn=list_past_generations,
        outputs=[past_files],
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=6767)
