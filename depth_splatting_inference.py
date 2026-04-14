import gc
import cv2
import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.cm as cm
from fire import Fire
from decord import VideoReader, cpu

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_VDA_ROOT = os.path.join(_REPO_ROOT, "dependency", "Video-Depth-Anything")
_VDA_CHECKPOINTS = os.path.join(_VDA_ROOT, "checkpoints")
if _VDA_ROOT not in sys.path:
    sys.path.insert(0, _VDA_ROOT)

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames

from Forward_Warp import forward_warp


def depth_sequence_to_vis(depths_thw: np.ndarray) -> np.ndarray:
    """RGB visualization [T,H,W,3] float in [0, 1], inferno colormap (global min/max)."""
    colormap = np.array(cm.get_cmap("inferno").colors)
    d_min, d_max = float(depths_thw.min()), float(depths_thw.max())
    span = d_max - d_min if d_max > d_min else 1e-6
    out = np.empty((*depths_thw.shape, 3), dtype=np.float32)
    for i in range(depths_thw.shape[0]):
        dn = ((depths_thw[i] - d_min) / span * 255.0).astype(np.int64)
        dn = np.clip(dn, 0, colormap.shape[0] - 1)
        out[i] = colormap[dn]
    return out


_VDA_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


class VideoDepthAnythingDemo:
    """Metric Video Depth Anything: depth in meters (approx.), temporally aligned."""

    def __init__(
        self,
        checkpoint_path: str,
        encoder: str = "vitl",
        device: Optional[str] = None,
    ):
        if encoder not in _VDA_MODEL_CONFIGS:
            raise ValueError(f"encoder must be one of {list(_VDA_MODEL_CONFIGS)}, got {encoder}")
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = dev
        cfg = {**_VDA_MODEL_CONFIGS[encoder], "metric": True}
        self.model = VideoDepthAnything(**cfg)
        state = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.to(dev).eval()

    def infer(
        self,
        input_video_path: str,
        output_video_path: str,
        process_length: int = -1,
        max_res: int = 1024,
        target_fps: int = -1,
        input_size: int = 518,
        fp32: bool = False,
        save_depth: bool = False,
    ):
        vid_full = VideoReader(input_video_path, ctx=cpu(0))
        original_height, original_width = vid_full.get_batch([0]).shape[1:3]

        frames, target_fps = read_video_frames(
            input_video_path, process_length, target_fps, max_res
        )
        inf_h, inf_w = int(frames.shape[1]), int(frames.shape[2])

        with torch.inference_mode():
            depths, _ = self.model.infer_video_depth(
                frames, target_fps, input_size=input_size, device=self.device, fp32=fp32
            )

        if (inf_h, inf_w) != (original_height, original_width):
            t = torch.from_numpy(depths.astype(np.float32)).unsqueeze(1)
            t = F.interpolate(
                t,
                size=(original_height, original_width),
                mode="bilinear",
                align_corners=True,
            )
            depths = t.numpy()[:, 0, :, :]

        vis = depth_sequence_to_vis(depths)

        if save_depth:
            save_path = os.path.join(
                os.path.dirname(output_video_path),
                os.path.splitext(os.path.basename(output_video_path))[0],
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez_compressed(save_path + ".npz", depth=depths)

        return depths, vis


class ForwardWarpStereo(nn.Module):
    def __init__(self, eps=1e-6, occlu_map=False):
        super(ForwardWarpStereo, self).__init__()
        self.eps = eps
        self.occlu_map = occlu_map
        self.fw = forward_warp()

    def forward(self, im, disp):
        im = im.contiguous()
        disp = disp.contiguous()
        weights_map = disp - disp.min()
        weights_map = (1.414) ** weights_map
        flow = -disp.squeeze(1)
        dummy_flow = torch.zeros_like(flow, requires_grad=False)
        flow = torch.stack((flow, dummy_flow), dim=-1)
        res_accum = self.fw(im * weights_map, flow)
        mask = self.fw(weights_map, flow)
        mask.clamp_(min=self.eps)
        res = res_accum / mask
        if not self.occlu_map:
            return res
        ones = torch.ones_like(disp, requires_grad=False)
        occlu_map = self.fw(ones, flow)
        occlu_map.clamp_(0.0, 1.0)
        occlu_map = 1.0 - occlu_map
        return res, occlu_map


def DepthSplatting(
    input_video_path,
    output_video_path,
    video_depth,
    depth_vis,
    process_length,
    batch_size,
    *,
    use_metric_depth: bool = True,
    focal_length_px: Optional[float] = None,
    horizontal_fov_deg: Optional[float] = None,
    baseline_m: Optional[float] = None,
    min_depth_m: float = 1e-3,
    max_disp_px: Optional[float] = None,
    # legacy normalized depth: depth in [0,1], disparity = (2*d-1)*max_disp_norm
    max_disp_norm: float = 20.0,
):
    """
    Depth-based splatting to synthesize a right view.

    Metric mode (use_metric_depth=True):
        video_depth: [T,H,W] depth in meters (approx.).
        disparity (pixels) = focal_length_px * baseline_m / max(Z, min_depth_m).
        Provide focal_length_px **or** horizontal_fov_deg (f derived from frame width).

    Legacy mode (use_metric_depth=False):
        video_depth: [T,H,W] in [0, 1]; disp = (depth*2 - 1) * max_disp_norm.
    """
    vid_reader = VideoReader(input_video_path, ctx=cpu(0))
    original_fps = vid_reader.get_avg_fps()

    total_frames = len(vid_reader)
    num_frames = total_frames if process_length == -1 else min(process_length, total_frames)
    video_depth = video_depth[:num_frames]
    depth_vis = depth_vis[:num_frames]

    stereo_projector = ForwardWarpStereo(occlu_map=True).cuda()

    first_frame = vid_reader[0].asnumpy()
    height, width, _ = first_frame.shape

    if use_metric_depth:
        if baseline_m is None or baseline_m <= 0:
            raise ValueError("metric splatting requires baseline_m > 0 (meters)")
        if focal_length_px is not None and focal_length_px > 0:
            f_px = float(focal_length_px)
        elif horizontal_fov_deg is not None and horizontal_fov_deg > 0:
            f_px = (width * 0.5) / np.tan(np.deg2rad(float(horizontal_fov_deg) * 0.5))
        else:
            raise ValueError("metric splatting requires focal_length_px > 0 or horizontal_fov_deg > 0")

    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        original_fps,
        (width * 2, height * 2),
    )

    for i in range(0, num_frames, batch_size):
        batch_indices = list(range(i, min(i + batch_size, num_frames)))
        batch_frames = vid_reader.get_batch(batch_indices).asnumpy() / 255.0
        batch_depth = video_depth[i : i + batch_size]
        batch_depth_vis = depth_vis[i : i + batch_size]

        left_video = torch.from_numpy(batch_frames).permute(0, 3, 1, 2).float().cuda()
        disp_map = torch.from_numpy(batch_depth).unsqueeze(1).float().cuda()

        if use_metric_depth:
            z = torch.clamp(disp_map, min=float(min_depth_m))
            disp_map = (f_px * float(baseline_m)) / z
            if max_disp_px is not None and max_disp_px > 0:
                disp_map = torch.clamp(disp_map, max=float(max_disp_px))
        else:
            disp_map = disp_map * 2.0 - 1.0
            disp_map = disp_map * float(max_disp_norm)

        with torch.no_grad():
            right_video, occlusion_mask = stereo_projector(left_video, disp_map)

        right_video = right_video.cpu().permute(0, 2, 3, 1).numpy()
        occlusion_mask = occlusion_mask.cpu().permute(0, 2, 3, 1).numpy().repeat(3, axis=-1)

        for j in range(len(batch_frames)):
            video_grid_top = np.concatenate([batch_frames[j], batch_depth_vis[j]], axis=1)
            video_grid_bottom = np.concatenate([occlusion_mask[j], right_video[j]], axis=1)
            video_grid = np.concatenate([video_grid_top, video_grid_bottom], axis=0)

            video_grid_uint8 = np.clip(video_grid * 255.0, 0, 255).astype(np.uint8)
            video_grid_bgr = cv2.cvtColor(video_grid_uint8, cv2.COLOR_RGB2BGR)
            out.write(video_grid_bgr)

        del left_video, disp_map, right_video, occlusion_mask
        torch.cuda.empty_cache()
        gc.collect()

    out.release()


def main(
    input_video_path: str,
    output_video_path: str,
    video_depth_checkpoint: str = "",
    encoder: str = "vitl",
    horizontal_fov_deg: float = 55.0,
    focal_length_px: float = 0.0,
    ipd_mm: float = 63.0,
    max_disp_px: float = 500.0,
    min_depth_m: float = 1e-3,
    process_length: int = -1,
    batch_size: int = 10,
    max_res: int = 1024,
    input_size: int = 518,
    fp32: bool = False,
):
    ckpt = video_depth_checkpoint or os.environ.get(
        "VIDEO_DEPTH_CHECKPOINT",
        os.path.join(_VDA_CHECKPOINTS, "metric_video_depth_anything_vitl.pth"),
    )
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"Metric Video Depth Anything weights not found: {ckpt}\n"
            "Set VIDEO_DEPTH_CHECKPOINT or pass --video_depth_checkpoint. "
            "Default: dependency/Video-Depth-Anything/checkpoints/ (run get_weights.sh there)."
        )

    demo = VideoDepthAnythingDemo(checkpoint_path=ckpt, encoder=encoder)
    video_depth, depth_vis = demo.infer(
        input_video_path,
        output_video_path,
        process_length=int(process_length),
        max_res=int(max_res),
        input_size=int(input_size),
        fp32=bool(fp32),
    )

    f_px = float(focal_length_px) if focal_length_px and focal_length_px > 0 else None
    hfov = float(horizontal_fov_deg) if f_px is None else None

    DepthSplatting(
        input_video_path,
        output_video_path,
        video_depth,
        depth_vis,
        int(process_length),
        int(batch_size),
        use_metric_depth=True,
        focal_length_px=f_px,
        horizontal_fov_deg=hfov,
        baseline_m=float(ipd_mm) * 1e-3,
        min_depth_m=float(min_depth_m),
        max_disp_px=float(max_disp_px) if max_disp_px and max_disp_px > 0 else None,
    )


if __name__ == "__main__":
    Fire(main)
