import gc
import cv2
import os
import sys
from typing import Optional, Tuple

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
from video_depth_anything.video_depth_stream import VideoDepthAnything as VideoDepthAnythingStreaming
from utils.dc_utils import read_video_frames, ensure_even

from Forward_Warp import forward_warp


def depth_batch_to_vis_numpy(batch_thw: np.ndarray, d_min: float, d_max: float) -> np.ndarray:
    """batch_thw: (B,H,W) -> (B,H,W,3) float [0,1] inferno."""
    colormap = np.array(cm.get_cmap("inferno").colors)
    span = (d_max - d_min) if d_max > d_min else 1e-6
    dn = ((batch_thw - d_min) / span * 255.0).astype(np.int64)
    dn = np.clip(dn, 0, colormap.shape[0] - 1)
    return colormap[dn].astype(np.float32)


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


class VideoDepthAnythingStreamingDemo:
    """Metric Video Depth Anything with **streaming** inference (low RAM, long videos)."""

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
        cfg = {**_VDA_MODEL_CONFIGS[encoder]}
        self.model = VideoDepthAnythingStreaming(**cfg)
        state = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.to(dev).eval()

    def reset_streaming(self) -> None:
        m = self.model
        m.transform = None
        m.frame_id_list = []
        m.frame_cache_list = []
        m.id = -1

    def infer_streaming(
        self,
        input_video_path: str,
        memmap_path: str,
        process_length: int = -1,
        max_res: int = 1024,
        target_fps: int = -1,
        input_size: int = 518,
        fp32: bool = False,
    ) -> Tuple[np.memmap, Tuple[float, float], np.ndarray]:
        """
        Stream frames through VDA; write depths to float32 memmap (T, orig_h, orig_w).
        Returns (memmap, (d_min, d_max), frames_idx) — memmap shape (T, oh, ow).
        Caller must delete memmap and unlink memmap_path when done.
        """
        self.reset_streaming()
        vid = VideoReader(input_video_path, ctx=cpu(0))
        sample = vid.get_batch([0]).asnumpy()
        oh, ow = int(sample.shape[1]), int(sample.shape[2])

        inf_h, inf_w = oh, ow
        if max_res > 0 and max(inf_h, inf_w) > max_res:
            scale = max_res / max(oh, ow)
            inf_h = ensure_even(round(oh * scale))
            inf_w = ensure_even(round(ow * scale))

        avg_fps = vid.get_avg_fps()
        fps = avg_fps if target_fps == -1 else float(target_fps)
        stride = max(round(avg_fps / fps), 1)
        frames_idx = list(range(0, len(vid), stride))
        if process_length != -1 and process_length < len(frames_idx):
            frames_idx = frames_idx[: int(process_length)]
        T = len(frames_idx)
        if T == 0:
            raise ValueError("No frames to process (empty video or process_length=0)")

        if os.path.isfile(memmap_path):
            os.unlink(memmap_path)
        depth_mm = np.memmap(memmap_path, dtype=np.float32, mode="w+", shape=(T, oh, ow))

        d_min = np.float32(np.inf)
        d_max = np.float32(-np.inf)

        with torch.inference_mode():
            for t, fi in enumerate(frames_idx):
                frame = vid[int(fi)].asnumpy()
                if (inf_h, inf_w) != (oh, ow):
                    frame = cv2.resize(frame, (inf_w, inf_h), interpolation=cv2.INTER_AREA)
                depth = self.model.infer_video_depth_one(
                    frame, input_size=input_size, device=self.device, fp32=fp32
                )
                if (inf_h, inf_w) != (oh, ow):
                    dt = torch.from_numpy(depth[None, None].astype(np.float32)).to(self.device)
                    depth = (
                        F.interpolate(dt, size=(oh, ow), mode="bilinear", align_corners=True)
                        .cpu()
                        .numpy()[0, 0]
                    )
                depth_mm[t] = depth
                d_min = np.minimum(d_min, float(depth.min()))
                d_max = np.maximum(d_max, float(depth.max()))

        depth_mm.flush()
        idx_arr = np.asarray(frames_idx, dtype=np.int64)
        return depth_mm, (float(d_min), float(d_max)), idx_arr


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
    depth_vis_bounds: Optional[Tuple[float, float]] = None,
    depth_frame_indices: Optional[np.ndarray] = None,
    # legacy normalized depth: depth in [0,1], disparity = (2*d-1)*max_disp_norm
    max_disp_norm: float = 20.0,
):
    """
    Depth-based splatting to synthesize a right view.

    Metric mode (use_metric_depth=True):
        video_depth: [T,H,W] depth in meters (approx.), ndarray or memmap.
        disparity (pixels) = focal_length_px * baseline_m / max(Z, min_depth_m).

    depth_vis may be None if depth_vis_bounds is set (inferno vis computed per batch).

    depth_frame_indices: length-T vector of source frame indices (required when depth used
    striding / subset). If None, uses consecutive frames 0..T-1 capped by process_length.
    """
    vid_reader = VideoReader(input_video_path, ctx=cpu(0))
    original_fps = vid_reader.get_avg_fps()

    if depth_frame_indices is not None:
        depth_frame_indices = np.asarray(depth_frame_indices, dtype=np.int64)
        num_iter = int(depth_frame_indices.shape[0])
    else:
        total_frames = len(vid_reader)
        num_iter = total_frames if process_length == -1 else min(process_length, total_frames)
        video_depth = video_depth[:num_iter]
        if depth_vis is not None:
            depth_vis = depth_vis[:num_iter]

    if depth_vis is None and depth_vis_bounds is None:
        raise ValueError("Provide depth_vis or depth_vis_bounds (global min/max for colormap).")

    stereo_projector = ForwardWarpStereo(occlu_map=True).cuda()

    first_idx = int(depth_frame_indices[0]) if depth_frame_indices is not None else 0
    first_frame = vid_reader[first_idx].asnumpy()
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

    d_lo, d_hi = (0.0, 1.0)
    if depth_vis_bounds is not None:
        d_lo, d_hi = float(depth_vis_bounds[0]), float(depth_vis_bounds[1])

    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        original_fps,
        (width * 2, height * 2),
    )

    for i in range(0, num_iter, batch_size):
        end = min(i + batch_size, num_iter)
        if depth_frame_indices is not None:
            batch_src_idx = depth_frame_indices[i:end].tolist()
        else:
            batch_src_idx = list(range(i, end))
        batch_frames = vid_reader.get_batch(batch_src_idx).asnumpy() / 255.0
        batch_depth = np.asarray(video_depth[i:end])
        if depth_vis is not None:
            batch_depth_vis = depth_vis[i:end]
        else:
            batch_depth_vis = depth_batch_to_vis_numpy(batch_depth, d_lo, d_hi)

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

    mmap_path = os.path.splitext(output_video_path)[0] + ".depth.mmap"
    demo = VideoDepthAnythingStreamingDemo(checkpoint_path=ckpt, encoder=encoder)
    depth_mm, bounds, fidx = demo.infer_streaming(
        input_video_path,
        mmap_path,
        process_length=int(process_length),
        max_res=int(max_res),
        input_size=int(input_size),
        fp32=bool(fp32),
    )

    f_px = float(focal_length_px) if focal_length_px and focal_length_px > 0 else None
    hfov = float(horizontal_fov_deg) if f_px is None else None

    try:
        DepthSplatting(
            input_video_path,
            output_video_path,
            depth_mm,
            None,
            int(process_length),
            int(batch_size),
            use_metric_depth=True,
            focal_length_px=f_px,
            horizontal_fov_deg=hfov,
            baseline_m=float(ipd_mm) * 1e-3,
            min_depth_m=float(min_depth_m),
            max_disp_px=float(max_disp_px) if max_disp_px and max_disp_px > 0 else None,
            depth_vis_bounds=bounds,
            depth_frame_indices=fidx,
        )
    finally:
        del depth_mm
        gc.collect()
        try:
            os.unlink(mmap_path)
        except OSError:
            pass


if __name__ == "__main__":
    Fire(main)
