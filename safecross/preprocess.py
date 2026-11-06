"""Frame preprocessing that reproduces torchvision's SSD input transform exactly.

torchvision's ``ssdlite320_mobilenet_v3_large`` wraps the network in a
``GeneralizedRCNNTransform(min_size=320, max_size=320, fixed_size=(320, 320))``
with ``image_mean=[0.5]*3`` and ``image_std=[0.5]*3``, and the pretrained
weights' own transform only does ``uint8 -> float`` (divide by 255). So the
full chain is:

    BGR uint8 frame
      -> RGB
      -> float32 / 255
      -> bilinear resize to 320x320 (align_corners=False, no antialias,
         aspect ratio NOT preserved -- SSD distorts instead of letterboxing)
      -> (x - 0.5) / 0.5

``normalize`` and ``resize`` are both affine/linear, so swapping their order is
numerically equivalent up to float rounding; we resize first because it is
cheaper. The resize itself is implemented here in NumPy with torch's own
``area_pixel_compute_source_index`` formula rather than ``cv2.resize``, so the
exported .pte gets bit-comparable input to the eager model (``cv2.resize``
differs by up to ~1 LSB at borders). ``tests/test_preprocess.py`` asserts the
match against ``torch.nn.functional.interpolate``.
"""

from __future__ import annotations

from typing import Iterable, NamedTuple

import numpy as np

__all__ = [
    "Rect",
    "resize_bilinear",
    "normalize",
    "preprocess_frame",
    "preprocess_crop",
    "tile_rects",
]


class Rect(NamedTuple):
    """Integer pixel crop rectangle, right/bottom exclusive."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def _bilinear_index(in_size: int, out_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Neighbour indices and weights for torch bilinear, align_corners=False.

    Mirrors ATen's ``area_pixel_compute_source_index`` with
    ``align_corners=False``: ``src = scale * (dst + 0.5) - 0.5`` clamped at 0,
    then floor/ceil with the upper index clamped to ``in_size - 1``.

    The arithmetic is done in float32 on purpose. ATen uses ``accscalar_t``,
    which is ``float`` for a float input, so computing the weights in float64
    here would leave a ~5e-5 discrepancy against torch on awkward scale factors
    (for example 768 -> 320). float32 reproduces torch's rounding instead.
    """
    scale = np.float32(in_size) / np.float32(out_size)
    dst = np.arange(out_size, dtype=np.float32)
    src = scale * (dst + np.float32(0.5)) - np.float32(0.5)
    np.clip(src, np.float32(0.0), None, out=src)
    i0 = np.floor(src).astype(np.int64)
    frac = (src - i0.astype(np.float32)).astype(np.float32)
    i1 = np.minimum(i0 + 1, in_size - 1)
    i0 = np.minimum(i0, in_size - 1)
    return i0, i1, frac


def resize_bilinear(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize an HWC float32 image to (out_h, out_w) like torch bilinear."""
    if image.ndim != 3:
        raise ValueError(f"expected HWC image, got shape {image.shape}")
    in_h, in_w = image.shape[:2]
    if (in_h, in_w) == (out_h, out_w):
        return np.ascontiguousarray(image, dtype=np.float32)
    image = np.ascontiguousarray(image, dtype=np.float32)

    y0, y1, wy = _bilinear_index(in_h, out_h)
    x0, x1, wx = _bilinear_index(in_w, out_w)

    wx_ = wx[None, :, None]
    wy_ = wy[:, None, None]

    rows0 = image[y0]            # (out_h, in_w, C)
    rows1 = image[y1]
    top = rows0[:, x0] * (1.0 - wx_) + rows0[:, x1] * wx_
    bottom = rows1[:, x0] * (1.0 - wx_) + rows1[:, x1] * wx_
    return (top * (1.0 - wy_) + bottom * wy_).astype(np.float32, copy=False)


def normalize(image: np.ndarray, mean: Iterable[float], std: Iterable[float]) -> np.ndarray:
    """Apply per-channel ``(x - mean) / std`` to an HWC float image."""
    mean_a = np.asarray(list(mean), dtype=np.float32).reshape(1, 1, -1)
    std_a = np.asarray(list(std), dtype=np.float32).reshape(1, 1, -1)
    return ((image - mean_a) / std_a).astype(np.float32, copy=False)


def _to_nchw(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None, ...], dtype=np.float32)


def preprocess_frame(
    frame_bgr: np.ndarray,
    size: tuple[int, int] = (320, 320),
    mean: Iterable[float] = (0.5, 0.5, 0.5),
    std: Iterable[float] = (0.5, 0.5, 0.5),
) -> np.ndarray:
    """BGR uint8 frame (as OpenCV returns it) -> (1, 3, H, W) float32 tensor."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 BGR frame, got shape {frame_bgr.shape}")
    rgb = frame_bgr[:, :, ::-1].astype(np.float32) / 255.0
    out_h, out_w = size
    resized = resize_bilinear(rgb, out_h, out_w)
    return _to_nchw(normalize(resized, mean, std))


def preprocess_crop(
    frame_bgr: np.ndarray,
    rect: Rect,
    size: tuple[int, int] = (320, 320),
    mean: Iterable[float] = (0.5, 0.5, 0.5),
    std: Iterable[float] = (0.5, 0.5, 0.5),
) -> np.ndarray:
    """Preprocess one crop of a frame (used by tiled inference)."""
    crop = frame_bgr[rect.y0 : rect.y1, rect.x0 : rect.x1]
    if crop.size == 0:
        raise ValueError(f"empty crop for rect {rect}")
    return preprocess_frame(crop, size=size, mean=mean, std=std)


def tile_rects(
    height: int,
    width: int,
    grid: tuple[int, int] = (2, 2),
    overlap: float = 0.15,
    include_full_frame: bool = True,
) -> list[Rect]:
    """Split a frame into an overlapping grid of crops.

    Overlap matters: an object sitting on a tile seam would otherwise be cut in
    half in every tile. ``include_full_frame`` prepends the whole frame so that
    objects larger than one tile are still detected; the merge step in
    ``postprocess.merge_detections`` de-duplicates across all of them.
    """
    rows, cols = int(grid[0]), int(grid[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"tile grid must be positive, got {grid}")
    rects: list[Rect] = []
    if include_full_frame:
        rects.append(Rect(0, 0, width, height))

    tile_h = height / rows
    tile_w = width / cols
    pad_y = tile_h * overlap
    pad_x = tile_w * overlap
    for r in range(rows):
        for c in range(cols):
            y0 = int(round(max(0.0, r * tile_h - pad_y)))
            y1 = int(round(min(float(height), (r + 1) * tile_h + pad_y)))
            x0 = int(round(max(0.0, c * tile_w - pad_x)))
            x1 = int(round(min(float(width), (c + 1) * tile_w + pad_x)))
            if y1 > y0 and x1 > x0:
                rects.append(Rect(x0, y0, x1, y1))
    return rects
