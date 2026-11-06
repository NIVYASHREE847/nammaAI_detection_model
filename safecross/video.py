"""Frame sources: a video file or a folder of images, behind one iterator.

Kept separate from ``preprocess`` so that the pipeline, the exporter, the
benchmark and the evaluation script all read frames the same way and agree on
frame indices -- the frame index in ``outputs/counts.csv`` must mean the same
thing as the frame index in a ground-truth CSV.

Frame indices are always indices into the *source*, counted from 0, before any
``--stride`` subsampling. So ``--stride 5`` yields frames 0, 5, 10, ... and the
CSV says so.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

__all__ = ["Frame", "SourceInfo", "probe_source", "iter_frames", "sample_frames", "VideoWriter"]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_IMAGE_FPS = 25.0


@dataclass
class Frame:
    """One decoded frame. ``image`` is BGR uint8, as OpenCV delivers it."""

    index: int
    timestamp_s: float
    image: np.ndarray


@dataclass
class SourceInfo:
    kind: str                # "video" or "images"
    path: Path
    frame_count: int         # 0 when a video refuses to report its length
    fps: float
    width: int
    height: int
    fps_assumed: bool = False  # True for image folders: there is no real frame rate

    def describe(self) -> str:
        fps = f"{self.fps:.3f} fps" + (" (assumed)" if self.fps_assumed else "")
        count = self.frame_count if self.frame_count else "unknown"
        return f"{self.kind} {self.path.name}: {self.width}x{self.height}, {count} frames, {fps}"


def _is_image_dir(path: Path) -> bool:
    return path.is_dir()


def _list_images(path: Path) -> list[Path]:
    files = [p for p in sorted(path.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
    if not files:
        raise RuntimeError(f"no images with {sorted(IMAGE_SUFFIXES)} in {path}")
    return files


def probe_source(source: str | os.PathLike[str], image_fps: float = DEFAULT_IMAGE_FPS) -> SourceInfo:
    """Read the source's geometry without consuming it."""
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"source not found: {path}")

    if _is_image_dir(path):
        files = _list_images(path)
        first = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
        if first is None:
            raise RuntimeError(f"could not decode {files[0]}")
        return SourceInfo("images", path, len(files), float(image_fps),
                          first.shape[1], first.shape[0], fps_assumed=True)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    assumed = fps <= 0.0
    if assumed:
        fps = float(image_fps)
    return SourceInfo("video", path, max(count, 0), fps, width, height, fps_assumed=assumed)


def iter_frames(
    source: str | os.PathLike[str],
    stride: int = 1,
    max_frames: int | None = None,
    image_fps: float = DEFAULT_IMAGE_FPS,
) -> Iterator[Frame]:
    """Yield every ``stride``-th frame, at most ``max_frames`` of them."""
    stride = max(1, int(stride))
    info = probe_source(source, image_fps=image_fps)
    produced = 0

    if info.kind == "images":
        for index, file in enumerate(_list_images(info.path)):
            if index % stride:
                continue
            image = cv2.imread(str(file), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"could not decode {file}")
            yield Frame(index, index / info.fps, image)
            produced += 1
            if max_frames is not None and produced >= max_frames:
                return
        return

    cap = cv2.VideoCapture(str(info.path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video {info.path}")
    try:
        index = 0
        while True:
            ok, image = cap.read()
            if not ok:
                return
            if index % stride == 0:
                yield Frame(index, index / info.fps, image)
                produced += 1
                if max_frames is not None and produced >= max_frames:
                    return
            index += 1
    finally:
        cap.release()


def sample_frames(
    source: str | os.PathLike[str],
    count: int,
    image_fps: float = DEFAULT_IMAGE_FPS,
) -> list[Frame]:
    """Return up to ``count`` frames spread evenly across the whole source.

    Decoding is sequential rather than seek-based: frame-accurate seeking is
    unreliable across containers and codecs, and a mis-seek would silently
    calibrate INT8 on the wrong frames.
    """
    count = max(0, int(count))
    if count == 0:
        return []
    info = probe_source(source, image_fps=image_fps)

    if info.frame_count > 0:
        wanted = sorted(
            set(np.linspace(0, info.frame_count - 1, num=min(count, info.frame_count)).astype(int))
        )
        wanted_set = set(int(i) for i in wanted)
        out = [f for f in iter_frames(source, image_fps=image_fps) if f.index in wanted_set]
        return out[:count]

    # Unknown length: take the first `count` frames and say so via the caller's log.
    return list(iter_frames(source, max_frames=count, image_fps=image_fps))


class VideoWriter:
    """Thin ``cv2.VideoWriter`` wrapper that opens lazily on the first frame.

    Lazy opening means the writer learns the frame size from the first
    annotated frame instead of guessing it, which matters because the annotated
    frame keeps the source resolution.
    """

    def __init__(self, path: str | os.PathLike[str], fps: float, fourcc: str = "mp4v") -> None:
        self.path = Path(path)
        self.fps = float(fps) if fps and fps > 0 else DEFAULT_IMAGE_FPS
        self.fourcc = fourcc
        self._writer: cv2.VideoWriter | None = None
        self.frames_written = 0

    def write(self, frame_bgr: np.ndarray) -> None:
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame_bgr.shape[:2]
            self._writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*self.fourcc), self.fps, (width, height)
            )
            if not self._writer.isOpened():
                raise RuntimeError(
                    f"cv2.VideoWriter could not open {self.path} with fourcc {self.fourcc!r}"
                )
        self._writer.write(frame_bgr)
        self.frames_written += 1

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
