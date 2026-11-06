"""Tests for preprocessing, including parity with torch's bilinear resize."""

from __future__ import annotations

import numpy as np
import pytest

from safecross.preprocess import (
    Rect,
    normalize,
    preprocess_crop,
    preprocess_frame,
    resize_bilinear,
    tile_rects,
)

torch = pytest.importorskip("torch", reason="torch parity checks need torch installed")


def _random_image(height: int, width: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((height, width, 3), dtype=np.float32)


@pytest.mark.parametrize(
    "in_hw,out_hw",
    [((432, 768), (320, 320)), ((720, 1280), (320, 320)), ((97, 53), (320, 320)),
     ((320, 320), (320, 320)), ((100, 100), (50, 50))],
)
def test_resize_matches_torch_interpolate(in_hw, out_hw):
    """The NumPy resize must reproduce F.interpolate(mode='bilinear', align_corners=False)."""
    image = _random_image(*in_hw)
    mine = resize_bilinear(image, *out_hw)
    reference = torch.nn.functional.interpolate(
        torch.from_numpy(image.transpose(2, 0, 1))[None],
        size=list(out_hw), mode="bilinear", align_corners=False,
    )[0].permute(1, 2, 0).numpy()
    assert mine.shape == reference.shape
    assert np.max(np.abs(mine - reference)) < 1e-5


def test_resize_is_identity_at_matching_size():
    image = _random_image(32, 32)
    assert np.array_equal(resize_bilinear(image, 32, 32), image)


def test_resize_rejects_non_hwc():
    with pytest.raises(ValueError):
        resize_bilinear(np.zeros((10, 10), dtype=np.float32), 5, 5)


def test_normalize_matches_the_affine_definition():
    image = _random_image(4, 4)
    out = normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    assert np.allclose(out, (image - 0.5) / 0.5, atol=1e-6)
    assert out.dtype == np.float32


def test_preprocess_frame_shape_range_and_channel_order():
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    frame[..., 0] = 255        # pure blue in BGR
    out = preprocess_frame(frame)
    assert out.shape == (1, 3, 320, 320)
    assert out.dtype == np.float32
    # After BGR->RGB the blue channel is index 2, and (1-0.5)/0.5 = 1
    assert np.allclose(out[0, 2], 1.0, atol=1e-5)
    assert np.allclose(out[0, 0], -1.0, atol=1e-5)


def test_preprocess_frame_matches_torchvision_transform():
    """End-to-end check against GeneralizedRCNNTransform from the real model."""
    tv_detection = pytest.importorskip("torchvision.models.detection")
    weights = tv_detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = tv_detection.ssdlite320_mobilenet_v3_large(weights=weights).eval()

    rng = np.random.default_rng(1)
    frame_bgr = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)

    rgb_chw = torch.from_numpy(frame_bgr[:, :, ::-1].copy().transpose(2, 0, 1)).float() / 255.0
    reference = model.transform([rgb_chw])[0].tensors.numpy()
    mine = preprocess_frame(frame_bgr)
    assert mine.shape == reference.shape
    assert np.max(np.abs(mine - reference)) < 1e-4


def test_preprocess_rejects_grayscale():
    with pytest.raises(ValueError):
        preprocess_frame(np.zeros((10, 10), dtype=np.uint8))


def test_preprocess_crop_uses_only_the_rect():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[50:, :, :] = 255
    top = preprocess_crop(frame, Rect(0, 0, 100, 50))
    bottom = preprocess_crop(frame, Rect(0, 50, 100, 100))
    assert np.allclose(top, -1.0, atol=1e-5)
    assert np.allclose(bottom, 1.0, atol=1e-5)


def test_tile_rects_cover_the_frame_and_overlap():
    rects = tile_rects(400, 600, grid=(2, 2), overlap=0.1, include_full_frame=False)
    assert len(rects) == 4
    assert all(r.x0 >= 0 and r.y0 >= 0 and r.x1 <= 600 and r.y1 <= 400 for r in rects)
    # Union covers every pixel.
    covered = np.zeros((400, 600), dtype=bool)
    for r in rects:
        covered[r.y0:r.y1, r.x0:r.x1] = True
    assert covered.all()
    # Neighbours overlap, otherwise an object on the seam is lost.
    assert rects[0].x1 > rects[1].x0
    assert rects[0].y1 > rects[2].y0


def test_tile_rects_can_prepend_the_full_frame():
    rects = tile_rects(100, 100, grid=(2, 2), overlap=0.0, include_full_frame=True)
    assert len(rects) == 5
    assert rects[0] == Rect(0, 0, 100, 100)


def test_tile_rects_rejects_bad_grid():
    with pytest.raises(ValueError):
        tile_rects(100, 100, grid=(0, 2))
