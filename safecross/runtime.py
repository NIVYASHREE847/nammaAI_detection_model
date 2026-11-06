"""Model runtimes: ExecuTorch (.pte) for the real pipeline, eager torch for parity.

Both runtimes expose the same call signature::

    cls_logits, bbox_regression = runtime(nchw_float32_numpy_array)

so the rest of the pipeline never learns which one it is talking to.

``ExecuTorchRuntime`` is the one that matters: it loads a ``.pte`` produced by
``export/export_model.py`` and runs it on the ExecuTorch CPU runtime with the
XNNPACK backend. No CUDA, no torchvision, and the same program file runs
unchanged on a Raspberry Pi (ExecuTorch ships manylinux aarch64 wheels, and the
same .pte also loads in the C++ runner).

``EagerRuntime`` needs torch + torchvision and exists only so that
``export/export_model.py`` and ``tests/`` can prove the .pte agrees with
torchvision's own output. It is also a usable fallback for development before a
.pte exists (``--model eager``).

torch is imported lazily inside each class so that a NumPy-only consumer can
import this module.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "make_raw_ssd",
    "BaseRuntime",
    "EagerRuntime",
    "ExecuTorchRuntime",
    "build_runtime",
    "load_anchors",
]

INPUT_SIZE = (320, 320)


def load_anchors(path: str | os.PathLike[str]) -> np.ndarray:
    """Load the (3234, 4) anchor table written at export time."""
    anchors = np.load(str(path))
    if anchors.ndim != 2 or anchors.shape[1] != 4:
        raise ValueError(f"bad anchor file {path}: shape {anchors.shape}")
    return anchors.astype(np.float32)


def build_raw_ssd() -> tuple[Any, np.ndarray, dict[str, Any]]:
    """Build the exportable wrapper, its anchors and its metadata.

    Returns ``(module, anchors, meta)``. Used by the exporter and by
    ``EagerRuntime``; needs torch + torchvision.
    """
    import torch
    from torchvision.models.detection import (
        SSDLite320_MobileNet_V3_Large_Weights,
        ssdlite320_mobilenet_v3_large,
    )
    from torchvision.models.detection.image_list import ImageList

    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weights).eval()

    wrapper = make_raw_ssd(model).eval()

    # Anchors are a pure function of the input size (320x320 is fixed here), so
    # we generate them once with torchvision's own DefaultBoxGenerator and ship
    # them next to the .pte. No hand-derived anchor maths anywhere.
    with torch.no_grad():
        dummy = torch.zeros(1, 3, *INPUT_SIZE)
        features = list(model.backbone(dummy).values())
        anchors = model.anchor_generator(ImageList(dummy, [INPUT_SIZE]), features)[0]
        head_out = model.head(features)

    meta = {
        "model": "ssdlite320_mobilenet_v3_large",
        "weights": str(weights),
        "input_size": list(INPUT_SIZE),
        "image_mean": list(model.transform.image_mean),
        "image_std": list(model.transform.image_std),
        "box_weights": list(model.box_coder.weights),
        "num_anchors": int(anchors.shape[0]),
        "num_classes": int(head_out["cls_logits"].shape[-1]),
        "categories": list(weights.meta["categories"]),
        "torchvision_defaults": {
            "score_thresh": float(model.score_thresh),
            "nms_thresh": float(model.nms_thresh),
            "topk_candidates": int(model.topk_candidates),
            "detections_per_img": int(model.detections_per_img),
        },
        "outputs": ["cls_logits", "bbox_regression"],
    }
    return wrapper, anchors.detach().cpu().numpy().astype(np.float32), meta


_RAW_SSD_CLS = None

_RAW_SSD_DOC = """Backbone + SSD head only: raw ``(cls_logits, bbox_regression)``.

Everything torchvision does around the network -- the input transform, the
anchor generator, softmax, score threshold, NMS and the box rescale -- is
deliberately left out, because those parts are either data-dependent
(NMS produces dynamic shapes) or trivially portable NumPy. What stays inside
the exported graph is one static-shape convolutional network, which is exactly
what the XNNPACK backend wants.
"""


def raw_ssd_class():
    """Return (and cache) the ``torch.nn.Module`` subclass used for export."""
    global _RAW_SSD_CLS
    if _RAW_SSD_CLS is not None:
        return _RAW_SSD_CLS

    import torch

    class _RawSSD(torch.nn.Module):
        __doc__ = _RAW_SSD_DOC

        def __init__(self, model: "torch.nn.Module") -> None:
            super().__init__()
            self.backbone = model.backbone
            self.head = model.head

        def forward(self, x: "torch.Tensor"):
            features = list(self.backbone(x).values())
            out = self.head(features)
            return out["cls_logits"], out["bbox_regression"]

    _RAW_SSD_CLS = _RawSSD
    return _RAW_SSD_CLS


def make_raw_ssd(model):
    """Wrap a torchvision SSD model so it emits raw head outputs."""
    return raw_ssd_class()(model)


class BaseRuntime:
    """Common interface. ``name`` and ``precision`` land in the report."""

    name: str = "base"
    precision: str = "fp32"
    model_path: Path | None = None

    def __call__(self, nchw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @property
    def size_mb(self) -> float:
        if self.model_path is None or not Path(self.model_path).exists():
            return float("nan")
        return Path(self.model_path).stat().st_size / (1024 * 1024)

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class EagerRuntime(BaseRuntime):
    """torchvision model in eager mode. Reference path, not the deployment path."""

    name = "eager"
    precision = "fp32"

    def __init__(self) -> None:
        import torch

        self._torch = torch
        self.module, self.anchors, self.meta = build_raw_ssd()
        torch.set_grad_enabled(False)

    def __call__(self, nchw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch = self._torch
        with torch.no_grad():
            cls_logits, bbox = self.module(torch.from_numpy(np.ascontiguousarray(nchw)))
        return cls_logits.numpy(), bbox.numpy()

    @property
    def size_mb(self) -> float:
        # Parameter footprint, for a like-for-like line next to the .pte sizes.
        total = sum(p.numel() * p.element_size() for p in self.module.parameters())
        return total / (1024 * 1024)


class ExecuTorchRuntime(BaseRuntime):
    """Runs a ``.pte`` program through the ExecuTorch runtime (XNNPACK backend)."""

    name = "executorch"

    def __init__(self, pte_path: str | os.PathLike[str], precision: str = "fp32") -> None:
        import torch  # ExecuTorch's Python bindings exchange torch.Tensor objects

        self._torch = torch
        self.model_path = Path(pte_path)
        self.precision = precision
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"{self.model_path} not found. Run:  python export/export_model.py"
            )
        self._method, self._flavour = self._load(self.model_path)

    @staticmethod
    def _load(path: Path):
        """Load via the documented API, falling back to the pybind shim."""
        errors: list[str] = []
        try:
            from executorch.runtime import Runtime

            runtime = Runtime.get()
            program = runtime.load_program(path)
            return program.load_method("forward"), "runtime"
        except Exception as exc:  # pragma: no cover - environment dependent
            errors.append(f"executorch.runtime: {type(exc).__name__}: {exc}")
        try:
            from executorch.extension.pybindings.portable_lib import (  # type: ignore
                _load_for_executorch,
            )

            return _load_for_executorch(str(path)), "portable_lib"
        except Exception as exc:  # pragma: no cover - environment dependent
            errors.append(f"portable_lib: {type(exc).__name__}: {exc}")
        raise RuntimeError("could not load .pte:\n  " + "\n  ".join(errors))

    def __call__(self, nchw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tensor = self._torch.from_numpy(np.ascontiguousarray(nchw, dtype=np.float32))
        if self._flavour == "runtime":
            outputs = self._method.execute([tensor])
        else:  # pragma: no cover - fallback path
            outputs = self._method.forward([tensor])
        cls_logits, bbox = outputs[0], outputs[1]
        to_np = lambda t: t.numpy() if hasattr(t, "numpy") else np.asarray(t)
        return to_np(cls_logits), to_np(bbox)


def build_runtime(
    kind: str,
    cfg: dict[str, Any] | None = None,
    *,
    fp32_pte: str | os.PathLike[str] | None = None,
    int8_pte: str | os.PathLike[str] | None = None,
) -> BaseRuntime:
    """Create a runtime from ``--model {fp32,int8,eager}``."""
    kind = kind.lower()
    if kind == "eager":
        return EagerRuntime()
    if cfg is not None:
        fp32_pte = fp32_pte or cfg["model"]["fp32_pte"]
        int8_pte = int8_pte or cfg["model"]["int8_pte"]
    if kind == "fp32":
        if fp32_pte is None:
            raise ValueError("fp32 runtime needs fp32_pte")
        return ExecuTorchRuntime(fp32_pte, precision="fp32")
    if kind == "int8":
        if int8_pte is None:
            raise ValueError("int8 runtime needs int8_pte")
        return ExecuTorchRuntime(int8_pte, precision="int8")
    raise ValueError(f"unknown model kind {kind!r}; expected fp32, int8 or eager")
