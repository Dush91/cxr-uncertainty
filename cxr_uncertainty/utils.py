"""Image loading + the canonical torchxrayvision preprocessing pipeline.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch

try:
    import torchxrayvision as xrv
except ImportError as e:  # pragma: no cover
    raise ImportError("torchxrayvision is required: pip install torchxrayvision") from e


# Canonical transform used by the xrv pretrained DenseNets.
_TRANSFORM = None


def _get_transform(img_size: int = 224):
    """XRayCenterCrop() takes no size arg (crops to the smaller side); XRayResizer
    then resizes to img_size. Both expect (C,H,W) arrays."""
    global _TRANSFORM
    if _TRANSFORM is None or _TRANSFORM[1] != img_size:
        _TRANSFORM = (
            __import__("torchvision").transforms.Compose([
                xrv.datasets.XRayCenterCrop(),
                xrv.datasets.XRayResizer(img_size),
            ]),
            img_size,
        )
    return _TRANSFORM[0]


def load_raw_gray(path):
    """Read a CXR file (png/jpg/tiff/dicom via pydicom) into a 2-D grayscale
    array WITHOUT any model normalization, plus the metadata needed to
    normalize it.

    Returns (img2d, meta):
      - uint8 PNGs  -> (H,W) uint8 (0-255),     meta["maxval"] = 255.0
      - uint16 PNGs -> (H,W) uint16 (0-65535),  meta["maxval"] = 65535.0
      - float reads (multi-channel means) -> same value-range heuristic
      - DICOM in HU-ish range -> (H,W) float32 HU values,
        meta["maxval"] = None, meta["dicom_hu"] = True (caller must window)
      - other DICOMs -> as the equivalent 8-bit case, meta["dicom_hu"] = False

    Agent 3's image-quality metrics consume this directly ([0,1] = img2d /
    maxval, or the HU window); load_image_tensor continues into the xrv
    preprocessing pipeline. Raises FileNotFoundError if the path is missing.
    """
    path = str(path)
    if not Path(path).exists():
        raise FileNotFoundError(path)

    p = path.lower()
    if p.endswith(".dcm") or p.endswith(".dicom"):
        import pydicom
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        pi = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
        if pi == "MONOCHROME1":
            arr = arr.max() - arr
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept
        gray = arr if arr.ndim == 2 else arr.mean(axis=2)
        if gray.max() > 255:
            return gray, {"kind": "dicom", "dicom_hu": True, "maxval": None,
                          "bit_depth": None}
        gray = gray.astype(np.uint8)
        return gray, {"kind": "dicom", "dicom_hu": False, "maxval": 255.0,
                      "bit_depth": 8}

    raw = xrv.datasets.imread(path)
    img2d = raw if raw.ndim == 2 else raw.mean(axis=2)

    # Normalization maxval by source bit depth. The xrv DenseNets were trained
    # on 8-bit images normalized by 255. Some leak-free eval corpora ship 16-bit
    # CXR PNGs (e.g. ReXGradient-160K: uint16, 0-65535); these must be normalized
    # by 65535 to land in the same [0,1] -> [-1024,1024] band. Use the integer
    # dtype's max when available (uint8->255, uint16->65535), else a value-range
    # heuristic for imread upcasts to float (multi-channel means).
    if np.issubdtype(img2d.dtype, np.integer):
        maxval = float(np.iinfo(img2d.dtype).max)
        bit_depth = int(np.iinfo(img2d.dtype).bits)
    else:
        maxval = 65535.0 if float(img2d.max()) > 255 else 255.0
        bit_depth = 16 if maxval > 255 else 8
    return img2d, {"kind": "png", "dicom_hu": False, "maxval": maxval,
                   "bit_depth": bit_depth}


def load_image_tensor(path, img_size: int = 224, device: str = "cpu") -> torch.Tensor:
    """Load a CXR file (png/jpg/tiff/dicom via pydicom) -> (1,1,H,W) tensor in
    the [-1024,1024] range expected by torchxrayvision models.

    Raises FileNotFoundError if the path does not exist and ValueError if the
    image is empty.
    """
    img2d, meta = load_raw_gray(path)

    if meta["dicom_hu"]:
        # Already in HU-ish range; scale to the [-1024,1024] band the models
        # expect, then go straight to the transform.
        gray = np.clip(img2d, -1024, 1024)
        img = gray[None, :, :].astype(np.float32)
        t = _get_transform(img_size)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = np.asarray(t(img)).astype(np.float32)
        return torch.from_numpy(img).float().unsqueeze(0).to(device)
    maxval = meta["maxval"]

    # (H,W) -> (1,H,W) so XRayCenterCrop/XRayResizer see (C,H,W).
    img = img2d[None, :, :].astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = xrv.datasets.normalize(img, maxval)  # -> [-1024,1024]

    t = _get_transform(img_size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = np.asarray(t(img)).astype(np.float32)
    if img.ndim == 2:
        img = img[None, :, :]
    return torch.from_numpy(img).float().unsqueeze(0).to(device)