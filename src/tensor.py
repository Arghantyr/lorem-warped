"""Tensor operations for PyTorch image warps."""
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from PIL import Image


def warp_image(image: torch.Tensor, field: torch.Tensor, intensity: float) -> torch.Tensor:
    """Backward bilinear warp of a float CHW image using a unit-amplitude field."""
    if intensity == 0:
        return image.clone()
    _, height, width = image.shape
    y = torch.arange(height, device=image.device, dtype=image.dtype)[:, None]
    x = torch.arange(width, device=image.device, dtype=image.dtype)[None, :]
    grid = _pixel_coordinates_to_grid(
        y - intensity * field[..., 0], x - intensity * field[..., 1], height, width,
    )
    warped = torch.nn.functional.grid_sample(
        image.unsqueeze(0), grid.unsqueeze(0),
        mode="bilinear", padding_mode="border", align_corners=True,
    )
    return warped.squeeze(0).clamp(0.0, 1.0)

def _as_float_chw(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 3 or any(size == 0 for size in image.shape):
        raise ValueError("image must have nonempty shape (channels, height, width)")
    if not image.is_floating_point():
        image = image.float() / 255.0
    return image.clamp(0.0, 1.0)

def _validate_centres(
    centres: torch.Tensor,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    if centres.ndim != 2 or centres.shape[1] != 2:
        raise ValueError("centres must have shape (N, 2) using (y, x) coordinates")
    centres = centres.to(device=device, dtype=torch.float32)
    if not torch.isfinite(centres).all():
        raise ValueError("centre coordinates must be finite")
    if torch.any(centres[:, 0] < 0) or torch.any(centres[:, 0] >= height):
        raise ValueError("centre y coordinates must be inside the image")
    if torch.any(centres[:, 1] < 0) or torch.any(centres[:, 1] >= width):
        raise ValueError("centre x coordinates must be inside the image")
    return centres

def _pixel_coordinates_to_grid(y: torch.Tensor, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    grid_y = 2.0 * y / max(height - 1, 1) - 1.0
    grid_x = 2.0 * x / max(width - 1, 1) - 1.0
    return torch.stack((grid_x, grid_y), dim=-1)

def _load_image(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        return image_to_tensor(image)

def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a Pillow image to a float RGB CHW tensor."""
    return torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).float() / 255.0

def _save_image(image: torch.Tensor, path: Path) -> None:
    array = (image.permute(1, 2, 0).mul(255).round().byte().cpu().numpy())
    Image.fromarray(array).save(path)
