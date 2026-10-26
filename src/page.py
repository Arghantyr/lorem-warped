"""Page operations for PyTorch image warps."""
from __future__ import annotations

import math
from pathlib import Path
import torch
from .tensor import _as_float_chw, warp_image


def page_warp_field(image: torch.Tensor, warp_function) -> torch.Tensor:
    """Evaluate a page-wide function on normalized (y, x) coordinates in [0, 1].

    The function returns (dy, dx) displacement components in pixel-amplitude
    units. Coordinates are full H×W tensors on the image device; the resulting
    H×W×2 field can be reused with ``warp_image`` for an intensity sweep.
    """
    image = _as_float_chw(image)
    _, height, width = image.shape
    y, x = torch.meshgrid(
        torch.linspace(0, 1, height, device=image.device, dtype=image.dtype),
        torch.linspace(0, 1, width, device=image.device, dtype=image.dtype),
        indexing="ij",
    )
    components = warp_function(y, x)
    if not isinstance(components, (tuple, list)) or len(components) != 2:
        raise ValueError("warp function must return (dy, dx) tensors")
    if any(not isinstance(c, torch.Tensor) or c.shape != (height, width)
           or c.device != image.device or not torch.is_floating_point(c) for c in components):
        raise ValueError("warp components must be H×W floating tensors on the image device")
    field = torch.stack(components, dim=-1)
    if not torch.isfinite(field).all():
        raise ValueError("warp components must be finite")
    return field

def warp_page(image: torch.Tensor, warp_function, intensity: float = 3.0,
              mask: str | Path | torch.Tensor | None = None) -> torch.Tensor:
    """Warp the entire page with a coordinate function and pixel amplitude.

    Positive dy/dx move visible content down/right (backward bilinear sampling).
    For legibility, keep the displacement small compared with glyph height and
    make its spatial variation slow compared with character width.
    """
    if not math.isfinite(intensity) or intensity < 0:
        raise ValueError("intensity must be finite and non-negative")
    image = _as_float_chw(image)
    field = page_warp_field(image, warp_function)
    return warp_image(image, field, intensity, mask=mask)

def _page_envelope(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Pin all four page edges while allowing a smooth interior deformation."""
    return torch.sin(math.pi * y) * torch.sin(math.pi * x)

def horizontal_wave(y: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Gently move text left/right in three broad waves down the page."""
    envelope = _page_envelope(y, x)
    return torch.zeros_like(y), envelope * torch.sin(6 * math.pi * y)

def vertical_wave(y: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Gently bow baselines up/down over two broad waves across the page."""
    envelope = _page_envelope(y, x)
    return envelope * torch.sin(4 * math.pi * x), torch.zeros_like(x)

def gentle_pinch(y: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Low-strength radial compression toward the page centre, tapered at edges."""
    envelope = _page_envelope(y, x)
    return envelope * (0.5 - y), envelope * (0.5 - x)

def sine_lens(
    y: torch.Tensor, x: torch.Tensor, A: int = 2, B: int = 2,
    phase_x: float = 0.0, phase_y: float = 0.0,
    *, amplitude_x: float = 1.0, amplitude_y: float = 1.0,
    frequency_x: float | None = None, frequency_y: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Whole-page sinusoidal lens with independent X/Y gain and frequency.

    ``frequency_x/y`` override the legacy integer lens counts ``A/B`` and
    may be fractional. Each axis has unit gain by default; final pixel movement
    is the component gain times the caller's global ``intensity``. Signed gain
    reverses one axis and zero disables it. A phase of 0.25 shifts that axis
    by a quarter cycle. Fractional frequency or nonzero phase tapers the field
    at the page boundary; old integer, zero-phase fields remain unchanged.
    """
    if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in (A, B)):
        raise ValueError("A and B must be positive integer lens counts")
    if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p)
           for p in (phase_x, phase_y, amplitude_x, amplitude_y)):
        raise ValueError("phases and axis amplitudes must be finite numbers")
    frequency_x = A if frequency_x is None else frequency_x
    frequency_y = B if frequency_y is None else frequency_y
    if any(isinstance(f, bool) or not isinstance(f, (int, float)) or not math.isfinite(f) or f <= 0
           for f in (frequency_x, frequency_y)):
        raise ValueError("axis frequencies must be finite and positive")
    phase_x, phase_y = phase_x % 1, phase_y % 1
    cell_x, cell_y = frequency_x * x + 2 * phase_x, frequency_y * y + 2 * phase_y
    envelope = torch.sin(math.pi * cell_x) * torch.sin(math.pi * cell_y)
    if phase_x or phase_y or not float(frequency_x).is_integer() or not float(frequency_y).is_integer():
        envelope = envelope * _page_envelope(y, x)
    return (amplitude_y * envelope * (0.5 - torch.remainder(cell_y, 1)),
            amplitude_x * envelope * (0.5 - torch.remainder(cell_x, 1)))

def overlay_warped_grid(
    image: torch.Tensor, field: torch.Tensor | None = None, intensity: float = 0.0,
    spacing: int = 40, opacity: float = 0.45,
) -> torch.Tensor:
    """Composite a semitransparent light-gray grid onto an already warped page.

    ``field`` is the same (dy, dx) backward field used to warp the page;
    ``intensity`` is its pixel multiplier. Omit field for the initial, straight
    grid on the original image. The grid is sampled with ``warp_image`` so its
    deformations match the text exactly. Returns a new CHW image tensor.
    """
    image = _as_float_chw(image)
    _, height, width = image.shape
    if not isinstance(spacing, int) or spacing <= 0:
        raise ValueError("spacing must be a positive integer")
    if not math.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError("opacity must be finite and between 0 and 1")
    if not math.isfinite(intensity) or intensity < 0:
        raise ValueError("intensity must be finite and non-negative")
    if field is not None and (field.shape != (height, width, 2)
                              or field.device != image.device or not torch.isfinite(field).all()):
        raise ValueError("field must be finite H×W×2 on the image device")
    mask = image.new_zeros((1, height, width))
    mask[:, ::spacing, :] = 1
    mask[:, :, ::spacing] = 1
    if field is not None:
        mask = warp_image(mask, field, intensity)
    return image * (1 - opacity * mask) + 0.7 * opacity * mask

PAGE_WARPS = {
    "horizontal_wave": horizontal_wave,
    "vertical_wave": vertical_wave,
    "gentle_pinch": gentle_pinch,
    "sine_lens": sine_lens,
}
