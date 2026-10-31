"""PyTorch tensor image warps, reusable as callable transforms or from a CLI.

Images use CHW tensors with values in [0, 1]; integer images are interpreted
as 8-bit pixels. Pixel coordinates and displacement fields use (y, x).
"""
from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Callable

import torch

from .dataset_generator import generate_dataset
from .core import (
    CentreDistribution, DistortionConfig, PAGE_WARPS, displacement_field,
    distort_image, gentle_pinch, horizontal_wave, custom_wave, image_to_tensor,
    make_centres, make_nonoverlapping_centres, make_page_centres,
    overlay_warped_grid, page_warp_field, parabolic_profile, refract_image,
    refraction_distortion_matrices, refractive_index_field, render_lorem_page,
    sine_lens, validate_nonoverlap, vertical_wave, warp_image, warp_page,
)


@dataclass(frozen=True)
class WarpTransform:
    """Callable PyTorch transform for a single CHW image.

    ``mode='droplet'`` uses ``config``; page modes use ``intensity`` in pixels.
    ``sine_lens`` accepts lens settings via ``lens_options`` (e.g. A=3,
    frequency_x=2.5); ``refractive_index`` accepts optical parameters via
    ``refraction_options``. A custom normalized-coordinate callback can be
    supplied as ``warp_function`` with ``mode='custom'``.
    """

    mode: str = "droplet"
    intensity: float | None = None
    wavelength: float | None = None
    edge_taper: bool = True
    shift_X: float = 0.0
    shift_Y: float = 0.0
    config: DistortionConfig = field(default_factory=DistortionConfig)
    lens_options: dict = field(default_factory=dict)
    refraction_options: dict = field(default_factory=dict)
    warp_function: Callable | None = None
    mask: str | Path | torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.mode not in (*PAGE_WARPS, "droplet", "refractive_index", "custom"):
            raise ValueError(f"unsupported mode: {self.mode}")
        if self.intensity is not None and (isinstance(self.intensity, bool) or not isinstance(self.intensity, (int, float)) or not math.isfinite(self.intensity) or self.intensity < 0):
            raise ValueError("intensity must be finite and non-negative")
        if self.mode == "custom" and not callable(self.warp_function):
            raise ValueError("custom mode requires warp_function")
        if self.mode != "custom" and self.warp_function is not None:
            raise ValueError("warp_function requires custom mode")
        if self.wavelength is not None:
            if self.mode not in ("horizontal_wave", "vertical_wave"):
                raise ValueError("wavelength requires a wave mode")
            if (isinstance(self.wavelength, bool) or not isinstance(self.wavelength, (int, float))
                    or not math.isfinite(self.wavelength) or self.wavelength <= 0):
                raise ValueError("wavelength must be a finite positive number of pixels")
        if not isinstance(self.edge_taper, bool):
            raise ValueError("edge_taper must be a boolean")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in (self.shift_X, self.shift_Y)):
            raise ValueError("shift_X and shift_Y must be finite pixel values")
        if self.mode in ("droplet", "refractive_index") and (not self.edge_taper or self.shift_X or self.shift_Y):
            raise ValueError("edge_taper and shift require a page warp mode")
        if self.lens_options and self.mode != "sine_lens":
            raise ValueError("lens_options require sine_lens mode")
        if self.refraction_options and self.mode != "refractive_index":
            raise ValueError("refraction_options require refractive_index mode")
        if self.mode == "refractive_index" and self.intensity is not None:
            raise ValueError("intensity cannot be applied to a physical refraction model")
        self.config.validate()

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Return a warped CHW tensor on the input device, preserving gradients."""
        if self.mode == "droplet":
            config = self.config if self.intensity is None else replace(self.config, intensity=self.intensity)
            return distort_image(image, config, mask=self.mask)
        if self.mode == "refractive_index":
            return refract_image(image, mask=self.mask, **self.refraction_options)
        function = self.warp_function if self.mode == "custom" else PAGE_WARPS[self.mode]
        if self.mode == "sine_lens":
            function = partial(sine_lens, **self.lens_options)
        elif self.wavelength is not None:
            function = partial(function, wavelength=self.wavelength)
        if self.mode != "custom" or not self.edge_taper:
            function = partial(function, edge_taper=self.edge_taper)
        return warp_page(image, function, 3.0 if self.intensity is None else self.intensity,
                         mask=self.mask, shift_X=self.shift_X, shift_Y=self.shift_Y)


__all__ = [
    "WarpTransform", "CentreDistribution", "DistortionConfig", "PAGE_WARPS", "generate_dataset",
    "displacement_field", "distort_image", "gentle_pinch", "horizontal_wave", "custom_wave",
    "image_to_tensor", "make_centres", "make_nonoverlapping_centres", "make_page_centres",
    "overlay_warped_grid", "page_warp_field", "parabolic_profile", "refract_image",
    "refraction_distortion_matrices", "refractive_index_field", "render_lorem_page",
    "sine_lens", "validate_nonoverlap", "vertical_wave", "warp_image", "warp_page",
]
