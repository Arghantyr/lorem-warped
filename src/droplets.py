"""Droplets operations for PyTorch image warps."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
import torch
from .tensor import _as_float_chw, _validate_centres, warp_image


class CentreDistribution(str, Enum):
    RANDOM = "random"
    SQUARE = "square"
    TRIANGLE = "triangle"

@dataclass(frozen=True)
class DistortionConfig:
    centre_count: int = 3
    distribution: CentreDistribution = CentreDistribution.RANDOM
    radius: float = 80.0
    intensity: float = 12.0
    seed: int | None = 7
    profile: str = "cosine"

    def validate(self) -> None:
        if self.profile not in ("cosine", "parabolic"):
            raise ValueError("profile must be cosine or parabolic")
        if self.centre_count < 0:
            raise ValueError("centre_count must be non-negative")
        if not math.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("radius must be finite and positive")
        if not math.isfinite(self.intensity) or self.intensity < 0:
            raise ValueError("intensity must be finite and non-negative")

def distort_image(
    image: torch.Tensor,
    config: DistortionConfig,
    centres: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply radial droplet distortions to a CHW image tensor.

    Args:
        image: Float or integer tensor shaped ``(channels, height, width)``.
        config: Distortion parameters. ``intensity`` is the displacement
            amplitude in pixels.
        centres: Optional tensor shaped ``(N, 2)`` containing ``(y, x)``
            centre coordinates. If omitted, centres are generated from the
            selected distribution.

    Returns:
        A float tensor with the same shape as ``image`` and values in [0, 1].
    """
    config.validate()
    image = _as_float_chw(image)
    _, height, width = image.shape

    if centres is None:
        centres = make_centres(config, height, width, device=image.device)
    centres = _validate_centres(centres, height, width, image.device)

    if centres.numel() == 0 or config.intensity == 0:
        return image.clone()

    field = displacement_field(image, config.radius, centres, profile=config.profile)
    return warp_image(image, field, config.intensity)

def displacement_field(
    image: torch.Tensor, radius: float, centres: torch.Tensor,
    profile: str = "cosine",
) -> torch.Tensor:
    """Unit-amplitude (dy, dx) field, reusable for an intensity sweep.

    Evaluate each circular profile only in its clipped bounding box, rather
    than over the entire image. Overlapping droplet displacements add.
    Bounding boxes use detached coordinates; field arithmetic retains centre
    gradients away from the profile's nondifferentiable support boundaries.
    """
    if profile not in ("cosine", "parabolic"):
        raise ValueError("profile must be cosine or parabolic")
    if profile == "parabolic":
        validate_nonoverlap(centres, radius)
    _, height, width = image.shape
    field = image.new_zeros((height, width, 2))
    # One device-to-host transfer, not one synchronization per coordinate.
    positions = centres.detach().cpu().tolist()
    for (bound_y, bound_x), (centre_y, centre_x) in zip(positions, centres):
        top = max(0, math.ceil(bound_y - radius))
        bottom = min(height, math.floor(bound_y + radius) + 1)
        left = max(0, math.ceil(bound_x - radius))
        right = min(width, math.floor(bound_x + radius) + 1)
        if top >= bottom or left >= right:
            continue
        dy = torch.arange(top, bottom, device=image.device, dtype=image.dtype)[:, None] - centre_y
        dx = torch.arange(left, right, device=image.device, dtype=image.dtype)[None, :] - centre_x
        distance = (dy.square() + dx.square()).sqrt()
        magnitude = (parabolic_profile(distance, 1.0, radius) if profile == "parabolic"
                     else 1 + torch.cos(2 * math.pi * distance / radius))
        scale = torch.where(distance <= radius, magnitude, 0) / distance.clamp_min(1e-6)
        field[top:bottom, left:right, 0] += scale * dy
        field[top:bottom, left:right, 1] += scale * dx
    return field

def make_centres(
    config: DistortionConfig,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate centre coordinates as an ``(N, 2)`` tensor of ``(y, x)``."""
    if config.centre_count == 0:
        return torch.empty((0, 2), device=device, dtype=torch.float32)
    if config.distribution == CentreDistribution.RANDOM:
        generator = torch.Generator(device=device)
        if config.seed is not None:
            generator.manual_seed(config.seed)
        margin_y = min(config.radius, height / 2)
        margin_x = min(config.radius, width / 2)
        ys = torch.rand(config.centre_count, generator=generator, device=device)
        xs = torch.rand(config.centre_count, generator=generator, device=device)
        ys = margin_y + ys * max(height - 2 * margin_y, 0.0)
        xs = margin_x + xs * max(width - 2 * margin_x, 0.0)
        return torch.stack((ys, xs), dim=1)

    if config.distribution == CentreDistribution.SQUARE:
        return _grid_centres(config.centre_count, height, width, triangular=False, device=device)
    if config.distribution == CentreDistribution.TRIANGLE:
        return _grid_centres(config.centre_count, height, width, triangular=True, device=device)
    raise ValueError(f"unsupported distribution: {config.distribution}")

def _grid_centres(
    count: int,
    height: int,
    width: int,
    triangular: bool,
    device: torch.device,
) -> torch.Tensor:
    """Fill a regular rectangular or equilateral-triangle lattice row-wise."""
    columns = max(1, math.ceil(math.sqrt(count * width / max(height, 1))))
    rows = max(1, math.ceil(count / columns))
    spacing_x = width / (columns + 1)
    spacing_y = height / (rows + 1)
    if triangular:
        # In an equilateral triangular lattice, vertical spacing is
        # horizontal spacing multiplied by sqrt(3) / 2.
        spacing_y = spacing_x * math.sqrt(3.0) / 2.0
        if spacing_y * (rows + 1) > height:
            spacing_y = height / (rows + 1)
            spacing_x = 2.0 * spacing_y / math.sqrt(3.0)

    points: list[tuple[float, float]] = []
    for row in range(rows):
        row_offset = spacing_x / 2 if triangular and row % 2 else 0.0
        for column in range(columns):
            if len(points) == count:
                break
            x = (column + 1) * spacing_x + row_offset
            y = (row + 1) * spacing_y
            if x < width and y < height:
                points.append((y, x))
    return torch.tensor(points, dtype=torch.float32, device=device)

def make_page_centres(
    height: int,
    width: int,
    radius: float,
    distribution: CentreDistribution | str,
    seed: int = 7,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Fill the page with centres spaced one radius apart.

    Regular grids overlap droplets for dense coverage. Triangle rows have
    sqrt(3)/2 times the horizontal spacing. Random uses one seeded, uniformly
    sampled point per radius-sized cell (stratified random, not a lattice).
    Unlike make_centres, the count is determined by page size and radius.
    """
    if height <= 0 or width <= 0 or not math.isfinite(radius) or radius <= 0:
        raise ValueError("page dimensions and radius must be positive and finite")
    distribution = CentreDistribution(distribution)
    columns = math.ceil(width / radius)
    row_spacing = radius * (math.sqrt(3) / 2 if distribution == CentreDistribution.TRIANGLE else 1)
    rows = math.ceil(height / row_spacing)
    row, column = torch.meshgrid(
        torch.arange(rows, device=device), torch.arange(columns, device=device), indexing="ij",
    )
    if distribution == CentreDistribution.RANDOM:
        generator = torch.Generator(device=device).manual_seed(seed)
        jitter = torch.rand((rows, columns, 2), generator=generator, device=device)
        y = (row + jitter[..., 0]) * (height / rows)
        x = (column + jitter[..., 1]) * (width / columns)
    else:
        y = row * row_spacing + (height - (rows - 1) * row_spacing) / 2
        x = column * radius + (width - (columns - 1) * radius) / 2
        if distribution == CentreDistribution.TRIANGLE:
            x = x + (row % 2) * (radius / 2)
    centres = torch.stack((y, x), dim=-1).reshape(-1, 2)
    return centres[(centres[:, 0] < height) & (centres[:, 1] < width)]

def parabolic_profile(distance: torch.Tensor, amplitude: float, radius: float) -> torch.Tensor:
    """f(c) = max(A - d*c², 0), where d = A/r²; A is in pixels."""
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    if not math.isfinite(amplitude) or amplitude < 0:
        raise ValueError("amplitude must be finite and non-negative")
    return (amplitude - (amplitude / radius**2) * distance.square()).clamp_min(0)

def validate_nonoverlap(centres: torch.Tensor, radius: float) -> None:
    """Reject intersecting disk interiors using a spatial hash, not an N² matrix.

    Tangency is allowed: the parabolic field is zero at the circle boundary.
    Check the actual stored coordinates, so rounding cannot hide an overlap.
    """
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    if centres.ndim != 2 or centres.shape[1] != 2 or not torch.isfinite(centres).all():
        raise ValueError("centres must be a finite (N, 2) tensor")
    diameter = 2 * radius
    buckets = {}
    for y, x in centres.detach().cpu().tolist():
        row, column = math.floor(y / diameter), math.floor(x / diameter)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for other_y, other_x in buckets.get((row + dy, column + dx), ()):
                    if (y - other_y)**2 + (x - other_x)**2 < diameter**2:
                        raise ValueError("distortion fields overlap: centre separation must be >= 2*radius")
        buckets.setdefault((row, column), []).append((y, x))

def make_nonoverlapping_centres(
    height: int, width: int, radius: float,
    distribution: CentreDistribution | str, seed: int = 7,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Page-wide disjoint disks, fully inside the image.

    Grids use diameter spacing plus a tiny float32 rounding guard. Random
    places one point in each >=3r cell, at least r from every cell boundary.
    This is stratified random placement with a guaranteed separation, not
    unconstrained uniform sampling or maximum-density random packing.
    """
    if not math.isfinite(radius) or radius <= 0 or min(height, width) < 2 * radius:
        raise ValueError("radius must be positive and its diameter must fit the page")
    distribution = CentreDistribution(distribution)
    guard = max(height, width, radius) * torch.finfo(torch.float32).eps * 8
    spacing = 2 * radius + guard
    if distribution == CentreDistribution.RANDOM:
        rows = max(1, math.floor(height / (3 * radius)))
        columns = max(1, math.floor(width / (3 * radius)))
        row, column = torch.meshgrid(torch.arange(rows), torch.arange(columns), indexing="ij")
        cell_y, cell_x = height / rows, width / columns
        jitter = torch.rand((rows, columns, 2), generator=torch.Generator().manual_seed(seed))
        y = row * cell_y + radius + jitter[..., 0] * (cell_y - 2 * radius)
        x = column * cell_x + radius + jitter[..., 1] * (cell_x - 2 * radius)
    else:
        row_spacing = spacing * (math.sqrt(3) / 2 if distribution == CentreDistribution.TRIANGLE else 1)
        rows = math.floor((height - 2 * radius) / row_spacing) + 1
        columns = math.floor((width - 2 * radius) / spacing) + 1
        row, column = torch.meshgrid(torch.arange(rows), torch.arange(columns), indexing="ij")
        y = row * row_spacing + (height - (rows - 1) * row_spacing) / 2
        x = column * spacing + (width - (columns - 1) * spacing) / 2
        if distribution == CentreDistribution.TRIANGLE:
            x = x + (row % 2) * (spacing / 2)
    centres = torch.stack((y, x), dim=-1).reshape(-1, 2).to(device)
    limits = centres.new_tensor([height - radius, width - radius])
    centres = centres[((centres >= radius) & (centres <= limits)).all(dim=1)]
    validate_nonoverlap(centres, radius)
    return centres
