"""Optics operations for PyTorch image warps."""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import torch
from .tensor import _as_float_chw, warp_image, load_mask
from .page import overlay_warped_grid


def _validate_refraction(thickness_mm: float, n_x: float, n_y: float,
                         n_z: float | None) -> float:
    """Validate a uniform, axis-aligned, plane-parallel lossless slab."""
    n_z = n_x if n_z is None else n_z
    for name, value in (("thickness_mm", thickness_mm), ("n_x", n_x),
                        ("n_y", n_y), ("n_z", n_z)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or (value < 0 if name == "thickness_mm" else value <= 0)):
            raise ValueError(f"{name} must be finite and physically valid")
    return n_z

def refraction_distortion_matrices(
    *, thickness_mm: float = 5.0, n_x: float = 1.3,
    n_y: float = 2.5, n_z: float | None = None,
    direction_xyz: tuple[float, float, float] = (-1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Two affine transforms in physical mm for a planar anisotropic slab.

    Direction is the air-side ray pointing from image toward the observer;
    its Z component must be positive. Tangential wavevectors are continuous
    across the plane interface. Solve the Fresnel determinant for each forward
    mode's k_z, obtain its electric eigenvector and energy-flow direction
    S=E×(k×E). The output translations are the apparent source offsets
    relative to the same ray in air, not pixel units. The 2x2 blocks are
    identity because a uniform slab cannot create spatially varying warps.
    Mode 0 is more X-polarized; mode 1 more Y-polarized.
    """
    n_z = _validate_refraction(thickness_mm, n_x, n_y, n_z)
    if (not isinstance(direction_xyz, (tuple, list)) or len(direction_xyz) != 3
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                   for v in direction_xyz) or direction_xyz[2] <= 0):
        raise ValueError("direction_xyz must have three finite values and positive Z")
    d = np.asarray(direction_xyz, dtype=np.float64)
    d /= np.linalg.norm(d)
    qx, qy = d[:2]
    eps = np.array([n_x * n_x, n_y * n_y, n_z * n_z])
    if max(eps) - min(eps) < 1e-12:
        u = eps[0] - qx*qx - qy*qy
        if u <= 0:
            raise ValueError("no propagating transmitted mode for this direction")
        offset = thickness_mm * (np.array([qx, qy]) / math.sqrt(u) - d[:2] / d[2])
        transforms = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1)
        transforms[:, :2, 2] = torch.from_numpy(offset).repeat(2, 1)
        return transforms.float()

    def determinant(u: float) -> float:
        q = np.array([qx, qy, math.sqrt(u)])
        return float(np.linalg.det(np.eye(3) * (qx*qx + qy*qy + u)
                                   - np.outer(q, q) - np.diag(eps)))

    f0, f1, f2 = determinant(0), determinant(1), determinant(2)
    a, b = (f2 - 2*f1 + f0) / 2, f1 - f0 - (f2 - 2*f1 + f0) / 2
    roots = np.roots([a, b, f0])
    if any(abs(u.imag) > 1e-8 or u.real <= 0 for u in roots):
        raise ValueError("no two propagating transmitted modes for this direction")
    modes = []
    for u in roots.real:
        q = np.array([qx, qy, math.sqrt(float(u))])
        matrix = np.eye(3) * float(q @ q) - np.outer(q, q) - np.diag(eps)
        _, _, vh = np.linalg.svd(matrix)
        e = vh[-1]
        s = np.cross(e, np.cross(q, e))
        if s[2] <= 0:
            raise ValueError("transmitted mode must carry energy toward positive Z")
        offset = thickness_mm * (s[:2] / s[2] - d[:2] / d[2])
        x_fraction = e[0]**2 / (e[0]**2 + e[1]**2)
        modes.append((x_fraction, offset))
    modes.sort(key=lambda mode: mode[0], reverse=True)
    transforms = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1)
    for i, (_, offset) in enumerate(modes):
        transforms[i, :2, 2] = torch.from_numpy(offset)
    return transforms.float()

def refractive_index_field(
    image: torch.Tensor, *, mode: str = "x", thickness_mm: float = 5.0,
    n_x: float = 1.3, n_y: float = 2.5, n_z: float | None = None,
    direction_xyz: tuple[float, float, float] = (-1.0, 1.0, 1.0),
    page_width_mm: float = 210.0, page_height_mm: float = 297.0,
) -> torch.Tensor:
    """Constant (dy, dx) backward-sampler field for one polarization mode.

    Page dimensions set physical mm/px separately in X and Y; they describe
    the raster's full extent, not an artificial optical warp amplitude.
    """
    if mode not in ("x", "y"):
        raise ValueError("mode must be x or y")
    for size in (page_width_mm, page_height_mm):
        if isinstance(size, bool) or not isinstance(size, (int, float)) or not math.isfinite(size) or size <= 0:
            raise ValueError("page_width_mm and page_height_mm must be finite positive numbers")
    transforms = refraction_distortion_matrices(
        thickness_mm=thickness_mm, n_x=n_x, n_y=n_y, n_z=n_z,
        direction_xyz=direction_xyz)
    image = _as_float_chw(image)
    _, height, width = image.shape
    dx = transforms[0 if mode == "x" else 1, 0, 2] * width / page_width_mm
    dy = transforms[0 if mode == "x" else 1, 1, 2] * height / page_height_mm
    return torch.stack((dy, dx)).to(device=image.device, dtype=image.dtype).expand(height, width, 2)

def refract_image(
    image: torch.Tensor, *, polarization_deg: float = 45.0,
    thickness_mm: float = 5.0, n_x: float = 1.3, n_y: float = 2.5,
    n_z: float | None = None, show_grid: bool = False, grid_spacing: int = 40,
    direction_xyz: tuple[float, float, float] = (-1.0, 1.0, 1.0),
    page_width_mm: float = 210.0, page_height_mm: float = 297.0,
    mask: str | Path | torch.Tensor | None = None,
) -> torch.Tensor:
    """Two separately back-sampled polarization views through a uniform slab.

    A4 is the default physical extent. Mode offsets use the exact Fresnel
    dispersion relation and energy-flow directions. At oblique incidence,
    X/Y cos²/sin² mode powers and normal-incidence Fresnel transmission below
    are approximations to full polarized interface coupling; the ray positions
    are physical, but the relative ghost brightness is not an exact prediction.
    The lossless, incoherent plate transmission sums internal reflections:
    R=((n-1)/(n+1))², T=(1-R)/(1+R). Coherent interference and absorption
    cannot be determined from the supplied variables.
    """
    if (isinstance(polarization_deg, bool) or not isinstance(polarization_deg, (int, float))
            or not math.isfinite(polarization_deg)):
        raise ValueError("polarization_deg must be a finite angle in degrees")
    image = _as_float_chw(image)
    if mask is not None:
        mask = load_mask(mask, image)
    options = dict(thickness_mm=thickness_mm, n_x=n_x, n_y=n_y, n_z=n_z,
                   direction_xyz=direction_xyz, page_width_mm=page_width_mm,
                   page_height_mm=page_height_mm)
    x_field = refractive_index_field(image, mode="x", **options)
    y_field = refractive_index_field(image, mode="y", **options)
    if show_grid:
        image = overlay_warped_grid(image, spacing=grid_spacing)
    if thickness_mm == 0:
        return image.clone()
    theta = math.radians(polarization_deg)
    def transmission(n: float) -> float:
        reflection = ((n - 1) / (n + 1)) ** 2
        return (1 - reflection) / (1 + reflection)
    x_gain = math.cos(theta) ** 2 * transmission(n_x)
    y_gain = math.sin(theta) ** 2 * transmission(n_y)
    if not x_field[0, 0].any() and not y_field[0, 0].any():
        return image * (x_gain + y_gain)
    return warp_image(image, x_field, 1, mask=mask) * x_gain + warp_image(image, y_field, 1, mask=mask) * y_gain
