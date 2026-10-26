"""Command-line interface for tensor image transformations."""
from __future__ import annotations

import argparse
import json
import math
import sys
from functools import partial
from pathlib import Path

from .dataset_generator import DEFAULT_LOCALES, generate_dataset
from .core import (
    CentreDistribution, DistortionConfig, PAGE_WARPS, _load_image, _save_image,
    image_to_tensor, render_lorem_page, sine_lens, refraction_distortion_matrices,
    refract_image, page_warp_field, warp_image, overlay_warped_grid,
    make_nonoverlapping_centres, distort_image,
)
from .tensor import load_mask


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "dataset-generator":
        dataset_parser = argparse.ArgumentParser(prog="lorem-warped dataset-generator")
        dataset_parser.add_argument("--locales", nargs="+", default=list(DEFAULT_LOCALES))
        dataset_parser.add_argument("--words-per-locale", type=int, default=100)
        dataset_parser.add_argument("--words-per-paragraph", type=int, default=100)
        dataset_parser.add_argument("--total-characters-per-paragraph", type=int)
        dataset_parser.add_argument("--font-size", type=float, default=12)
        dataset_parser.add_argument("--dpi", type=int, default=300)
        dataset_parser.add_argument("--seed", type=int, help="Reproduce the same Faker words across runs.")
        dataset_parser.add_argument("--output-root", type=Path, default=Path("."))
        options = dataset_parser.parse_args(argv[1:])
        try:
            output = generate_dataset(**vars(options))
        except ValueError as exc:
            dataset_parser.error(str(exc))
        print(f"Saved dataset to {output}")
        return

    defaults = DistortionConfig()
    parser = argparse.ArgumentParser(description="Apply PyTorch droplet distortion to an image.")
    parser.add_argument("--input", type=Path, help="Input image. Omit to use the Lorem ipsum example.")
    parser.add_argument("--output", type=Path, default=Path("distorted.png"))
    parser.add_argument("--mask", type=Path,
                        help="Same-size grayscale image: black disables displacement, white keeps it, gray scales it.")
    parser.add_argument("--centres", type=int, default=defaults.centre_count)
    parser.add_argument("--distribution", choices=[item.value for item in CentreDistribution], default=defaults.distribution.value)
    parser.add_argument("--radius", type=float, default=defaults.radius)
    parser.add_argument("--intensity", type=float, default=None,
                        help="Pixel warp multiplier for geometric modes only.")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--profile", choices=("cosine", "parabolic"), default=defaults.profile)
    parser.add_argument("--page-warp", choices=(*PAGE_WARPS, "refractive_index"),
                        help="Warp the whole page with a smooth coordinate function instead of droplets.")
    parser.add_argument("--lens-a", type=int, metavar="COLUMNS",
                        help="Sine lens columns A (positive integer; default: 2).")
    parser.add_argument("--lens-b", type=int, metavar="ROWS",
                        help="Sine lens rows B (positive integer; default: 2).")
    parser.add_argument("--lens-phase-x", type=float, metavar="CYCLES",
                        help="Shift sine lens X by a fraction of a cycle; 0.25 = quarter-cycle.")
    parser.add_argument("--lens-phase-y", type=float, metavar="CYCLES",
                        help="Shift sine lens Y by a fraction of a cycle; 0.25 = quarter-cycle.")
    parser.add_argument("--lens-amplitude-x", type=float, metavar="GAIN",
                        help="X displacement gain (default: 1; multiplied by --intensity).")
    parser.add_argument("--lens-amplitude-y", type=float, metavar="GAIN",
                        help="Y displacement gain (default: 1; multiplied by --intensity).")
    parser.add_argument("--lens-frequency-x", type=float, metavar="FREQUENCY",
                        help="Positive X lens frequency; overrides --lens-a (default: 2).")
    parser.add_argument("--lens-frequency-y", type=float, metavar="FREQUENCY",
                        help="Positive Y lens frequency; overrides --lens-b (default: 2).")
    parser.add_argument("--refraction-thickness-mm", type=float, default=None)
    parser.add_argument("--refraction-n-x", type=float, default=None)
    parser.add_argument("--refraction-n-y", type=float, default=None)
    parser.add_argument("--refraction-n-z", type=float, default=None,
                        help="Axial index (default: same as --refraction-n-x).")
    parser.add_argument("--refraction-polarization-deg", type=float, default=None,
                        help="Incident linear polarization from +X toward +Y (default: 45 degrees).")
    parser.add_argument("--refraction-direction", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="Air-side ray toward observer; default (-1, 1, 1).")
    parser.add_argument("--refraction-matrices-mm", action="store_true",
                        help="Print two physical-mm mode transforms as JSON without rendering an image.")
    parser.add_argument("--page-width-mm", type=float,
                        help="Physical image width (default: A4 portrait, 210 mm).")
    parser.add_argument("--page-height-mm", type=float,
                        help="Physical image height (default: A4 portrait, 297 mm).")
    parser.add_argument("--show-grid", action="store_true",
                        help="Overlay a translucent light-gray grid, warped with the page.")
    parser.add_argument("--show-original-grid", type=Path, metavar="PATH",
                        help="Save the original text with the straight reference grid to PATH.")
    parser.add_argument("--grid-spacing", type=int, default=40,
                        help="Spacing between grid lines in source pixels (default: 40).")
    parser.add_argument("--nonoverlapping-page", action="store_true",
                        help="Fill the page with disjoint droplets; overrides --centres.")
    args = parser.parse_args(argv)
    if args.page_warp and args.nonoverlapping_page:
        parser.error("--page-warp and --nonoverlapping-page cannot be combined")
    if args.page_warp != "sine_lens" and any(value is not None for value in
            (args.lens_a, args.lens_b, args.lens_phase_x, args.lens_phase_y,
             args.lens_amplitude_x, args.lens_amplitude_y,
             args.lens_frequency_x, args.lens_frequency_y)):
        parser.error("--lens-* options require --page-warp sine_lens")
    if args.page_warp == "sine_lens" and (args.lens_a is not None and args.lens_a <= 0
                                           or args.lens_b is not None and args.lens_b <= 0):
        parser.error("--lens-a and --lens-b must be positive integers")
    if any(value is not None and not math.isfinite(value) for value in
           (args.lens_phase_x, args.lens_phase_y, args.lens_amplitude_x, args.lens_amplitude_y,
            args.lens_frequency_x, args.lens_frequency_y)):
        parser.error("lens phase, amplitude, and frequency must be finite")
    if any(value is not None and value <= 0 for value in
           (args.lens_frequency_x, args.lens_frequency_y)):
        parser.error("lens frequencies must be positive")
    refraction_options = (args.refraction_thickness_mm, args.refraction_n_x, args.refraction_n_y,
                          args.refraction_n_z, args.refraction_polarization_deg)
    if args.page_warp != "refractive_index" and (any(value is not None for value in refraction_options)
                                                    or args.refraction_direction is not None
                                                    or args.refraction_matrices_mm
                                                    or args.page_width_mm is not None
                                                    or args.page_height_mm is not None):
        parser.error("--refraction-* options require --page-warp refractive_index")
    if args.page_warp == "refractive_index" and args.intensity is not None:
        parser.error("--intensity cannot create a physical warp in a uniform slab")
    if not args.page_warp and (args.show_grid or args.show_original_grid):
        parser.error("grid options require --page-warp")
    if args.grid_spacing <= 0:
        parser.error("--grid-spacing must be positive")
    if args.show_original_grid == args.output:
        parser.error("--show-original-grid must differ from --output")

    if args.refraction_matrices_mm:
        if args.mask is not None:
            parser.error("--mask requires image rendering, not --refraction-matrices-mm")
        try:
            matrices = refraction_distortion_matrices(
                thickness_mm=5 if args.refraction_thickness_mm is None else args.refraction_thickness_mm,
                n_x=1.3 if args.refraction_n_x is None else args.refraction_n_x,
                n_y=2.5 if args.refraction_n_y is None else args.refraction_n_y,
                n_z=args.refraction_n_z,
                direction_xyz=(-1, 1, 1) if args.refraction_direction is None
                else args.refraction_direction)
        except ValueError as exc:
            parser.error(str(exc))
        print(json.dumps({"unit": "mm", "matrices_mm": matrices.tolist()}))
        return

    source = _load_image(args.input) if args.input else image_to_tensor(render_lorem_page())
    try:
        mask = load_mask(args.mask, source) if args.mask is not None else None
    except (OSError, ValueError, TypeError) as exc:
        parser.error(f"mask: {exc}")
    if args.page_warp:
        if args.page_warp == "refractive_index":
            try:
                output = refract_image(
                    source, polarization_deg=45 if args.refraction_polarization_deg is None
                    else args.refraction_polarization_deg,
                    show_grid=args.show_grid, grid_spacing=args.grid_spacing,
                    thickness_mm=5 if args.refraction_thickness_mm is None else args.refraction_thickness_mm,
                    n_x=1.3 if args.refraction_n_x is None else args.refraction_n_x,
                    n_y=2.5 if args.refraction_n_y is None else args.refraction_n_y,
                    n_z=args.refraction_n_z,
                    direction_xyz=(-1, 1, 1) if args.refraction_direction is None
                    else args.refraction_direction,
                    page_width_mm=210 if args.page_width_mm is None else args.page_width_mm,
                    page_height_mm=297 if args.page_height_mm is None else args.page_height_mm,
                    mask=mask,
                )
            except ValueError as exc:
                parser.error(str(exc))
        else:
            intensity = defaults.intensity if args.intensity is None else args.intensity
            function = (partial(sine_lens, A=2 if args.lens_a is None else args.lens_a,
                            B=2 if args.lens_b is None else args.lens_b,
                            phase_x=0.0 if args.lens_phase_x is None else args.lens_phase_x,
                            phase_y=0.0 if args.lens_phase_y is None else args.lens_phase_y,
                            amplitude_x=1.0 if args.lens_amplitude_x is None else args.lens_amplitude_x,
                            amplitude_y=1.0 if args.lens_amplitude_y is None else args.lens_amplitude_y,
                            frequency_x=args.lens_frequency_x, frequency_y=args.lens_frequency_y)
                        if args.page_warp == "sine_lens" else PAGE_WARPS[args.page_warp])
            field = page_warp_field(source, function)
            if mask is not None:
                field = field * mask.unsqueeze(-1)
            output = warp_image(source, field, intensity)
            if args.show_grid:
                output = overlay_warped_grid(output, field, intensity, spacing=args.grid_spacing)
        if args.show_original_grid:
            _save_image(overlay_warped_grid(source, spacing=args.grid_spacing), args.show_original_grid)
        _save_image(output, args.output)
        print(f"Saved distorted image to {args.output}")
        return
    config = DistortionConfig(
        centre_count=args.centres,
        distribution=CentreDistribution(args.distribution),
        radius=args.radius,
        intensity=defaults.intensity if args.intensity is None else args.intensity,
        seed=args.seed,
        profile=args.profile,
    )
    centres = None
    if args.nonoverlapping_page:
        centres = make_nonoverlapping_centres(
            source.shape[1], source.shape[2], args.radius, args.distribution, args.seed, source.device,
        )
    _save_image(distort_image(source, config, centres, mask=mask), args.output)
    print(f"Saved distorted image to {args.output}")


