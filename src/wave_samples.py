"""Generate separate 5×5 pixel-wavelength/intensity wave comparison grids."""
from __future__ import annotations

import argparse
from functools import partial
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .page import PAGE_WARPS, custom_wave, page_warp_field
from .tensor import _load_image, _save_image, warp_image

PERCENTAGES = (50, 100, 150, 200, 250)
INTENSITIES = (10, 20, 30, 40, 50)
WAVES = ("horizontal_wave", "vertical_wave")
CUSTOM_WAVES = ("custom_wave_fixed_vertical", "custom_wave_fixed_horizontal",
                "custom_wave_symmetric")


def generate_wave_grids(input_path: str | Path, output_dir: str | Path,
                        font_size_px: float = 28) -> Path:
    """Render 25 full-resolution samples and one labeled grid per wave direction.

    Wavelength is 50–250% of the source image's font size in pixel units;
    intensity is the maximum horizontal/vertical displacement in pixels.
    """
    return _generate_grids(input_path, output_dir, font_size_px, custom=False)


def generate_custom_wave_grids(input_path: str | Path, output_dir: str | Path,
                               font_size_px: float = 28) -> Path:
    """Render three 5×5 grids: fixed vertical, fixed horizontal, and symmetric.

    In the fixed cases the opposite wave has the user-controlled wavelength;
    the symmetric case applies that same pixel wavelength to X and Y.
    """
    return _generate_grids(input_path, output_dir, font_size_px, custom=True)


def _generate_grids(input_path: str | Path, output_dir: str | Path,
                    font_size_px: float, *, custom: bool) -> Path:
    if (isinstance(font_size_px, bool) or not isinstance(font_size_px, (int, float))
            or not math.isfinite(font_size_px) or font_size_px <= 0):
        raise ValueError("font_size_px must be finite and positive")
    input_path, output_dir = Path(input_path), Path(output_dir)
    source = _load_image(input_path)
    wavelengths = [font_size_px * percent / 100 for percent in PERCENTAGES]
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    font = ImageFont.load_default()
    thumb_w, thumb_h = 240, round(240 * source.shape[1] / source.shape[2])
    margin, label_h = 16, 24
    for mode in CUSTOM_WAVES if custom else WAVES:
        grid = Image.new("RGB", (margin + 5 * (thumb_w + margin),
                                 label_h + 5 * (thumb_h + label_h + margin)), "white")
        draw = ImageDraw.Draw(grid)
        for col, (percent, wavelength) in enumerate(zip(PERCENTAGES, wavelengths)):
            draw.text((margin + col * (thumb_w + margin), 4),
                      (f"X=Y {percent}% ({wavelength:g}px)" if mode == "custom_wave_symmetric"
                       else f"wavelength {percent}% ({wavelength:g}px)"), fill="black", font=font)
            fixed_axis = ("symmetric" if mode == "custom_wave_symmetric" else
                          mode.removeprefix("custom_wave_fixed_")) if custom else ""
            function = (partial(custom_wave, wavelength=wavelength, fixed_axis=fixed_axis)
                        if custom else partial(PAGE_WARPS[mode], wavelength=wavelength))
            field = page_warp_field(source, function)
            for row, intensity in enumerate(INTENSITIES):
                filename = f"{mode}/w{percent:03d}pct_{wavelength:g}px_i{intensity:03d}px.png"
                path = output_dir / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                rendered = warp_image(source, field, intensity)
                _save_image(rendered, path)
                sample = {"mode": mode, "wavelength_percent": percent,
                          "wavelength_px": wavelength, "intensity_px": intensity,
                          "file": filename}
                if custom:
                    sample["fixed_axis"] = fixed_axis
                    if fixed_axis == "symmetric":
                        sample.update(wavelength_x_px=wavelength, wavelength_y_px=wavelength)
                    else:
                        sample["fixed_wavelength_px"] = source.shape[2 if fixed_axis == "vertical" else 1] * .25
                samples.append(sample)
                thumbnail = Image.open(path).convert("RGB")
                thumbnail.thumbnail((thumb_w, thumb_h))
                x = margin + col * (thumb_w + margin)
                y = label_h + row * (thumb_h + label_h + margin)
                draw.text((x, y), f"intensity {intensity}px", fill="black", font=font)
                grid.paste(thumbnail, (x, y + label_h))
        grid.save(output_dir / f"{mode}_grid.png")
    manifest = {"source": str(input_path), "font_size_px": font_size_px,
                "wavelength_percent": list(PERCENTAGES), "wavelength_px": wavelengths,
                "intensity_px": list(INTENSITIES), "samples": samples}
    if custom:
        manifest["fixed_wave_period"] = {"vertical_px": source.shape[2] * .25,
                                         "horizontal_px": source.shape[1] * .25}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("original.png"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--custom-wave", action="store_true",
                        help="Compare a user-periodic wave against a fixed quarter-page wave on each axis.")
    parser.add_argument("--font-size-px", type=float, default=28,
                        help="Font size in pixels in the input image (default: 28 for original.png).")
    args = parser.parse_args()
    output_dir = args.output_dir or Path("examples/custom-wave-wavelength-grid" if args.custom_wave
                                       else "examples/wave-wavelength-grid")
    generate = generate_custom_wave_grids if args.custom_wave else generate_wave_grids
    print(generate(args.input, output_dir, args.font_size_px))


if __name__ == "__main__":
    main()
