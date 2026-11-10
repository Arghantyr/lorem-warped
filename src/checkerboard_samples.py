"""Ten checkerboard XOR examples from 4× to ¼× the source font height.

From the repository root:
    python -m src.checkerboard_samples --input original.png \
        --crop 70 104 590 564 --output-dir examples/checkerboard-xor
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from . import ImageTransform, image_to_tensor


def generate_xor_checkerboard_samples(
    input_path: str | Path, output_dir: str | Path, font_height_px: float = 28,
    crop: tuple[int, int, int, int] | None = None,
) -> Path:
    """Save ten binary masks, their XOR images, and a labeled comparison sheet."""
    if (isinstance(font_height_px, bool) or not isinstance(font_height_px, (int, float))
            or not math.isfinite(font_height_px) or font_height_px <= 0):
        raise ValueError("font_height_px must be finite and positive")
    input_path, output_dir = Path(input_path), Path(output_dir)
    with Image.open(input_path) as original:
        if crop is not None and (len(crop) != 4 or not (0 <= crop[0] < crop[2] <= original.width
                                                  and 0 <= crop[1] < crop[3] <= original.height)):
            raise ValueError("crop must be inside the input image with positive dimensions")
        source = (original.crop(crop) if crop is not None else original).convert("RGB")
    image = image_to_tensor(source)
    height, width = image.shape[1:]
    # Log spacing gives similar relative changes at large and small square sizes.
    sizes = [round(4 * font_height_px * (1 / 16) ** (index / 9)) for index in range(10)]
    if sizes[-1] < 1 or len(set(sizes)) != 10:
        raise ValueError("font_height_px is too small for ten distinct pixel-sized squares")
    output_dir.mkdir(parents=True, exist_ok=True)
    source.save(output_dir / "source.png")
    y = torch.arange(height)[:, None]
    x = torch.arange(width)[None, :]
    thumb_width = 260
    thumb_height = round(thumb_width * height / width)
    gap, label_height = 16, 24
    row_height = thumb_height + label_height + gap
    sheet = Image.new("RGB", (3 * thumb_width + 4 * gap,
                               label_height + 10 * row_height + gap), "white")
    draw = ImageDraw.Draw(sheet)
    for column, heading in enumerate(("Source", "Binary checkerboard", "XOR result")):
        draw.text((gap + column * (thumb_width + gap), 4), heading, fill="black")
    samples = []
    for index, size in enumerate(sizes):
        mask = ((x // size + y // size) % 2).bool()
        mask_name = f"mask_{index + 1:02d}_{size}px.png"
        output_name = f"xor_{index + 1:02d}_{size}px.png"
        mask_image = Image.fromarray(mask.byte().numpy() * 255, "L")
        result = ImageTransform.logical_xor(image, mask)
        output_image = Image.fromarray((result.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        mask_image.save(output_dir / mask_name)
        output_image.save(output_dir / output_name)
        samples.append({"square_px": size, "mask": mask_name, "output": output_name})
        row_y = label_height + index * row_height
        for column, panel in enumerate((source, mask_image, output_image)):
            px = gap + column * (thumb_width + gap)
            draw.text((px, row_y), f"{index + 1:02d}: {size}px ({size / font_height_px:g}× font)",
                      fill="black")
            sheet.paste(panel.resize((thumb_width, thumb_height),
                                     Image.Resampling.NEAREST if column == 1 else Image.Resampling.LANCZOS),
                        (px, row_y + label_height))
    sheet.save(output_dir / "contact_sheet.png")
    manifest = {"source": str(input_path), "crop": crop, "font_height_px": font_height_px,
                "source_file": "source.png", "contact_sheet": "contact_sheet.png",
                "samples": samples}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("original.png"))
    parser.add_argument("--output-dir", type=Path, default=Path("examples/checkerboard-xor"))
    parser.add_argument("--font-height-px", type=float, default=28,
                        help="Source body font height in pixels (original.png uses 28).")
    parser.add_argument("--crop", nargs=4, type=int, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    args = parser.parse_args()
    print(generate_xor_checkerboard_samples(args.input, args.output_dir,
                                            args.font_height_px, tuple(args.crop) if args.crop else None))


if __name__ == "__main__":
    main()
