"""Build the ASD-STE100-style manual and real PyTorch comparison galleries.

Run from the repository root after installing the dev extra:
python docs/build_manual.py --input original.png --output docs/lorem-warped-manual.pdf
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from functools import partial
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pymupdf
import torch
from PIL import Image as PILImage, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                                PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import (CentreDistribution, DistortionConfig, WarpTransform,
                 custom_wave as pixel_custom_wave, image_to_tensor, make_centres, overlay_warped_grid)
from src.wave_samples import generate_custom_wave_grids, generate_wave_grids, PERCENTAGES, INTENSITIES
from manual_atlas import build_atlas, overlay_droplet_guides, source_paragraph


LENS = dict(frequency_x=2.5, frequency_y=1.5, phase_x=0.1,
            phase_y=0.2, amplitude_x=1.2, amplitude_y=0.8)
STRENGTHS = {
    "droplet": (0.5, 1, 2, 3, 5),
    "sine_lens": (0.5, 4, 32, 256, 1024),
    "refractive_index": (0.5, 2, 5, 15, 40),
    "horizontal_wave": (2, 8, 24, 60, 120),
    "vertical_wave": (2, 8, 24, 60, 120),
    "gentle_pinch": (8, 32, 80, 160, 320),
    "custom": (2, 5, 10, 20, 30),
}


def transform_for(mode: str, value: float) -> WarpTransform:
    if mode == "droplet":
        return WarpTransform(mode=mode, config=DistortionConfig(
            centre_count=3, radius=80, intensity=value, seed=7, profile="cosine"))
    if mode == "sine_lens":
        return WarpTransform(mode=mode, intensity=value, lens_options=LENS)
    if mode == "refractive_index":
        return WarpTransform(mode=mode, refraction_options=dict(
            thickness_mm=value, n_x=1.3, n_y=2.5, n_z=1.3,
            polarization_deg=45, direction_xyz=(-1, 1, 1),
            page_width_mm=210, page_height_mm=297))
    if mode == "custom":
        return WarpTransform(mode=mode, intensity=value,
                             warp_function=partial(pixel_custom_wave, wavelength=28,
                                                   fixed_axis="symmetric"))
    if mode in ("horizontal_wave", "vertical_wave"):
        return WarpTransform(mode=mode, intensity=value, wavelength=28)
    return WarpTransform(mode=mode, intensity=value)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#526070"))
    canvas.drawString(18 * mm, 13 * mm, "lorem-warped  |  Operator manual  |  selected STE principles")
    canvas.drawRightString(192 * mm, 13 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build(input_path: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    generated = output.parent / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    # Earlier builds wrote an unmarked crop here. Do not leave a stale
    # displayed-looking "original" beside the new gray-guide controls.
    (generated / "original.png").unlink(missing_ok=True)
    gallery_dir = output.parent / "images"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    with PILImage.open(input_path) as source:
        image = image_to_tensor(source)
        source_rgb = source.convert("RGB")
    width, height = source_rgb.size
    crop_w, crop_h = min(width, 520), min(height, 460)
    left = min(width - crop_w, round(width * 0.058))
    top = min(height - crop_h, round(height * 0.065))
    crop_box = (left, top, left + crop_w, top + crop_h)
    # Keep grid spacing proportional to the input, including small test images.
    radius = max(12, min(70, min(width, height) // 17))
    dense_count = max(9, round(width * height / radius**2))
    # Leave space between arrangement centres so their 3 px warp remains
    # readable; the separate opening overview retains the dense layout.
    distribution_count = max(9, round(dense_count * 0.18))
    distribution_radius = max(12, min(50, min(width, height) // 24))

    def guide_panel(settings: DistortionConfig, name: str) -> dict:
        centres = make_centres(settings, height, width, device=image.device)
        guide = generated / f"{name}_original_guide.png"
        overlay_droplet_guides(source_rgb, centres.tolist(), settings.radius,
                               show_centres=True).crop(crop_box).save(guide)
        return {"path": str(guide), "role": "original", "overlay_alpha": .3,
                "distribution": settings.distribution.value, "radius": settings.radius,
                "count": settings.centre_count, "actual_count": len(centres)}

    def droplet_panel(distribution: CentreDistribution, count: int, name: str,
                      intensity: float, radius_px: float) -> tuple[dict, dict]:
        settings = DistortionConfig(centre_count=count, distribution=distribution,
                                    radius=radius_px, intensity=intensity, seed=7, profile="cosine")
        guide = guide_panel(settings, name)
        output_image = WarpTransform(mode="droplet", config=settings)(image)
        pixels = output_image.mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
        full_path = generated / f"{name}.png"
        crop_path = generated / f"{name}_crop.png"
        PILImage.fromarray(pixels).save(full_path)
        PILImage.fromarray(pixels).crop(crop_box).save(crop_path)
        return guide, {"distribution": distribution.value, "role": "warped",
                "path": str(crop_path),
                "full_path": str(full_path), "count": count,
                "actual_count": guide["actual_count"], "radius": radius_px,
                "intensity": intensity}

    overview_original, overview_square = droplet_panel(
        CentreDistribution.SQUARE, dense_count, "overview_dense_square", .5, radius)
    overview = {"distribution": "square", "panels": [
        {"label": "Original + gray droplet guide (30%)", **overview_original},
        {"label": "Dense square grid: 0.5 px", **overview_square}],
        "count": dense_count, "radius": radius, "intensity": .5}
    distributions = {"panels": [], "count": distribution_count,
                     "radius": distribution_radius, "intensity": 3}
    for arrangement in CentreDistribution:
        original_panel, warped_panel = droplet_panel(
            arrangement, distribution_count,
            f"droplet_distribution_{arrangement.value}", 3, distribution_radius)
        name = {"random": "Random", "square": "Square grid",
                "triangle": "Triangle (hexagonal)"}[arrangement.value]
        original_panel["label"] = f"{name} centres: original (gray 30%)"
        warped_panel["label"] = f"{name}: 3 px warp"
        distributions["panels"].extend((original_panel, warped_panel))

    def comparison_image(panels: list[dict], name: str) -> Path:
        cell_w, cell_h, margin, label_h = 360, 320, 12, 32
        rows = (len(panels) + 1) // 2
        collage = PILImage.new("RGB", (2 * (cell_w + margin) + margin,
                                      rows * (cell_h + label_h + margin) + margin), "#eef2f5")
        draw = ImageDraw.Draw(collage)
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
        for index, panel in enumerate(panels):
            x = margin + (index % 2) * (cell_w + margin)
            y = margin + (index // 2) * (cell_h + label_h + margin)
            with PILImage.open(panel["path"]) as crop:
                scaled = crop.copy()
                scaled.thumbnail((cell_w, cell_h), PILImage.Resampling.LANCZOS)
                collage.paste(scaled, (x + (cell_w - scaled.width) // 2, y))
            draw.text((x + 6, y + cell_h + 6), panel["label"],
                      fill="#17324d", font=font)
        path = gallery_dir / name
        collage.save(path)
        return path

    overview["readme_image"] = str(comparison_image(overview["panels"], "overview_dense_square.png"))
    distributions["readme_image"] = str(comparison_image(
        distributions["panels"], "droplet_distribution_grid.png"))
    atlas = build_atlas(gallery_dir, generated)
    gallery_guide = guide_panel(DistortionConfig(
        centre_count=3, distribution=CentreDistribution.RANDOM, radius=80,
        intensity=0, seed=7, profile="cosine"), "gallery")
    grid_spacing, grid_opacity = 28, .24
    gridded_source = overlay_warped_grid(image, spacing=grid_spacing,
                                         opacity=grid_opacity)
    grid_original = generated / "gallery_original_square_grid.png"
    grid_pixels = gridded_source.mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    PILImage.fromarray(grid_pixels).crop(crop_box).save(grid_original)
    galleries = {}
    for mode, values in STRENGTHS.items():
        unit = "mm thickness" if mode == "refractive_index" else "px intensity"
        panels = ([{"label": "Original + gray droplet guide (30%)", "value": 0,
                    **gallery_guide}] if mode == "droplet" else
                  [{"label": "Original + straight square grid", "value": 0,
                    "path": str(grid_original), "role": "original",
                    "guide": "straight_square_grid"}])
        for value in values:
            result = transform_for(mode, value)(image)
            pixels = result.mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
            full = PILImage.fromarray(pixels)
            suffix = str(value).replace(".", "_")
            full_path = generated / f"{mode}_{suffix}.png"
            preview_path = generated / f"{mode}_{suffix}_crop.png"
            full.save(full_path)
            if mode == "droplet":
                full.crop(crop_box).save(preview_path)
            else:
                # Warp the gridded source, not a straight grid drawn afterward.
                guided = transform_for(mode, value)(gridded_source)
                guided_pixels = guided.mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
                PILImage.fromarray(guided_pixels).crop(crop_box).save(preview_path)
            panels.append({"label": f"{value:g} {'mm' if mode == 'refractive_index' else 'px'}",
                           "value": value, "path": str(preview_path),
                           "full_path": str(full_path),
                           **({"guide": "warped_square_grid"} if mode != "droplet" else {})})
        galleries[mode] = {"unit": unit, "panels": panels,
                           **({"grid_spacing": grid_spacing, "grid_opacity": grid_opacity}
                              if mode != "droplet" else {})}
        # The README uses exactly the same six real crops as the PDF.
        cell_w, cell_h, margin, label_h = 360, 320, 12, 32
        collage = PILImage.new("RGB", (2 * (cell_w + margin) + margin,
                                      3 * (cell_h + label_h + margin) + margin), "#eef2f5")
        draw = ImageDraw.Draw(collage)
        font = ImageFont.load_default()
        for index, panel in enumerate(panels):
            x = margin + (index % 2) * (cell_w + margin)
            y = margin + (index // 2) * (cell_h + label_h + margin)
            with PILImage.open(panel["path"]) as crop:
                scaled = crop.copy()
                scaled.thumbnail((cell_w, cell_h), PILImage.Resampling.LANCZOS)
                collage.paste(scaled, (x + (cell_w - scaled.width) // 2, y))
            draw.text((x + 6, y + cell_h + 6), panel["label"] + (" (no warp)" if index == 0 else ""),
                      fill="#17324d", font=font)
        collage_path = gallery_dir / f"{mode}_grid.png"
        collage.save(collage_path)
        galleries[mode]["readme_grid"] = str(collage_path)

    # The mask demonstration uses a readable crop of the same input. Its own
    # full-range gradient spans that crop; the bundled example spans original.png.
    mask_source = generated / "mask_source.png"
    mask_path = generated / "mask_gradient.png"
    checkerboard_path = generated / "mask_checkerboard.png"
    unmasked_path = generated / "unmasked_sine_lens.png"
    masked_path = generated / "mask_sine_lens.png"
    checkerboard_warp_path = generated / "checkerboard_sine_lens.png"
    source_rgb.crop(crop_box).save(mask_source)
    gradient = PILImage.new("L", (crop_w, 1))
    gradient.putdata([round(255 * x / max(crop_w - 1, 1)) for x in range(crop_w)])
    gradient.resize((crop_w, crop_h), PILImage.Resampling.NEAREST).save(mask_path)
    yy, xx = np.indices((crop_h, crop_w))
    checkerboard = (((yy * 7 // crop_h + xx * 7 // crop_h) % 2) * 255).astype(np.uint8)
    PILImage.fromarray(checkerboard, mode="L").save(checkerboard_path)
    mask_intensity = 128
    sample_tensor = image_to_tensor(source_rgb.crop(crop_box))
    unmasked = WarpTransform(mode="sine_lens", intensity=mask_intensity)(sample_tensor)
    unmasked_pixels = unmasked.mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    PILImage.fromarray(unmasked_pixels).save(unmasked_path)
    masked = WarpTransform(mode="sine_lens", intensity=mask_intensity,
                           mask=mask_path)(sample_tensor)
    masked_pixels = masked.mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    PILImage.fromarray(masked_pixels).save(masked_path)
    checker_warp = WarpTransform(mode="sine_lens", intensity=mask_intensity,
                                 mask=checkerboard_path)(sample_tensor)
    checker_pixels = checker_warp.mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    PILImage.fromarray(checker_pixels).save(checkerboard_warp_path)
    mask_section = {"source_path": str(mask_source), "mask_path": str(mask_path),
                    "unmasked_path": str(unmasked_path), "masked_path": str(masked_path),
                    "checkerboard_path": str(checkerboard_path),
                    "checkerboard_warp_path": str(checkerboard_warp_path),
                    "checker_square_size_px": crop_h / 7, "intensity": mask_intensity,
                    "field": "sine_lens", "mask_values": [0, 255]}

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ManualTitle", parent=styles["Title"], fontSize=22,
                              leading=27, textColor=colors.HexColor("#17324d"), spaceAfter=14))
    styles.add(ParagraphStyle(name="ManualHeading", parent=styles["Heading1"], fontSize=14,
                              leading=18, textColor=colors.HexColor("#17324d"),
                              spaceBefore=10, spaceAfter=8, keepWithNext=True))
    styles.add(ParagraphStyle(name="ManualBody", parent=styles["BodyText"], fontSize=9.5,
                              leading=14, spaceAfter=7))
    styles.add(ParagraphStyle(name="ManualCode", fontName="Courier", fontSize=8, leading=11,
                              backColor=colors.HexColor("#edf2f5"), borderPadding=6, spaceAfter=7))
    styles.add(ParagraphStyle(name="ManualSubpoint", parent=styles["Heading2"],
                              fontSize=10, leading=13, spaceBefore=8,
                              spaceAfter=4, keepWithNext=True))
    styles.add(ParagraphStyle(name="ManualCaption", parent=styles["BodyText"], fontSize=9,
                              leading=11, alignment=TA_CENTER, textColor=colors.HexColor("#354859")))
    story = []

    def heading(text):
        story.append(Paragraph(text, styles["ManualHeading"]))

    def paragraph(text):
        story.append(Paragraph(text, styles["ManualBody"]))

    def code(text, subpoint):
        # Each example states the interface before its code block.
        story.append(Paragraph(subpoint, styles["ManualSubpoint"]))
        story.append(Paragraph(escape(text).replace(" ", "&nbsp;").replace("\n", "<br/>"),
                               styles["ManualCode"]))

    def cell(panel):
        with PILImage.open(panel["path"]) as crop:
            w, h = crop.size
        scale = min(73 * mm / w, 52 * mm / h)
        return [Image(panel["path"], width=w * scale, height=h * scale),
                Spacer(1, 2 * mm), Paragraph(panel["label"], styles["ManualCaption"])]

    def figure(path, max_w=73 * mm, max_h=52 * mm):
        with PILImage.open(path) as picture:
            w, h = picture.size
        scale = min(max_w / w, max_h / h)
        return Image(str(path), width=w * scale, height=h * scale)

    def framed_table(rows, widths, heights):
        table = Table(rows, colWidths=widths, rowHeights=heights)
        table.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), .5, colors.HexColor("#b6c7d3")),
            ("INNERGRID", (0, 0), (-1, -1), .5, colors.HexColor("#b6c7d3")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        return table

    story.append(Paragraph("Image transformations with PyTorch", styles["ManualTitle"]))
    paragraph("Operator manual | lorem-warped | Version 1.4")
    paragraph("This manual uses selected writing principles from ASD-STE100 Issue 9. It is not certified ASD-STE100 documentation. Formal conformity needs a controlled-dictionary and rule review.")
    heading("1. Package introduction")
    paragraph("lorem-warped applies independent PyTorch image transformations to a single RGB image. It includes droplet layouts, page waves, a six-control sine lens, a gentle pinch, a refractive-index slab, and a Python-only custom callback. Input and output are CHW tensors; the CLI writes PNG files.")
    paragraph("This document includes the input-generation recipe, controls, code, and images for each transform. The shown input is a 1200-by-1600 RGB page with a 28 px body font; the following setup recreates it exactly. All study settings are printed beside their images.")
    heading("1.1 Distortion overview")
    paragraph("A distortion moves source pixels to new positions. The left image is the unwarped original with its actual droplet regions shaded gray at 30% opacity. The right image uses a dense square-grid droplet warp. The guide is display-only: it is not drawn on the transform input or output.")
    overview_table = Table([[cell(panel) for panel in overview["panels"]]],
                           colWidths=[84 * mm, 84 * mm], rowHeights=[70 * mm])
    overview_table.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), .5, colors.HexColor("#b6c7d3")),
                                        ("INNERGRID", (0, 0), (-1, -1), .5, colors.HexColor("#b6c7d3")),
                                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                        ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story.append(overview_table)
    paragraph(f"For this input, request {dense_count} square-grid centres, a radius of {radius} pixels, and an intensity of 0.5 pixels. This low intensity keeps the dense example legible. More centres make the grid denser. Radius sets each droplet's size. No droplet example in this manual exceeds 5 pixels of intensity.")
    paragraph("Choose a transform: droplets give local circles; a sine lens and other page warps move larger page regions; refraction makes two translated views. Choose random, square, or triangle (hexagonal) centres for droplets. Set centre count, radius, intensity, seed, and profile. The other modes do not use centre distribution.")
    story.append(PageBreak())
    heading("2. Distortion types")
    paragraph("Read this tree from left to right. The middle column applies each actual transform to the same dense, text-free square grid. The right column applies it to the same short paragraph. Grid lines move with the transform; they are not droplet centres. Droplet layouts use 3 px intensity, and a uniform slab shifts lines without bending them.")
    tree_rows: list[list] = [[Paragraph("Type", styles["ManualCaption"]),
                  Paragraph("Distorted square grid", styles["ManualCaption"]),
                  Paragraph("Distorted text", styles["ManualCaption"])]]
    for index, preview in enumerate(atlas["tree"]["previews"]):
        tree_name = ("|-- droplets<br/>&nbsp;&nbsp;&nbsp;&nbsp;|-- random" if index == 0 else
                     "|&nbsp;&nbsp;&nbsp;|-- square grid" if index == 1 else
                     "|&nbsp;&nbsp;&nbsp;`-- triangular grid" if index == 2 else
                     "|-- " + escape(preview["label"]))
        label = Paragraph("<font face='Courier'>" + tree_name + "</font>",
                          styles["ManualBody"])
        tree_rows.append([label, figure(preview["grid_path"], 42 * mm, 20 * mm),
                          figure(preview["path"], 42 * mm, 20 * mm)])
    tree_table = framed_table(tree_rows, [70 * mm, 49 * mm, 49 * mm],
                              [9 * mm] + [23 * mm] * 9)
    tree_table.setStyle(TableStyle([("TOPPADDING", (0, 0), (-1, -1), 2),
                                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    story.append(tree_table)
    paragraph("The slab view is a uniform translation of two polarization modes. Its preview does not imply a spatial lens. A custom callback has no CLI command.")
    story.append(PageBreak())
    heading("3. Install and prepare")
    paragraph("Install Python 3.10 or later. Open a shell in the lorem-warped source checkout root, create an environment, and install the project dependencies. Keep this directory as the working directory: the recipes import the src package from the checkout. No separate sample image or study file is required.")
    code("python -m venv .venv\n. .venv/bin/activate\npython -m pip install -e '.[dev]'\npython -m src --help", "a. CLI")
    paragraph("On Windows PowerShell, activate .venv\\Scripts\\Activate.ps1. Use the PyTorch installation selector if you need a specific GPU build.")
    paragraph("Create the image shown in the full-page studies with the package's built-in page renderer. Its 28 px body text is the basis of the 14, 28, 42, 56, and 70 px periods. Create it before either the CLI or script recipes. You may substitute your own RGB image; keep masks the same size as that image.")
    code("python -c \"from src.demo import render_lorem_page; render_lorem_page().save('original.png')\"", "b. CLI: create the input")
    heading("4. Script setup")
    paragraph("In script examples, read the image once. The following setup also creates the input if absent. Each transform starts from source, not from a previous output. PIL images become float CHW tensors from 0 to 1.")
    code("from PIL import Image\nfrom src.demo import render_lorem_page\nfrom src import (CentreDistribution, DistortionConfig,\n                 WarpTransform, image_to_tensor)\nrender_lorem_page().save('original.png')\nsource = image_to_tensor(Image.open('original.png'))\ndef save(path, result):\n    pixels = result.clamp(0, 1).mul(255).round().byte()\n    pixels = pixels.permute(1, 2, 0).cpu().numpy()\n    Image.fromarray(pixels).save(path)", "a. in script")
    paragraph("The custom-wave example imports partial and the supplied custom_wave callback. The CLI runs on CPU. Move the tensor to a supported GPU in your script if necessary.")
    paragraph("Geometric fields use pixel displacement. The sampler reads source pixels at output position minus intensity times field. It uses bilinear sampling, border padding, and aligned corners. Strong fields can fold or repeat text. If the package cannot be imported, activate the environment and install it from the project root. For a different physical page size, change both page dimensions in the refractive recipe.")
    story.append(PageBreak())
    heading("4.1 Create the four-line comparison source")
    paragraph("The compact 4-by-4 studies use this separate 300-by-184 source. Create it once. The straight reference grid is part of the source tensor, so text and lines are warped together. The full-page five-by-five studies instead use original.png.")
    code("from PIL import ImageDraw, ImageFont\nfrom src import overlay_warped_grid\nmini = Image.new('RGB', (300, 184), 'white')\npen = ImageDraw.Draw(mini)\nfont = ImageFont.truetype('DejaVuSans.ttf', 28)\nfor row, line in enumerate(('Lorem ipsum', 'dolor sit amet',\n                            'consectetur', 'adipiscing elit')):\n    pen.text((10, 10 + row*34), line, font=font,\n             fill='#17252e')\nmini.save('mini-source.png')\nmini_grid = overlay_warped_grid(image_to_tensor(mini),\n                                spacing=28, opacity=.24)\nsave('mini-grid.png', mini_grid)", "a. in script")


    # These two-parameter grids use the documented 28 px body font of original.png.
    # The generators render each cell with the real pixel-wavelength field.
    wave_root = Path("examples/wave-wavelength-grid")
    custom_root = Path("examples/custom-wave-wavelength-grid")
    generate_wave_grids(input_path, wave_root, font_size_px=28)
    generate_custom_wave_grids(input_path, custom_root, font_size_px=28)
    def pdf_wave_sheet(root: Path, mode: str) -> Path:
        manifest = json.loads((root / "manifest.json").read_text())
        crop = (80, 120, 320, 280)
        if width < crop[2] or height < crop[3]:
            raise ValueError("wavelength study needs an image at least 320 by 280 pixels")
        sheet = PILImage.new("RGB", (1272, 1004), "white")
        pen = ImageDraw.Draw(sheet)
        label_font = ImageFont.truetype("DejaVuSans.ttf", 17)
        for column, percent in enumerate(PERCENTAGES):
            pen.text((12 + column * 252, 4), f"{percent}% ({28 * percent / 100:g} px)",
                     fill="black", font=label_font)
            for row, intensity in enumerate(INTENSITIES):
                entry = next(e for e in manifest["samples"] if e["mode"] == mode
                             and e["wavelength_percent"] == percent and e["intensity_px"] == intensity)
                with PILImage.open(root / entry["file"]) as full:
                    sample = full.crop(crop)
                x, y = 12 + column * 252, 24 + row * 196
                pen.text((x, y), f"{intensity} px", fill="black", font=label_font)
                sheet.paste(sample, (x, y + 24))
        dest = generated / f"pdf_study_{mode}.png"
        sheet.save(dest)
        return dest

    wave_studies = [
        ("Horizontal wave", wave_root, "horizontal_wave"),
        ("Vertical wave", wave_root, "vertical_wave"),
        ("Fixed vertical custom wave", custom_root, "custom_wave_fixed_vertical"),
        ("Fixed horizontal custom wave", custom_root, "custom_wave_fixed_horizontal"),
        ("Symmetric custom wave", custom_root, "custom_wave_symmetric"),
    ]
    wave_studies = [(name, pdf_wave_sheet(root, mode)) for name, root, mode in wave_studies]
    wavelength_values = ", ".join(f"{28 * p / 100:g}" for p in PERCENTAGES)
    intensity_values = ", ".join(map(str, INTENSITIES))
    def append_wave_study(number, index, name, grid_path):
        story.append(PageBreak())
        heading(f"{number}.{index} Two-parameter wavelength study: {name}")
        paragraph(f"The unchanged input has a 28 px body font. Its 50%, 100%, 150%, 200%, and 250% periods are {wavelength_values} px. Columns vary wavelength; rows vary intensity: {intensity_values} px. Each panel crops the same region of a full-page warp.")
        variant = ("vertical" if name == "Fixed vertical custom wave" else
                   "horizontal" if name == "Fixed horizontal custom wave" else
                   "symmetric")
        if name == "Fixed vertical custom wave":
            paragraph("The vertical component has a fixed period of 25% of image width (300 px). The horizontal component uses the column wavelength along Y.")
        elif name == "Fixed horizontal custom wave":
            paragraph("The horizontal component has a fixed period of 25% of image height (400 px). The vertical component uses the column wavelength along X.")
        elif name == "Symmetric custom wave":
            paragraph("X and Y have the same pixel wavelength in every column.")
        else:
            paragraph("The built-in wave uses the stated pixel wavelength and keeps the page-edge envelope fixed.")
        if "custom" in name.lower():
            paragraph("Custom callbacks have no CLI mode. Run this Python-only script after the setup in section 4.")
            code(f"from functools import partial\nfrom src import custom_wave\nfor w in (14, 28, 42, 56, 70):\n    for i in (10, 20, 30, 40, 50):\n        field = partial(custom_wave, wavelength=w,\n                        fixed_axis='{variant}')\n        save(f'custom_{variant}_{{w}}_{{i}}.png',\n             WarpTransform(mode='custom', intensity=i,\n                           warp_function=field)(source))", "a. in script")
        else:
            mode_name = "horizontal_wave" if name == "Horizontal wave" else "vertical_wave"
            code(f"for w in 14 28 42 56 70; do\n  for i in 10 20 30 40 50; do\n    python -m src --input original.png \\\n      --output {mode_name}_${{w}}_${{i}}.png \\\n      --page-warp {mode_name} --wavelength $w --intensity $i\n  done\ndone", "a. CLI (POSIX shell)")
            code(f"for w in (14, 28, 42, 56, 70):\n    for i in (10, 20, 30, 40, 50):\n        save(f'{mode_name}_{{w}}_{{i}}.png',\n             WarpTransform(mode='{mode_name}',\n                 wavelength=w, intensity=i)(source))", "b. in script")
        story.append(figure(grid_path, 168 * mm, 160 * mm))
        paragraph("All row and column values appear on this page. Strong settings can fold or repeat letters.")

    descriptions = {
        "droplet": "A droplet warp moves pixels around circular centres. Choose random, square, or triangle (hexagonal) centres. Radius sets each circle size. Intensity sets the movement in pixels. The seed fixes random centres. The cosine profile permits overlaps.",
        "sine_lens": "A sine lens moves the page in two axes. X and Y frequencies, phases, and gains give six shape parameters. The intensity multiplies the displacement in pixels. Keep the six parameters fixed in this comparison.",
        "refractive_index": "A uniform anisotropic slab makes two translated polarization views. Thickness sets the translation in millimetres. The refractive indices and physical page size stay fixed. Do not use pixel intensity for this physical model.",
        "horizontal_wave": "A horizontal wave moves text left and right. Wavelength is the period along Y in pixels. These examples use 28 px. The page edges stay in place. Intensity sets movement in pixels.",
        "vertical_wave": "A vertical wave moves baselines up and down. Wavelength is the period along X in pixels. These examples use 28 px. The page edges stay in place. Intensity sets movement in pixels.",
        "gentle_pinch": "A gentle pinch moves text toward the page centre. Its field is small. Use a larger pixel intensity to see the effect.",
        "custom": "A custom wave returns Y and X displacement tensors. The gallery holds a symmetric 28 px wavelength in both axes and varies only intensity. Separate studies compare fixed vertical, fixed horizontal, and symmetric wavelengths. Custom callbacks are Python-only.",
    }
    titles = {
        "droplet": "Droplet transformation", "sine_lens": "Six-parameter sine lens",
        "refractive_index": "Refractive-index transformation",
        "horizontal_wave": "Horizontal wave", "vertical_wave": "Vertical wave",
        "gentle_pinch": "Gentle pinch", "custom": "Custom wave (Python only)",
    }
    for number, mode in enumerate(STRENGTHS, start=5):
        gallery = galleries[mode]
        chapter = atlas["modes"][mode]
        story.append(PageBreak())
        heading(f"{number}. How {titles[mode].lower()} works")
        paragraph({
            "droplet": "Each centre has a radius. The cosine profile gives a radial displacement that fades to zero at the circle edge. Overlapping fields add before the image is sampled once. Pixel intensity multiplies the field. The right-hand plot shows the measured displacement vectors and shaded influence on a square reference grid.",
            "sine_lens": "The sine lens combines a two-axis sinusoidal envelope with local cell-centre directions. X and Y frequencies set the cell counts. Phase moves the lobes; axis gains set the X and Y movement. Pixel intensity multiplies the field.",
            "refractive_index": "The uniform slab yields two distinct constant translations: one for each polarization mode. Thickness changes their offsets. Polarization changes their weights. The right-hand plot shows both vectors, not a varying lens field. Displacement is converted from millimetres with the page dimensions.",
            "horizontal_wave": "The horizontal wave repeats along Y with a wavelength measured in source-image pixels. This gallery uses 28 px. The smooth edge envelope pins page edges. Pixel intensity multiplies the horizontal displacement.",
            "vertical_wave": "The vertical wave repeats along X with a wavelength measured in source-image pixels. This gallery uses 28 px. The smooth edge envelope pins page edges. Pixel intensity multiplies the vertical displacement.",
            "gentle_pinch": "The field points toward the page centre. A smooth envelope holds page edges fixed. The built-in transform exposes pixel intensity, but not a separate width or frequency control.",
            "custom": "The callback combines vertical (X-periodic) and horizontal (Y-periodic) waves into a (dy, dx) field. Choose a wavelength in pixels and fix one wave at 25% of its page dimension, or use the same wavelength on X and Y. Pixel intensity multiplies both components. Only the Python API accepts the callback.",
        }[mode])
        story.append(framed_table([[
            [figure(chapter["mini"]), Paragraph("Distorted paragraph", styles["ManualCaption"])],
            [figure(chapter["field"]), Paragraph("Measured displacement function", styles["ManualCaption"])]]],
            [84 * mm, 84 * mm], [72 * mm]))
        paragraph(f"The left panel uses {chapter['preview_settings']}. The plot shows its actual pixel displacement on a square sampling grid. Arrow lengths are amplified for display. The plot is not an extra transformation.")
        if mode != "droplet":
            story.append(figure(chapter["reference_grid"], 84 * mm, 52 * mm))
            paragraph("The small overlay above uses a dense square reference grid to show how this effect samples the image. It is not a square array of droplet centres. For the slab, the two mode translations remain constant across the page.")
        if mode == "refractive_index":
            story.append(PageBreak())
            heading(f"{number}.0 Ray geometry and optical indicatrix")
            paragraph("X and Y run along the slab's two flat edges; Z points through its thickness. The incoming beam tilts 45 degrees away from straight-through Z toward negative X, with no travel along Y: halfway between straight through and sideways. In this uniaxial example, the blue Ordinary beam and orange Extraordinary beam follow different paths, then leave parallel. Their paths come from the slab solver; the picture is not to scale.")
            story.append(figure(chapter["schematic"]["path"], 168 * mm, 84 * mm))
            paragraph("At right, imagine slicing the optical indicatrix (an egg-shaped index map) across the direction the light travels. This cyan cross-section is at 90 degrees to the transmitted wave normal; its colored arrows mark two ways the light can wiggle (polarization). Here n_x = n_y = 1.5, n_z = 2.0, thickness = 5 mm, and incident direction = (-1, 0, 1). The separate n_x-by-n_y studies can be biaxial, so they use X/Y eigenmode labels instead of ordinary/extraordinary.")
            paragraph("The diagram shows the principal axes and the two polarization directions. The ray positions come from the slab solver; polarization weights at oblique incidence are approximate. The schematic and its parameters are completely described here.")
        story.append(PageBreak())
        heading(f"{number}.1 {titles[mode]}: strength examples")
        paragraph(descriptions[mode])
        gallery["explanation"] = descriptions[mode]
        if mode == "droplet":
            paragraph("The first panel is the unwarped original with display-only gray droplet guides at 30% opacity. The other panels warp the unmarked source. Only pixel intensity changes; no example exceeds 5 pixels.")
        else:
            paragraph(f"The first panel shows a straight, dense square grid over the original. Each output uses one {mode} transform on both the text and this grid; the lines bend or translate with the text. The plain, ungridded output remains available to CLI and script users. Within these panels only {gallery['unit']} changes. Strong settings may reduce legibility.")
        table = Table([[cell(gallery["panels"][row * 2 + col]) for col in range(2)]
                       for row in range(3)], colWidths=[84 * mm, 84 * mm], rowHeights=[68 * mm] * 3)
        table.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), .5, colors.HexColor("#b6c7d3")),
                                   ("INNERGRID", (0, 0), (-1, -1), .5, colors.HexColor("#b6c7d3")),
                                   ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                   ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        story.append(table)
        story.append(PageBreak())
        heading(f"{number}.2 Run {titles[mode].lower()}")
        if mode == "droplet":
            code("python -m src --input original.png \\\n  --output droplet.png --centres 3 --distribution random \\\n  --radius 80 --intensity 5 --seed 7 --profile cosine", "a. CLI")
            code("transform = WarpTransform(mode='droplet',\n    config=DistortionConfig(centre_count=3, radius=80,\n                            intensity=5, seed=7))\nsave('droplet.png', transform(source))", "b. in script")
            paragraph("The opening overview uses a dense square arrangement at 0.5 px intensity. Its gray guide is display-only; the distorted image is generated by these settings.")
            code(f"python -m src --input original.png \\\n  --centres {dense_count} --distribution square \\\n  --radius {radius} --intensity 0.5 --seed 7 \\\n  --profile cosine --output dense-overview.png", "c. CLI: overview")
            code(f"config = DistortionConfig(centre_count={dense_count},\n    distribution=CentreDistribution.SQUARE,\n    radius={radius}, intensity=0.5, seed=7,\n    profile='cosine')\nsave('dense-overview.png',\n     WarpTransform(mode='droplet', config=config)(source))", "d. in script: overview")
            paragraph("The compact study uses the mini-source.png text from section 4.1, eight square-distributed centres, and the listed radius/intensity pairs. Unlike page-warps, droplet samples do not contain the straight reference grid. Gray circles on zero-warp controls are display-only annotations.")
            code("for radius in 30 45 60 75; do\n  for i in 0 1 3 5; do\n    python -m src --input mini-source.png \\\n      --centres 8 --distribution square --seed 7 \\\n      --profile cosine --radius $radius --intensity $i \\\n      --output droplet_${radius}_${i}.png\n  done\ndone", "e. CLI: four-by-four study")
            code("small = image_to_tensor(Image.open('mini-source.png'))\nfor radius in (30, 45, 60, 75):\n    for i in (0, 1, 3, 5):\n        config = DistortionConfig(centre_count=8,\n            distribution=CentreDistribution.SQUARE,\n            radius=radius, intensity=i, seed=7,\n            profile='cosine')\n        save(f'droplet_{radius}_{i}.png',\n             WarpTransform(mode='droplet',\n                           config=config)(small))", "f. in script: four-by-four study")
        elif mode == "sine_lens":
            paragraph("Set the six lens parameters to X/Y frequencies 2.5/1.5, phases 0.1/0.2 cycles, and gains 1.2/0.8. Use 32 pixels for this example. A frequency must be positive.")
            code("python -m src --input original.png \\\n  --output sine_lens.png --page-warp sine_lens \\\n  --intensity 32 --lens-frequency-x 2.5 \\\n  --lens-frequency-y 1.5 --lens-phase-x 0.1 \\\n  --lens-phase-y 0.2 --lens-amplitude-x 1.2 \\\n  --lens-amplitude-y 0.8", "a. CLI")
            code("transform = WarpTransform(mode='sine_lens', intensity=32,\n    lens_options=dict(frequency_x=2.5, frequency_y=1.5,\n        phase_x=0.1, phase_y=0.2,\n        amplitude_x=1.2, amplitude_y=0.8))\nsave('sine_lens.png', transform(source))", "b. in script")
            paragraph("To reproduce the four-by-four frequency/intensity study, start from mini-grid.png. Hold Y frequency 1.5, X/Y phases 0.1/0.2 and gains 1.2/0.8 fixed.")
            code("for f in 1 1.5 2 2.5; do\n  for i in 0 15 40 80; do\n    python -m src --input mini-grid.png \\\n      --page-warp sine_lens --lens-frequency-x $f \\\n      --lens-frequency-y 1.5 --lens-phase-x .1 \\\n      --lens-phase-y .2 --lens-amplitude-x 1.2 \\\n      --lens-amplitude-y .8 --intensity $i \\\n      --output lens_${f}_${i}.png\n  done\ndone", "c. CLI: four-by-four study")
            code("small = image_to_tensor(Image.open('mini-grid.png'))\nfor f in (1, 1.5, 2, 2.5):\n    for i in (0, 15, 40, 80):\n        opts = dict(frequency_x=f, frequency_y=1.5,\n            phase_x=.1, phase_y=.2,\n            amplitude_x=1.2, amplitude_y=.8)\n        save(f'lens_{f}_{i}.png',\n             WarpTransform(mode='sine_lens',\n                 intensity=i, lens_options=opts)(small))", "d. in script: four-by-four study")
        elif mode == "refractive_index":
            paragraph("Thickness and page dimensions use millimetres. The angle uses degrees. The slab gives constant translations, not a spatially varying lens. The polarization weights are approximate at oblique incidence.")
            code("python -m src --input original.png \\\n  --output refraction.png --page-warp refractive_index \\\n  --refraction-thickness-mm 5 --refraction-n-x 1.3 \\\n  --refraction-n-y 2.5 --refraction-n-z 1.3 \\\n  --refraction-polarization-deg 45 \\\n  --refraction-direction -1 1 1 \\\n  --page-width-mm 210 --page-height-mm 297", "a. CLI")
            code("transform = WarpTransform(mode='refractive_index',\n    refraction_options=dict(thickness_mm=5, n_x=1.3,\n        n_y=2.5, n_z=1.3, polarization_deg=45,\n        direction_xyz=(-1, 1, 1),\n        page_width_mm=210, page_height_mm=297))\nsave('refraction.png', transform(source))", "b. in script")
            paragraph("The three compact studies use thickness 2, 5 or 10 mm. In each one, rows and columns independently set n_x and n_y; n_z remains 1.3. No pixel intensity is used.")
            code("for thickness in 2 5 10; do\n  for nx in 1 1.2 1.5 2; do\n    for ny in 1 1.2 1.5 2; do\n      python -m src --input mini-grid.png \\\n        --page-warp refractive_index \\\n        --refraction-thickness-mm $thickness \\\n        --refraction-n-x $nx --refraction-n-y $ny \\\n        --refraction-n-z 1.3 \\\n        --refraction-polarization-deg 45 \\\n        --refraction-direction -1 1 1 \\\n        --page-width-mm 210 --page-height-mm 297 \\\n        --output slab_${thickness}_${nx}_${ny}.png\n    done\n  done\ndone", "c. CLI: three four-by-four studies")
            code("small = image_to_tensor(Image.open('mini-grid.png'))\nfor thickness in (2, 5, 10):\n    for nx in (1, 1.2, 1.5, 2):\n        for ny in (1, 1.2, 1.5, 2):\n            opts = dict(thickness_mm=thickness,\n                n_x=nx, n_y=ny, n_z=1.3,\n                polarization_deg=45,\n                direction_xyz=(-1, 1, 1),\n                page_width_mm=210, page_height_mm=297)\n            save(f'slab_{thickness}_{nx}_{ny}.png',\n                WarpTransform(mode='refractive_index',\n                    refraction_options=opts)(small))", "d. in script: three studies")
        elif mode == "custom":
            paragraph("The custom mode has no CLI option. Set wavelength=28 pixels and intensity=20 pixels for each of the three variants. A fixed vertical wave has a 25% of image width period; a fixed horizontal wave has a 25% of image height period. Symmetric uses 28 px on both axes.")
            code("from functools import partial\nfrom src import custom_wave\nfor variant in ('vertical', 'horizontal', 'symmetric'):\n    field = partial(custom_wave, wavelength=28,\n                    fixed_axis=variant)\n    transform = WarpTransform(mode='custom', intensity=20,\n                              warp_function=field)\n    save('custom_' + variant + '.png', transform(source))", "a. in script")
            paragraph("The compact four-by-four studies below use an independent X/Y-period callback. Recreate its fixed four-line input with section 4.1. The callback receives normalized (y, x) coordinates and returns unitless (dy, dx) components; WarpTransform applies pixel intensity. No CLI equivalent exists for this two-axis callback.")
            code("import torch\nfrom functools import partial\ndef xy_wave(y, x, wx, wy):\n    edge = torch.sin(torch.pi*y) * torch.sin(torch.pi*x)\n    dy = edge * torch.sin(2*torch.pi*x*299/wx)\n    dx = edge * torch.sin(2*torch.pi*y*183/wy)\n    return dy, dx\nsmall = image_to_tensor(Image.open('mini-grid.png'))\nfor strength in (5, 10, 20):\n    for wx in (14, 28, 42, 56):\n        for wy in (14, 28, 42, 56):\n            field = partial(xy_wave, wx=wx, wy=wy)\n            transform = WarpTransform(mode='custom',\n                intensity=strength, warp_function=field)\n            save(f'xy_{strength}_{wx}_{wy}.png',\n                 transform(small))", "b. in script: independent X/Y study")
            paragraph("The five-by-five wavelength-by-intensity studies later in this chapter use the supplied custom_wave callback.")
        else:
            strength = {"horizontal_wave": 20, "vertical_wave": 20,
                        "gentle_pinch": 160}[mode]
            wave_cli = " --wavelength 28" if mode in ("horizontal_wave", "vertical_wave") else ""
            wave_script = "wavelength=28, " if wave_cli else ""
            code(f"python -m src --input original.png \\\n  --output {mode}.png --page-warp {mode} --wavelength 28 \\\n  --intensity {strength}" if wave_cli else
                 f"python -m src --input original.png \\\n  --output {mode}.png --page-warp {mode} \\\n  --intensity {strength}", "a. CLI")
            code(f"transform = WarpTransform(mode='{mode}', {wave_script}intensity={strength})\nsave('{mode}.png', transform(source))", "b. in script")
            if mode in ("horizontal_wave", "vertical_wave"):
                paragraph("To reproduce the compact 4-by-4 study, use the mini-grid.png source from section 4.1. Rows use 14, 28, 42, 56 px wavelengths; columns use 0, 12, 28, 50 px intensity. These commands apply the built-in transform to the text and straight grid together.")
                code(f"for w in 14 28 42 56; do\n  for i in 0 12 28 50; do\n    python -m src --input mini-grid.png \\\n      --page-warp {mode} --wavelength $w \\\n      --intensity $i --output study_{mode}_${{w}}_${{i}}.png\n  done\ndone", "c. CLI: four-by-four study")
                code(f"small = image_to_tensor(Image.open('mini-grid.png'))\nfor w in (14, 28, 42, 56):\n    for i in (0, 12, 28, 50):\n        transform = WarpTransform(mode='{mode}',\n            wavelength=w, intensity=i)\n        save(f'study_{mode}_{{w}}_{{i}}.png',\n             transform(small))", "d. in script: four-by-four study")
            elif mode == "gentle_pinch":
                paragraph("The compact 4-by-4 pinch study varies a Python-only envelope power; the built-in CLI has intensity but no envelope-power option. Recreate mini-grid.png with section 4.1 and use this custom field for that study.")
                code("import torch\nfrom functools import partial\ndef pinch_power_field(y, x, power):\n    edge = (torch.sin(torch.pi*y) *\n            torch.sin(torch.pi*x)).clamp_min(0)\n    edge = edge.pow(power)\n    return edge*(.5-y), edge*(.5-x)\nsmall = image_to_tensor(Image.open('mini-grid.png'))\nfor power in (.6, 1, 1.5, 2):\n    for i in (0, 90, 180, 300):\n        field = partial(pinch_power_field, power=power)\n        save(f'pinch_{power}_{i}.png',\n             WarpTransform(mode='custom', intensity=i,\n                           warp_function=field)(small))", "c. in script: envelope-power study")
        if mode == "custom":
            paragraph("The script writes a PNG. Its callback is Python-only; no equivalent CLI option exists.")
        else:
            paragraph("The command writes a PNG. On PowerShell, enter the CLI command on one line. Do not use Unix backslash line continuation there.")
        for study in (chapter["thickness_studies"] if mode == "refractive_index" else
                      chapter["intensity_studies"] if mode == "custom" else
                      [chapter["sweep"]]):
            story.append(PageBreak())
            heading(f"{number}.3 Two-variable legibility study" +
                    (f": {study['thickness_mm']} mm" if mode == "refractive_index" else
                     f": {study['intensity_px']} px" if mode == "custom" else ""))
            first_axis, second_axis = study["axes"]
            paragraph(f"Rows change {first_axis}; columns change {second_axis}. The 4-by-4 table uses the same four-line source paragraph, unchanged in font, size, and style. Each label lists the row value, then the column value. Compare the letters, spaces, and repeated strokes.")
            if mode == "droplet":
                paragraph("Zero-warp controls show gray droplet regions at 30% opacity. Other cells use actual droplet warps at no more than 5 pixels of intensity.")
            elif mode == "refractive_index":
                paragraph("Every cell starts with the same dense square reference grid. The uniform slab translates the grid with the text; grid lines stay straight, though their contrast can change. These lines do not represent droplet centres.")
            elif mode == "custom":
                paragraph("The text and straight source grid stay fixed. Each output uses the same Python-only callback formula on both text and grid; only its X and Y pixel wavelengths change. Every cell is warped; no zero-intensity control appears on this page.")
            else:
                paragraph("Every cell starts with the same dense square reference grid. The transform warps the grid with the text; zero-intensity cells keep straight lines. These lines do not represent droplet centres.")
            if mode == "refractive_index":
                paragraph(f"Keep slab thickness at {study['thickness_mm']} mm and polarization at 45 degrees. Change n_x down the rows and n_y across the columns: 1, 1.2, 1.5, 2.0. The indices are dimensionless; n_z remains 1.3. A uniform slab produces translated views, not a curved lens. No artistic pixel intensity is used.")
            elif mode in ("horizontal_wave", "vertical_wave"):
                paragraph("Each row uses a pixel wavelength calculated from the 28 px source font: 50%, 100%, 150%, and 200% give 14, 28, 42, and 56 px. The built-in wave renders both text and grid with that wavelength; columns vary pixel intensity. The following five-by-five study extends the sweep to 250% and 50 px intensity.")
            elif mode == "gentle_pinch":
                paragraph("This study varies the Python-only pinch-envelope power and pixel intensity. The text itself never changes.")
            elif mode == "custom":
                paragraph(f"Keep pixel intensity at {study['intensity_px']} throughout this page. X wavelength runs 14, 28, 42, and 56 px down the rows; Y wavelength runs 14, 28, 42, and 56 px across the columns. These values are 50%, 100%, 150%, and 200% of the 28 px source font. This independently parameterized callback is Python-only.")
            else:
                paragraph("Keep all other transform settings fixed. The first column includes a zero-intensity control where applicable.")
            study_cells = []
            for row in range(4):
                study_cells.append([])
                for column in range(4):
                    panel = study["panels"][row * 4 + column]
                    study_cells[-1].append([
                        figure(panel["path"], 38 * mm, 26 * mm),
                        Paragraph(panel["label"], styles["ManualCaption"])])
            story.append(framed_table(study_cells, [42 * mm] * 4, [48 * mm] * 4))
        if mode == "droplet":
            story.append(PageBreak())
            heading(f"{number}.4 Shaded droplet regions")
            paragraph("Each gray circle has 30% opacity and shows one droplet's radius. Each gray dot marks a centre. The same original text lies under every shaded map. These are centre-placement guides, not warped output images.")
            maps = atlas["distributions"]["maps"]
            story.append(framed_table([[
                [figure(item["path"], 53 * mm, 35 * mm),
                 Paragraph(item["distribution"].title(), styles["ManualCaption"])]
                for item in maps]], [56 * mm] * 3, [51 * mm]))
            paragraph("Random uses the seed to place centres. Square uses aligned rows and columns. Triangle offsets alternate rows to make a hexagonal neighbour pattern. The shaded regions can overlap with a cosine profile; parabolic profiles need nonoverlapping centres.")
            story.append(PageBreak())
            heading(f"{number}.5 Droplet arrangements")
            paragraph("The distribution controls centre positions. Random selects seeded positions. Square places centres in aligned rows and columns. Triangle offsets alternate rows and makes a hexagonal neighbour pattern. In the CLI, select the hexagonal pattern with --distribution triangle, not hex.")
            paragraph(f"Compare the same text fragment. Each row pairs an unwarped original with its actual droplet regions shaded gray at 30% opacity (left) and the warped output (right). Request {distribution_count} centres for each layout. Keep radius {distribution_radius} pixels, intensity 3 pixels, seed 7, and the cosine profile fixed. Only the distribution changes. The overlays are display-only. A shifted triangle row can have fewer centres near an edge.")
            arrangement_table = Table(
                [[cell(distributions["panels"][row * 2 + col]) for col in range(2)]
                 for row in range(3)], colWidths=[84 * mm, 84 * mm], rowHeights=[65 * mm] * 3)
            arrangement_table.setStyle(TableStyle([
                ("BOX", (0, 0), (-1, -1), .5, colors.HexColor("#b6c7d3")),
                ("INNERGRID", (0, 0), (-1, -1), .5, colors.HexColor("#b6c7d3")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
            story.append(arrangement_table)
            story.append(PageBreak())
            heading(f"{number}.6 Select a droplet distribution")
            paragraph("Use one of these three names. Keep all other controls fixed to compare the centre layouts. The CLI commands below use the count and radius shown on the preceding page.")
            for arrangement in CentreDistribution:
                command = (f"python -m src --input original.png " + "\\\n"
                           + f"  --output droplets_{arrangement.value}.png --centres {distribution_count} " + "\\\n"
                           + f"  --distribution {arrangement.value} --radius {distribution_radius} " + "\\\n"
                           + "  --intensity 3 --seed 7 --profile cosine")
                code(command, f"{arrangement.value}. CLI")
            code(f"for name in ('random', 'square', 'triangle'):\n"
                 f"    config = DistortionConfig(centre_count={distribution_count},\n"
                 f"        distribution=CentreDistribution(name), radius={distribution_radius},\n"
                 f"        intensity=3, seed=7, profile='cosine')\n"
                 f"    save('droplets_' + name + '.png',\n"
                 f"         WarpTransform(mode='droplet', config=config)(source))",
                 "d. in script")
            paragraph("The script uses the imports and save function from section 4. A fixed centre count controls square and triangle density. A fixed seed repeats the random pattern. The cosine profile permits overlaps. A parabolic profile requires disjoint circles; --nonoverlapping-page makes disjoint page-wide layouts and overrides --centres.")

        if mode == "sine_lens":
            story.append(PageBreak())
            heading("6.4 Mask the displacement")
            paragraph("Use a grayscale mask with exactly the same width and height as the image. At each output pixel, black (0) cancels both displacement components; white (255) keeps the full field. Gray scales both components continuously: 128 gives about half strength. The sampler reads the original at the resulting coordinates. It does not blend an unwarped and a warped image, and it does not change image alpha.")
            paragraph("The three results use the same text crop and sine-lens field at 128 px intensity. Row 1 compares the original with an unmasked warp. Row 2 shows a left-to-right gradient mask and its result. Row 3 shows a black-and-white checkerboard mask and its result; each square has side length one-seventh of the crop height in both axes. Only the mask changes. The following recipes create both masks from the input image; no separate mask file is required.")
            def mask_cell(path, caption):
                return [figure(path, 73 * mm, 34 * mm), Paragraph(caption, styles["ManualCaption"])]
            mask_panels = [
                [mask_cell(mask_source, "Original text"), mask_cell(unmasked_path, "Without mask")],
                [mask_cell(mask_path, "Gradient mask"), mask_cell(masked_path, "With gradient mask")],
                [mask_cell(checkerboard_path, "Checkerboard mask"),
                 mask_cell(checkerboard_warp_path, "With checkerboard mask")],
            ]
            story.append(framed_table(mask_panels, [84 * mm] * 2, [44 * mm] * 3))
            paragraph("Use the recipes in 6.5 to render this exact crop, both masks, and their warped results. A mask must match its input; it is never resized. The mask also works with droplets, page/custom warps, and refraction. Refraction retains optical transmission.")
            story.append(PageBreak())
            heading("6.5 Reproduce the mask comparison")
            paragraph("This script extracts the same 520-by-460 crop from the generated original.png. It creates both grayscale masks and produces all three transformed views shown in 6.4. Run it after the imports and save helper in section 4.")
            code("from pathlib import Path\nimport numpy as np\ncrop = Image.open('original.png').convert('RGB')\ncrop = crop.crop((70, 104, 590, 564))\ncrop.save('mask-source.png')\nw, h = crop.size\ngradient = Image.new('L', (w, 1))\ngradient.putdata([round(255*x/(w-1)) for x in range(w)])\ngradient.resize((w, h)).save('gradient-mask.png')\ny, x = np.indices((h, w))\nchecks = ((y*7//h + x*7//h) % 2 * 255).astype('uint8')\nImage.fromarray(checks, 'L').save('checks-mask.png')\nsmall = image_to_tensor(crop)\nfor name, mask in (('plain', None),\n                   ('gradient', Path('gradient-mask.png')),\n                   ('checks', Path('checks-mask.png'))):\n    save('mask-' + name + '.png',\n         WarpTransform(mode='sine_lens', intensity=128,\n                       mask=mask)(small))", "a. in script")
            paragraph("After the script creates mask-source.png and the two masks, these equivalent CLI commands generate the three views without any bundled example assets. Each command starts from the same source crop.")
            code("python -m src --input mask-source.png --page-warp sine_lens \\\n  --intensity 128 --output mask-plain.png\npython -m src --input mask-source.png --page-warp sine_lens \\\n  --intensity 128 --mask gradient-mask.png \\\n  --output mask-gradient.png\npython -m src --input mask-source.png --page-warp sine_lens \\\n  --intensity 128 --mask checks-mask.png \\\n  --output mask-checks.png", "b. CLI")

        if mode in ("horizontal_wave", "vertical_wave", "custom"):
            names = ({"horizontal_wave": ("Horizontal wave",),
                      "vertical_wave": ("Vertical wave",),
                      "custom": ("Fixed vertical custom wave",
                                 "Fixed horizontal custom wave", "Symmetric custom wave")})[mode]
            for index, name in enumerate(names, start=4):
                grid_path = next(path for label, path in wave_studies if label == name)
                append_wave_study(number, index, name, grid_path)

    # One controlled source and wave strength for all four taper/shift cases.
    shift_grid = overlay_warped_grid(image_to_tensor(source_paragraph()),
                                     spacing=28, opacity=.24)
    shift_source_path = generated / "edge_taper_shift_source.png"
    PILImage.fromarray(shift_grid.mul(255).round().byte().permute(1, 2, 0).cpu().numpy()).save(shift_source_path)
    with PILImage.open(shift_source_path) as shift_input:
        shift_source = image_to_tensor(shift_input)
    shift_panels = []
    for label, dx, dy in (("edge_taper=True", 0, 0),
                          ("X shifted transform", 18, 0),
                          ("Y shifted transform", 0, 12),
                          ("XY shifted transform", 18, 12)):
        path = generated / f"edge_taper_shift_{dx}_{dy}.png"
        result = WarpTransform(mode="horizontal_wave", wavelength=28, intensity=4,
                               edge_taper=True, shift_X=dx, shift_Y=dy)(shift_source)
        PILImage.fromarray(result.mul(255).round().byte().permute(1, 2, 0).cpu().numpy()).save(path)
        shift_panels.append({"label": label, "path": str(path), "shift_X": dx,
                             "shift_Y": dy, "edge_taper": dx == dy == 0,
                             "mode": "horizontal_wave", "wavelength": 28, "intensity": 4})
    story.append(PageBreak())
    heading("12. Edge taper and pixel shifts")
    paragraph("Compare the same four-line text and straight reference grid in all four panels. Use horizontal_wave, wavelength 28 px, and intensity 4 px. The first panel uses edge_taper=True and zero shifts: wave displacement is exactly 0 px at each page edge. X and Y are right and down. Negative values move left and up.")
    paragraph("A nonzero shift_X or shift_Y overrides edge_taper=True: the wave itself becomes untapered before the pixel translation is added. These shifts are absolute pixels, not multiples of intensity. Both the wave and the translation are scaled by a displacement mask if one is supplied. Droplets and physical refraction do not accept these page-warp controls.")
    shift_rows = []
    for start in (0, 2):
        shift_rows.append([[figure(panel["path"], 73 * mm, 45 * mm),
                            Paragraph(f"{panel['label']}<br/>shift_X={panel['shift_X']} px; shift_Y={panel['shift_Y']} px",
                                      styles["ManualCaption"])]
                           for panel in shift_panels[start:start + 2]])
    story.append(framed_table(shift_rows, [84 * mm] * 2, [62 * mm] * 2))
    paragraph("Create mini-grid.png with section 4.1. All four images above use that same input; only the two shift values change. The code explicitly requests tapering so the shift override is visible.")
    code("python -m src --input mini-grid.png --output shift-xy.png \\\n  --page-warp horizontal_wave --wavelength 28 \\\n  --intensity 4 --edge-taper --shift-X 18 --shift-Y 12", "a. CLI: XY shift")
    code("small = image_to_tensor(Image.open('mini-grid.png'))\nfor name, dx, dy in (('taper', 0, 0), ('x', 18, 0),\n                     ('y', 0, 12), ('xy', 18, 12)):\n    transform = WarpTransform(mode='horizontal_wave',\n        wavelength=28, intensity=4, edge_taper=True,\n        shift_X=dx, shift_Y=dy)\n    save('shift-' + name + '.png', transform(small))", "b. in script: all four panels")

    doc = BaseDocTemplate(str(output), pagesize=A4, leftMargin=18 * mm,
                          rightMargin=18 * mm, topMargin=17 * mm, bottomMargin=20 * mm,
                          title="lorem-warped: PyTorch Image Transformations", author="lorem-warped contributors")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id="Manual", frames=[frame], onPage=footer))
    doc.build(story)
    with pymupdf.open(output) as pdf:
        wave_pages = [i + 1 for i, page in enumerate(pdf)
                      if "Two-parameter wavelength study" in page.get_text()]
        if len(wave_pages) != 5 or any(len(pdf[n - 1].get_image_info()) != 1 for n in wave_pages):
            raise RuntimeError("The five wavelength-by-intensity grids must each occupy one illustrated page")
        shift_hits = [i + 1 for i, page in enumerate(pdf)
                      if page.search_for("12. Edge taper and pixel shifts")]
        if len(shift_hits) != 1 or len(pdf[shift_hits[0] - 1].get_image_info()) != 4:
            raise RuntimeError("The four taper/shift panels must share one illustrated page")
        shift_page = pdf[shift_hits[0] - 1].get_text()
        if "b. in script: all four panels" not in shift_page or "shift_Y=12" not in shift_page:
            raise RuntimeError("The taper/shift recipe must remain on its illustrated page")
        mask_hits = [index + 1 for index, page in enumerate(pdf)
                     if page.search_for("6.4 Mask the displacement")]
        if len(mask_hits) != 1 or len(pdf[mask_hits[0] - 1].get_image_info()) != 6:
            raise RuntimeError("The mask chapter must show a 3-by-2 original, unmasked, gradient and checkerboard comparison")
        mask_section["page"] = mask_hits[0]
        overview_hits = [index + 1 for index, page in enumerate(pdf)
                         if page.search_for("1.1 Distortion overview")]
        distribution_hits = [index + 1 for index, page in enumerate(pdf)
                             if page.search_for("5.5 Droplet arrangements")]
        if overview_hits != [1] or len(pdf[0].get_image_info()) != 2:
            raise RuntimeError("The overview must have two panels on the first page")
        if len(distribution_hits) != 1 or len(pdf[distribution_hits[0] - 1].get_image_info()) != 6:
            raise RuntimeError("The droplet arrangements must have six panels on one page")
        overview["page"] = 1
        distributions["page"] = distribution_hits[0]
        tree_hits = [index + 1 for index, page in enumerate(pdf)
                     if page.search_for("2. Distortion types")]
        if len(tree_hits) != 1 or len(pdf[tree_hits[0] - 1].get_image_info()) != 18:
            raise RuntimeError("The distortion tree must contain nine grid and nine text previews")
        atlas["tree"]["page"] = tree_hits[0]
        shade_hits = [index + 1 for index, page in enumerate(pdf)
                      if page.search_for("5.4 Shaded droplet regions")]
        if len(shade_hits) != 1 or len(pdf[shade_hits[0] - 1].get_image_info()) != 3:
            raise RuntimeError("The three shaded distribution maps must share one page")
        atlas["distributions"]["page"] = shade_hits[0]
        schematic = atlas["modes"]["refractive_index"]["schematic"]
        diagram_hits = [i + 1 for i, page in enumerate(pdf)
                        if page.search_for("7.0 Ray geometry and optical indicatrix")]
        if len(diagram_hits) != 1 or len(pdf[diagram_hits[0] - 1].get_image_info()) != 1:
            raise RuntimeError("The optical schematic must occupy one illustrated page")
        schematic["page"] = diagram_hits[0]
        for number, mode in enumerate(STRENGTHS, start=5):
            mechanics = [i + 1 for i, page in enumerate(pdf)
                         if page.search_for(f"{number}. How {titles[mode].lower()} works")]
            sweep = [i + 1 for i, page in enumerate(pdf) if page.search_for(f"{number}.3 Two-variable legibility study")]
            expected = 3 if mode in ("refractive_index", "custom") else 1
            if len(mechanics) != 1 or len(sweep) != expected or any(
                    len(pdf[page - 1].get_image_info()) != 16 for page in sweep):
                raise RuntimeError(f"Incomplete mechanics or 4-by-4 study: {mode}")
            atlas["modes"][mode]["mechanics_page"] = mechanics[0]
            if mode == "refractive_index":
                for study, page in zip(atlas["modes"][mode]["thickness_studies"], sweep):
                    study["page"] = page
                atlas["modes"][mode]["sweep"]["page"] = sweep[1]
            elif mode == "custom":
                for study, page in zip(atlas["modes"][mode]["intensity_studies"], sweep):
                    study["page"] = page
                atlas["modes"][mode]["sweep"]["page"] = sweep[1]
            else:
                atlas["modes"][mode]["sweep"]["page"] = sweep[0]
        for mode, gallery in galleries.items():
            hits = [index + 1 for index, page in enumerate(pdf)
                    if page.search_for(gallery["explanation"])]
            if len(hits) != 1 or len(pdf[hits[0] - 1].get_image_info()) != 6:
                raise RuntimeError(f"Incomplete one-page gallery for {mode}: {hits}")
            gallery["page"] = hits[0]
    manifest = {"input": str(input_path.resolve()), "crop_xyxy": crop_box,
                "wave_studies": [{"mode": name, "image": str(path), "page": n,
                                  "source_crop_xyxy": [80, 120, 320, 280],
                                  "wavelength_px": [28 * p / 100 for p in PERCENTAGES],
                                  "intensity_px": list(INTENSITIES)}
                                 for (name, path), n in zip(wave_studies, wave_pages)],
                "lens_parameters": LENS, "overview": overview,
                "distributions": distributions, "galleries": galleries,
                "atlas": atlas, "mask": mask_section,
                "edge_taper_shift_examples": {"page": shift_hits[0], "source": str(shift_source_path),
                                              "panels": shift_panels}}
    (generated / "manual-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Built {output} with {len(galleries)} six-panel transformation galleries")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("original.png"))
    parser.add_argument("--output", type=Path, default=Path("docs/lorem-warped-manual.pdf"))
    args = parser.parse_args()
    build(args.input, args.output)


if __name__ == "__main__":
    main()
