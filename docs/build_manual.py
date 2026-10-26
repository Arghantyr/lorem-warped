"""Build the ASD-STE100-style manual and real PyTorch comparison galleries.

Run from the repository root after installing the dev extra:
python docs/build_manual.py --input original.png --output docs/lorem-warped-manual.pdf
"""
from __future__ import annotations

import argparse
import json
import math
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

from lorem_warped import (CentreDistribution, DistortionConfig, WarpTransform,
                                   image_to_tensor, make_centres, overlay_warped_grid)
from manual_atlas import build_atlas, overlay_droplet_guides


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


def custom_wave(y, x):
    envelope = torch.sin(math.pi * y) * torch.sin(math.pi * x)
    return (envelope * torch.sin(4 * math.pi * x),
            envelope * torch.sin(4 * math.pi * y))


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
        return WarpTransform(mode=mode, intensity=value, warp_function=custom_wave)
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
    paragraph("Package structure: src/lorem_warped/ holds the library and CLI. examples/ holds a runnable script. docs/build_manual.py rebuilds this PDF and its real examples. tests/ checks the interfaces. original.png is the sample input. No command in this manual requires you to replace that input.")
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
    paragraph("Install Python 3.10 or later. Open a shell in the repository root. Make a virtual environment. Install the project with its manual tools.")
    code("python -m venv .venv\n. .venv/bin/activate\npython -m pip install -e '.[dev]'\npython -m lorem_warped --help", "a. CLI")
    paragraph("On Windows PowerShell, activate .venv\\Scripts\\Activate.ps1. Use the PyTorch installation selector if you need a specific GPU build.")
    paragraph("Use original.png as the input. Do not overwrite it. The CLI uses a generated page if you omit --input. Python images use float CHW tensors from 0 to 1.")
    heading("4. Script setup")
    paragraph("In the script examples, read the image one time. Use this function to save each output tensor as a PNG. Each transform starts from source, not from the output of another transform.")
    code("from PIL import Image\nfrom lorem_warped import (\n    CentreDistribution, DistortionConfig,\n    WarpTransform, image_to_tensor)\nsource = image_to_tensor(Image.open('original.png'))\ndef save(path, result):\n    pixels = result.mul(255).round().byte()\n    pixels = pixels.permute(1, 2, 0).cpu().numpy()\n    Image.fromarray(pixels).save(path)", "a. in script")
    paragraph("A custom function also needs math and torch imports. The custom code example includes them. The CLI runs on CPU. Move the tensor to a supported GPU in your script if necessary.")

    descriptions = {
        "droplet": "A droplet warp moves pixels around circular centres. Choose random, square, or triangle (hexagonal) centres. Radius sets each circle size. Intensity sets the movement in pixels. The seed fixes random centres. The cosine profile permits overlaps.",
        "sine_lens": "A sine lens moves the page in two axes. X and Y frequencies, phases, and gains give six shape parameters. The intensity multiplies the displacement in pixels. Keep the six parameters fixed in this comparison.",
        "refractive_index": "A uniform anisotropic slab makes two translated polarization views. Thickness sets the translation in millimetres. The refractive indices and physical page size stay fixed. Do not use pixel intensity for this physical model.",
        "horizontal_wave": "A horizontal wave moves text left and right in broad bands. The page edges stay in place. The intensity sets the movement in pixels.",
        "vertical_wave": "A vertical wave moves baselines up and down across the page. The page edges stay in place. The intensity sets the movement in pixels.",
        "gentle_pinch": "A gentle pinch moves text toward the page centre. Its field is small. Use a larger pixel intensity to see the effect.",
        "custom": "A custom wave reads normalized Y and X coordinates and returns Y and X displacement tensors. Independent X and Y cycle counts shape the field. The 5, 10, and 20 px examples use a fixed 2-by-2 cycle pattern. The custom mode has no CLI option.",
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
            "horizontal_wave": "A fixed three-cycle wave moves text horizontally. The function has a smooth edge envelope, so page edges stay fixed. Pixel intensity multiplies the horizontal displacement.",
            "vertical_wave": "A fixed two-cycle wave moves text vertically. The function has a smooth edge envelope, so page edges stay fixed. Pixel intensity multiplies the vertical displacement.",
            "gentle_pinch": "The field points toward the page centre. A smooth envelope holds page edges fixed. The built-in transform exposes pixel intensity, but not a separate width or frequency control.",
            "custom": "The callback computes independently parameterized X and Y waves from normalized coordinates and returns a (dy, dx) field. Pixel intensity multiplies it. Only the Python API accepts this callback.",
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
            paragraph("Schematic terminology: Tulane University, Uniaxial Minerals, Uniaxial Indicatrix, Optic Sign, and Ray Path. The image uses this manual's optical solver for the ray positions; polarization weights at oblique incidence remain approximate.")
            paragraph("Optical indicatrix reference: https://www2.tulane.edu/~sanelson/eens211/uniaxial_minerals.htm")
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
            code("python -m lorem_warped --input original.png \\\n  --output droplet.png --centres 3 --distribution random \\\n  --radius 80 --intensity 5 --seed 7 --profile cosine", "a. CLI")
            code("transform = WarpTransform(mode='droplet',\n    config=DistortionConfig(centre_count=3, radius=80,\n                            intensity=5, seed=7))\nsave('droplet.png', transform(source))", "b. in script")
        elif mode == "sine_lens":
            paragraph("Set the six lens parameters to X/Y frequencies 2.5/1.5, phases 0.1/0.2 cycles, and gains 1.2/0.8. Use 32 pixels for this example. A frequency must be positive.")
            code("python -m lorem_warped --input original.png \\\n  --output sine_lens.png --page-warp sine_lens \\\n  --intensity 32 --lens-frequency-x 2.5 \\\n  --lens-frequency-y 1.5 --lens-phase-x 0.1 \\\n  --lens-phase-y 0.2 --lens-amplitude-x 1.2 \\\n  --lens-amplitude-y 0.8", "a. CLI")
            code("transform = WarpTransform(mode='sine_lens', intensity=32,\n    lens_options=dict(frequency_x=2.5, frequency_y=1.5,\n        phase_x=0.1, phase_y=0.2,\n        amplitude_x=1.2, amplitude_y=0.8))\nsave('sine_lens.png', transform(source))", "b. in script")
        elif mode == "refractive_index":
            paragraph("Thickness and page dimensions use millimetres. The angle uses degrees. The slab gives constant translations, not a spatially varying lens. The polarization weights are approximate at oblique incidence.")
            code("python -m lorem_warped --input original.png \\\n  --output refraction.png --page-warp refractive_index \\\n  --refraction-thickness-mm 5 --refraction-n-x 1.3 \\\n  --refraction-n-y 2.5 --refraction-n-z 1.3 \\\n  --refraction-polarization-deg 45 \\\n  --refraction-direction -1 1 1 \\\n  --page-width-mm 210 --page-height-mm 297", "a. CLI")
            code("transform = WarpTransform(mode='refractive_index',\n    refraction_options=dict(thickness_mm=5, n_x=1.3,\n        n_y=2.5, n_z=1.3, polarization_deg=45,\n        direction_xyz=(-1, 1, 1),\n        page_width_mm=210, page_height_mm=297))\nsave('refraction.png', transform(source))", "b. in script")
        elif mode == "custom":
            paragraph("The custom mode has no CLI option. Use the Python API to supply the function below.")
            code("import math\nimport torch\ndef custom_wave(cycles_x, cycles_y):\n    def field(y, x):\n        edge = torch.sin(math.pi * y) * torch.sin(math.pi * x)\n        return (edge * torch.sin(2 * math.pi * cycles_x * x),\n                edge * torch.sin(2 * math.pi * cycles_y * y))\n    return field\ntransform = WarpTransform(mode='custom', intensity=10,\n                          warp_function=custom_wave(2, 2))\nsave('custom.png', transform(source))", "a. in script")
            paragraph("In the three 4-by-4 studies, keep intensity fixed at 5, 10, or 20 px, then set cycles_x and cycles_y to 1, 2, 3, or 4. Run each combination on the same source image. No CLI cycle control exists.")
        else:
            strength = {"horizontal_wave": 60, "vertical_wave": 60,
                        "gentle_pinch": 160}[mode]
            code(f"python -m lorem_warped --input original.png \\\n  --output {mode}.png --page-warp {mode} \\\n  --intensity {strength}", "a. CLI")
            code(f"transform = WarpTransform(mode='{mode}', intensity={strength})\nsave('{mode}.png', transform(source))", "b. in script")
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
                paragraph("The text and straight source grid stay fixed. Each output uses the same callback formula on both text and grid; only its X and Y wave cycles change. Every cell is warped; no zero-intensity control appears on this page.")
            else:
                paragraph("Every cell starts with the same dense square reference grid. The transform warps the grid with the text; zero-intensity cells keep straight lines. These lines do not represent droplet centres.")
            if mode == "refractive_index":
                paragraph(f"Keep slab thickness at {study['thickness_mm']} mm and polarization at 45 degrees. Change n_x down the rows and n_y across the columns: 1, 1.2, 1.5, 2.0. The indices are dimensionless; n_z remains 1.3. A uniform slab produces translated views, not a curved lens. No artistic pixel intensity is used.")
            elif mode in ("horizontal_wave", "vertical_wave", "gentle_pinch"):
                paragraph("The built-in preset exposes only intensity. This study uses a Python callback with an adjustable wave-cycle count or pinch-envelope power; it is a parameterized variant, not an extra preset CLI flag. The input text itself never changes.")
            elif mode == "custom":
                paragraph(f"Keep pixel intensity at {study['intensity_px']} throughout this page. X cycles run 1 to 4 down the rows; Y cycles run 1 to 4 across the columns. This is a Python-only callback, not a built-in wave CLI option.")
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
                command = (f"python -m lorem_warped --input original.png " + "\\\n"
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

    story.append(PageBreak())
    heading("12. Sampling, diagnosis, and references")
    paragraph("Geometric fields use pixel displacement. The sampler reads source pixels at output position minus intensity times field. It uses bilinear sampling, border padding, and aligned corners. Strong fields can fold or repeat text.")
    paragraph("If the package cannot be imported, activate the correct environment and install it with python -m pip install -e '.[dev]'. For a different physical page size, change both page dimensions in the refractive example.")
    code("python docs/build_manual.py --input original.png \\\n  --output docs/lorem-warped-manual.pdf\npython -m pytest -q", "a. CLI")
    paragraph("Writing reference: ASD-STE100 Issue 9 (January 15, 2025), https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf. Software reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html. Formal STE conformity is not claimed.")

    story.append(PageBreak())
    heading("13. Mask the displacement")
    paragraph("Use a grayscale mask with exactly the same width and height as the image. At each output pixel, black (0) cancels both displacement components; white (255) keeps the full field. Gray scales both components continuously: 128 gives about half strength. The sampler reads the original at the resulting coordinates. It does not blend an unwarped and a warped image, and it does not change image alpha.")
    paragraph("All three results use the same text crop and sine-lens field at 128 px intensity. Row 1 compares the original with an unmasked warp. Row 2 shows a left-to-right gradient mask and its result. Row 3 shows a black-and-white checkerboard mask and its result; each square has side length one-seventh of the crop height in both axes. Only the mask changes. The bundled examples/mask-gradient.png matches the full 1200-by-1600 original.png.")
    def mask_cell(path, caption):
        return [figure(path, 73 * mm, 34 * mm), Paragraph(caption, styles["ManualCaption"])]
    mask_panels = [
        [mask_cell(mask_source, "Original text"), mask_cell(unmasked_path, "Without mask")],
        [mask_cell(mask_path, "Gradient mask"), mask_cell(masked_path, "With gradient mask")],
        [mask_cell(checkerboard_path, "Checkerboard mask"),
         mask_cell(checkerboard_warp_path, "With checkerboard mask")],
    ]
    story.append(framed_table(mask_panels, [84 * mm] * 2, [44 * mm] * 3))
    paragraph("Run this command on the bundled full-size image. To use another input, make a mask of its exact size; the program does not resize the mask. Omit --mask to keep the unmasked behavior.")
    code("python -m lorem_warped --input original.png \\\n  --mask examples/mask-gradient.png --page-warp sine_lens \\\n  --intensity 128 --output masked.png", "a. CLI")
    code("from pathlib import Path\ntransform = WarpTransform(mode='sine_lens', intensity=128,\n    mask=Path('examples/mask-gradient.png'))\nsave('masked.png', transform(source))", "b. in script")
    paragraph("The mask also works with droplets, page/custom warps, and refraction. Refraction retains optical transmission. Python accepts normalized H-by-W tensors; the CLI converts color files to grayscale. Mismatched sizes fail.")

    doc = BaseDocTemplate(str(output), pagesize=A4, leftMargin=18 * mm,
                          rightMargin=18 * mm, topMargin=17 * mm, bottomMargin=20 * mm,
                          title="lorem-warped: PyTorch Image Transformations", author="lorem-warped contributors")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id="Manual", frames=[frame], onPage=footer))
    doc.build(story)
    with pymupdf.open(output) as pdf:
        mask_hits = [index + 1 for index, page in enumerate(pdf)
                     if page.search_for("13. Mask the displacement")]
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
                "lens_parameters": LENS, "overview": overview,
                "distributions": distributions, "galleries": galleries,
                "atlas": atlas, "mask": mask_section}
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
