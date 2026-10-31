"""Render real, small PyTorch examples for the illustrated operator manual.

A page-reference grid is a sampled visual guide, not a droplet-centre grid.
The uniform optical slab has two constant translation fields, not a lens field.
"""
from __future__ import annotations

import math
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
if __package__:
    from .optics_schematic import build_schematic
else:
    from optics_schematic import build_schematic

from src import (
    CentreDistribution, DistortionConfig, PAGE_WARPS, WarpTransform,
    custom_wave, displacement_field, image_to_tensor, make_centres, overlay_warped_grid,
    page_warp_field, refract_image, refractive_index_field, sine_lens,
)

PARAGRAPH = ("Lorem ipsum", "dolor sit amet", "consectetur", "adipiscing elit")
FONT_PX = 28
MODES = ("droplet", "sine_lens", "refractive_index", "horizontal_wave",
         "vertical_wave", "gentle_pinch", "custom")
LABELS = {"droplet": "Droplet", "sine_lens": "Sine lens",
          "refractive_index": "Refractive-index slab",
          "horizontal_wave": "Horizontal wave", "vertical_wave": "Vertical wave",
          "gentle_pinch": "Gentle pinch", "custom": "Custom wave"}


def source_paragraph(font_px: int = FONT_PX) -> Image.Image:
    """Use the same four-line excerpt for every demonstrated transformation."""
    image = Image.new("RGB", (300, 184), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("DejaVuSans.ttf", font_px)
    step = font_px + 6
    for row, line in enumerate(PARAGRAPH):
        draw.text((10, 10 + row * step), line, font=font, fill="#17252e")
    return image


def pixels(tensor: torch.Tensor) -> Image.Image:
    data = tensor.detach().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(data, "RGB")


def custom_function(wavelength_x: float, wavelength_y: float, width: int, height: int):
    """Independent pixel periods along X/Y with a common edge taper (dy, dx)."""
    def field(y: torch.Tensor, x: torch.Tensor):
        envelope = torch.sin(math.pi * y) * torch.sin(math.pi * x)
        return (envelope * torch.sin(2 * math.pi * x * (width - 1) / wavelength_x),
                envelope * torch.sin(2 * math.pi * y * (height - 1) / wavelength_y))
    return field


def variable_pinch(y: torch.Tensor, x: torch.Tensor, *, envelope_power: float):
    envelope = (torch.sin(math.pi * y) * torch.sin(math.pi * x)).clamp_min(0)
    envelope = envelope.pow(envelope_power)
    return envelope * (0.5 - y), envelope * (0.5 - x)


def render(mode: str, image: Image.Image, *, intensity: float | None = None,
           radius: float = 28, count: int = 8, distribution: str = "square",
           frequency_x: float = 1.5, thickness_mm: float = 5,
           polarization_deg: float = 30, wavelength_x: float = 28,
           wavelength_y: float = 28,
           n_x: float = 1.3, n_y: float = 2.5,
           wave_wavelength: float | None = None,
           pinch_power: float | None = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Return the actual transformed tensor and its actual pixel displacement field(s)."""
    tensor = image_to_tensor(image)
    if mode == "droplet":
        strength = 2.5 if intensity is None else intensity
        config = DistortionConfig(centre_count=count, distribution=CentreDistribution(distribution),
                                  radius=radius, intensity=strength, seed=7, profile="cosine")
        centres = make_centres(config, tensor.shape[1], tensor.shape[2], device=tensor.device)
        field = displacement_field(tensor, radius, centres, profile="cosine")
        return WarpTransform(mode="droplet", config=config)(tensor), [field * strength]
    if mode == "refractive_index":
        fields = [refractive_index_field(
                  tensor, mode=axis, thickness_mm=thickness_mm,
                  n_x=n_x, n_y=n_y, n_z=1.3,
                  direction_xyz=(-1, 1, 1),
                  page_width_mm=210, page_height_mm=297)
                  for axis in ("x", "y")]
        options = dict(thickness_mm=thickness_mm, n_x=n_x, n_y=n_y, n_z=1.3,
                       polarization_deg=polarization_deg, direction_xyz=(-1, 1, 1),
                       page_width_mm=210, page_height_mm=297)
        return WarpTransform(mode=mode, refraction_options=options)(tensor), fields
    strength = {"sine_lens": 20, "horizontal_wave": 8, "vertical_wave": 10,
                "gentle_pinch": 150, "custom": 8}[mode] if intensity is None else intensity
    if mode == "sine_lens":
        options = dict(frequency_x=frequency_x, frequency_y=1.5,
                       phase_x=0.1, phase_y=0.2,
                       amplitude_x=1.2, amplitude_y=0.8)
        function = partial(sine_lens, **options)
        transform = WarpTransform(mode=mode, intensity=strength, lens_options=options)
    elif mode == "custom":
        function = custom_function(wavelength_x, wavelength_y, image.width, image.height)
        transform = WarpTransform(mode=mode, intensity=strength, warp_function=function)
    elif mode in ("horizontal_wave", "vertical_wave"):
        period = 28 if wave_wavelength is None else wave_wavelength
        function = partial(PAGE_WARPS[mode], wavelength=period)
        transform = WarpTransform(mode=mode, intensity=strength, wavelength=period)
    elif mode == "gentle_pinch" and pinch_power is not None:
        function = partial(variable_pinch, envelope_power=pinch_power)
        transform = WarpTransform(mode="custom", intensity=strength, warp_function=function)
    else:
        function = PAGE_WARPS[mode]
        transform = WarpTransform(mode=mode, intensity=strength)
    return transform(tensor), [page_warp_field(tensor, function) * strength]


def function_plot(fields: list[torch.Tensor], dest: Path) -> None:
    """Show the measured pixel field on a square reference grid.

    Colored vectors are rescaled for visibility; labels disclose that fact.
    """
    components = [field.detach().cpu().numpy() for field in fields]
    height, width = components[0].shape[:2]
    magnitude = np.maximum.reduce([np.linalg.norm(field, axis=-1) for field in components])
    max_magnitude = max(float(np.max(magnitude)), 1e-6)
    heat = np.clip(magnitude / max_magnitude, 0, 1)
    color = np.stack((255 - 57 * heat, 255 - 21 * heat, 255 - 8 * heat), axis=-1).astype("uint8")
    image = Image.fromarray(color, "RGB")
    draw = ImageDraw.Draw(image)
    # A wavelength can put every 28 px sample on a zero of the wave.
    # Use a noncommensurate grid in that case so arrows reveal both the
    # magnitude and the changing direction of the actual pixel field.
    spacing, origin = 28, 14
    samples = [(y, x) for y in range(origin, height, spacing)
               for x in range(origin, width, spacing)]
    if samples and not any(
        math.hypot(float(field[y, x, 0]), float(field[y, x, 1])) >= 0.05
        for field in components for y, x in samples
    ):
        spacing, origin = 24, 12
    for x in range(origin, width, spacing):
        draw.line((x, 0, x, height), fill="#b1c3cf", width=1)
    for y in range(origin, height, spacing):
        draw.line((0, y, width, y), fill="#b1c3cf", width=1)
    for y in range(origin, height, spacing):
        for x in range(origin, width, spacing):
            for index, field in enumerate(components):
                dy, dx = field[y, x]
                length = math.hypot(float(dy), float(dx))
                if length < 0.05:
                    continue
                scale = min(13 / length, 2.3)
                offset = -3 if len(components) > 1 and index == 0 else 3 if len(components) > 1 else 0
                x0, y0 = x + offset, y + offset
                color_name = "#176a9b" if index == 0 else "#bc5632"
                draw.line((x0, y0, x0 + dx * scale, y0 + dy * scale), fill=color_name, width=2)
                draw.ellipse((x0 + dx * scale - 2, y0 + dy * scale - 2,
                              x0 + dx * scale + 2, y0 + dy * scale + 2), fill=color_name)
    image.save(dest)


def collage(panels: list[dict], columns: int, path: Path) -> None:
    cell_w, cell_h, label_h, margin = 300, 184, 28, 10
    rows = math.ceil(len(panels) / columns)
    canvas = Image.new("RGB", (columns * (cell_w + margin) + margin,
                               rows * (cell_h + label_h + margin) + margin), "#eaf0f4")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("DejaVuSans.ttf", 16)
    for index, panel in enumerate(panels):
        x = margin + index % columns * (cell_w + margin)
        y = margin + index // columns * (cell_h + label_h + margin)
        with Image.open(panel["path"]) as source:
            canvas.paste(source.convert("RGB"), (x, y))
        draw.text((x + 3, y + cell_h + 3), panel["label"], font=font, fill="#18374a")
    canvas.save(path)


def overlay_droplet_guides(base: Image.Image, centres, radius: float,
                           *, show_centres: bool = False) -> Image.Image:
    """Mark actual droplet regions in gray at 30% alpha on an unwarped source.

    Composite one mask, not one circle at a time: overlapping circles remain
    30% opaque rather than becoming darker at their intersections.
    """
    overlay = Image.new("RGBA", base.size)
    draw = ImageDraw.Draw(overlay)
    points = list(centres)
    for y, x in points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                     fill=(128, 128, 128, round(255 * .3)))
    if show_centres:
        for y, x in points:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2),
                         fill=(64, 64, 64, round(255 * .3)))
    return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")


def shaded_centres(base: Image.Image, arrangement: str, dest: Path) -> dict:
    settings = DistortionConfig(centre_count=12, distribution=CentreDistribution(arrangement),
                                radius=30, intensity=3, seed=7, profile="cosine")
    centres = make_centres(settings, base.height, base.width, torch.device("cpu"))
    overlay_droplet_guides(base, centres.tolist(), 30, show_centres=True).save(dest)
    return {"distribution": arrangement, "count": len(centres), "radius": 30,
            "overlay_alpha": .3, "role": "original", "path": str(dest)}


def build_atlas(images: Path, generated: Path) -> dict:
    """Build all visible atlas assets and a manifest of two-axis sweep settings."""
    images.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)
    base = source_paragraph()
    base.save(generated / "mini_source.png")
    data = {"source": str(generated / "mini_source.png"), "modes": {},
            "tree": {"previews": []}, "distributions": {"maps": []}}
    for mode in MODES:
        result, fields = render(mode, base, intensity=80 if mode == "sine_lens" else None)
        mini = images / f"{mode}_mini.png"
        pixels(result).save(mini)
        field_image = images / f"{mode}_function.png"
        function_plot(fields, field_image)
        reference = images / f"{mode}_reference_grid.png"
        if mode == "refractive_index":
            grid = refract_image(image_to_tensor(base), thickness_mm=5,
                                 n_x=1.3, n_y=2.5, n_z=1.3, polarization_deg=30,
                                 direction_xyz=(-1, 1, 1), page_width_mm=210,
                                 page_height_mm=297, show_grid=True, grid_spacing=28)
        else:
            grid = overlay_warped_grid(result, fields[0], 1, spacing=28, opacity=0.24)
        pixels(grid).save(reference)
        wavelengths = tuple(FONT_PX * percent / 100 for percent in (50, 100, 150, 200))
        if mode == "droplet":
            rows, cols, axes = (30, 45, 60, 75), (0, 1, 3, 5), ("radius (px)", "intensity (px)")
        elif mode == "sine_lens":
            rows, cols, axes = (1, 1.5, 2, 2.5), (0, 15, 40, 80), ("X frequency", "intensity (px)")
        elif mode == "refractive_index":
            rows, cols, axes = (1, 1.2, 1.5, 2), (1, 1.2, 1.5, 2), ("n_x", "n_y")
        elif mode == "gentle_pinch":
            rows, cols, axes = (0.6, 1, 1.5, 2), (0, 90, 180, 300), ("pinch envelope power", "intensity (px)")
        elif mode == "custom":
            rows, cols, axes = wavelengths, wavelengths, ("X wavelength (px)", "Y wavelength (px)")
        elif mode == "horizontal_wave":
            rows, cols, axes = wavelengths, (0, 12, 28, 50), ("Y wavelength (px)", "intensity (px)")
        else:
            rows, cols, axes = wavelengths, (0, 12, 28, 50), ("X wavelength (px)", "intensity (px)")
        studies = []
        study_values = ((2, 5, 10) if mode == "refractive_index" else
                        (5, 10, 20) if mode == "custom" else (None,))
        for setting in study_values:
            study_panels = []
            for row, first in enumerate(rows):
                for column, second in enumerate(cols):
                    opts = ({"radius": first, "intensity": second} if mode == "droplet" else
                            {"frequency_x": first, "intensity": second} if mode == "sine_lens" else
                            {"thickness_mm": setting, "polarization_deg": 45,
                             "n_x": first, "n_y": second} if mode == "refractive_index" else
                            {"wavelength_x": first, "wavelength_y": second,
                             "intensity": setting} if mode == "custom" else
                            {"pinch_power": first, "intensity": second} if mode == "gentle_pinch" else
                            {"wave_wavelength": first, "intensity": second})
                    suffix = (f"_{setting}mm" if mode == "refractive_index" else
                              f"_{setting}px" if mode == "custom" else "")
                    path = generated / f"study_{mode}{suffix}_{row}_{column}.png"
                    zero_warp = mode not in ("refractive_index", "custom") and second == 0
                    if mode == "droplet":
                        transformed, _ = render(mode, base, **opts)
                        shown = pixels(transformed)
                        if zero_warp:
                            guide_config = DistortionConfig(
                                centre_count=8, distribution=CentreDistribution.SQUARE,
                                radius=first, intensity=0, seed=7, profile="cosine")
                            centres = make_centres(guide_config, shown.height, shown.width,
                                                   torch.device("cpu"))
                            shown = overlay_droplet_guides(
                                shown, centres.tolist(), guide_config.radius, show_centres=True)
                    else:
                        straight = overlay_warped_grid(image_to_tensor(base), spacing=28,
                                                       opacity=.24)
                        gridded, _ = render(mode, pixels(straight), **opts)
                        shown = pixels(gridded)
                    shown.save(path)
                    label = (f"{first:g} px | {second:g} px" if mode in
                             ("custom", "horizontal_wave", "vertical_wave") else
                             f"{first:g} | {second:g}")
                    study_panels.append({"row": row, "column": column, "first": first,
                                         "second": second, "label": label,
                                         "path": str(path), "role": "original" if zero_warp else "warped",
                                         **({"guide": "straight_square_grid" if zero_warp else
                                            "warped_square_grid"} if mode != "droplet" else
                                            {"overlay_alpha": .3} if zero_warp else {})})
            readme_name = (f"{mode}_sweep_{setting}mm_4x4.png" if mode == "refractive_index" else
                           f"{mode}_sweep_{setting}px_4x4.png" if mode == "custom" else
                           f"{mode}_sweep_4x4.png")
            readme_sweep = images / readme_name
            collage(study_panels, 4, readme_sweep)
            studies.append({"axes": axes, "rows": rows, "columns": cols,
                            "panels": study_panels, "readme_image": str(readme_sweep),
                            **({"thickness_mm": setting} if mode == "refractive_index" else
                               {"intensity_px": setting} if mode == "custom" else {})})
        main_study = studies[1] if mode in ("refractive_index", "custom") else studies[0]
        if mode in ("refractive_index", "custom"):
            # Preserve links to the original central 4-by-4 figure.
            collage(main_study["panels"], 4, images / f"{mode}_sweep_4x4.png")
        data["modes"][mode] = {"mini": str(mini), "field": str(field_image),
                                **({"preview_intensity": 2.5} if mode == "droplet" else {}),
                                "reference_grid": str(reference), "sweep": main_study,
                                **({"thickness_studies": studies} if mode == "refractive_index" else {}),
                                **({"intensity_studies": studies} if mode == "custom" else {}),
                                "preview_settings": {
                                    "droplet": "square; 8 centres; radius 28 px; intensity 2.5 px",
                                    "sine_lens": "X/Y frequency 1.5/1.5; intensity 80 px",
                                    "refractive_index": "thickness 5 mm; polarization 30 degrees",
                                    "horizontal_wave": "Y wavelength 28 px; intensity 8 px",
                                    "vertical_wave": "X wavelength 28 px; intensity 10 px",
                                    "gentle_pinch": "intensity 150 px",
                                    "custom": "X/Y wavelength 28/28 px; intensity 8 px",
                                }[mode]}
        if mode == "refractive_index":
            data["modes"][mode]["schematic"] = build_schematic(
                images / "refractive_index_ray_indicatrix.png")
    for arrangement in CentreDistribution:
        mini = images / f"droplet_{arrangement.value}_mini.png"
        pixels(render("droplet", base, distribution=arrangement.value,
                      count=6, radius=26, intensity=3)[0]).save(mini)
        map_path = generated / f"shaded_{arrangement.value}.png"
        data["distributions"]["maps"].append(shaded_centres(base, arrangement.value, map_path))
    shade = images / "droplet_centres_shaded.png"
    collage([{**item, "label": item["distribution"].title()} for item in
             data["distributions"]["maps"]], 3, shade)
    data["distributions"]["readme_image"] = str(shade)
    # Keep the grid free of text: apply the exact preview transform to its
    # pixels, rather than drawing straight lines over a finished warp.
    grid_spacing = 16
    grid_source = Image.new("RGB", base.size, "white")
    grid_draw = ImageDraw.Draw(grid_source)
    for x in range(0, grid_source.width, grid_spacing):
        grid_draw.line((x, 0, x, grid_source.height), fill="#809aaa", width=1)
    for y in range(0, grid_source.height, grid_spacing):
        grid_draw.line((0, y, grid_source.width, y), fill="#809aaa", width=1)
    grid_source_path = generated / "tree_square_grid_original.png"
    grid_source.save(grid_source_path)
    data["tree"].update(grid_source=str(grid_source_path), grid_spacing=grid_spacing)
    data["tree"]["previews"] = [
        {"label": f"Droplet > {name}", "path": str(images / f"droplet_{name}_mini.png"),
         "intensity": 3, "mode": "droplet", "distribution": name}
        for name in ("random", "square", "triangle")]
    data["tree"]["previews"] += [
        {"label": LABELS[mode], "path": data["modes"][mode]["mini"], "mode": mode,
         **({"intensity": 80} if mode == "sine_lens" else {})}
        for mode in MODES if mode != "droplet"]
    for preview in data["tree"]["previews"]:
        mode = preview["mode"]
        options = ({"distribution": preview["distribution"], "count": 6,
                    "radius": 26, "intensity": 3} if mode == "droplet" else
                   {"intensity": preview["intensity"]} if mode == "sine_lens" else {})
        name = (f"droplet_{preview['distribution']}" if mode == "droplet" else mode)
        grid_path = images / f"{name}_grid_mini.png"
        pixels(render(mode, grid_source, **options)[0]).save(grid_path)
        preview.update(grid_path=str(grid_path), grid_guide="warped_square_grid")
    return data
