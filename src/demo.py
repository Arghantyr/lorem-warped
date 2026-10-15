"""Demo operations for PyTorch image warps."""
from __future__ import annotations

import math
from PIL import Image, ImageDraw, ImageFont


def render_lorem_page(
    width: int = 1200,
    height: int = 1600,
    paragraphs: int = 6,
) -> Image.Image:
    """Render a two-column page with paragraphs containing 50 words each."""
    page = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(page)
    font = _load_font(28)
    heading_font = _load_font(42)
    margin = 80
    column_gap = 50
    column_width = (width - 2 * margin - column_gap) // 2
    draw.text((margin, 35), "Lorem ipsum", fill="black", font=heading_font)

    lorem_words = (
        "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
        "tempor incididunt ut labore et dolore magna aliqua Ut enim ad minim "
        "veniam quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea "
        "commodo consequat Duis aute irure dolor in reprehenderit voluptate "
        "velit esse cillum dolore eu fugiat nulla pariatur Excepteur sint occaecat "
        "cupidatat non proident sunt culpa qui officia deserunt mollit anim id est laborum"
    ).split()
    words = (lorem_words * math.ceil(50 / len(lorem_words)))[:50]
    paragraph = " ".join(words) + "."
    lines = _wrap_text(draw, paragraph, font, column_width)

    x_positions = (margin, margin + column_width + column_gap)
    y = 120
    column = 0
    line_height = 38
    paragraph_height = len(lines) * line_height + 28
    for _ in range(paragraphs):
        if y + paragraph_height > height - margin and column == 0:
            column = 1
            y = 120
        for line in lines:
            draw.text((x_positions[column], y), line, fill="black", font=font)
            y += line_height
        y += 28
    return page

def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines: list[str] = []
    line = ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= width:
            line = candidate
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()
