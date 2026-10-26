"""Render localized Faker paragraphs into an indexed image/text dataset."""
from __future__ import annotations

import hashlib
import io
import json
import math
import random
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

from faker import Faker
from faker.config import AVAILABLE_LOCALES
from PIL import Image, ImageDraw, ImageFont

DEFAULT_LOCALES = ("en_US", "ja_JP", "fr_FR", "es_ES", "ru_RU", "ar_AA", "de_DE", "el_GR")


def _locale_words(locale: str, count: int, seed: int | None) -> list[str]:
    faker = Faker(locale)
    if seed is not None:
        faker.seed_instance(seed)
    # Faker items can contain spaces or repeat; build a unique token vocabulary.
    unique: dict[str, None] = {}
    for _ in range(30):
        for item in faker.words(nb=max(count, 2 * (count - len(unique)))):
            for token in item.split():
                unique[token] = None
        if len(unique) >= count:
            return list(unique)[:count]
    raise ValueError(f"could not obtain {count} unique words for {locale}")


def _mixed_paragraphs(
    pools: dict[str, list[str]], total_words: int, max_words: int,
    max_chars: int | None, rng: random.Random,
) -> list[list[tuple[str, str]]]:
    locale_count = len(pools)
    if max_words < locale_count:
        raise ValueError("words_per_paragraph must allow at least one word from each locale")
    shortest = {locale: min(map(len, words)) for locale, words in pools.items()}
    if max_chars is not None and sum(shortest.values()) + locale_count - 1 > max_chars:
        raise ValueError("total_characters_per_paragraph is too small to include every locale")
    available = {locale: words.copy() for locale, words in pools.items()}
    paragraphs: list[list[tuple[str, str]]] = []
    remaining = total_words
    while remaining:
        paragraph: list[tuple[str, str]] = []
        counts = dict.fromkeys(pools, 0)
        length = 0
        target = math.ceil(remaining / math.ceil(remaining / max_words))
        while remaining and len(paragraph) < target:
            # Leave enough words to make every remaining paragraph multilingual.
            if max_chars is not None and len(paragraph) >= locale_count and remaining <= 2 * locale_count:
                break
            room = None if max_chars is None else max_chars - length - bool(paragraph)
            candidates = {
                locale: [word for word in available[locale] if room is None or (
                    len(word) <= room and
                    length + len(word) + bool(paragraph) +
                    sum(shortest[other] + 1 for other in pools if other != locale and not counts[other])
                    <= (max_chars if max_chars is not None else 0)
                )]
                for locale in pools
            }
            eligible = [locale for locale, words in candidates.items() if words]
            if not eligible:
                break
            missing = [locale for locale in eligible if not counts[locale]]
            if missing and target - len(paragraph) <= len(missing):
                eligible = missing
            least = min(counts[locale] for locale in eligible)
            locale = rng.choice([locale for locale in eligible if counts[locale] == least])
            word = rng.choice(candidates[locale])
            available[locale].remove(word)
            if not available[locale]:
                available[locale] = pools[locale].copy()
            paragraph.append((locale, word))
            counts[locale] += len(word)
            length += len(word) + (len(paragraph) > 1)
            remaining -= 1
        if len(paragraph) < locale_count:
            raise ValueError("paragraph limits cannot fit a word from each locale")
        paragraphs.append(paragraph)
    return paragraphs


def _font(locale: str, size: int) -> ImageFont.FreeTypeFont:
    language = locale.split("_")[0]
    try:
        result = subprocess.run(
            ["fc-match", "-f", "%{file}", f":lang={language}"],
            capture_output=True, text=True, check=True,
        )
        candidates = (result.stdout, "DejaVuSans.ttf")
    except (OSError, subprocess.CalledProcessError):
        candidates = ("DejaVuSans.ttf",)
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    raise RuntimeError("No usable TrueType font found; install a Unicode font for image rendering")


def _render(words: list[tuple[str, str]], font_size: float, dpi: int) -> bytes:
    pixel_size = max(1, round(font_size * dpi / 72))
    fonts = {locale: _font(locale, pixel_size) for locale, _ in words}
    width = round(210 * dpi / 25.4)  # A4 width, with a height fitted to this paragraph.
    margin = max(16, round(12 * dpi / 25.4))
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    width = max(width, math.ceil(max(probe.textlength(word, font=fonts[locale])
                                     for locale, word in words)) + 2 * margin)
    available = width - 2 * margin
    lines: list[list[tuple[str, str]]] = []
    line: list[tuple[str, str]] = []
    line_width = 0.0
    space = probe.textlength(" ", font=next(iter(fonts.values())))
    for locale, word in words:
        word_width = probe.textlength(word, font=fonts[locale])
        if line and line_width + space + word_width > available:
            lines.append(line)
            line, line_width = [], 0.0
        line.append((locale, word))
        line_width += word_width + (space if len(line) > 1 else 0)
    if line:
        lines.append(line)
    step = max(1, round(pixel_size * 1.5))
    image = Image.new("RGB", (width, 2 * margin + len(lines) * step), "white")
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(lines):
        x = float(margin)
        for locale, word in line:
            font = fonts[locale]
            draw.text((x, margin + index * step), word, font=font, fill="black")
            x += draw.textlength(word, font=font) + space
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=95, dpi=(dpi, dpi))
    return stream.getvalue()


def generate_dataset(
    locales: Sequence[str] = DEFAULT_LOCALES,
    words_per_locale: int = 100,
    words_per_paragraph: int = 100,
    total_characters_per_paragraph: int | None = None,
    font_size: float = 12,
    dpi: int = 300,
    output_root: str | Path = ".",
    seed: int | None = None,
) -> Path:
    """Generate paragraph images, paired UTF-8 text and a SHA-1 registry.

    ``total_characters_per_paragraph`` optionally limits paragraph length,
    including separating spaces; words are never split. Each locale supplies
    ``words_per_locale`` unique vocabulary words; paragraphs randomly sample
    that vocabulary and balance each locale's character contribution. The next unused ``dataset-N`` directory
    is created under ``output_root``. Font size is in points, dpi in pixels/inch.
    Supply ``seed`` to reproduce the words without changing Faker's global RNG.
    """
    if isinstance(locales, (str, bytes)) or not locales or any(
        not isinstance(locale, str) or locale not in AVAILABLE_LOCALES for locale in locales
    ):
        raise ValueError("locales must be a nonempty list of supported Faker locale codes")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in (words_per_locale, words_per_paragraph, dpi)):
        raise ValueError("words_per_locale, words_per_paragraph and dpi must be positive integers")
    if total_characters_per_paragraph is not None and (
        isinstance(total_characters_per_paragraph, bool)
        or not isinstance(total_characters_per_paragraph, int)
        or total_characters_per_paragraph <= 0
    ):
        raise ValueError("total_characters_per_paragraph must be a positive integer")
    if isinstance(font_size, bool) or not isinstance(font_size, (float, int)) or not math.isfinite(font_size) or font_size <= 0:
        raise ValueError("font_size must be a finite positive number of points")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError("seed must be an integer or None")
    if len(set(locales)) != len(locales):
        raise ValueError("locales must not contain duplicates")
    pools = {locale: _locale_words(locale, words_per_locale, seed) for locale in locales}
    content = _mixed_paragraphs(
        pools, len(locales) * words_per_locale, words_per_paragraph,
        total_characters_per_paragraph, random.Random(seed),
    )
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        output = root / f"dataset-{number}"
        try:
            output.mkdir()
            break
        except FileExistsError:
            number += 1
    try:
        registry: dict[str, dict[str, str]] = {}
        for words in content:
            text = " ".join(word for _, word in words)
            image = _render(words, font_size, dpi)
            digest = hashlib.sha1(image).hexdigest()
            if digest in registry:
                raise RuntimeError("two paragraphs rendered to the same image; dataset would lose a text pair")
            image_name = f"pimg-{digest}.jpg"
            text_name = f"ptxt-{digest}.txt"
            (output / image_name).write_bytes(image)
            (output / text_name).write_text(text, encoding="utf-8")
            registry[digest] = {"image": image_name, "text": text_name}
        (output / "registry.json").write_text(
            json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(output)
        raise
    return output
