# CLI reference

From the source checkout, run the CLI with `python -m src` (or `.venv/bin/python -m src` in the project virtual environment).

```sh
python -m src --help
```

The CLI uses droplets when you do not set `--page-warp`. The default output path is `distorted.png`. The CLI makes a Lorem ipsum page when you do not set `--input`.

## Input and output options

| Option | Default | Use |
| --- | --- | --- |
| `--input PATH` | Lorem ipsum page | Read an image file. The CLI converts it to RGB. |
| `--output PATH` | `distorted.png` | Save the output image file. |
| `--page-warp NAME` | Droplets | Select `horizontal_wave`, `vertical_wave`, `gentle_pinch`, `sine_lens`, or `refractive_index`. |
| `--intensity NUMBER` | 12 pixels for geometric modes | Set geometric displacement in pixels. Do not use it for refraction. |
| `--wavelength PIXELS` | Legacy page-relative period | Set a finite, positive source-pixel wave period; requires `horizontal_wave` or `vertical_wave`. |
| `--edge-taper` / `--no-edge-taper` | On | Enable or disable the geometric page-edge envelope. A nonzero pixel shift overrides `--edge-taper`. |
| `--shift-X PIXELS` | `0` | Translate a geometric page warp right by a finite number of pixels (negative moves left). `--shift-x` is an alias. |
| `--shift-Y PIXELS` | `0` | Translate a geometric page warp down by a finite number of pixels (negative moves up). `--shift-y` is an alias. |

The Python `WarpTransform` uses 3 pixels by default for geometric page modes. The CLI uses 12 pixels by default. Set `--intensity` to make a CLI result match an explicit Python setting.

## Droplet options

These options set the droplet field when you omit `--page-warp`.

| Option | Default | Use |
| --- | --- | --- |
| `--centres COUNT` | `3` | Set the number of generated centers. |
| `--distribution NAME` | `random` | Select `random`, `square`, or `triangle`. |
| `--radius PIXELS` | `80` | Set the droplet radius. |
| `--seed INTEGER` | `7` | Set the random seed. |
| `--profile NAME` | `cosine` | Select `cosine` or `parabolic`. |
| `--nonoverlapping-page` | Off | Fill the page with separate droplets. This option overrides `--centres`. |

The parabolic profile rejects overlapping droplets. Use `--nonoverlapping-page` when you need disjoint droplets across the page. The droplet diameter must fit inside the image.

```sh
python -m src --input source.png --output droplets.png \
  --centres 4 --radius 20 --intensity 5 --seed 7
```

```sh
python -m src --input source.png --output separate.png \
  --profile parabolic --radius 20 --nonoverlapping-page
```

Do not combine `--nonoverlapping-page` with `--page-warp`. The CLI rejects this combination.

## Page-warp options

Use `--page-warp NAME` to select a page function. `horizontal_wave` and `vertical_wave` accept `--wavelength PIXELS` as well as `--intensity`; `gentle_pinch` uses intensity, taper, and shifts. Horizontal waves move text left/right with a period along Y; vertical waves move it up/down with a period along X. Wavelength is a distance between repetitions in source pixels, not the number of cycles. Omitting it preserves the original three-cycle horizontal or two-cycle vertical page-wide wave.

```sh
python -m src --input source.png --output horizontal-wave.png \
  --page-warp horizontal_wave --wavelength 28 --intensity 20
python -m src --input source.png --output vertical-wave.png \
  --page-warp vertical_wave --wavelength 28 --intensity 20
```

`--wavelength` rejects zero, negative, and nonfinite values, and cannot be used with droplets, pinch, lens, or refraction. The CLI does not accept Python callbacks, so the combined `custom_wave` variants are Python-only; see the [Python API](API_REFERENCE.md).

Wave displacement is zero at every boundary pixel when tapering is enabled and both shifts are zero. `--no-edge-taper` removes the envelope. If either shift is nonzero, the wave becomes untapered even if `--edge-taper` is specified; the constant pixel translation is then added after intensity scaling. Shifts work even at `--intensity 0`. A displacement mask scales both wave and translation. These flags require a geometric `--page-warp`; they do not apply to droplets or physical refraction.

```sh
python -m src --input source.png --output taper.png \
  --page-warp horizontal_wave --wavelength 28 --intensity 20 --edge-taper
python -m src --input source.png --output shift-x.png \
  --page-warp horizontal_wave --wavelength 28 --intensity 20 --shift-X 18
python -m src --input source.png --output shift-y.png \
  --page-warp horizontal_wave --wavelength 28 --intensity 20 --shift-Y 12
python -m src --input source.png --output shift-xy.png \
  --page-warp horizontal_wave --wavelength 28 --intensity 20 --shift-X 18 --shift-Y 12
```

### Lens options

These options require `--page-warp sine_lens`.

| Option | Default | Use |
| --- | --- | --- |
| `--lens-a COUNT` | `2` | Set the column count. Use a positive integer. |
| `--lens-b COUNT` | `2` | Set the row count. Use a positive integer. |
| `--lens-amplitude-x GAIN` | `1` | Multiply X field displacement. A negative value reverses it. |
| `--lens-amplitude-y GAIN` | `1` | Multiply Y field displacement. A negative value reverses it. |
| `--lens-frequency-x VALUE` | Use `--lens-a` | Replace the X count with a positive frequency. |
| `--lens-frequency-y VALUE` | Use `--lens-b` | Replace the Y count with a positive frequency. |
| `--lens-phase-x CYCLES` | `0` | Shift the X field by a fraction of a full cycle. |
| `--lens-phase-y CYCLES` | `0` | Shift the Y field by a fraction of a full cycle. |

The gain values multiply the field before the global pixel intensity. A phase of `0.25` is one quarter of a full cycle. A fractional frequency or nonzero phase adds an edge taper by default; `--no-edge-taper` removes that extra envelope.

```sh
python -m src --input source.png --output lens.png \
  --page-warp sine_lens --intensity 20 \
  --lens-a 2 --lens-b 3 --lens-frequency-x 2.5
```

Do not use lens options with another mode. The CLI rejects nonfinite values and nonpositive frequencies. The lens counts must be positive integers.

### Grid options

These options require a page-warp mode.

| Option | Default | Use |
| --- | --- | --- |
| `--show-grid` | Off | Add a light-gray grid to the warped output. |
| `--show-original-grid PATH` | No reference file | Save the original image with a straight grid at `PATH`. |
| `--grid-spacing PIXELS` | `40` | Set the source-pixel distance between grid lines. |

```sh
python -m src --input source.png --output wave-grid.png \
  --page-warp vertical_wave --intensity 5 \
  --show-grid --show-original-grid source-grid.png --grid-spacing 40
```

Use a positive integer for spacing. Use different paths for `--output` and `--show-original-grid`. The original-grid option writes its file even if you do not set `--show-grid`.

## Refraction options

These options require `--page-warp refractive_index`. They do not apply to geometric lens or droplet modes.

| Option | Default | Use |
| --- | --- | --- |
| `--refraction-thickness-mm VALUE` | `5` | Set the slab thickness in millimeters. |
| `--refraction-n-x VALUE` | `1.3` | Set the X principal index. |
| `--refraction-n-y VALUE` | `2.5` | Set the Y principal index. |
| `--refraction-n-z VALUE` | Same as X index | Set the Z principal index. |
| `--refraction-polarization-deg VALUE` | `45` | Set the polarization angle in degrees. |
| `--refraction-direction X Y Z` | `-1 1 1` | Set the air-side direction toward the observer. |
| `--page-width-mm VALUE` | `210` | Set the physical width of the raster. |
| `--page-height-mm VALUE` | `297` | Set the physical height of the raster. |
| `--refraction-matrices-mm` | Off | Print two affine matrices as JSON and do not save an image. |

The thickness must be nonnegative and finite. The indices and page dimensions must be positive and finite. The direction must have three finite values and a positive Z value.

```sh
python -m src --input source.png --output refraction.png \
  --page-warp refractive_index \
  --refraction-thickness-mm 5 \
  --refraction-n-x 1.3 --refraction-n-y 2.5 \
  --page-width-mm 210 --page-height-mm 297
```

This command prints matrices in physical millimeters. It does not need an input image or a page size:

```sh
python -m src --page-warp refractive_index --refraction-matrices-mm
```

The JSON object has `unit` and `matrices_mm` keys. It has two 3-by-3 matrices. The translation values in these matrices are millimeters, not pixels.

A uniform slab does not make a spatially varying warp at normal incidence. The model uses approximate oblique polarization weights. Do not use `--intensity` to change the refraction result.

## Command errors

The CLI rejects unsupported combinations before it writes an output file. It also rejects invalid wavelength values, lens frequencies, lens counts, grid spacing, and several physical values. An absent input file or an output path error can produce a file-system exception.

Use `--help` to see the option names in the installed version. Use the [Python API](API_REFERENCE.md) if you need a custom page function.
