# User guide

This guide gives procedures for Python code and CLI commands. For exact parameter names, see the [Python API](API_REFERENCE.md) and [CLI reference](CLI_REFERENCE.md).

## Image format

The Python API takes one image tensor in CHW order: `(channels, height, width)`. All dimensions must have a nonzero size. A batch in NCHW order is not supported.

Use float pixel values from 0 through 1. Integer tensors are divided by 255. The functions clamp float values outside this range. They return float tensors on the input device.

The CLI opens an image file with Pillow. It converts the image to RGB. It saves an RGB image file at the output path. Select a file extension that Pillow can write.

## Install and check the package

1. Open a terminal in the package directory.
2. Install the package in your Python environment:

   ```sh
   python -m pip install -e .
   ```

3. From the source checkout, confirm the module CLI:

   ```sh
   python -m src --help
   ```

The package requires Python 3.10 or later. It also requires PyTorch, NumPy, and Pillow. The installation command installs these declared dependencies.

## Apply a page warp in Python

1. Import the transform.
2. Make or load one CHW image tensor.
3. Select a mode and an intensity.
4. Call the transform with the image.

```python
import torch
from src import WarpTransform

image = torch.rand(3, 256, 256)
transform = WarpTransform(mode="horizontal_wave", wavelength=28, intensity=20)
warped = transform(image)
assert warped.shape == image.shape
```

The result stays on the input device. Float image gradients can flow through geometric page warps. A droplet transform can also retain center gradients away from its support boundary.

`horizontal_wave` moves text left/right and repeats along image Y; `vertical_wave` moves text up/down and repeats along image X. The optional `wavelength` is the period in source-image pixels, independent of the pixel-displacement `intensity`. Use `WarpTransform(mode="vertical_wave", wavelength=28, intensity=20)` for the other direction. Omitting wavelength preserves the original three-cycle horizontal or two-cycle vertical page-wide wave.

### Compare wave wavelengths and intensities

If the source text has a 28px font, periods at 50%, 100%, 150%, 200%, and 250% of that size are 14, 28, 42, 56, and 70px. Compare each period against independent displacement intensities of 10, 20, 30, 40, and 50px. From the source checkout, generate labeled 5×5 grids and full-resolution images for both built-in directions:

```sh
python -m src.wave_samples --input original.png --font-size-px 28
```

Pass `--output-dir PATH` to choose a destination. The generator writes `manifest.json` with each sample's source-pixel wavelength and intensity. Use the actual font size of your own input for `--font-size-px`.

### Combine waves with a Python callback

`custom_wave` moves pixels on both axes and is available through the Python API only. One component can have a fixed quarter-page period, or both can use the same source-pixel wavelength:

```python
from functools import partial
from src import WarpTransform, custom_wave

for variant in ("vertical", "horizontal", "symmetric"):
    callback = partial(custom_wave, wavelength=28, fixed_axis=variant)
    warped = WarpTransform(mode="custom", intensity=20, warp_function=callback)(image)
```

With `fixed_axis="vertical"`, the up/down component repeats every 25% of image width along X, while the left/right component uses `wavelength` along Y. With `fixed_axis="horizontal"`, the left/right component repeats every 25% of image height along Y, while the up/down component uses `wavelength` along X. `symmetric` uses the same pixel period along both axes. To generate all three 5×5 comparison grids, run `python -m src.wave_samples --custom-wave --input original.png --font-size-px 28`. There is no `--page-warp custom_wave` option; the top-level CLI cannot accept a callback.

## Apply a droplet warp in Python

The default mode is `droplet`. Use `DistortionConfig` to set the droplet positions and shape.

```python
from src import DistortionConfig, WarpTransform

config = DistortionConfig(
    centre_count=4,
    distribution="random",
    radius=20,
    intensity=5,
    seed=7,
    profile="cosine",
)
transform = WarpTransform(config=config)
warped = transform(image)
```

The `distribution` field also accepts a `CentreDistribution` value. An explicit `WarpTransform(intensity=8, config=config)` replaces `config.intensity` for that transform call.

The `parabolic` profile does not permit droplet overlap. Use separated centers or a nonoverlapping center generator. The `cosine` profile adds displacement where droplets overlap.

## Apply a lens warp in Python

The `sine_lens` mode is a geometric warp. `A` sets the column count. `B` sets the row count. The pixel intensity is a separate value.

```python
lens = WarpTransform(
    mode="sine_lens",
    intensity=20,
    lens_options={"A": 2, "B": 3, "frequency_x": 2.5},
)
warped_lens = lens(image)
```

A frequency value replaces the count for its axis. A phase value of `0.25` moves that axis by one quarter of a full cycle. `amplitude_x` and `amplitude_y` multiply the related field components.

High intensity can duplicate letters or move them into a border. Check the output image before you use a high-intensity setting.

## Apply the refraction model in Python

The `refractive_index` mode models a uniform, plane-parallel anisotropic slab. The mode has two polarization views. It combines their sampled images with approximate power weights.

```python
physical = WarpTransform(
    mode="refractive_index",
    refraction_options={
        "thickness_mm": 5,
        "n_x": 1.3,
        "n_y": 2.5,
        "polarization_deg": 45,
        "page_width_mm": 210,
        "page_height_mm": 297,
    },
)
refracted = physical(image)
```

The default view direction is `(-1, 1, 1)` from the image toward the observer. At normal incidence, use `direction_xyz=(0, 0, 1)`. A uniform slab cannot make a spatially varying warp at normal incidence.

The optical offset has a physical unit of millimeters. The code uses the page width and height to convert that offset to pixels. It uses a separate scale for each axis. `WarpTransform.intensity` is not available in this mode.

The model does not calculate a spatial index distribution. The mode weights use an approximation for oblique incidence. Do not treat the output as an exact prediction of ghost brightness. An observer that accepts only normal rays does not collect the oblique output ray.

## Use a custom page function

A custom function takes two normalized coordinate tensors, `y` and `x`. Each tensor has shape `(height, width)`. Each coordinate goes from 0 through 1.

The function returns two tensors in `(dy, dx)` order. Each output tensor must have the same shape and device as the input coordinates. Each value must be finite.

```python
import torch
from src import WarpTransform


def shift_right(y, x):
    return torch.zeros_like(y), torch.ones_like(x)

custom = WarpTransform(
    mode="custom",
    intensity=2,
    warp_function=shift_right,
)
shifted = custom(image)
```

A positive `dx` moves visible content to the right. The sampler reads an input pixel to the left of each output pixel. The sampler uses bilinear interpolation and repeats border pixels.

## Apply a page warp with the CLI

1. Put an image file at `source.png`.
2. Run this command:

   ```sh
   python -m src --input source.png --output wave.png \
     --page-warp horizontal_wave --wavelength 28 --intensity 20
   ```

3. Open `wave.png` to check the result. For up/down movement instead, choose `--page-warp vertical_wave` with the same wavelength and intensity flags.

Omit `--input` to create a Lorem ipsum page. Its default size is 1200 by 1600 pixels. The default output path is `distorted.png` in the current directory.

## Add a diagnostic grid

The grid helps you see the movement of the image. The grid lines use the same geometric field as the image.

```sh
python -m src --input source.png --output wave-grid.png \
  --page-warp vertical_wave --intensity 5 --show-grid \
  --show-original-grid source-grid.png --grid-spacing 40
```

The CLI writes a second file at `source-grid.png`. That file shows the straight grid on the original image. Give it a path different from `--output`.

For refraction, the CLI puts the grid on the source before it samples each polarization view. This keeps the grid and the image in the same optical path.

## Select a mode

| Mode | Function | Key controls |
| --- | --- | --- |
| `droplet` | Moves pixels around selected centers. | `centre_count`, `radius`, `profile`, `intensity` |
| `horizontal_wave` | Moves pixels left and right, varying along Y. | `wavelength` in source pixels (optional), `intensity` |
| `vertical_wave` | Moves pixels up and down, varying along X. | `wavelength` in source pixels (optional), `intensity` |
| `gentle_pinch` | Moves pixels toward the page center. | `intensity` |
| `sine_lens` | Makes a repeated geometric lens field. | lens counts, gains, frequencies, phases, `intensity` |
| `refractive_index` | Samples two optical views through a uniform slab. | thickness, indices, direction, page size, polarization |
| `custom` | Uses a Python coordinate function, including `custom_wave`. | function, `intensity`; callback-specific `wavelength` and `fixed_axis` |

The CLI uses droplets if you omit `--page-warp`. The CLI cannot take a Python custom function.

## Existing image examples

The package directory has four example manifests:

| Manifest | Content |
| --- | --- |
| `manifest.json` | Earlier droplet examples and their center coordinates. |
| `functional_page/manifest.json` | Page warps, grids, lens settings, and the refraction view. |
| `full_page/manifest.json` | Dense droplet fields with overlaps. |
| `nonoverlapping/parabolic/manifest.json` | Separate parabolic droplets and their center files. |

Read a manifest before you compare example images. The manifest gives the settings and relative image paths. Some files come from earlier workflows. They are not installed with the Python package.

The scripts `generate_page_warp_examples.py` and `generate_distortion_examples.py` are in the parent directory. Run them only if you want to make the example images again. The page generator writes to `functional_page/` by default. The droplet generator writes to `nonoverlapping/` by default.

## Resolve common errors

- If you see a CHW shape error, remove the batch dimension or process each image separately.
- If a custom function fails, check the field shape, device, type, and finite values.
- If parabolic droplets overlap, use `make_nonoverlapping_centres` for the image size.
- If a page warp does not move pixels, increase geometric intensity or inspect the source image.
- If a wave period is rejected, use a finite positive pixel value and select `horizontal_wave` or `vertical_wave`; pass a custom-wave period into its Python callback instead.
- If the optical output has no local distortion, check the slab geometry and the viewing direction.
- If the CLI rejects lens flags, select `--page-warp sine_lens`.
- If the CLI rejects refraction flags, select `--page-warp refractive_index`.
- If the CLI rejects a grid flag, select a page-warp mode.

Do not use `--intensity` with the refraction mode. Change physical inputs only when they describe the real slab and page.
