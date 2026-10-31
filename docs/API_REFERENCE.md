# Python API reference

This reference describes the names exported by `src` when running from the source checkout. All image functions take one CHW tensor. A field has shape `(height, width, 2)` and stores `(dy, dx)` values.

## Input and output contract

- A float input is clamped to `[0, 1]`. An integer input is converted to float and divided by 255.
- An image must have three nonempty dimensions: channels, height, and width.
- An image result has the same CHW shape as the input. The result uses the input device.
- A geometric warp samples the input with bilinear interpolation. It repeats border pixels outside the image.
- A positive `dy` moves visible content down. A positive `dx` moves visible content right.
- Do not pass an NCHW batch to these functions. Call a transform once for each image.

## `WarpTransform`

```python
WarpTransform(
    mode="droplet",
    intensity=None,
    wavelength=None,
    edge_taper=True,
    shift_X=0.0,
    shift_Y=0.0,
    config=DistortionConfig(),
    lens_options={},
    refraction_options={},
    warp_function=None,
)
```

`WarpTransform` is a callable object. Call it as `transform(image)`. It returns a warped float tensor. It does not change the input tensor in place.

| Argument | Use |
| --- | --- |
| `mode` | Selects `droplet`, one page mode, `refractive_index`, or `custom`. |
| `intensity` | Sets geometric displacement in pixels. `None` uses the mode default. |
| `wavelength` | Positive, finite source-pixel period for `horizontal_wave` or `vertical_wave` only. `None` retains the original page-relative wave. |
| `edge_taper` | Boolean, default `True`. Tapers built-in geometric page warps at the image boundary. `False` removes the page-edge envelope. A nonzero shift overrides `True` and disables tapering. On a custom callback, `False` requires that the callback accept `edge_taper`. |
| `shift_X`, `shift_Y` | Finite pixel translations, default `0.0`. Positive X moves right; positive Y moves down. Applied after multiplying the wave field by intensity, so shifts still work with zero intensity. Any nonzero shift disables tapering for built-in page warps and `custom_wave`. Not available for droplets or physical refraction. |
| `config` | Supplies the settings for `droplet`. |
| `lens_options` | Supplies keyword arguments to `sine_lens`. Use only with `sine_lens`. |
| `refraction_options` | Supplies keyword arguments to `refract_image`. Use only with `refractive_index`. |
| `warp_function` | Supplies a coordinate function. Use only with `custom`. |

For droplets, the default intensity is `config.intensity`, which starts at 12 pixels. An explicit transform intensity replaces it. For geometric page modes, the default transform intensity is 3 pixels.

The refraction mode does not accept a transform intensity. It uses thickness and page size to set the image displacement. `WarpTransform` checks its mode and option groups during construction. Some option values are checked when the transform runs.

For example, `WarpTransform(mode="horizontal_wave", wavelength=28, intensity=20)(image)` moves text left/right, repeating every 28 source pixels along Y. Use `vertical_wave` for up/down movement repeating along X. `wavelength` is not an intensity and cannot be set on other modes, including `custom`; use a configured callback for custom waves instead.

## Droplet settings

```python
DistortionConfig(
    centre_count=3,
    distribution=CentreDistribution.RANDOM,
    radius=80.0,
    intensity=12.0,
    seed=7,
    profile="cosine",
)
```

| Field | Meaning | Limit |
| --- | --- | --- |
| `centre_count` | Number of centers made when you do not supply centers. | Nonnegative integer. |
| `distribution` | Center layout. | `RANDOM`, `SQUARE`, or `TRIANGLE`. |
| `radius` | Droplet radius in pixels. | Finite and greater than zero. |
| `intensity` | Droplet displacement multiplier in pixels. | Finite and nonnegative. |
| `seed` | Seed for random centers. | Integer or `None`. |
| `profile` | Droplet field formula. | `cosine` or `parabolic`. |

`CentreDistribution` has three values: `random`, `square`, and `triangle`. Use `DistortionConfig.validate()` to check its supported settings. `profile="parabolic"` rejects overlapping centers. `profile="cosine"` adds overlapping fields.

### `distort_image(image, config, centres=None)`

This function applies a droplet warp. `centres` is an optional `(N, 2)` tensor in `(y, x)` order. Each center must be inside the image. The result is a float CHW tensor.

```python
import torch
from src import DistortionConfig, distort_image

image = torch.rand(3, 64, 96)
centres = torch.tensor([[20.0, 30.0], [40.0, 70.0]])
config = DistortionConfig(radius=8, intensity=3)
warped = distort_image(image, config, centres)
```

The function makes centers when `centres` is `None`. Zero centers or zero intensity returns a copy of the normalized input.

### Center and profile functions

| Function | Result and main inputs |
| --- | --- |
| `make_centres(config, height, width, device)` | Makes `config.centre_count` `(y, x)` centers. |
| `make_page_centres(height, width, radius, distribution, seed=7, device="cpu")` | Fills a page at approximately one-radius spacing. Centers can overlap. |
| `make_nonoverlapping_centres(height, width, radius, distribution, seed=7, device="cpu")` | Makes disjoint droplets that fit inside the image. |
| `validate_nonoverlap(centres, radius)` | Raises `ValueError` when two droplet interiors overlap. |
| `parabolic_profile(distance, amplitude, radius)` | Returns `max(amplitude × (1 − distance² / radius²), 0)`. |
| `displacement_field(image, radius, centres, profile="cosine")` | Makes a unit-amplitude `(dy, dx)` field. |

The nonoverlapping generator requires a diameter that fits inside the page. Adjacent droplets may touch. The page generator can make centers that overlap, so do not use it directly for a parabolic field.

The cosine field has magnitude `1 + cos(2π × distance / radius)` inside each radius. Its magnitude is zero outside each radius. The direction is radial, and the exact center stays fixed. `displacement_field` can be reused with different intensities.

## Page-warp functions

### `page_warp_field(image, warp_function)`

This function evaluates a coordinate function at every output pixel. It returns an `(H, W, 2)` field. The coordinate function receives normalized `(y, x)` image coordinates.

### `warp_page(image, warp_function, intensity=3.0, mask=None, *, shift_X=0.0, shift_Y=0.0)`

This function makes a page field and applies it to the image. Its intensity is nonnegative and finite. The field components set the direction; intensity sets the displacement scale in pixels.

For supported page functions, a nonzero pixel shift overrides an explicitly bound `edge_taper=True`: the wave envelope is disabled before the constant translation is added. A mask, if supplied, scales both the warp and the translation. An arbitrary custom callback without an `edge_taper` argument retains its own field shape.

### `warp_image(image, field, intensity)`

This lower-level function applies a field with backward bilinear sampling. It multiplies the field by intensity. Supply a normalized float CHW image and a compatible field.

`warp_image` does not check every field or intensity value. Use `warp_page`, `distort_image`, or a validated field when you need input checks. A zero intensity returns a copy of the input image.

### Built-in coordinate functions

`PAGE_WARPS` maps four mode names to their coordinate functions:

| Name | Field direction |
| --- | --- |
| `horizontal_wave(y, x, wavelength=None, *, edge_taper=True)` | Moves image content left or right; period varies along Y. |
| `vertical_wave(y, x, wavelength=None, *, edge_taper=True)` | Moves image content up or down; period varies along X. |
| `gentle_pinch(y, x, *, edge_taper=True)` | Moves image content toward the page center. |
| `sine_lens(y, x, ..., edge_taper=True)` | Makes a repeated geometric lens field. |

The built-in functions return `(dy, dx)` tensors. With `edge_taper=True` and no shifts, waves have exactly zero displacement on every boundary pixel. Use `edge_taper=False` to remove their page-edge envelope. These coordinate functions return unit fields; pixel-valued shifts belong to `warp_page` or `WarpTransform`, not the coordinate-function arguments.

For the built-in waves, `wavelength` is a positive, finite number of **source-image pixels**, not a cycle count or fraction of the page. Omitting it preserves the legacy three-cycle horizontal or two-cycle vertical wave across the whole image. Pass a period to `WarpTransform`, or bind it when calling `warp_page` or `page_warp_field` directly:

```python
from functools import partial
import torch
from src import horizontal_wave, vertical_wave, warp_page

image = torch.rand(3, 256, 256)
horizontal = warp_page(image, partial(horizontal_wave, wavelength=28), intensity=20)
vertical = warp_page(image, partial(vertical_wave, wavelength=28), intensity=20)
shifted = warp_page(image, partial(horizontal_wave, wavelength=28, edge_taper=True),
                    intensity=20, shift_X=18, shift_Y=12)  # disables taper
```

### `custom_wave(y, x, *, wavelength, fixed_axis="vertical", edge_taper=True)`

This Python-only coordinate function combines horizontal and vertical waves and returns `(dy, dx)`. The period is a positive, finite source-pixel value; both components have a page-edge taper by default, which `edge_taper=False` disables. `fixed_axis="vertical"` holds the vertical-wave period at 25% of image width (along X) and applies `wavelength` to the horizontal wave (along Y). `fixed_axis="horizontal"` holds the horizontal-wave period at 25% of image height (along Y) and applies `wavelength` to the vertical wave (along X). `fixed_axis="symmetric"` uses `wavelength` on both X and Y, with no fixed quarter-page period.

```python
from functools import partial
from src import WarpTransform, custom_wave

wave = partial(custom_wave, wavelength=28, fixed_axis="symmetric")
warped = WarpTransform(mode="custom", intensity=20, warp_function=wave)(image)
```

`custom_wave` is a callback, not a `PAGE_WARPS` mode. There is no `--page-warp custom_wave` or CLI custom-function option. The transform's `wavelength` argument is for the two built-in wave modes only.

### `sine_lens` options

```python
sine_lens(
    y, x,
    A=2, B=2,
    phase_x=0.0, phase_y=0.0,
    *, amplitude_x=1.0, amplitude_y=1.0,
    frequency_x=None, frequency_y=None,
    edge_taper=True,
)
```

`A` sets the column count. `B` sets the row count. Each count must be a positive integer. A frequency value replaces its related count. Frequencies must be positive, finite numbers.

Each axis amplitude is a signed field gain. Zero turns off that field component. Phase is a fraction of a full cycle. A phase value of `0.25` is a quarter-cycle shift. `sine_lens` keeps the old zero-phase, integer-frequency field shape.

A nonzero phase or a fractional frequency adds a page-edge taper by default. `edge_taper=False` disables this extra envelope; the underlying lens cell envelope remains. Intensity is not a `sine_lens` argument. Pass it to `warp_page` or `WarpTransform`.

### `overlay_warped_grid(image, field=None, intensity=0.0, spacing=40, opacity=0.45)`

This function adds a light-gray grid to the image. If `field` is present, it samples the grid with the same field and intensity. The spacing unit is source pixels.

`spacing` must be a positive integer. `opacity` must be in `[0, 1]`. The field must be finite and have shape `(H, W, 2)` on the image device.

## Refraction functions

The refraction functions use a uniform, plane-parallel anisotropic slab. Their default page size is 210 mm by 297 mm. Their default direction is `(-1, 1, 1)` toward the observer.

The index values `n_x`, `n_y`, and `n_z` describe three principal axes. `n_z=None` uses `n_x`. The thickness must be finite and nonnegative. Each index and each page dimension must be finite and positive.

The slab has no spatial index variation. At normal incidence, it does not make a spatially varying warp. At oblique incidence, each polarization mode can have a different constant translation.

### `refraction_distortion_matrices(...)`

```python
refraction_distortion_matrices(
    thickness_mm=5.0,
    n_x=1.3,
    n_y=2.5,
    n_z=None,
    direction_xyz=(-1.0, 1.0, 1.0),
)
```

The function returns two affine matrices with shape `(2, 3, 3)`. Their translation values use millimeters, not pixels. Mode 0 is more X-polarized. Mode 1 is more Y-polarized.

The direction must have three finite values. Its Z value must be positive. A uniform slab makes identity 2-by-2 blocks, not a spatially varying field.

### `refractive_index_field(image, *, mode="x", ...)`

This function converts one mode translation to an `(H, W, 2)` pixel field. `mode` must be `x` or `y`. The function uses `width / page_width_mm` and `height / page_height_mm` separately.

The field uses `(dy, dx)` component order. Each pixel has the same two values. The page dimensions describe the physical size of the full raster.

### `refract_image(image, *, polarization_deg=45.0, ...)`

This function samples the X and Y modes separately. It combines their images with polarization weights. It also applies an approximate transmission factor for each mode.

`refract_image` accepts the slab parameters, page dimensions, direction, `show_grid=False`, and `grid_spacing=40`. A zero thickness returns a copy of the normalized input. A finite polarization angle is required.

The positions follow the uniform-slab modal ray calculation. The weights approximate oblique interface behavior. The function does not calculate coherent interference or absorption. Do not use the image as an exact prediction of brightness.

## Image conversion and demonstration

| Function | Result |
| --- | --- |
| `image_to_tensor(pillow_image)` | Converts a Pillow image to an RGB float CHW tensor in `[0, 1]`. |
| `render_lorem_page(width=1200, height=1600, paragraphs=6)` | Returns a two-column RGB Pillow page. |

## Errors and limits

Most configuration errors raise `ValueError`. The function can raise another Python exception for a wrong object type or unsupported operation. In particular, `warp_image` is a low-level helper with fewer checks.

The package does not provide a batch transform, an exact optical brightness model, or a CLI option for a custom Python function. It does not put gradients through integer conversion. Check the output when a high intensity can make text hard to read.
