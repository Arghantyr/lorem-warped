"""Compatibility exports; implementation lives in focused modules."""
from .tensor import _as_float_chw, _load_image, _save_image, _validate_centres, _pixel_coordinates_to_grid, warp_image, image_to_tensor  # noqa: F401
from .droplets import CentreDistribution, DistortionConfig, distort_image, displacement_field, make_centres, _grid_centres, make_page_centres, parabolic_profile, validate_nonoverlap, make_nonoverlapping_centres  # noqa: F401
from .page import page_warp_field, warp_page, _page_envelope, horizontal_wave, vertical_wave, gentle_pinch, sine_lens, overlay_warped_grid, PAGE_WARPS  # noqa: F401
from .optics import _validate_refraction, refraction_distortion_matrices, refractive_index_field, refract_image  # noqa: F401
from .demo import render_lorem_page, _wrap_text, _load_font  # noqa: F401
