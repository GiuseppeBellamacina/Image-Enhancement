"""
Degradation methods for images.
"""

from .gaussian_noise import add_gaussian_noise

# from .motion_blur import add_motion_blur
# from .jpeg_compression import jpeg_compress
# from .salt_and_pepper import add_salt_and_pepper

__all__ = [
    "add_gaussian_noise",
    # 'add_motion_blur',
    # 'jpeg_compress',
    # 'add_salt_and_pepper'
]
