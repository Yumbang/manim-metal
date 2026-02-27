"""manim-metal: Metal renderer add-on for Manim Community Edition.

Importing this module patches manim to support ``config.renderer = "metal"``.
"""

from manim_metal.patch import apply_patches

apply_patches()

__version__ = "0.1.0"
