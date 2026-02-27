"""MetalCamera — manages Metal textures as frame buffer and renders mobjects."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import Metal
import numpy as np
from manim._config import config
from manim.mobject.types.vectorized_mobject import VMobject
from manim.utils.color import color_to_int_rgba
from manim.utils.family import extract_mobject_family_members
from manim.utils.iterables import list_difference_update
from PIL import Image

from manim_metal.metal_context import MetalContext
from manim_metal.utils import (
    build_world_to_ndc_matrix,
    vmobject_to_stroke_quads,
    vmobject_to_triangles,
)

if TYPE_CHECKING:
    from manim.mobject.mobject import Mobject
    from manim.typing import PixelArray


class MetalCamera:
    """Minimal camera that renders mobjects via Metal.

    Mirrors the subset of :class:`manim.camera.camera.Camera` that
    :class:`CairoRenderer` actually calls.

    Parameters
    ----------
    pixel_width, pixel_height
        Render target dimensions in pixels.
    frame_width, frame_height
        Visible area in manim world units.
    frame_rate
        Frames per second.
    background_color, background_opacity
        Background color and opacity.
    """

    def __init__(
        self,
        pixel_width: int | None = None,
        pixel_height: int | None = None,
        frame_width: float | None = None,
        frame_height: float | None = None,
        frame_rate: float | None = None,
        background_color=None,
        background_opacity: float | None = None,
        use_z_index: bool = True,
        **kwargs: Any,
    ) -> None:
        from manim.utils.color import ManimColor

        self.pixel_width = pixel_width or config["pixel_width"]
        self.pixel_height = pixel_height or config["pixel_height"]
        self.frame_width = frame_width or config["frame_width"]
        self.frame_height = frame_height or config["frame_height"]
        self.frame_rate = frame_rate or config["frame_rate"]
        self.use_z_index = use_z_index
        self.image_mode = "RGBA"
        self.n_channels = 4
        self.pixel_array_dtype = "uint8"

        if background_color is None:
            self._background_color = ManimColor.parse(config["background_color"])
        else:
            self._background_color = ManimColor.parse(background_color)

        self._background_opacity = (
            background_opacity if background_opacity is not None else config["background_opacity"]
        )

        # Background RGBA as normalized floats
        bg_int = color_to_int_rgba(self._background_color, self._background_opacity)
        self._bg_color_float = tuple(c / 255.0 for c in bg_int)

        # Initialize Metal context
        self.ctx = MetalContext(self.pixel_width, self.pixel_height)

        # Initialize pixel array (RGBA uint8)
        self.pixel_array: PixelArray = np.zeros(
            (self.pixel_height, self.pixel_width, 4), dtype=np.uint8
        )
        # Fill with background color
        self.pixel_array[:, :] = color_to_int_rgba(self._background_color, self._background_opacity)
        self.background = self.pixel_array.copy()

        # Pre-compute MVP matrix
        self._mvp = build_world_to_ndc_matrix(self.frame_width, self.frame_height, 0.0, 0.0)

    @property
    def background_color(self):
        return self._background_color

    @background_color.setter
    def background_color(self, color):
        from manim.utils.color import ManimColor

        self._background_color = ManimColor.parse(color)
        bg_int = color_to_int_rgba(self._background_color, self._background_opacity)
        self._bg_color_float = tuple(c / 255.0 for c in bg_int)
        self.background[:, :] = bg_int

    # ------------------------------------------------------------------
    # CairoRenderer-facing interface
    # ------------------------------------------------------------------

    def reset(self) -> MetalCamera:
        """Clear render target to background color."""
        self.ctx.clear_render_target(self._bg_color_float)
        # Reset pixel_array to background
        self.pixel_array[:, :, :] = self.background[:, :, :]
        return self

    def set_frame_to_background(self, background: PixelArray) -> None:
        """Set the render target from a pre-rendered static background."""
        self.pixel_array[:, :, :] = background[:, :, :]
        # Upload the background to the Metal render target
        self._upload_pixel_array_to_texture()

    def capture_mobjects(
        self,
        mobjects: Iterable[Mobject],
        include_submobjects: bool = True,
        excluded_mobjects: list | None = None,
        **kwargs: Any,
    ) -> None:
        """Render mobjects into the Metal render target."""
        mobjects = self._get_mobjects_to_display(
            mobjects,
            include_submobjects=include_submobjects,
            excluded_mobjects=excluded_mobjects,
        )

        if not mobjects:
            self._readback_texture()
            return

        # Encode all VMobject draw calls in a single command buffer
        cmd = self.ctx.command_queue.commandBuffer()
        rpd = self.ctx.make_render_pass_descriptor(clear=False)
        encoder = cmd.renderCommandEncoderWithDescriptor_(rpd)
        encoder.setViewport_(
            (0.0, 0.0, float(self.pixel_width), float(self.pixel_height), 0.0, 1.0)
        )

        mvp_buffer = self.ctx.make_buffer(self._mvp)

        for mob in mobjects:
            if isinstance(mob, VMobject):
                self._encode_vmobject(encoder, mob, mvp_buffer)

        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        # Read back to pixel_array
        self._readback_texture()

    def get_image(self, pixel_array: PixelArray | None = None) -> Image.Image:
        """Return a PIL Image from the current pixel array."""
        if pixel_array is None:
            pixel_array = self.pixel_array
        return Image.fromarray(pixel_array, mode=self.image_mode)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_mobjects_to_display(
        self,
        mobjects: Iterable[Mobject],
        include_submobjects: bool = True,
        excluded_mobjects: list | None = None,
    ) -> list[Mobject]:
        if include_submobjects:
            mobjects = extract_mobject_family_members(
                mobjects,
                use_z_index=self.use_z_index,
                only_those_with_points=True,
            )
            if excluded_mobjects:
                all_excluded = extract_mobject_family_members(
                    excluded_mobjects,
                    use_z_index=self.use_z_index,
                )
                mobjects = list_difference_update(mobjects, all_excluded)
        return list(mobjects)

    def _encode_vmobject(self, encoder, vmob: VMobject, mvp_buffer) -> None:
        """Encode fill and stroke draw calls for a single VMobject."""
        points = vmob.points
        if len(points) < 4:
            return

        # --- Fill ---
        fill_rgba = vmob.get_fill_rgbas()
        if len(fill_rgba) > 0:
            fill_color = fill_rgba[0]
            if fill_color[3] > 0:  # Has visible fill
                self._encode_fill(encoder, points, fill_color, mvp_buffer)

        # --- Stroke ---
        stroke_rgba = vmob.get_stroke_rgbas()
        stroke_width = vmob.get_stroke_width()
        if len(stroke_rgba) > 0 and stroke_width > 0:
            stroke_color = stroke_rgba[0]
            if stroke_color[3] > 0:
                # Convert stroke width from pixels to scene units
                scene_stroke_width = stroke_width * self.frame_width / self.pixel_width
                self._encode_stroke(encoder, points, stroke_color, scene_stroke_width, mvp_buffer)

    def _encode_fill(self, encoder, points, fill_color, mvp_buffer) -> None:
        """Encode stencil-then-cover fill for a VMobject."""
        triangles = vmobject_to_triangles(points)
        if len(triangles) == 0:
            return

        # Pack uniforms: mvp (float4x4) + color (float4)
        color_arr = np.array(fill_color, dtype=np.float32)
        uniforms = np.concatenate([self._mvp.flatten(), color_arr])
        uniform_buffer = self.ctx.make_buffer(uniforms)

        vertex_buffer = self.ctx.make_buffer(triangles)
        n_vertices = len(triangles)

        # Pass 1: Stencil — toggle stencil bits with triangle fan
        encoder.setRenderPipelineState_(self.ctx._fill_stencil_pso)
        encoder.setDepthStencilState_(self.ctx._stencil_increment_dss)
        encoder.setStencilReferenceValue_(0)
        encoder.setVertexBuffer_offset_atIndex_(vertex_buffer, 0, 0)
        encoder.setVertexBuffer_offset_atIndex_(uniform_buffer, 0, 1)
        encoder.drawPrimitives_vertexStart_vertexCount_(
            0,  # MTLPrimitiveTypeTriangle
            0,
            n_vertices,
        )

        # Pass 2: Cover — draw bounding quad where stencil != 0
        bbox = self._bounding_quad(points)
        bbox_buffer = self.ctx.make_buffer(bbox)
        encoder.setRenderPipelineState_(self.ctx._fill_cover_pso)
        encoder.setDepthStencilState_(self.ctx._stencil_nonzero_dss)
        encoder.setStencilReferenceValue_(0)
        encoder.setVertexBuffer_offset_atIndex_(bbox_buffer, 0, 0)
        encoder.setVertexBuffer_offset_atIndex_(uniform_buffer, 0, 1)
        encoder.setFragmentBuffer_offset_atIndex_(uniform_buffer, 0, 1)
        encoder.drawPrimitives_vertexStart_vertexCount_(
            0,  # MTLPrimitiveTypeTriangle
            0,
            6,
        )

    def _encode_stroke(self, encoder, points, stroke_color, stroke_width, mvp_buffer) -> None:
        """Encode stroke quads for a VMobject."""
        quads = vmobject_to_stroke_quads(points, stroke_width)
        if len(quads) == 0:
            return

        color_arr = np.array(stroke_color, dtype=np.float32)
        uniforms = np.concatenate([self._mvp.flatten(), color_arr])
        uniform_buffer = self.ctx.make_buffer(uniforms)

        vertex_buffer = self.ctx.make_buffer(quads)
        n_vertices = len(quads)

        encoder.setRenderPipelineState_(self.ctx._stroke_pso)
        encoder.setDepthStencilState_(self.ctx._stencil_disabled_dss)
        encoder.setVertexBuffer_offset_atIndex_(vertex_buffer, 0, 0)
        encoder.setVertexBuffer_offset_atIndex_(uniform_buffer, 0, 1)
        encoder.setFragmentBuffer_offset_atIndex_(uniform_buffer, 0, 1)
        encoder.drawPrimitives_vertexStart_vertexCount_(
            0,  # MTLPrimitiveTypeTriangle
            0,
            n_vertices,
        )

    def _bounding_quad(self, points) -> np.ndarray:
        """Compute an axis-aligned bounding quad (2 triangles) from VMobject points."""
        xy = points[:, :2]
        x_min, y_min = xy.min(axis=0)
        x_max, y_max = xy.max(axis=0)
        # Slight padding to ensure coverage
        pad = 0.01
        x_min -= pad
        y_min -= pad
        x_max += pad
        y_max += pad
        # Two triangles forming the quad
        return np.array(
            [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_min],
                [x_max, y_max],
                [x_min, y_max],
            ],
            dtype=np.float32,
        )

    def _readback_texture(self) -> None:
        """Read the Metal render target into self.pixel_array."""
        self.pixel_array[:, :, :] = self.ctx.render_texture_to_numpy()

    def _upload_pixel_array_to_texture(self) -> None:
        """Upload self.pixel_array (RGBA) to the Metal render target (BGRA)."""
        # Convert RGBA -> BGRA
        bgra = self.pixel_array.copy()
        bgra[:, :, [0, 2]] = bgra[:, :, [2, 0]]
        bytes_per_row = self.pixel_width * 4
        mtl_region = Metal.MTLRegionMake2D(0, 0, self.pixel_width, self.pixel_height)
        self.ctx.render_target.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
            mtl_region, 0, bgra.tobytes(), bytes_per_row
        )
