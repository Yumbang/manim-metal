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


# ---------------------------------------------------------------------------
# Draw operation types for two-pass rendering
# ---------------------------------------------------------------------------

# Each draw op is a tuple: (kind, ...) where kind is one of:
_OP_FILL_STENCIL = 0  # (kind, vert_offset, vert_count, uniform_offset)
_OP_FILL_COVER = 1  # (kind, bbox_offset, uniform_offset)
_OP_STROKE = 2  # (kind, vert_offset, vert_count, uniform_offset)


# ---------------------------------------------------------------------------
# Encoder state tracker — avoids redundant pyobjc calls
# ---------------------------------------------------------------------------


class _EncoderStateTracker:
    """Thin wrapper around a render command encoder that skips redundant state calls."""

    __slots__ = ("_encoder", "_pso", "_dss", "_stencil_ref")

    def __init__(self, encoder) -> None:
        self._encoder = encoder
        self._pso = None
        self._dss = None
        self._stencil_ref = None

    def set_pipeline(self, pso) -> None:
        if pso is not self._pso:
            self._encoder.setRenderPipelineState_(pso)
            self._pso = pso

    def set_depth_stencil(self, dss) -> None:
        if dss is not self._dss:
            self._encoder.setDepthStencilState_(dss)
            self._dss = dss

    def set_stencil_ref(self, ref: int) -> None:
        if ref != self._stencil_ref:
            self._encoder.setStencilReferenceValue_(ref)
            self._stencil_ref = ref

    @property
    def encoder(self):
        return self._encoder


# ---------------------------------------------------------------------------
# Geometry cache — skip tessellation for static objects
# ---------------------------------------------------------------------------


class _GeometryCache:
    """Cache tessellated geometry keyed on VMobject identity.

    Each entry stores a reference to the points array alongside the cached
    result.  On lookup we use ``is`` to verify the points object is literally
    the same — this is safe against CPython ``id()`` reuse because holding
    the reference prevents the old array from being garbage-collected.

    When Manim animates a VMobject it replaces ``.points`` with a new array,
    so the ``is`` check fails and we re-tessellate.
    """

    __slots__ = ("_fill", "_stroke")

    def __init__(self) -> None:
        # id(vmob) -> (points_array_ref, cached_result)
        self._fill: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        # id(vmob) -> (points_array_ref, stroke_width, cached_result)
        self._stroke: dict[int, tuple[np.ndarray, float, np.ndarray]] = {}

    def get_fill(self, vmob: VMobject) -> np.ndarray | None:
        entry = self._fill.get(id(vmob))
        if entry is not None and entry[0] is vmob.points:
            return entry[1]
        return None

    def put_fill(self, vmob: VMobject, triangles: np.ndarray) -> None:
        self._fill[id(vmob)] = (vmob.points, triangles)

    def get_stroke(self, vmob: VMobject, stroke_width: float) -> np.ndarray | None:
        entry = self._stroke.get(id(vmob))
        if entry is not None and entry[0] is vmob.points and entry[1] == stroke_width:
            return entry[2]
        return None

    def put_stroke(
        self, vmob: VMobject, stroke_width: float, quads: np.ndarray
    ) -> None:
        self._stroke[id(vmob)] = (vmob.points, stroke_width, quads)

    def clear(self) -> None:
        self._fill.clear()
        self._stroke.clear()


# ---------------------------------------------------------------------------
# MetalCamera
# ---------------------------------------------------------------------------


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
            background_opacity
            if background_opacity is not None
            else config["background_opacity"]
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
        self.pixel_array[:, :] = color_to_int_rgba(
            self._background_color, self._background_opacity
        )
        self.background = self.pixel_array.copy()

        # Pre-compute MVP matrix and its flattened form
        self._mvp = build_world_to_ndc_matrix(
            self.frame_width, self.frame_height, 0.0, 0.0
        )
        self._mvp_flat = self._mvp.flatten()  # 16 floats, cached

        # Geometry cache
        self._geo_cache = _GeometryCache()

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
        """Render mobjects into the Metal render target.

        Uses a two-pass approach:
          Pass 1 — stage all geometry/uniform data into a BufferPool bytearray.
          Pass 2 — create one MTLBuffer, then encode all draw commands.
        """
        mobjects = self._get_mobjects_to_display(
            mobjects,
            include_submobjects=include_submobjects,
            excluded_mobjects=excluded_mobjects,
        )

        if not mobjects:
            self._readback_texture()
            return

        # --- Pass 1: Stage geometry and uniforms ---
        pool = self.ctx.buffer_pool
        pool.reset()

        draw_ops: list[tuple] = []
        for mob in mobjects:
            if isinstance(mob, VMobject):
                self._stage_vmobject(mob, pool, draw_ops)

        if not draw_ops:
            self._readback_texture()
            return

        # Create single shared MTLBuffer from staged data
        shared_buf = pool.finalize()

        # --- Pass 2: Encode draw commands ---
        cmd = self.ctx.command_queue.commandBuffer()
        rpd = self.ctx.make_render_pass_descriptor(
            clear=True, clear_color=self._bg_color_float
        )
        raw_encoder = cmd.renderCommandEncoderWithDescriptor_(rpd)
        raw_encoder.setViewport_(
            (
                0.0,
                0.0,
                float(self.pixel_width),
                float(self.pixel_height),
                0.0,
                1.0,
            )
        )

        tracker = _EncoderStateTracker(raw_encoder)

        for op in draw_ops:
            self._execute_draw_op(tracker, raw_encoder, shared_buf, op)

        raw_encoder.endEncoding()
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
    # Internal helpers — Pass 1: staging
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

    def _stage_vmobject(
        self, vmob: VMobject, pool, draw_ops: list[tuple]
    ) -> None:
        """Stage fill and stroke data for a VMobject into the buffer pool."""
        points = vmob.points
        if len(points) < 4:
            return

        # --- Fill ---
        fill_rgba = vmob.get_fill_rgbas()
        if len(fill_rgba) > 0:
            fill_color = fill_rgba[0]
            if fill_color[3] > 0:
                self._stage_fill(vmob, points, fill_color, pool, draw_ops)

        # --- Stroke ---
        stroke_rgba = vmob.get_stroke_rgbas()
        stroke_width = vmob.get_stroke_width()
        if len(stroke_rgba) > 0 and stroke_width > 0:
            stroke_color = stroke_rgba[0]
            if stroke_color[3] > 0:
                scene_stroke_width = stroke_width * 0.01
                self._stage_stroke(
                    vmob, points, stroke_color, scene_stroke_width, pool, draw_ops
                )

    def _stage_fill(self, vmob, points, fill_color, pool, draw_ops) -> None:
        """Stage stencil-then-cover fill data."""
        # Check geometry cache
        triangles = self._geo_cache.get_fill(vmob)
        if triangles is None:
            triangles = vmobject_to_triangles(points)
            self._geo_cache.put_fill(vmob, triangles)

        if len(triangles) == 0:
            return

        # Pack uniforms: mvp (16 floats) + color (4 floats)
        color_arr = np.array(fill_color, dtype=np.float32)
        uniforms = np.empty(20, dtype=np.float32)
        uniforms[:16] = self._mvp_flat
        uniforms[16:20] = color_arr

        uniform_off = pool.stage(uniforms)
        vertex_off = pool.stage(triangles)
        n_vertices = len(triangles)

        # Fill stencil pass
        draw_ops.append(
            (_OP_FILL_STENCIL, vertex_off, n_vertices, uniform_off)
        )

        # Fill cover pass — bounding quad
        bbox = self._bounding_quad(points)
        bbox_off = pool.stage(bbox)
        draw_ops.append((_OP_FILL_COVER, bbox_off, uniform_off))

    def _stage_stroke(
        self, vmob, points, stroke_color, stroke_width, pool, draw_ops
    ) -> None:
        """Stage stroke quad data."""
        # Check geometry cache
        quads = self._geo_cache.get_stroke(vmob, stroke_width)
        if quads is None:
            quads = vmobject_to_stroke_quads(points, stroke_width)
            self._geo_cache.put_stroke(vmob, stroke_width, quads)

        if len(quads) == 0:
            return

        # Pack uniforms
        color_arr = np.array(stroke_color, dtype=np.float32)
        uniforms = np.empty(20, dtype=np.float32)
        uniforms[:16] = self._mvp_flat
        uniforms[16:20] = color_arr

        uniform_off = pool.stage(uniforms)
        vertex_off = pool.stage(quads)
        n_vertices = len(quads)

        draw_ops.append((_OP_STROKE, vertex_off, n_vertices, uniform_off))

    # ------------------------------------------------------------------
    # Internal helpers — Pass 2: encoding
    # ------------------------------------------------------------------

    def _execute_draw_op(self, tracker, encoder, buf, op) -> None:
        """Dispatch a single draw operation."""
        kind = op[0]
        if kind == _OP_FILL_STENCIL:
            _, vert_off, vert_count, uni_off = op
            tracker.set_pipeline(self.ctx._fill_stencil_pso)
            tracker.set_depth_stencil(self.ctx._stencil_increment_dss)
            tracker.set_stencil_ref(0)
            encoder.setVertexBuffer_offset_atIndex_(buf, vert_off, 0)
            encoder.setVertexBuffer_offset_atIndex_(buf, uni_off, 1)
            encoder.drawPrimitives_vertexStart_vertexCount_(
                Metal.MTLPrimitiveTypeTriangle, 0, vert_count
            )
        elif kind == _OP_FILL_COVER:
            _, bbox_off, uni_off = op
            tracker.set_pipeline(self.ctx._fill_cover_pso)
            tracker.set_depth_stencil(self.ctx._stencil_nonzero_dss)
            tracker.set_stencil_ref(0)
            encoder.setVertexBuffer_offset_atIndex_(buf, bbox_off, 0)
            encoder.setVertexBuffer_offset_atIndex_(buf, uni_off, 1)
            encoder.setFragmentBuffer_offset_atIndex_(buf, uni_off, 1)
            encoder.drawPrimitives_vertexStart_vertexCount_(
                Metal.MTLPrimitiveTypeTriangle, 0, 6
            )
        elif kind == _OP_STROKE:
            _, vert_off, vert_count, uni_off = op
            tracker.set_pipeline(self.ctx._stroke_pso)
            tracker.set_depth_stencil(self.ctx._stencil_disabled_dss)
            encoder.setVertexBuffer_offset_atIndex_(buf, vert_off, 0)
            encoder.setVertexBuffer_offset_atIndex_(buf, uni_off, 1)
            encoder.setFragmentBuffer_offset_atIndex_(buf, uni_off, 1)
            encoder.drawPrimitives_vertexStart_vertexCount_(
                Metal.MTLPrimitiveTypeTriangle, 0, vert_count
            )

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

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
