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

from manim_metal import native_encoder
from manim_metal.metal_context import MetalContext
from manim_metal.utils import (
    batch_tessellate,
    build_rotation_matrix,
    build_world_to_ndc_matrix,
    compute_globally_smooth_normals,
    compute_stroke_face_normals,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    from manim.mobject.mobject import Mobject
    from manim.typing import PixelArray


# ---------------------------------------------------------------------------
# Draw operation types for two-pass rendering
# ---------------------------------------------------------------------------

# Each draw op is a tuple: (kind, vert_offset, vert_count, uniform_offset)
_OP_FILL_STENCIL = 0  # stencil-mark pass: fan triangles, stencil invert, no color/depth write
_OP_FILL_COVER = 1  # fill+depth pass: same triangles, color write + depth write, stencil NZ→0
_OP_STROKE = 2  # stroke pass: stroke quads, depth test, no stencil
_OP_FILL_COVER_LIT = 3  # lit cover pass: LitVertex buffer (pos+normal), Blinn-Phong shading
_OP_STROKE_LIT = 4  # lit stroke pass: LitVertex buffer (pos+normal), Blinn-Phong shading


# ---------------------------------------------------------------------------
# Uniform buffer layout constants
# ---------------------------------------------------------------------------

# Metal float3x3 is stored as 3 × float4 (each column padded to 16 bytes).
# Total uniform struct (must match shader Uniforms):
#   float4x4 mvp         = 64 bytes (16 floats)
#   float4   color       = 16 bytes (4 floats)
#   float3x3 rotation    = 48 bytes (3 columns × 16 bytes each)
#   float3   frame_center= 12 bytes + 4 pad = 16 bytes
#   float2   frame_shape = 8 bytes
#   float    focal_dist  = 4 bytes
#   float    zoom        = 4 bytes
#   uint     is_3d       = 4 bytes
#   padding to 256       = remaining bytes
# Total meaningful: 64 + 16 + 48 + 16 + 8 + 4 + 4 + 4 = 164 bytes
# Padded to 256-byte alignment for Metal buffer offsets.
_UNIFORM_SIZE = 256  # bytes
_UNIFORM_FLOATS = _UNIFORM_SIZE // 4  # 64 float32s


def _pack_uniforms_2d(mvp_flat: np.ndarray, color: np.ndarray, buf: np.ndarray) -> None:
    """Pack 2D uniform data into the pre-allocated buffer."""
    buf[:16] = mvp_flat
    buf[16:20] = color
    # is_3d = 0 (default from zeros)


def _pack_uniforms_3d(
    mvp_flat: np.ndarray,
    color: np.ndarray,
    rotation: np.ndarray,
    frame_center: np.ndarray,
    frame_shape: np.ndarray,
    focal_distance: float,
    zoom: float,
    buf: np.ndarray,
) -> None:
    """Pack 3D uniform data into the pre-allocated buffer.

    Metal's float3x3 is stored as 3 columns, each padded to float4 (16 bytes).
    So a 3x3 matrix uses 12 floats in memory (3 × 4).
    """
    buf[:16] = mvp_flat
    buf[16:20] = color
    # rotation: 3x3 matrix → 3 columns × float4 (padded)
    # Column 0: indices 20-22, pad at 23
    buf[20:23] = rotation[:, 0].astype(np.float32)
    buf[23] = 0.0
    # Column 1: indices 24-26, pad at 27
    buf[24:27] = rotation[:, 1].astype(np.float32)
    buf[27] = 0.0
    # Column 2: indices 28-30, pad at 31
    buf[28:31] = rotation[:, 2].astype(np.float32)
    buf[31] = 0.0
    # frame_center: float3 + pad
    buf[32:35] = frame_center.astype(np.float32)
    buf[35] = 0.0
    # frame_shape: float2
    buf[36:38] = frame_shape.astype(np.float32)
    # focal_distance, zoom
    buf[38] = np.float32(focal_distance)
    buf[39] = np.float32(zoom)
    # is_3d flag (reinterpret as uint32 = 1)
    buf[40] = np.float32(0.0)  # placeholder — set via view below
    # use_lighting at [41] — set via _set_use_lighting_flag
    # Lighting params at [42..53] — set via _pack_lighting_params


def _set_is_3d_flag(buf: np.ndarray, value: int) -> None:
    """Set the is_3d uint flag in the uniform buffer."""
    # buf is float32; we need to write a uint32 at index 40
    buf.view(np.uint32)[40] = value


def _set_use_lighting_flag(buf: np.ndarray, value: int) -> None:
    """Set the use_lighting uint flag in the uniform buffer."""
    # buf is float32; we need to write a uint32 at index 41
    buf.view(np.uint32)[41] = value


def _needs_lighting(vmob: VMobject) -> bool:
    """Return True if *vmob* should use the lit (Blinn-Phong) rendering path.

    Criteria (any one is sufficient):
      - Instance of a 3D surface class (Surface, Sphere, Torus, Cylinder, Cone)
      - Has a ``shade_in_3d`` attribute set to True
    """
    # Check shade_in_3d attribute (set by some Manim 3D mobjects)
    if getattr(vmob, "shade_in_3d", False):
        return True

    # Check for known 3D surface types (lazy import to avoid circular deps)
    try:
        from manim.mobject.three_d.three_dimensions import (
            Cone,
            Cylinder,
            Sphere,
            Surface,
            Torus,
        )

        if isinstance(vmob, (Surface, Sphere, Torus, Cylinder, Cone)):
            return True
    except ImportError:
        pass

    return False


def _interleave_pos_normal(positions: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Interleave (N, 3) positions and (N, 3) normals into LitVertex format.

    Returns an (N, 6) float32 array: [px, py, pz, nx, ny, nz] per vertex.
    When reinterpreted as raw bytes, each vertex is 24 bytes — matching the
    GPU-side ``LitVertex`` struct (two ``packed_float3``).
    """
    n = len(positions)
    lit = np.empty((n, 6), dtype=np.float32)
    lit[:, :3] = positions
    lit[:, 3:] = normals
    return lit


def _pack_lighting_params(
    buf: np.ndarray,
    ambient_strength: float,
    diffuse_strength: float,
    light_position: np.ndarray,
    light_color: np.ndarray,
    specular_strength: float,
    shininess: float,
) -> None:
    """Pack Blinn-Phong lighting parameters into the uniform buffer.

    See lighting.h for byte offset documentation.  The buffer is a float32
    array of 64 elements (256 bytes total).

    Offsets (float32 index):
      [42] ambient_strength
      [43] diffuse_strength
      [44:47] light_position (float3, pad at [47])
      [48:51] light_color    (float3, pad at [51])
      [52] specular_strength
      [53] shininess
    """
    buf[42] = np.float32(ambient_strength)
    buf[43] = np.float32(diffuse_strength)
    buf[44:47] = light_position[:3].astype(np.float32)
    # buf[47] = 0.0 — already zero from np.zeros
    buf[48:51] = light_color[:3].astype(np.float32)
    # buf[51] = 0.0 — already zero from np.zeros
    buf[52] = np.float32(specular_strength)
    buf[53] = np.float32(shininess)


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

    def put_stroke(self, vmob: VMobject, stroke_width: float, quads: np.ndarray) -> None:
        self._stroke[id(vmob)] = (vmob.points, stroke_width, quads)

    def clear(self) -> None:
        self._fill.clear()
        self._stroke.clear()


# ---------------------------------------------------------------------------
# MetalCamera
# ---------------------------------------------------------------------------


class MetalCamera:
    """Camera that renders mobjects via Metal, with ThreeDCamera-compatible API.

    Mirrors the subset of :class:`manim.camera.camera.Camera` that
    :class:`CairoRenderer` actually calls, plus ThreeDCamera's ValueTracker
    properties for 3D camera control.

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
        # Match Cairo: adjust frame_height to preserve aspect ratio (frame_width is primary)
        aspect_ratio = self.pixel_width / self.pixel_height
        self.frame_height = self.frame_width / aspect_ratio
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

        # Pre-compute MVP matrix and its flattened form
        self._mvp = build_world_to_ndc_matrix(self.frame_width, self.frame_height, 0.0, 0.0)
        self._mvp_flat = self._mvp.flatten()  # 16 floats, cached

        # Geometry cache
        self._geo_cache = _GeometryCache()

        # Lazy clear flag — avoids redundant GPU round-trip in reset()
        self._needs_clear = True

        # Pre-allocated uniform buffer (256-byte aligned)
        self._uniform_buf = np.zeros(_UNIFORM_FLOATS, dtype=np.float32)

        # --- 3D camera state (ThreeDCamera-compatible) ---
        self._init_3d_camera()

    # ------------------------------------------------------------------
    # 3D camera initialization and API (ThreeDCamera-compatible)
    # ------------------------------------------------------------------

    def _init_3d_camera(self) -> None:
        """Initialize ValueTracker properties matching ThreeDCamera's interface."""
        from manim.mobject.value_tracker import ValueTracker

        self.phi_tracker = ValueTracker(0)
        self.theta_tracker = ValueTracker(0)
        self.gamma_tracker = ValueTracker(0)
        self.zoom_tracker = ValueTracker(1)
        self.focal_distance_tracker = ValueTracker(20.0)
        # _frame_center as a Mobject with .points so move_to() works
        from manim.mobject.mobject import Mobject

        self._frame_center = Mobject()
        self._frame_center.points = np.array([[0.0, 0.0, 0.0]])

        # Fixed orientation/in-frame mobject tracking
        self.fixed_orientation_mobjects: dict = {}
        self.fixed_in_frame_mobjects: set = set()

        # Cached rotation matrix (invalidated when camera params change)
        self._rotation_matrix: np.ndarray | None = None

        # --- Lighting state ---
        self._init_lighting()

    @property
    def phi(self) -> float:
        return self.phi_tracker.get_value()

    @property
    def theta(self) -> float:
        return self.theta_tracker.get_value()

    @property
    def gamma(self) -> float:
        return self.gamma_tracker.get_value()

    @property
    def zoom(self) -> float:
        return self.zoom_tracker.get_value()

    @property
    def focal_distance(self) -> float:
        return self.focal_distance_tracker.get_value()

    def get_phi(self) -> float:
        return self.phi_tracker.get_value()

    def get_theta(self) -> float:
        return self.theta_tracker.get_value()

    def get_gamma(self) -> float:
        return self.gamma_tracker.get_value()

    def get_zoom(self) -> float:
        return self.zoom_tracker.get_value()

    def get_focal_distance(self) -> float:
        return self.focal_distance_tracker.get_value()

    def set_phi(self, value: float) -> None:
        self.phi_tracker.set_value(value)

    def set_theta(self, value: float) -> None:
        self.theta_tracker.set_value(value)

    def set_gamma(self, value: float) -> None:
        self.gamma_tracker.set_value(value)

    def set_zoom(self, value: float) -> None:
        self.zoom_tracker.set_value(value)

    def set_focal_distance(self, value: float) -> None:
        self.focal_distance_tracker.set_value(value)

    def get_value_trackers(self) -> list:
        """Return all camera ValueTrackers (matches ThreeDCamera's interface)."""
        return [
            self.phi_tracker,
            self.theta_tracker,
            self.focal_distance_tracker,
            self.gamma_tracker,
            self.zoom_tracker,
        ]

    @property
    def frame_center(self):
        return self._frame_center.points[0]

    @frame_center.setter
    def frame_center(self, point) -> None:
        self._frame_center.move_to(point)

    def add_fixed_orientation_mobjects(
        self,
        *mobjects: Mobject,
        use_static_center_func: bool = False,
        center_func=None,
    ) -> None:
        def _make_center_func(m):
            def _center():
                return m.get_center()

            return _center

        for mob in mobjects:
            for submob in mob.get_family():
                if use_static_center_func:
                    func = _make_center_func(submob)
                elif center_func is not None:
                    func = center_func
                else:
                    func = _make_center_func(submob)
                self.fixed_orientation_mobjects[submob] = func

    def add_fixed_in_frame_mobjects(self, *mobjects: Mobject) -> None:
        for mob in mobjects:
            for submob in mob.get_family():
                self.fixed_in_frame_mobjects.add(submob)

    def remove_fixed_orientation_mobjects(self, *mobjects: Mobject) -> None:
        for mob in mobjects:
            for submob in mob.get_family():
                self.fixed_orientation_mobjects.pop(submob, None)

    def remove_fixed_in_frame_mobjects(self, *mobjects: Mobject) -> None:
        for mob in mobjects:
            for submob in mob.get_family():
                self.fixed_in_frame_mobjects.discard(submob)

    @property
    def _is_3d_active(self) -> bool:
        """Check if any 3D camera parameter is non-default."""
        return (
            self.phi_tracker.get_value() != 0
            or self.theta_tracker.get_value() != 0
            or self.gamma_tracker.get_value() != 0
        )

    def _get_rotation_matrix(self) -> np.ndarray:
        """Build and cache the 3x3 rotation matrix from current camera angles."""
        return build_rotation_matrix(
            self.phi_tracker.get_value(),
            self.theta_tracker.get_value(),
            self.gamma_tracker.get_value(),
        )

    # ------------------------------------------------------------------
    # Lighting initialization and API
    # ------------------------------------------------------------------

    def _init_lighting(self) -> None:
        """Initialize lighting ValueTrackers for Cairo-matching shading.

        Defaults match Cairo ThreeDCamera's ``get_shaded_rgb`` formula:
        ``light = 0.5 * (n·L)^3``, asymmetric shadow (half intensity).
        Light position matches ``light_source_start_point = 9*DOWN + 7*LEFT + 10*OUT``.

        The shader uses ``diffuse_strength`` as the intensity coefficient and
        ``shininess`` as the exponent.  ``ambient_strength`` and
        ``specular_strength`` are retained for API compatibility but unused
        by the current Cairo-matching shader.
        """
        from manim.mobject.value_tracker import ValueTracker

        # Light position matching Cairo ThreeDCamera:
        # 9*DOWN + 7*LEFT + 10*OUT = (-7, -9, 10)
        self._light_position = np.array([-7.0, -9.0, 10.0], dtype=np.float64)
        # Light color as a numpy array (RGB, normalized)
        self._light_color = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        # Cairo formula: light = intensity * (n·L)^exponent
        # intensity = diffuse_strength (default 0.5 matching Cairo's coefficient)
        # exponent  = shininess        (default 3.0 matching Cairo's cubic power)
        self.ambient_strength_tracker = ValueTracker(0.5)
        self.diffuse_strength_tracker = ValueTracker(0.5)
        self.specular_strength_tracker = ValueTracker(0.0)
        self.shininess_tracker = ValueTracker(3.0)

    # --- Light position ---

    def set_light_position(self, pos: np.ndarray | list | tuple) -> None:
        self._light_position = np.array(pos, dtype=np.float64)

    def get_light_position(self) -> np.ndarray:
        return self._light_position.copy()

    # --- Light color ---

    def set_light_color(self, color: np.ndarray | list | tuple) -> None:
        self._light_color = np.array(color, dtype=np.float64)

    def get_light_color(self) -> np.ndarray:
        return self._light_color.copy()

    # --- Ambient strength ---

    @property
    def ambient_strength(self) -> float:
        return self.ambient_strength_tracker.get_value()

    def set_ambient_strength(self, val: float) -> None:
        self.ambient_strength_tracker.set_value(val)

    def get_ambient_strength(self) -> float:
        return self.ambient_strength_tracker.get_value()

    # --- Diffuse strength ---

    @property
    def diffuse_strength(self) -> float:
        return self.diffuse_strength_tracker.get_value()

    def set_diffuse_strength(self, val: float) -> None:
        self.diffuse_strength_tracker.set_value(val)

    def get_diffuse_strength(self) -> float:
        return self.diffuse_strength_tracker.get_value()

    # --- Specular strength ---

    @property
    def specular_strength(self) -> float:
        return self.specular_strength_tracker.get_value()

    def set_specular_strength(self, val: float) -> None:
        self.specular_strength_tracker.set_value(val)

    def get_specular_strength(self) -> float:
        return self.specular_strength_tracker.get_value()

    # --- Shininess ---

    @property
    def shininess(self) -> float:
        return self.shininess_tracker.get_value()

    def set_shininess(self, val: float) -> None:
        self.shininess_tracker.set_value(val)

    def get_shininess(self) -> float:
        return self.shininess_tracker.get_value()

    # ------------------------------------------------------------------
    # CairoRenderer-facing interface
    # ------------------------------------------------------------------

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

    def reset(self) -> MetalCamera:
        """Mark render target for clearing on next capture.

        The actual GPU clear happens in :meth:`capture_mobjects` via the
        render pass load action, avoiding a redundant GPU round-trip.
        """
        self._needs_clear = True
        return self

    def reset_rotation_matrix(self) -> None:
        """Recompute the rotation matrix (called by ThreeDCamera.capture_mobjects)."""
        self._rotation_matrix = self._get_rotation_matrix()

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
        # Update rotation matrix if 3D is active
        if self._is_3d_active:
            self.reset_rotation_matrix()

        mobjects = self._get_mobjects_to_display(
            mobjects,
            include_submobjects=include_submobjects,
            excluded_mobjects=excluded_mobjects,
        )

        if not mobjects:
            if self._needs_clear:
                self.ctx.clear_render_target(self._bg_color_float)
                self._needs_clear = False
            self._readback_texture()
            return

        # --- Pass 1: Stage geometry and uniforms ---
        pool = self.ctx.buffer_pool
        pool.reset()

        draw_ops: list[tuple] = []
        vmobs = [m for m in mobjects if isinstance(m, VMobject) and len(m.points) >= 4]
        self._stage_all_vmobjects(vmobs, pool, draw_ops)

        if not draw_ops:
            if self._needs_clear:
                self.ctx.clear_render_target(self._bg_color_float)
                self._needs_clear = False
            self._readback_texture()
            return

        # Create single shared MTLBuffer from staged data
        shared_buf = pool.finalize()

        # --- Pass 2: Encode draw commands ---
        cmd = self.ctx.command_queue.commandBuffer()
        rpd = self.ctx.make_render_pass_descriptor(clear=True, clear_color=self._bg_color_float)
        self._needs_clear = False
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

        # Bind shared buffer once — draw ops only update offsets
        raw_encoder.setVertexBuffer_offset_atIndex_(shared_buf, 0, 0)
        raw_encoder.setVertexBuffer_offset_atIndex_(shared_buf, 0, 1)
        raw_encoder.setFragmentBuffer_offset_atIndex_(shared_buf, 0, 1)

        if native_encoder.is_available():
            ops_array = np.array(draw_ops, dtype=np.int32)
            native_encoder.encode_draw_ops(raw_encoder, shared_buf, ops_array, self.ctx)
        else:
            tracker = _EncoderStateTracker(raw_encoder)
            for op in draw_ops:
                self._execute_draw_op(tracker, raw_encoder, op)

        raw_encoder.endEncoding()

        # Blit resolved texture → shared readback buffer (GPU untiles, zero-copy on UMA)
        self.ctx.blit_texture_to_readback(cmd)

        cmd.commit()
        cmd.waitUntilCompleted()

        # Zero-copy read from shared buffer into pixel_array
        np.copyto(self.pixel_array, self.ctx._readback_numpy)

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

    def _make_uniform_data(self, color: np.ndarray, use_lighting: bool = False) -> np.ndarray:
        """Build the uniform buffer for the current camera state and given color.

        Parameters
        ----------
        color
            RGBA color as float32 (4 elements).
        use_lighting
            If True, sets ``use_lighting=1`` in the uniform buffer and packs
            all Blinn-Phong lighting parameters from the camera's ValueTrackers.
        """
        buf = np.zeros(_UNIFORM_FLOATS, dtype=np.float32)

        if self._is_3d_active:
            rot = self._rotation_matrix
            if rot is None:
                rot = self._get_rotation_matrix()
            fc = self._frame_center.points[0]
            _pack_uniforms_3d(
                self._mvp_flat,
                color.astype(np.float32),
                rot,
                fc,
                np.array([self.frame_width, self.frame_height], dtype=np.float32),
                self.focal_distance_tracker.get_value(),
                self.zoom_tracker.get_value(),
                buf,
            )
            _set_is_3d_flag(buf, 1)
        else:
            _pack_uniforms_2d(self._mvp_flat, color.astype(np.float32), buf)
            # is_3d already 0 from zeros

        if use_lighting:
            _set_use_lighting_flag(buf, 1)
            _pack_lighting_params(
                buf,
                ambient_strength=self.ambient_strength_tracker.get_value(),
                diffuse_strength=self.diffuse_strength_tracker.get_value(),
                light_position=self._light_position,
                light_color=self._light_color,
                specular_strength=self.specular_strength_tracker.get_value(),
                shininess=self.shininess_tracker.get_value(),
            )

        return buf

    def _stage_all_vmobjects(self, vmobs: list[VMobject], pool, draw_ops: list[tuple]) -> None:
        """Stage fill and stroke data for all VMobjects, using batch tessellation.

        Objects that need lighting (3D surfaces) take a separate path:
        tessellated with normals, interleaved into LitVertex format, and
        drawn with the lit pipeline variants.
        """
        if not vmobs:
            return

        # Separate lit from unlit objects — they use different tessellation paths
        unlit_vmobs: list[VMobject] = []
        unlit_indices: list[int] = []  # index into vmobs
        lit_vmobs: list[VMobject] = []
        lit_indices: list[int] = []

        for i, vmob in enumerate(vmobs):
            if _needs_lighting(vmob):
                lit_vmobs.append(vmob)
                lit_indices.append(i)
            else:
                unlit_vmobs.append(vmob)
                unlit_indices.append(i)

        # We process lit and unlit objects separately, then interleave draw ops
        # in the original z-order by collecting per-object ops and merging.
        # Key: original index in vmobs -> list of draw ops for that object.
        per_object_ops: dict[int, list[tuple]] = {}

        # --- Unlit path (existing, unchanged) ---
        self._stage_unlit_vmobjects(unlit_vmobs, unlit_indices, pool, per_object_ops)

        # --- Lit path (Phase 2: positions + normals) ---
        self._stage_lit_vmobjects(lit_vmobs, lit_indices, pool, per_object_ops)

        # --- Merge draw ops in original z-order ---
        for i in range(len(vmobs)):
            ops = per_object_ops.get(i)
            if ops:
                draw_ops.extend(ops)

    def _stage_unlit_vmobjects(
        self,
        vmobs: list[VMobject],
        orig_indices: list[int],
        pool,
        per_object_ops: dict[int, list[tuple]],
    ) -> None:
        """Stage unlit VMobjects using the existing position-only pipeline."""
        if not vmobs:
            return

        # Gather per-object metadata and identify cache misses
        obj_meta: list[tuple] = []  # (vmob, fill_color|None, stroke_color|None, scene_sw)
        uncached_indices: list[int] = []
        uncached_items: list[tuple] = []

        for i, vmob in enumerate(vmobs):
            fill_color = None
            fill_rgba = vmob.get_fill_rgbas()
            if len(fill_rgba) > 0 and fill_rgba[0][3] > 0:
                fill_color = fill_rgba[0]

            stroke_color = None
            scene_sw = 0.0
            stroke_rgba = vmob.get_stroke_rgbas()
            sw = vmob.get_stroke_width()
            if len(stroke_rgba) > 0 and sw > 0 and stroke_rgba[0][3] > 0:
                stroke_color = stroke_rgba[0]
                scene_sw = sw * 0.01

            obj_meta.append((vmob, fill_color, stroke_color, scene_sw))

            # Check if tessellation is needed (cache miss for fill or stroke)
            need_fill = fill_color is not None and self._geo_cache.get_fill(vmob) is None
            need_stroke = (
                stroke_color is not None and self._geo_cache.get_stroke(vmob, scene_sw) is None
            )

            if need_fill or need_stroke:
                uncached_indices.append(i)
                uncached_items.append((vmob.points, scene_sw if stroke_color is not None else None))

        # Batch tessellation for all cache misses in one NumPy call
        if uncached_items:
            batch_results = batch_tessellate(uncached_items)
            for idx, (fill_tris, stroke_quads) in zip(uncached_indices, batch_results):
                vmob = vmobs[idx]
                _, fill_color, stroke_color, scene_sw = obj_meta[idx]
                if fill_color is not None:
                    self._geo_cache.put_fill(vmob, fill_tris)
                if stroke_color is not None and stroke_quads is not None:
                    self._geo_cache.put_stroke(vmob, scene_sw, stroke_quads)

        # Stage all objects from cache into the buffer pool
        for local_i, (vmob, fill_color, stroke_color, scene_sw) in enumerate(obj_meta):
            orig_idx = orig_indices[local_i]
            ops: list[tuple] = []

            # --- Fill ---
            if fill_color is not None:
                triangles = self._geo_cache.get_fill(vmob)
                if triangles is not None and len(triangles) > 0:
                    uniforms = self._make_uniform_data(fill_color)
                    uniform_off = pool.stage(uniforms)
                    vertex_off = pool.stage(triangles)

                    n_verts = len(triangles)
                    ops.append((_OP_FILL_STENCIL, vertex_off, n_verts, uniform_off))
                    ops.append((_OP_FILL_COVER, vertex_off, n_verts, uniform_off))

            # --- Stroke ---
            if stroke_color is not None:
                quads = self._geo_cache.get_stroke(vmob, scene_sw)
                if quads is not None and len(quads) > 0:
                    uniforms = self._make_uniform_data(stroke_color)
                    uniform_off = pool.stage(uniforms)
                    vertex_off = pool.stage(quads)

                    ops.append((_OP_STROKE, vertex_off, len(quads), uniform_off))

            if ops:
                per_object_ops[orig_idx] = ops

    def _stage_lit_vmobjects(
        self,
        vmobs: list[VMobject],
        orig_indices: list[int],
        pool,
        per_object_ops: dict[int, list[tuple]],
    ) -> None:
        """Stage lit VMobjects with normals for the Blinn-Phong pipeline.

        The stencil pass still uses position-only packed_float3 buffers (12B/vert)
        with the existing ``fill_stencil_vertex`` shader.

        The cover and stroke passes switch to the lit variants which read
        LitVertex buffers (24B/vert: [px, py, pz, nx, ny, nz]).
        """
        if not vmobs:
            return

        # Gather metadata and identify cache misses
        obj_meta: list[tuple] = []  # (vmob, fill_color|None, stroke_color|None, scene_sw)
        uncached_indices: list[int] = []
        uncached_items: list[tuple] = []

        for i, vmob in enumerate(vmobs):
            fill_color = None
            fill_rgba = vmob.get_fill_rgbas()
            if len(fill_rgba) > 0 and fill_rgba[0][3] > 0:
                fill_color = fill_rgba[0]

            stroke_color = None
            scene_sw = 0.0
            stroke_rgba = vmob.get_stroke_rgbas()
            sw = vmob.get_stroke_width()
            if len(stroke_rgba) > 0 and sw > 0 and stroke_rgba[0][3] > 0:
                # Skip thin default wireframe strokes (≤ 1.0) on lit surfaces.
                # Cairo renders these as sub-pixel lines that are invisible;
                # Metal quad-based strokes can't match that, so skip them.
                if sw > 1.0:
                    stroke_color = stroke_rgba[0]
                    scene_sw = sw * 0.01

            obj_meta.append((vmob, fill_color, stroke_color, scene_sw))

            # Check if tessellation is needed (cache miss for fill or stroke)
            need_fill = fill_color is not None and self._geo_cache.get_fill(vmob) is None
            need_stroke = (
                stroke_color is not None and self._geo_cache.get_stroke(vmob, scene_sw) is None
            )

            if need_fill or need_stroke:
                uncached_indices.append(i)
                uncached_items.append((vmob.points, scene_sw if stroke_color is not None else None))

        # Batch tessellation with normals for all cache misses
        if uncached_items:
            batch_results = batch_tessellate(uncached_items, compute_normals=True)
            for idx, result in zip(uncached_indices, batch_results):
                vmob = vmobs[idx]
                _, fill_color, stroke_color, scene_sw = obj_meta[idx]
                # compute_normals=True returns 4-tuples:
                # (fill_tris, fill_normals, stroke_quads, stroke_normals)
                fill_tris, fill_normals, stroke_quads, stroke_normals = result
                if fill_color is not None:
                    self._geo_cache.put_fill(vmob, fill_tris)
                if stroke_color is not None and stroke_quads is not None:
                    self._geo_cache.put_stroke(vmob, scene_sw, stroke_quads)

        # --- Phase 2: Collect fill arrays and compute globally smooth normals ---
        # Gather all fill triangle arrays so coincident vertices across
        # sub-mobject boundaries get averaged normals (eliminates embossing).
        fill_tri_list: list[npt.NDArray[np.float32] | None] = []
        for vmob, fill_color, _stroke_color, _scene_sw in obj_meta:
            if fill_color is not None:
                triangles = self._geo_cache.get_fill(vmob)
                if triangles is not None and len(triangles) > 0:
                    fill_tri_list.append(triangles)
                else:
                    fill_tri_list.append(None)
            else:
                fill_tri_list.append(None)

        # Compute globally smooth normals across all sub-mobjects at once
        non_empty_fills = [t for t in fill_tri_list if t is not None]
        if non_empty_fills:
            smooth_normals_list = compute_globally_smooth_normals(non_empty_fills)
        else:
            smooth_normals_list = []

        # --- Phase 3: Stage all lit objects into the buffer pool ---
        smooth_idx = 0  # index into smooth_normals_list (only non-empty fills)
        for local_i, (vmob, fill_color, stroke_color, scene_sw) in enumerate(obj_meta):
            orig_idx = orig_indices[local_i]
            ops: list[tuple] = []

            # --- Fill (lit) ---
            if fill_color is not None:
                triangles = fill_tri_list[local_i]
                if triangles is not None and len(triangles) > 0:
                    normals = smooth_normals_list[smooth_idx]
                    smooth_idx += 1

                    # Stencil pass uses position-only buffer (same as unlit)
                    stencil_uniforms = self._make_uniform_data(fill_color)
                    stencil_uniform_off = pool.stage(stencil_uniforms)
                    stencil_vertex_off = pool.stage(triangles)
                    n_verts = len(triangles)
                    ops.append((_OP_FILL_STENCIL, stencil_vertex_off, n_verts, stencil_uniform_off))

                    # Cover pass uses LitVertex buffer (pos + normal)
                    lit_verts = _interleave_pos_normal(triangles, normals)
                    lit_uniforms = self._make_uniform_data(fill_color, use_lighting=True)
                    lit_uniform_off = pool.stage(lit_uniforms)
                    lit_vertex_off = pool.stage(lit_verts)
                    ops.append((_OP_FILL_COVER_LIT, lit_vertex_off, n_verts, lit_uniform_off))
                else:
                    # fill_tri_list entry was None — skip smooth_idx
                    pass

            # --- Stroke (lit) ---
            if stroke_color is not None:
                quads = self._geo_cache.get_stroke(vmob, scene_sw)
                if quads is not None and len(quads) > 0:
                    # Compute normals for the cached stroke quads
                    stroke_normals = compute_stroke_face_normals(quads)

                    lit_stroke = _interleave_pos_normal(quads, stroke_normals)
                    lit_uniforms = self._make_uniform_data(stroke_color, use_lighting=True)
                    lit_uniform_off = pool.stage(lit_uniforms)
                    lit_vertex_off = pool.stage(lit_stroke)
                    ops.append((_OP_STROKE_LIT, lit_vertex_off, len(quads), lit_uniform_off))

            if ops:
                per_object_ops[orig_idx] = ops

    # ------------------------------------------------------------------
    # Internal helpers — Pass 2: encoding
    # ------------------------------------------------------------------

    def _execute_draw_op(self, tracker, encoder, op) -> None:
        """Dispatch a single draw operation.

        The shared buffer is already bound to vertex indices 0, 1 and
        fragment index 1.  We use the lightweight ``setVertexBufferOffset``
        / ``setFragmentBufferOffset`` to update offsets without re-binding.
        """
        kind = op[0]
        if kind == _OP_FILL_STENCIL:
            _, vert_off, vert_count, uni_off = op
            tracker.set_pipeline(self.ctx._fill_stencil_pso)
            tracker.set_depth_stencil(self.ctx._stencil_increment_dss)
            tracker.set_stencil_ref(0)
            encoder.setVertexBufferOffset_atIndex_(vert_off, 0)
            encoder.setVertexBufferOffset_atIndex_(uni_off, 1)
            encoder.drawPrimitives_vertexStart_vertexCount_(
                Metal.MTLPrimitiveTypeTriangle, 0, vert_count
            )
        elif kind == _OP_FILL_COVER:
            _, vert_off, vert_count, uni_off = op
            tracker.set_pipeline(self.ctx._fill_cover_pso)
            tracker.set_depth_stencil(self.ctx._stencil_nonzero_dss)
            tracker.set_stencil_ref(0)
            encoder.setVertexBufferOffset_atIndex_(vert_off, 0)
            encoder.setVertexBufferOffset_atIndex_(uni_off, 1)
            encoder.setFragmentBufferOffset_atIndex_(uni_off, 1)
            encoder.drawPrimitives_vertexStart_vertexCount_(
                Metal.MTLPrimitiveTypeTriangle, 0, vert_count
            )
        elif kind == _OP_STROKE:
            _, vert_off, vert_count, uni_off = op
            tracker.set_pipeline(self.ctx._stroke_pso)
            tracker.set_depth_stencil(self.ctx._stencil_disabled_dss)
            encoder.setVertexBufferOffset_atIndex_(vert_off, 0)
            encoder.setVertexBufferOffset_atIndex_(uni_off, 1)
            encoder.setFragmentBufferOffset_atIndex_(uni_off, 1)
            encoder.drawPrimitives_vertexStart_vertexCount_(
                Metal.MTLPrimitiveTypeTriangle, 0, vert_count
            )
        elif kind == _OP_FILL_COVER_LIT:
            # Lit cover pass: LitVertex buffer (24B/vert), Blinn-Phong shading
            _, vert_off, vert_count, uni_off = op
            tracker.set_pipeline(self.ctx._fill_cover_lit_pso)
            tracker.set_depth_stencil(self.ctx._stencil_nonzero_dss)
            tracker.set_stencil_ref(0)
            encoder.setVertexBufferOffset_atIndex_(vert_off, 0)
            encoder.setVertexBufferOffset_atIndex_(uni_off, 1)
            encoder.setFragmentBufferOffset_atIndex_(uni_off, 1)
            encoder.drawPrimitives_vertexStart_vertexCount_(
                Metal.MTLPrimitiveTypeTriangle, 0, vert_count
            )
        elif kind == _OP_STROKE_LIT:
            # Lit stroke pass: LitVertex buffer (24B/vert), Blinn-Phong shading
            _, vert_off, vert_count, uni_off = op
            tracker.set_pipeline(self.ctx._stroke_lit_pso)
            tracker.set_depth_stencil(self.ctx._stencil_disabled_dss)
            encoder.setVertexBufferOffset_atIndex_(vert_off, 0)
            encoder.setVertexBufferOffset_atIndex_(uni_off, 1)
            encoder.setFragmentBufferOffset_atIndex_(uni_off, 1)
            encoder.drawPrimitives_vertexStart_vertexCount_(
                Metal.MTLPrimitiveTypeTriangle, 0, vert_count
            )

    def _readback_texture(self) -> None:
        """Read the Metal render target directly into self.pixel_array."""
        self.ctx.render_texture_to_numpy(target=self.pixel_array)

    def _upload_pixel_array_to_texture(self) -> None:
        """Upload self.pixel_array (RGBA) to the Metal render target (RGBA)."""
        bytes_per_row = self.pixel_width * 4
        mtl_region = Metal.MTLRegionMake2D(0, 0, self.pixel_width, self.pixel_height)
        self.ctx.render_target.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
            mtl_region, 0, self.pixel_array.tobytes(), bytes_per_row
        )
