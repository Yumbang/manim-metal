"""Unit tests for uniform buffer packing -- Phase 1 existing layout and Phase 2 lighting extensions.

Tests are organized into three sections:
  1. Pure packing function tests -- call ``_pack_uniforms_2d`` / ``_pack_uniforms_3d``
     / ``_set_is_3d_flag`` directly with synthetic inputs.  No GPU or Metal device
     needed.  These run NOW.
  2. Metal device-dependent tests -- need a live ``MetalContext`` (shader compilation).
     Skipped if shaders fail to compile (e.g., another engineer's WIP include).
  3. Phase 2 lighting uniform tests -- verify that light parameters are packed at
     expected byte offsets once the CPU-side packing code is updated.  Marked
     ``@pytest.mark.skip`` until the implementation lands.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

# ---------------------------------------------------------------------------
# Import packing helpers directly -- these are pure Python/NumPy, no GPU needed.
# ---------------------------------------------------------------------------
from manim_metal.metal_camera import (
    _UNIFORM_FLOATS,
    _UNIFORM_SIZE,
    _pack_uniforms_2d,
    _pack_uniforms_3d,
    _set_is_3d_flag,
)
from manim_metal.utils import build_rotation_matrix, build_world_to_ndc_matrix

# =====================================================================
# Section 1: Pure packing function tests (no GPU required)
# =====================================================================


class TestUniformConstants:
    """Verify module-level uniform buffer constants."""

    def test_uniform_buffer_size_is_256_bytes(self):
        """The uniform buffer should be exactly 256 bytes (64 floats)."""
        assert _UNIFORM_SIZE == 256
        assert _UNIFORM_FLOATS == 64

    def test_uniform_size_multiple_of_256(self):
        """Metal requires buffer offsets to be 256-byte aligned."""
        assert _UNIFORM_SIZE % 256 == 0


class TestPackUniforms2D:
    """Verify _pack_uniforms_2d places data at the correct float indices."""

    def _make_buf(self) -> np.ndarray:
        return np.zeros(_UNIFORM_FLOATS, dtype=np.float32)

    def test_mvp_at_offset_0(self):
        """The 4x4 MVP matrix should occupy float indices 0-15 (bytes 0-63)."""
        mvp = build_world_to_ndc_matrix(14.222, 8.0)
        mvp_flat = mvp.flatten()
        color = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
        buf = self._make_buf()

        _pack_uniforms_2d(mvp_flat, color, buf)

        npt.assert_allclose(buf[:16], mvp_flat, atol=1e-10)
        # Scale factors should be non-zero
        assert buf[0] != 0.0, "MVP sx should be non-zero"
        assert buf[5] != 0.0, "MVP sy should be non-zero"

    def test_color_at_offset_64_bytes(self):
        """The RGBA color should occupy float indices 16-19 (bytes 64-79)."""
        mvp_flat = np.eye(4, dtype=np.float32).flatten()
        color = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32)
        buf = self._make_buf()

        _pack_uniforms_2d(mvp_flat, color, buf)

        npt.assert_allclose(buf[16:20], [0.2, 0.4, 0.6, 0.8], atol=1e-7)

    def test_remaining_bytes_are_zero(self):
        """In 2D mode, all indices after color (20+) should be zero."""
        mvp_flat = np.eye(4, dtype=np.float32).flatten()
        color = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
        buf = self._make_buf()

        _pack_uniforms_2d(mvp_flat, color, buf)

        npt.assert_allclose(buf[20:], 0.0, atol=0.0)

    def test_buffer_dtype_is_float32(self):
        """The buffer should be float32."""
        buf = self._make_buf()
        assert buf.dtype == np.float32

    def test_buffer_length_is_64_floats(self):
        """The buffer should have exactly 64 float32 elements (256 bytes)."""
        buf = self._make_buf()
        assert len(buf) == 64
        assert buf.nbytes == 256


class TestPackUniforms3D:
    """Verify _pack_uniforms_3d places data at correct float indices."""

    def _pack_3d(
        self,
        phi=0.5,
        theta=-0.3,
        gamma=0.1,
        color=None,
        frame_center=None,
        frame_shape=None,
        focal_distance=20.0,
        zoom=1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Helper: pack 3D uniforms with given params. Returns (buf, rotation)."""
        mvp_flat = build_world_to_ndc_matrix(14.222, 8.0).flatten()
        if color is None:
            color = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
        if frame_center is None:
            frame_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        if frame_shape is None:
            frame_shape = np.array([14.222, 8.0], dtype=np.float32)

        rotation = build_rotation_matrix(phi, theta, gamma)
        buf = np.zeros(_UNIFORM_FLOATS, dtype=np.float32)
        _pack_uniforms_3d(
            mvp_flat,
            color.astype(np.float32),
            rotation,
            frame_center,
            frame_shape,
            focal_distance,
            zoom,
            buf,
        )
        return buf, rotation

    def test_mvp_at_offset_0(self):
        """MVP should still be at float indices 0-15 in 3D mode."""
        buf, _ = self._pack_3d()
        mvp_expected = build_world_to_ndc_matrix(14.222, 8.0).flatten()
        npt.assert_allclose(buf[:16], mvp_expected, atol=1e-6)

    def test_color_at_offset_64_bytes(self):
        """Color should still be at float indices 16-19 in 3D mode."""
        color = np.array([0.1, 0.2, 0.3, 0.9], dtype=np.float32)
        buf, _ = self._pack_3d(color=color)
        npt.assert_allclose(buf[16:20], [0.1, 0.2, 0.3, 0.9], atol=1e-7)

    def test_rotation_column_0_at_indices_20_22(self):
        """Rotation column 0 should be at float indices 20-22 with pad at 23.

        Metal stores float3x3 as 3 x float4: each column has 3 data floats + 1 pad.
        """
        buf, rotation = self._pack_3d(phi=0.7, theta=-0.3, gamma=0.1)
        npt.assert_allclose(buf[20:23], rotation[:, 0].astype(np.float32), atol=1e-6)
        assert buf[23] == 0.0, "Pad after rotation column 0 should be 0"

    def test_rotation_column_1_at_indices_24_26(self):
        """Rotation column 1 should be at float indices 24-26 with pad at 27."""
        buf, rotation = self._pack_3d(phi=0.7, theta=-0.3, gamma=0.1)
        npt.assert_allclose(buf[24:27], rotation[:, 1].astype(np.float32), atol=1e-6)
        assert buf[27] == 0.0, "Pad after rotation column 1 should be 0"

    def test_rotation_column_2_at_indices_28_30(self):
        """Rotation column 2 should be at float indices 28-30 with pad at 31."""
        buf, rotation = self._pack_3d(phi=0.7, theta=-0.3, gamma=0.1)
        npt.assert_allclose(buf[28:31], rotation[:, 2].astype(np.float32), atol=1e-6)
        assert buf[31] == 0.0, "Pad after rotation column 2 should be 0"

    def test_frame_center_at_indices_32_34(self):
        """frame_center (float3) at float indices 32-34, pad at 35."""
        fc = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        buf, _ = self._pack_3d(frame_center=fc)
        npt.assert_allclose(buf[32:35], [1.0, 2.0, 3.0], atol=1e-6)
        assert buf[35] == 0.0, "Pad after frame_center should be 0"

    def test_frame_shape_at_indices_36_37(self):
        """frame_shape (float2) at float indices 36-37."""
        fs = np.array([14.222, 8.0], dtype=np.float32)
        buf, _ = self._pack_3d(frame_shape=fs)
        npt.assert_allclose(buf[36:38], [14.222, 8.0], atol=1e-3)

    def test_focal_distance_at_index_38(self):
        """focal_distance should be at float index 38."""
        buf, _ = self._pack_3d(focal_distance=15.0)
        npt.assert_allclose(buf[38], 15.0, atol=1e-6)

    def test_zoom_at_index_39(self):
        """zoom should be at float index 39."""
        buf, _ = self._pack_3d(zoom=2.5)
        npt.assert_allclose(buf[39], 2.5, atol=1e-6)

    def test_custom_focal_and_zoom(self):
        """Both focal_distance and zoom should be independently settable."""
        buf, _ = self._pack_3d(focal_distance=10.0, zoom=3.0)
        npt.assert_allclose(buf[38], 10.0, atol=1e-6)
        npt.assert_allclose(buf[39], 3.0, atol=1e-6)


class TestIs3dFlag:
    """Verify the is_3d uint32 flag packing at float index 40 (byte offset 160)."""

    def test_set_is_3d_flag_to_1(self):
        """Setting is_3d to 1 should write uint32(1) at float index 40."""
        buf = np.zeros(_UNIFORM_FLOATS, dtype=np.float32)
        _set_is_3d_flag(buf, 1)
        assert buf.view(np.uint32)[40] == 1

    def test_set_is_3d_flag_to_0(self):
        """Setting is_3d to 0 should write uint32(0) at float index 40."""
        buf = np.zeros(_UNIFORM_FLOATS, dtype=np.float32)
        _set_is_3d_flag(buf, 1)  # set first
        _set_is_3d_flag(buf, 0)  # then clear
        assert buf.view(np.uint32)[40] == 0

    def test_is_3d_flag_byte_offset(self):
        """The is_3d flag should be at byte offset 160 (float index 40 * 4)."""
        buf = np.zeros(_UNIFORM_FLOATS, dtype=np.float32)
        _set_is_3d_flag(buf, 1)
        # Verify via raw bytes
        raw = buf.view(np.uint8)
        # uint32 value 1 at byte 160 in little-endian
        assert raw[160] == 1
        assert raw[161] == 0
        assert raw[162] == 0
        assert raw[163] == 0

    def test_is_3d_does_not_affect_other_fields(self):
        """Setting is_3d should not corrupt adjacent fields."""
        buf = np.zeros(_UNIFORM_FLOATS, dtype=np.float32)
        # Put known values around index 40
        buf[39] = np.float32(2.5)  # zoom
        buf[41] = np.float32(42.0)  # next field (padding or future use)

        _set_is_3d_flag(buf, 1)

        npt.assert_allclose(buf[39], 2.5, atol=1e-7)
        npt.assert_allclose(buf[41], 42.0, atol=1e-7)
        assert buf.view(np.uint32)[40] == 1


class TestRotationMatrixProperties:
    """Verify the rotation matrix used for 3D uniform packing."""

    def test_identity_rotation_at_default_angles(self):
        """phi=0, theta=-PI/2, gamma=0 should produce identity-like behavior."""
        R = build_rotation_matrix(0, -np.pi / 2, 0)
        assert R.shape == (3, 3)
        # Orthogonal: R @ R.T = I
        npt.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        # Determinant = 1 (proper rotation)
        npt.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_arbitrary_rotation_is_orthogonal(self):
        """Any rotation matrix should be orthogonal with det=1."""
        for phi, theta, gamma in [(0.5, -0.3, 0.1), (1.2, 0.7, -0.5), (0, 0, 0)]:
            R = build_rotation_matrix(phi, theta, gamma)
            npt.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
            npt.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_packing_preserves_rotation_data(self):
        """After packing into the uniform buffer, rotation data should be recoverable."""
        phi, theta, gamma = 0.8, -0.4, 0.2
        rotation = build_rotation_matrix(phi, theta, gamma)
        mvp_flat = np.eye(4, dtype=np.float32).flatten()
        color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        frame_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        frame_shape = np.array([14.222, 8.0], dtype=np.float32)

        buf = np.zeros(_UNIFORM_FLOATS, dtype=np.float32)
        _pack_uniforms_3d(mvp_flat, color, rotation, frame_center, frame_shape, 20.0, 1.0, buf)

        # Extract rotation back from buffer (Metal float3x3 layout)
        recovered = np.zeros((3, 3), dtype=np.float32)
        recovered[:, 0] = buf[20:23]
        recovered[:, 1] = buf[24:27]
        recovered[:, 2] = buf[28:31]

        # Float64 -> float32 conversion may lose a bit of precision
        npt.assert_allclose(recovered, rotation.astype(np.float32), atol=1e-6)


class TestBufferPoolAlignmentPure:
    """Verify buffer pool alignment using its pure Python logic (no Metal device)."""

    def test_alignment_formula(self):
        """The 256-byte alignment formula should work correctly.

        The formula: next_offset = (offset + size + 255) & ~255
        """
        alignment = 256

        def align_up(offset: int, size: int) -> int:
            return (offset + size + alignment - 1) & ~(alignment - 1)

        # After staging 40 bytes at offset 0, next offset should be 256
        assert align_up(0, 40) == 256
        # After staging 256 bytes at offset 256, next should be 512
        assert align_up(256, 256) == 512
        # After staging 1 byte at offset 512, next should be 768
        assert align_up(512, 1) == 768
        # Staging exactly 256 at offset 0 -> 256
        assert align_up(0, 256) == 256


# =====================================================================
# Section 2: GPU-dependent tests (skip if shaders fail to compile)
# =====================================================================


def _can_create_metal_camera() -> bool:
    """Check whether MetalCamera can be instantiated (shaders compile OK)."""
    try:
        from manim import config

        original = config.renderer
        config.renderer = "metal"
        try:
            from manim_metal.metal_camera import MetalCamera

            MetalCamera()
            return True
        except Exception:
            return False
        finally:
            config.renderer = original
    except Exception:
        return False


_skip_no_gpu = pytest.mark.skipif(
    not _can_create_metal_camera(),
    reason="Metal shaders failed to compile (WIP include or no GPU)",
)


@pytest.fixture
def metal_config():
    """Set config to use Metal renderer, restore after test."""
    from manim import config

    original = config.renderer
    config.renderer = "metal"
    yield config
    config.renderer = original


@_skip_no_gpu
class TestMetalCameraDefaults:
    """Verify MetalCamera default values for 3D parameters (requires GPU)."""

    def test_default_focal_distance(self, metal_config):
        """Default focal_distance should be 20.0 (matching ThreeDCamera)."""
        from manim_metal.metal_camera import MetalCamera

        cam = MetalCamera()
        assert cam.focal_distance == 20.0

    def test_default_zoom(self, metal_config):
        """Default zoom should be 1.0."""
        from manim_metal.metal_camera import MetalCamera

        cam = MetalCamera()
        assert cam.zoom == 1.0

    def test_default_phi_theta_gamma_zero(self, metal_config):
        """Default camera angles should all be 0."""
        from manim_metal.metal_camera import MetalCamera

        cam = MetalCamera()
        assert cam.phi == 0.0
        assert cam.theta == 0.0
        assert cam.gamma == 0.0

    def test_default_frame_center_origin(self, metal_config):
        """Default frame center should be at the origin."""
        from manim_metal.metal_camera import MetalCamera

        cam = MetalCamera()
        npt.assert_allclose(cam.frame_center, [0.0, 0.0, 0.0], atol=1e-10)

    def test_default_not_3d(self, metal_config):
        """Camera should not be in 3D mode by default."""
        from manim_metal.metal_camera import MetalCamera

        cam = MetalCamera()
        assert not cam._is_3d_active


@_skip_no_gpu
class TestMetalCameraUniformOutput:
    """Verify _make_uniform_data output (requires GPU for MetalCamera init)."""

    def test_make_uniform_2d_output(self, metal_config):
        """_make_uniform_data in 2D mode should match _pack_uniforms_2d."""
        from manim_metal.metal_camera import MetalCamera

        cam = MetalCamera()
        color = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
        buf = cam._make_uniform_data(color)

        assert buf.dtype == np.float32
        assert len(buf) == _UNIFORM_FLOATS
        # is_3d should be 0
        assert buf.view(np.uint32)[40] == 0

    def test_make_uniform_3d_output(self, metal_config):
        """_make_uniform_data in 3D mode should set is_3d=1 and pack rotation."""
        from manim_metal.metal_camera import MetalCamera

        cam = MetalCamera()
        cam.set_phi(0.5)
        cam.reset_rotation_matrix()

        color = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        buf = cam._make_uniform_data(color)

        assert buf.view(np.uint32)[40] == 1
        # Rotation column 0 should be non-zero
        assert not np.allclose(buf[20:23], 0.0)


@_skip_no_gpu
class TestBufferPoolAlignment:
    """Verify that the buffer pool maintains 256-byte alignment."""

    def test_buffer_pool_stage_alignment(self, metal_config):
        """Each stage() call should return a 256-byte-aligned offset."""
        from manim_metal.metal_context import MetalContext

        ctx = MetalContext(64, 64)
        pool = ctx.buffer_pool
        pool.reset()

        offsets = []
        for size in [10, 100, 7, 256, 1]:
            data = np.zeros(size, dtype=np.float32)
            off = pool.stage(data)
            offsets.append(off)

        assert offsets[0] == 0
        for off in offsets[1:]:
            assert off % 256 == 0, f"Offset {off} is not 256-byte aligned"


# =====================================================================
# Section 3: Phase 2 lighting uniform tests (skip until implemented)
# =====================================================================

# The lighting.h shader header defines the extended Uniforms struct:
#
#   Offset  Size  Field
#   160     4B    uint     is_3d
#   164     4B    uint     use_lighting
#   168     4B    float    ambient_strength
#   172     4B    float    diffuse_strength
#   176     16B   float3   light_position (+4B pad)
#   192     16B   float3   light_color (+4B pad)
#   208     4B    float    specular_strength
#   212     4B    float    shininess
#   216     40B   padding
#   Total: 256B
#
# Float index mapping (divide byte offset by 4):
#   use_lighting:     index 41
#   ambient_strength: index 42
#   diffuse_strength: index 43
#   light_position:   indices 44-46 (pad at 47)
#   light_color:      indices 48-50 (pad at 51)
#   specular_strength: index 52
#   shininess:        index 53


class TestLightingUniformPacking:
    """Verify that lighting parameters are packed into the uniform buffer.

    These tests validate the CPU-side packing functions that write lighting fields
    into the uniform buffer. The byte offsets match the lighting.h Uniforms struct.
    """

    def test_use_lighting_flag_at_index_41(self):
        """use_lighting (uint) should be at float index 41 (byte 164)."""
        from manim_metal.metal_camera import _set_use_lighting_flag

        buf = np.zeros(_UNIFORM_FLOATS, dtype=np.float32)
        _set_use_lighting_flag(buf, 1)
        assert buf.view(np.uint32)[41] == 1
        # Verify byte offset 164
        raw = buf.view(np.uint8)
        assert raw[164] == 1
        assert raw[165] == 0

    def test_ambient_strength_at_index_42(self):
        """ambient_strength (float) should be at float index 42 (byte 168)."""
        from manim_metal.metal_camera import _pack_lighting_params

        buf = np.zeros(_UNIFORM_FLOATS, dtype=np.float32)
        _pack_lighting_params(
            buf,
            ambient_strength=0.2,
            diffuse_strength=0.7,
            light_position=np.array([-4.0, 5.0, 10.0]),
            light_color=np.array([1.0, 1.0, 1.0]),
            specular_strength=0.5,
            shininess=32.0,
        )
        npt.assert_allclose(buf[42], 0.2, atol=1e-6)

    def test_diffuse_strength_at_index_43(self):
        """diffuse_strength (float) should be at float index 43 (byte 172)."""
        from manim_metal.metal_camera import _pack_lighting_params

        buf = np.zeros(_UNIFORM_FLOATS, dtype=np.float32)
        _pack_lighting_params(
            buf,
            ambient_strength=0.2,
            diffuse_strength=0.7,
            light_position=np.array([-4.0, 5.0, 10.0]),
            light_color=np.array([1.0, 1.0, 1.0]),
            specular_strength=0.5,
            shininess=32.0,
        )
        npt.assert_allclose(buf[43], 0.7, atol=1e-6)

    def test_light_position_at_indices_44_46(self):
        """light_position (float3) should be at indices 44-46 (bytes 176-187), pad at 47."""
        from manim_metal.metal_camera import _pack_lighting_params

        buf = np.zeros(_UNIFORM_FLOATS, dtype=np.float32)
        light_pos = np.array([-4.0, 5.0, 10.0])
        _pack_lighting_params(
            buf,
            ambient_strength=0.2,
            diffuse_strength=0.7,
            light_position=light_pos,
            light_color=np.array([1.0, 1.0, 1.0]),
            specular_strength=0.5,
            shininess=32.0,
        )
        npt.assert_allclose(buf[44:47], light_pos.astype(np.float32), atol=1e-6)
        # Padding at index 47 should be 0
        npt.assert_allclose(buf[47], 0.0, atol=1e-10)

    def test_light_color_at_indices_48_50(self):
        """light_color (float3) should be at indices 48-50 (bytes 192-203), pad at 51."""
        from manim_metal.metal_camera import _pack_lighting_params

        buf = np.zeros(_UNIFORM_FLOATS, dtype=np.float32)
        light_color = np.array([0.8, 0.9, 1.0])
        _pack_lighting_params(
            buf,
            ambient_strength=0.2,
            diffuse_strength=0.7,
            light_position=np.array([-4.0, 5.0, 10.0]),
            light_color=light_color,
            specular_strength=0.5,
            shininess=32.0,
        )
        npt.assert_allclose(buf[48:51], light_color.astype(np.float32), atol=1e-6)
        # Padding at index 51 should be 0
        npt.assert_allclose(buf[51], 0.0, atol=1e-10)

    def test_specular_strength_at_index_52(self):
        """specular_strength (float) should be at index 52 (byte 208)."""
        from manim_metal.metal_camera import _pack_lighting_params

        buf = np.zeros(_UNIFORM_FLOATS, dtype=np.float32)
        _pack_lighting_params(
            buf,
            ambient_strength=0.2,
            diffuse_strength=0.7,
            light_position=np.array([-4.0, 5.0, 10.0]),
            light_color=np.array([1.0, 1.0, 1.0]),
            specular_strength=0.5,
            shininess=32.0,
        )
        npt.assert_allclose(buf[52], 0.5, atol=1e-6)

    def test_shininess_at_index_53(self):
        """shininess (float) should be at index 53 (byte 212)."""
        from manim_metal.metal_camera import _pack_lighting_params

        buf = np.zeros(_UNIFORM_FLOATS, dtype=np.float32)
        _pack_lighting_params(
            buf,
            ambient_strength=0.2,
            diffuse_strength=0.7,
            light_position=np.array([-4.0, 5.0, 10.0]),
            light_color=np.array([1.0, 1.0, 1.0]),
            specular_strength=0.5,
            shininess=32.0,
        )
        npt.assert_allclose(buf[53], 32.0, atol=1e-6)

    def test_uniform_still_256_bytes_with_lighting(self):
        """The uniform struct should remain exactly 256 bytes after adding lighting."""
        assert _UNIFORM_SIZE == 256
        assert _UNIFORM_SIZE % 256 == 0

    def test_lighting_fields_fit_in_padding(self):
        """All lighting fields must fit within bytes 164-255 (former padding region).

        Byte budget: 256 - 164 = 92 bytes available for lighting + padding.
        Lighting needs: use_lighting(4) + ambient(4) + diffuse(4) + light_pos(16)
                      + light_color(16) + specular(4) + shininess(4) = 52 bytes.
        Remaining padding: 92 - 52 = 40 bytes. Should be sufficient.
        """
        lighting_bytes_needed = 4 + 4 + 4 + 16 + 16 + 4 + 4  # = 52
        available_padding = 256 - 164  # = 92
        assert lighting_bytes_needed <= available_padding


@_skip_no_gpu
class TestLightingCameraAPI:
    """Verify MetalCamera light property getters/setters."""

    def test_default_light_position(self, metal_config):
        """MetalCamera should have a default light_position."""
        from manim_metal.metal_camera import MetalCamera

        cam = MetalCamera()
        pos = cam.get_light_position()
        assert isinstance(pos, np.ndarray)
        assert pos.shape == (3,)
        assert np.all(np.isfinite(pos))

    def test_set_light_position(self, metal_config):
        """Setting light_position should be retrievable."""
        from manim_metal.metal_camera import MetalCamera

        cam = MetalCamera()
        new_pos = np.array([5.0, 5.0, 10.0])
        cam.set_light_position(new_pos)
        npt.assert_allclose(cam.get_light_position(), new_pos, atol=1e-10)

    def test_default_ambient_strength(self, metal_config):
        """Default ambient_strength should be in (0, 1)."""
        from manim_metal.metal_camera import MetalCamera

        cam = MetalCamera()
        assert 0.0 < cam.ambient_strength <= 1.0

    def test_default_diffuse_strength(self, metal_config):
        """Default diffuse_strength should be in (0, 1)."""
        from manim_metal.metal_camera import MetalCamera

        cam = MetalCamera()
        assert 0.0 < cam.diffuse_strength <= 1.0

    def test_default_specular_strength(self, metal_config):
        """Default specular_strength should be in [0, 1]."""
        from manim_metal.metal_camera import MetalCamera

        cam = MetalCamera()
        assert 0.0 <= cam.specular_strength <= 1.0

    def test_default_shininess(self, metal_config):
        """Default shininess should be > 0 (typical: 16 or 32)."""
        from manim_metal.metal_camera import MetalCamera

        cam = MetalCamera()
        assert cam.shininess > 0.0

    def test_set_lighting_parameters(self, metal_config):
        """All lighting parameters should be settable via set_* methods."""
        from manim_metal.metal_camera import MetalCamera

        cam = MetalCamera()
        cam.set_ambient_strength(0.3)
        cam.set_diffuse_strength(0.6)
        cam.set_specular_strength(0.5)
        cam.set_shininess(64.0)

        npt.assert_allclose(cam.ambient_strength, 0.3, atol=1e-10)
        npt.assert_allclose(cam.diffuse_strength, 0.6, atol=1e-10)
        npt.assert_allclose(cam.specular_strength, 0.5, atol=1e-10)
        npt.assert_allclose(cam.shininess, 64.0, atol=1e-10)

    def test_lighting_not_applied_in_2d(self, metal_config):
        """In 2D mode, use_lighting=0 by default in the uniform buffer."""
        from manim_metal.metal_camera import MetalCamera

        cam = MetalCamera()
        assert not cam._is_3d_active

        color = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
        buf = cam._make_uniform_data(color)

        is_3d = buf.view(np.uint32)[40]
        assert is_3d == 0
        # use_lighting should also be 0 when not explicitly requested
        use_lighting = buf.view(np.uint32)[41]
        assert use_lighting == 0

    def test_make_uniform_data_with_lighting(self, metal_config):
        """_make_uniform_data(use_lighting=True) should pack all lighting params."""
        from manim_metal.metal_camera import MetalCamera

        cam = MetalCamera()
        cam.set_phi(0.5)
        cam.reset_rotation_matrix()

        color = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
        buf = cam._make_uniform_data(color, use_lighting=True)

        # use_lighting flag should be 1
        assert buf.view(np.uint32)[41] == 1
        # ambient_strength should be default (0.5)
        npt.assert_allclose(buf[42], 0.5, atol=1e-6)
        # diffuse_strength should be default (0.5, Cairo intensity coefficient)
        npt.assert_allclose(buf[43], 0.5, atol=1e-6)
        # light_position should be default (-7, -9, 10) matching Cairo ThreeDCamera
        npt.assert_allclose(buf[44:47], [-7.0, -9.0, 10.0], atol=1e-5)
        # light_color should be default (1, 1, 1)
        npt.assert_allclose(buf[48:51], [1.0, 1.0, 1.0], atol=1e-6)
        # specular_strength should be default (0.0, unused in Cairo formula)
        npt.assert_allclose(buf[52], 0.0, atol=1e-6)
        # shininess should be default (3.0, Cairo cubic exponent)
        npt.assert_allclose(buf[53], 3.0, atol=1e-6)
