"""Metal device, command queue, pipeline state, and offscreen render target management."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import Metal  # pyobjc-framework-Metal
import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

_SHADER_DIR = Path(__file__).parent / "shaders"

# Metal requires buffer offsets to be 256-byte aligned
_BUFFER_ALIGNMENT = 256
_INITIAL_POOL_SIZE = 16 * 1024 * 1024  # 16 MB


class BufferPool:
    """Staging allocator that collects frame data into a single MTLBuffer.

    All vertex/uniform data is written into a Python-side bytearray during
    the staging pass.  :meth:`finalize` creates a single ``MTLBuffer`` from
    the accumulated data—reducing per-object ``newBufferWithBytes`` calls to
    one call per frame.

    Parameters
    ----------
    device
        The Metal device to allocate from.
    initial_size
        Initial staging capacity in bytes.
    """

    def __init__(self, device, initial_size: int = _INITIAL_POOL_SIZE) -> None:
        self._device = device
        self._data = bytearray(initial_size)
        self._capacity = initial_size
        self._offset = 0

    def reset(self) -> None:
        """Rewind the allocator to the start. Call at the beginning of each frame."""
        self._offset = 0

    def stage(self, data: np.ndarray) -> int:
        """Copy *data* into the staging bytearray and return its byte offset.

        Parameters
        ----------
        data
            NumPy array to copy.

        Returns
        -------
        int
            Byte offset into the shared buffer (available after :meth:`finalize`).
        """
        byte_data = data.tobytes()
        size = len(byte_data)

        needed = self._offset + size
        if needed > self._capacity:
            self._grow(needed)

        self._data[self._offset : self._offset + size] = byte_data
        offset = self._offset

        # Advance with 256-byte alignment
        self._offset = (self._offset + size + _BUFFER_ALIGNMENT - 1) & ~(
            _BUFFER_ALIGNMENT - 1
        )
        return offset

    def finalize(self):
        """Create a single MTLBuffer from all staged data.

        Returns
        -------
        MTLBuffer or None
            The shared buffer, or ``None`` if nothing was staged.
        """
        if self._offset == 0:
            return None
        return self._device.newBufferWithBytes_length_options_(
            bytes(self._data[: self._offset]),
            self._offset,
            Metal.MTLResourceStorageModeShared,
        )

    def _grow(self, min_capacity: int) -> None:
        new_cap = self._capacity
        while new_cap < min_capacity:
            new_cap *= 2
        new_data = bytearray(new_cap)
        new_data[: self._offset] = self._data[: self._offset]
        self._data = new_data
        self._capacity = new_cap


class MetalContext:
    """Manages Metal device, command queue, compiled shaders, and render targets.

    Parameters
    ----------
    width
        Render target width in pixels.
    height
        Render target height in pixels.
    """

    MSAA_SAMPLE_COUNT = 4

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

        # Device & queue
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("No Metal-compatible GPU found")
        self.command_queue = self.device.newCommandQueue()

        # Compile shader libraries
        self._libraries: dict[str, Metal.MTLLibrary] = {}
        self._compile_shaders()

        # Create render targets (MSAA + resolve)
        self._create_render_target()
        self._create_msaa_target()
        self._create_stencil_target()

        # Create pipeline states
        self._fill_stencil_pso = self._make_fill_stencil_pipeline()
        self._fill_cover_pso = self._make_fill_cover_pipeline()
        self._stroke_pso = self._make_stroke_pipeline()

        # Depth-stencil states
        self._stencil_increment_dss = self._make_stencil_increment_state()
        self._stencil_nonzero_dss = self._make_stencil_nonzero_state()
        self._stencil_disabled_dss = self._make_stencil_disabled_state()

        # Buffer pool for sub-allocation
        self.buffer_pool = BufferPool(self.device)

    # ------------------------------------------------------------------
    # Shader compilation
    # ------------------------------------------------------------------

    def _compile_shaders(self) -> None:
        for name in ("fill", "stroke", "blit"):
            src_path = _SHADER_DIR / f"{name}.metal"
            source = src_path.read_text()
            library, error = self.device.newLibraryWithSource_options_error_(source, None, None)
            if error is not None:
                raise RuntimeError(f"Failed to compile {name}.metal: {error}")
            self._libraries[name] = library

    def _get_function(self, library_name: str, function_name: str):
        lib = self._libraries[library_name]
        fn = lib.newFunctionWithName_(function_name)
        if fn is None:
            raise RuntimeError(f"Function '{function_name}' not found in {library_name}.metal")
        return fn

    # ------------------------------------------------------------------
    # Render targets
    # ------------------------------------------------------------------

    def _create_render_target(self) -> None:
        """Non-MSAA resolve target (CPU-readable for readback)."""
        make_tex = Metal.MTLTextureDescriptor
        desc = make_tex.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
            Metal.MTLPixelFormatBGRA8Unorm,
            self.width,
            self.height,
            False,
        )
        desc.setUsage_(Metal.MTLTextureUsageRenderTarget | Metal.MTLTextureUsageShaderRead)
        desc.setStorageMode_(Metal.MTLStorageModeShared)  # UMA — CPU-readable
        self.render_target = self.device.newTextureWithDescriptor_(desc)

    def _create_msaa_target(self) -> None:
        """4x MSAA color texture — rendering goes here, then resolves to render_target."""
        desc = Metal.MTLTextureDescriptor.alloc().init()
        desc.setTextureType_(Metal.MTLTextureType2DMultisample)
        desc.setPixelFormat_(Metal.MTLPixelFormatBGRA8Unorm)
        desc.setWidth_(self.width)
        desc.setHeight_(self.height)
        desc.setSampleCount_(self.MSAA_SAMPLE_COUNT)
        desc.setUsage_(Metal.MTLTextureUsageRenderTarget)
        desc.setStorageMode_(Metal.MTLStorageModePrivate)
        self.msaa_target = self.device.newTextureWithDescriptor_(desc)

    def _create_stencil_target(self) -> None:
        """MSAA stencil texture (must match MSAA sample count)."""
        desc = Metal.MTLTextureDescriptor.alloc().init()
        desc.setTextureType_(Metal.MTLTextureType2DMultisample)
        desc.setPixelFormat_(Metal.MTLPixelFormatStencil8)
        desc.setWidth_(self.width)
        desc.setHeight_(self.height)
        desc.setSampleCount_(self.MSAA_SAMPLE_COUNT)
        desc.setUsage_(Metal.MTLTextureUsageRenderTarget)
        desc.setStorageMode_(Metal.MTLStorageModePrivate)
        self.stencil_target = self.device.newTextureWithDescriptor_(desc)

    # ------------------------------------------------------------------
    # Pipeline states
    # ------------------------------------------------------------------

    def _make_fill_stencil_pipeline(self):
        desc = Metal.MTLRenderPipelineDescriptor.alloc().init()
        desc.setSampleCount_(self.MSAA_SAMPLE_COUNT)
        desc.setVertexFunction_(self._get_function("fill", "fill_stencil_vertex"))
        desc.setFragmentFunction_(self._get_function("fill", "fill_stencil_fragment"))
        desc.colorAttachments().objectAtIndexedSubscript_(0).setPixelFormat_(
            Metal.MTLPixelFormatBGRA8Unorm
        )
        # Disable color writes for stencil pass
        desc.colorAttachments().objectAtIndexedSubscript_(0).setWriteMask_(
            Metal.MTLColorWriteMaskNone
        )
        desc.setStencilAttachmentPixelFormat_(Metal.MTLPixelFormatStencil8)
        pso, error = self.device.newRenderPipelineStateWithDescriptor_error_(desc, None)
        if error is not None:
            raise RuntimeError(f"Fill stencil pipeline creation failed: {error}")
        return pso

    def _make_fill_cover_pipeline(self):
        desc = Metal.MTLRenderPipelineDescriptor.alloc().init()
        desc.setSampleCount_(self.MSAA_SAMPLE_COUNT)
        desc.setVertexFunction_(self._get_function("fill", "fill_cover_vertex"))
        desc.setFragmentFunction_(self._get_function("fill", "fill_cover_fragment"))
        ca = desc.colorAttachments().objectAtIndexedSubscript_(0)
        ca.setPixelFormat_(Metal.MTLPixelFormatBGRA8Unorm)
        # Enable alpha blending
        ca.setBlendingEnabled_(True)
        ca.setSourceRGBBlendFactor_(Metal.MTLBlendFactorSourceAlpha)
        ca.setDestinationRGBBlendFactor_(Metal.MTLBlendFactorOneMinusSourceAlpha)
        ca.setSourceAlphaBlendFactor_(Metal.MTLBlendFactorOne)
        ca.setDestinationAlphaBlendFactor_(Metal.MTLBlendFactorOneMinusSourceAlpha)
        desc.setStencilAttachmentPixelFormat_(Metal.MTLPixelFormatStencil8)
        pso, error = self.device.newRenderPipelineStateWithDescriptor_error_(desc, None)
        if error is not None:
            raise RuntimeError(f"Fill cover pipeline creation failed: {error}")
        return pso

    def _make_stroke_pipeline(self):
        desc = Metal.MTLRenderPipelineDescriptor.alloc().init()
        desc.setSampleCount_(self.MSAA_SAMPLE_COUNT)
        desc.setVertexFunction_(self._get_function("stroke", "stroke_vertex"))
        desc.setFragmentFunction_(self._get_function("stroke", "stroke_fragment"))
        ca = desc.colorAttachments().objectAtIndexedSubscript_(0)
        ca.setPixelFormat_(Metal.MTLPixelFormatBGRA8Unorm)
        ca.setBlendingEnabled_(True)
        ca.setSourceRGBBlendFactor_(Metal.MTLBlendFactorSourceAlpha)
        ca.setDestinationRGBBlendFactor_(Metal.MTLBlendFactorOneMinusSourceAlpha)
        ca.setSourceAlphaBlendFactor_(Metal.MTLBlendFactorOne)
        ca.setDestinationAlphaBlendFactor_(Metal.MTLBlendFactorOneMinusSourceAlpha)
        desc.setStencilAttachmentPixelFormat_(Metal.MTLPixelFormatStencil8)
        pso, error = self.device.newRenderPipelineStateWithDescriptor_error_(desc, None)
        if error is not None:
            raise RuntimeError(f"Stroke pipeline creation failed: {error}")
        return pso

    # ------------------------------------------------------------------
    # Depth-stencil states
    # ------------------------------------------------------------------

    def _make_stencil_increment_state(self):
        """DSS for stencil pass: always pass, invert stencil on both front/back."""
        desc = Metal.MTLDepthStencilDescriptor.alloc().init()
        stencil_desc = Metal.MTLStencilDescriptor.alloc().init()
        stencil_desc.setStencilCompareFunction_(Metal.MTLCompareFunctionAlways)
        stencil_desc.setDepthStencilPassOperation_(Metal.MTLStencilOperationInvert)
        stencil_desc.setStencilFailureOperation_(Metal.MTLStencilOperationKeep)
        stencil_desc.setDepthFailureOperation_(Metal.MTLStencilOperationKeep)
        stencil_desc.setReadMask_(0xFF)
        stencil_desc.setWriteMask_(0xFF)
        desc.setFrontFaceStencil_(stencil_desc)
        desc.setBackFaceStencil_(stencil_desc)
        return self.device.newDepthStencilStateWithDescriptor_(desc)

    def _make_stencil_nonzero_state(self):
        """DSS for cover pass: pass where stencil != 0, then reset to 0."""
        desc = Metal.MTLDepthStencilDescriptor.alloc().init()
        stencil_desc = Metal.MTLStencilDescriptor.alloc().init()
        stencil_desc.setStencilCompareFunction_(Metal.MTLCompareFunctionNotEqual)
        stencil_desc.setDepthStencilPassOperation_(Metal.MTLStencilOperationZero)
        stencil_desc.setStencilFailureOperation_(Metal.MTLStencilOperationKeep)
        stencil_desc.setDepthFailureOperation_(Metal.MTLStencilOperationKeep)
        stencil_desc.setReadMask_(0xFF)
        stencil_desc.setWriteMask_(0xFF)
        desc.setFrontFaceStencil_(stencil_desc)
        desc.setBackFaceStencil_(stencil_desc)
        return self.device.newDepthStencilStateWithDescriptor_(desc)

    def _make_stencil_disabled_state(self):
        """DSS with stencil test disabled (for stroke rendering)."""
        desc = Metal.MTLDepthStencilDescriptor.alloc().init()
        return self.device.newDepthStencilStateWithDescriptor_(desc)

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def make_render_pass_descriptor(self, clear: bool = False, clear_color=None):
        """Create a render pass descriptor targeting the MSAA texture.

        Rendering goes into ``msaa_target``; at store time Metal resolves
        into ``render_target`` (non-MSAA, CPU-readable).

        Parameters
        ----------
        clear
            If True, clear the render target at load time.
        clear_color
            (r, g, b, a) tuple in [0, 1]. Used only if clear=True.
        """
        rpd = Metal.MTLRenderPassDescriptor.renderPassDescriptor()

        ca = rpd.colorAttachments().objectAtIndexedSubscript_(0)
        ca.setTexture_(self.msaa_target)
        ca.setResolveTexture_(self.render_target)
        if clear and clear_color is not None:
            ca.setLoadAction_(Metal.MTLLoadActionClear)
            ca.setClearColor_(Metal.MTLClearColor(*clear_color))
        else:
            ca.setLoadAction_(Metal.MTLLoadActionLoad)
        ca.setStoreAction_(Metal.MTLStoreActionMultisampleResolve)

        sa = rpd.stencilAttachment()
        sa.setTexture_(self.stencil_target)
        if clear:
            sa.setLoadAction_(Metal.MTLLoadActionClear)
            sa.setClearStencil_(0)
        else:
            sa.setLoadAction_(Metal.MTLLoadActionLoad)
        sa.setStoreAction_(Metal.MTLStoreActionDontCare)

        return rpd

    def make_buffer(self, data: npt.NDArray) -> Metal.MTLBuffer:
        """Create a Metal buffer from a numpy array (shared memory / zero-copy on UMA)."""
        byte_data = data.tobytes()
        return self.device.newBufferWithBytes_length_options_(
            byte_data,
            len(byte_data),
            Metal.MTLResourceStorageModeShared,
        )

    def render_texture_to_numpy(self) -> npt.NDArray[np.uint8]:
        """Read the render target into a NumPy RGBA array.

        Returns
        -------
        np.ndarray
            Shape (height, width, 4), dtype uint8, RGBA order.
        """
        w = self.width
        h = self.height
        bytes_per_row = w * 4

        buf = bytearray(h * bytes_per_row)
        region = Metal.MTLRegionMake2D(0, 0, w, h)
        self.render_target.getBytes_bytesPerRow_fromRegion_mipmapLevel_(
            buf, bytes_per_row, region, 0
        )

        # BGRA -> RGBA
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4).copy()
        arr[:, :, [0, 2]] = arr[:, :, [2, 0]]
        return arr

    def clear_render_target(self, color: tuple[float, float, float, float]) -> None:
        """Clear the render target to the given RGBA color."""
        cmd = self.command_queue.commandBuffer()
        rpd = self.make_render_pass_descriptor(clear=True, clear_color=color)
        enc = cmd.renderCommandEncoderWithDescriptor_(rpd)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
