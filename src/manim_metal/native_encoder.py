"""Native Metal draw-op encoder — compiles and loads fast_encode.m on first use.

Falls back to pure-Python encoding if compilation fails (e.g., no Xcode CLT).
"""

from __future__ import annotations

import ctypes
import logging
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_NATIVE_DIR = Path(__file__).parent / "native"
_SRC_PATH = _NATIVE_DIR / "fast_encode.m"
_LIB_PATH = _NATIVE_DIR / "fast_encode.dylib"

# Loaded shared library (None = not yet attempted, False = failed)
_lib: ctypes.CDLL | None | bool = None


def _compile() -> bool:
    """Compile fast_encode.m → fast_encode.dylib.  Returns True on success."""
    try:
        subprocess.run(
            [
                "clang",
                "-shared",
                "-fPIC",
                "-O2",
                "-framework",
                "Metal",
                "-framework",
                "Foundation",
                "-lobjc",
                "-o",
                str(_LIB_PATH),
                str(_SRC_PATH),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("Native encoder compilation failed: %s", exc)
        return False


def _load_lib() -> ctypes.CDLL | None:
    """Load (and compile if needed) the native encoder library."""
    global _lib  # noqa: PLW0603

    if _lib is not None:
        return _lib if _lib is not False else None

    # Compile if .dylib is missing or older than .m source
    need_compile = not _LIB_PATH.exists()
    if not need_compile:
        need_compile = _SRC_PATH.stat().st_mtime > _LIB_PATH.stat().st_mtime

    if need_compile:
        if not _compile():
            _lib = False
            return None

    try:
        lib = ctypes.CDLL(str(_LIB_PATH))
    except OSError as exc:
        logger.warning("Failed to load native encoder: %s", exc)
        _lib = False
        return None

    # void encode_draw_ops(void*, void*, const DrawOp*, int32_t,
    #                      void*, void*, void*, void*, void*, void*,
    #                      void*, void*)
    lib.encode_draw_ops.restype = None
    lib.encode_draw_ops.argtypes = [
        ctypes.c_void_p,  # encoder
        ctypes.c_void_p,  # buffer
        ctypes.c_void_p,  # ops array pointer
        ctypes.c_int32,  # n_ops
        ctypes.c_void_p,  # fill_stencil_pso
        ctypes.c_void_p,  # fill_cover_pso
        ctypes.c_void_p,  # stroke_pso
        ctypes.c_void_p,  # stencil_inc_dss
        ctypes.c_void_p,  # stencil_nz_dss
        ctypes.c_void_p,  # stencil_disabled_dss
        ctypes.c_void_p,  # fill_cover_lit_pso
        ctypes.c_void_p,  # stroke_lit_pso
    ]

    _lib = lib
    logger.info("Native Metal encoder loaded from %s", _LIB_PATH)
    return lib


def is_available() -> bool:
    """Return True if the native encoder can be loaded."""
    return _load_lib() is not None


def encode_draw_ops(
    encoder,
    shared_buf,
    ops_array: np.ndarray,
    ctx,
) -> None:
    """Encode all draw ops in one native call.

    Parameters
    ----------
    encoder
        A pyobjc ``MTLRenderCommandEncoder``.
    shared_buf
        The shared ``MTLBuffer`` with all vertex/uniform data.
    ops_array
        NumPy int32 array of shape (n_ops, 4), each row is
        [kind, vert_offset, vert_count, uniform_offset].
    ctx
        The ``MetalContext`` (provides PSOs and DSSs).
    """
    import objc

    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Native encoder not available")

    n_ops = len(ops_array)
    if n_ops == 0:
        return

    # Ensure contiguous int32
    ops = np.ascontiguousarray(ops_array, dtype=np.int32)

    lib.encode_draw_ops(
        objc.pyobjc_id(encoder),
        objc.pyobjc_id(shared_buf),
        ops.ctypes.data,
        n_ops,
        objc.pyobjc_id(ctx._fill_stencil_pso),
        objc.pyobjc_id(ctx._fill_cover_pso),
        objc.pyobjc_id(ctx._stroke_pso),
        objc.pyobjc_id(ctx._stencil_increment_dss),
        objc.pyobjc_id(ctx._stencil_nonzero_dss),
        objc.pyobjc_id(ctx._stencil_disabled_dss),
        objc.pyobjc_id(ctx._fill_cover_lit_pso),
        objc.pyobjc_id(ctx._stroke_lit_pso),
    )
