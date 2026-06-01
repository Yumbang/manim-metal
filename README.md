# manim-metal

A **Metal rendering backend for [Manim Community Edition](https://www.manim.community/)**, built
for Apple Silicon. It is a drop-in plugin (not a fork): `import manim_metal` monkey-patches Manim so
you can set `config.renderer = "metal"` and render the *same* scenes you already have on the GPU.

The design goal is **parity with the Cairo renderer** — Metal output should be essentially
pixel-identical to Cairo, not a different look. That goal is verified quantitatively and visually
(see [Cairo parity](#cairo-parity) below).

## Requirements

- macOS on Apple Silicon (a Metal-capable GPU)
- Python ≥ 3.13
- [`uv`](https://docs.astral.sh/uv/) for tooling
- Xcode Command Line Tools (for the optional native draw-op encoder; the renderer falls back to a
  pure-Python encoder if `clang` is unavailable)

## Install

```bash
uv sync
```

This installs Manim CE, the pyobjc Metal frameworks, and `mapbox-earcut`.

## Usage

Activate the backend by importing the package (this applies the patches) and selecting the renderer:

```python
import manim_metal  # noqa: F401 — applies the Metal patches at import time
from manim import Scene, Circle, BLUE, WHITE, config

config.renderer = "metal"

class Demo(Scene):
    def construct(self):
        c = Circle(radius=1.5).set_fill(BLUE, 0.7).set_stroke(WHITE, 4)
        self.play(Create(c))
```

From the CLI:

```bash
uv run manim --renderer metal -ql your_scene.py Demo      # video
uv run manim --renderer metal -s  -ql your_scene.py Demo  # last-frame image
```

The plugin is also registered as a Manim plugin entry point (`manim_metal`).

## What works

| Capability | Status vs Cairo |
|---|---|
| 2D shapes (fills, strokes, opacity, concave/holes, overlap) | pixel-identical interiors; only sub-1% anti-aliasing seams differ |
| `Text`, `Tex`, `MathTex` (incl. small sub/superscripts) | pixel-identical interiors |
| 3D unlit (rotated/perspective shapes, depth ordering) | pixel-identical interiors |
| 3D lit surfaces (`Sphere`, `Torus`, `Cylinder`, `Cone`, `Surface`) | essentially identical — Cairo-exact flat per-facet shading |
| Animation / video (Transform, Write, camera & object motion) | frame counts and per-frame output match Cairo |

The remaining difference everywhere is anti-aliasing at edges: Metal uses 8× MSAA, Cairo uses
analytic coverage, so shape/glyph *outlines* differ by a fraction of a pixel while interiors match.

### Known limitations

- **Gradient fills** render as a flat first-stop color (no per-vertex color path yet).
- **`ImageMobject`** (raster images) is not rendered — only `VMobject` geometry is supported.
- 3D lit facets use the *flat average* of Cairo's 2-stop facet gradient (sub-perceptual residual,
  visible only under amplified diff at e.g. a sphere's pole).
- FXAA post-processing exists but is **off by default** (Cairo has no post-process AA, so enabling
  it would diverge from the reference). Opt in with `MetalCamera(fxaa=True)`.

## How it works

Metal reuses Manim's **Cairo** `VMobject` code path (not the OpenGL path), so the patches mainly
teach Manim's Cairo-only assertions to also accept `RendererType.METAL`.

| File | Role |
|---|---|
| `patch.py` | Extends the `RendererType` enum; patches `Scene`/`ThreeDScene`/3D-mobject methods that branch on the renderer |
| `renderer.py` | `MetalRenderer`, duck-typed to `CairoRenderer` |
| `metal_camera.py` | `ThreeDCamera`-compatible camera; two-pass staging + draw-op encoding; Cairo-exact per-facet 3D shading |
| `metal_context.py` | Metal device, shader compilation, pipeline/depth-stencil states, MSAA + depth + stencil targets, zero-copy readback |
| `utils.py` | Bézier→triangle tessellation (ear-clipping for planar fills, fan for 3D), stroke quads, matrices |
| `shaders/` | `fill.metal` (stencil-then-cover), `stroke.metal`, `blit.metal`, `fxaa.metal` |
| `native/fast_encode.m` | Optional native draw-op encoder (one C call replaces thousands of pyobjc bridge crossings) |

Rendering pipeline: vectorized CPU tessellation → single shared `MTLBuffer` → stencil-then-cover
fill + anti-aliased stroke at 8× MSAA → GPU blit + zero-copy readback on the unified-memory
architecture.

## Cairo parity

Two harnesses measure and visualize how close Metal is to Cairo:

```bash
# Still frames: renders a scene battery with BOTH cameras and reports
# per-channel MAE, max diff, and interior-vs-edge divergence.
uv run python compare_metric.py --save-diffs        # writes cairo|metal|diff strips

# Video: renders an animation with both backends, diffs EVERY frame,
# and saves the worst-frame montage.
uv run python compare_video.py VTransform
```

Diff images/videos are written under `media_compare/` and `media_vcmp/` (git-ignored).

## Performance

Render-time (not just accuracy) is benchmarked against Cairo by
[`benchmark.py`](benchmark.py); full results in [BENCHMARK.md](BENCHMARK.md).
On an Apple M3 Pro, per-frame `capture_mobjects` time:

- **3D lit surfaces: ≈3–6× faster** — Cairo shades every facet on the CPU.
- **Static-geometry 2D & text: ≈1.4–2.5× faster**, and the lead grows with resolution.
- **Caveat:** when geometry changes every frame, the CPU tessellation (ear-clipping
  per shape/glyph) re-runs and can make Metal *slower* than Cairo for 2D/text. Only
  the moving objects re-tessellate, so real animations land between the cached and
  uncached numbers. This is the main remaining optimization target.

```bash
uv run python benchmark.py --write-md     # regenerate BENCHMARK.md
```

## Development

```bash
uv run pytest         # test suite
uv run ruff check     # lint
uv run ruff format    # format
uv run python verify.py --compare   # labeled Cairo-vs-Metal side-by-side renders
```

Use `uv` for all tooling (`uv pip`, `uv run`, `uv add`/`uv remove`).
