# Benchmark: Cairo vs Metal renderer

Per-frame render time for `camera.capture_mobjects` on identical mobjects (tessellation + rasterization/GPU + readback). Process startup and video muxing are excluded — they are identical for both backends. Reported value is the median of up to 11 iterations after 3 warmups.

**Hardware:** Apple M3 Pro, 11-core, 18 GB, macOS 26.5

- **Metal (warm)** — geometry cache hot: steady-state animation where geometry is static and only the camera moves (the common case).
- **Metal (cold)** — geometry cache cleared every frame: geometry changes every frame (e.g. a `Transform`). Cairo re-rasterizes every frame either way.

## Summary

- **3D lit surfaces are Metal's biggest win (≈3–6×).** Cairo shades every facet on the CPU; Metal does the projection on the GPU. Even with the cache cold, Metal stays well ahead.
- **Static-geometry 2D & text: Metal ≈1.4–2.5× faster**, and the lead grows with resolution (GPU rasterization scales better than Cairo's CPU rasterizer).
- **When geometry changes every frame, CPU tessellation dominates** and Metal can be *slower* than Cairo for 2D/text (the ear-clipping pass re-runs per shape/glyph). This is the main optimization target; 3D still wins because Cairo's facet shading is the larger cost. In practice only the *moving* objects re-tessellate, so real animations sit between the warm and cold columns.

## 480p (854×480)

| Workload | Cairo | Metal (warm) | Speedup | Metal (cold) | Speedup |
|---|--:|--:|--:|--:|--:|
| 2D shapes x50 | 2.54 ms | 1.72 ms | **1.5×** | 11.45 ms | 0.2× |
| 2D shapes x200 | 7.97 ms | 4.93 ms | **1.6×** | 41.42 ms | 0.2× |
| Text (~55 glyphs) | 4.06 ms | 1.95 ms | **2.1×** | 61.70 ms | 0.1× |
| MathTex | 1.55 ms | 1.07 ms | **1.4×** | 15.01 ms | 0.1× |
| 3D Sphere (lit) | 50.66 ms | 10.93 ms | **4.6×** | 16.77 ms | 3.0× |
| 3D scene x3 (lit) | 209.83 ms | 51.38 ms | **4.1×** | 76.65 ms | 2.7× |

## 1080p (1920×1080)

| Workload | Cairo | Metal (warm) | Speedup | Metal (cold) | Speedup |
|---|--:|--:|--:|--:|--:|
| 2D shapes x50 | 4.91 ms | 2.02 ms | **2.4×** | 11.15 ms | 0.4× |
| 2D shapes x200 | 11.44 ms | 5.44 ms | **2.1×** | 41.43 ms | 0.3× |
| Text (~55 glyphs) | 5.42 ms | 2.62 ms | **2.1×** | 67.35 ms | 0.1× |
| MathTex | 2.92 ms | 1.67 ms | **1.7×** | 16.48 ms | 0.2× |
| 3D Sphere (lit) | 68.59 ms | 11.35 ms | **6.0×** | 18.43 ms | 3.7× |
| 3D scene x3 (lit) | 243.70 ms | 52.53 ms | **4.6×** | 80.05 ms | 3.0× |

## 4K (3840×2160)

| Workload | Cairo | Metal (warm) | Speedup | Metal (cold) | Speedup |
|---|--:|--:|--:|--:|--:|
| 2D shapes x50 | 10.03 ms | 3.79 ms | **2.6×** | 13.42 ms | 0.7× |
| 2D shapes x200 | 20.08 ms | 7.51 ms | **2.7×** | 44.33 ms | 0.5× |
| Text (~55 glyphs) | 7.13 ms | 3.68 ms | **1.9×** | 63.77 ms | 0.1× |
| MathTex | 4.74 ms | 3.27 ms | **1.5×** | 16.26 ms | 0.3× |
| 3D Sphere (lit) | 108.09 ms | 22.35 ms | **4.8×** | 34.47 ms | 3.1× |
| 3D scene x3 (lit) | 297.43 ms | 117.73 ms | **2.5×** | 189.63 ms | 1.6× |

## Notes

- Methodology: `uv run python benchmark.py`. Both backends render the exact same `VMobject` list through their respective cameras; 3D uses Cairo `ThreeDCamera` vs `MetalCamera` with matching phi/theta/zoom.
- Numbers are render-only and will vary with thermal state and machine load.
- Accuracy (not speed) parity is measured separately by `compare_metric.py` / `compare_video.py`.
