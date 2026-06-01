// FXAA 3.11 Quality — post-process anti-aliasing compute kernel.
//
// Based on Timothy Lottes' FXAA algorithm (NVIDIA, 2011).
// Operates on the MSAA-resolved framebuffer to smooth remaining aliased edges
// that hardware multisampling cannot fully eliminate.
//
// Pipeline: MSAA render → resolve → FXAA compute → readback blit.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Tuning constants (Quality preset 12)
// ---------------------------------------------------------------------------

// Minimum luminance contrast to trigger edge detection.
// Lower = more aggressive (catches subtler edges).  1/6 is a good balance.
constant float FXAA_EDGE_THRESHOLD = 0.166;

// Absolute minimum threshold — avoids processing very dark areas where
// any noise would trigger false edges.
constant float FXAA_EDGE_THRESHOLD_MIN = 0.0833;

// Sub-pixel AA quality.  0 = disabled, 1 = maximum smoothing.
// 0.75 gives natural results without over-blurring.
constant float FXAA_SUBPIX_QUALITY = 0.75;

// Maximum search iterations along the detected edge.
// More steps = better long-edge AA, but diminishing returns past 12.
constant int FXAA_SEARCH_STEPS = 12;

// Step distances for progressive edge search.  Starts with unit steps
// for precision near the pixel, then widens for efficiency.
constant float SEARCH_STEPS[12] = {
    1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 4.0, 8.0
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static inline float luma(float3 rgb) {
    // ITU-R BT.601 luminance (perceptual weighting).
    return dot(rgb, float3(0.299, 0.587, 0.114));
}

// ---------------------------------------------------------------------------
// FXAA compute kernel
// ---------------------------------------------------------------------------

kernel void fxaa_kernel(
    texture2d<float, access::sample> input  [[texture(0)]],
    texture2d<float, access::write>  output [[texture(1)]],
    sampler                          samp   [[sampler(0)]],
    uint2                            gid    [[thread_position_in_grid]]
) {
    int w = input.get_width();
    int h = input.get_height();
    if (int(gid.x) >= w || int(gid.y) >= h) return;

    float2 invSize = float2(1.0 / float(w), 1.0 / float(h));
    float2 uv = (float2(gid) + 0.5) * invSize;

    // ----- 1. Sample center + 4 cardinal neighbors -----------------------
    float3 rgbM = input.sample(samp, uv).rgb;
    float lumaM = luma(rgbM);

    float lumaN = luma(input.sample(samp, uv + float2(0, -invSize.y)).rgb);
    float lumaS = luma(input.sample(samp, uv + float2(0,  invSize.y)).rgb);
    float lumaW = luma(input.sample(samp, uv + float2(-invSize.x, 0)).rgb);
    float lumaE = luma(input.sample(samp, uv + float2( invSize.x, 0)).rgb);

    float lumaMin = min(lumaM, min(min(lumaN, lumaS), min(lumaW, lumaE)));
    float lumaMax = max(lumaM, max(max(lumaN, lumaS), max(lumaW, lumaE)));
    float range   = lumaMax - lumaMin;

    // Skip if local contrast is below threshold — no edge here.
    if (range < max(FXAA_EDGE_THRESHOLD_MIN, lumaMax * FXAA_EDGE_THRESHOLD)) {
        output.write(float4(rgbM, 1.0), gid);
        return;
    }

    // ----- 2. Sample 4 diagonal neighbors for sub-pixel detection --------
    float lumaNW = luma(input.sample(samp, uv + float2(-invSize.x, -invSize.y)).rgb);
    float lumaNE = luma(input.sample(samp, uv + float2( invSize.x, -invSize.y)).rgb);
    float lumaSW = luma(input.sample(samp, uv + float2(-invSize.x,  invSize.y)).rgb);
    float lumaSE = luma(input.sample(samp, uv + float2( invSize.x,  invSize.y)).rgb);

    // ----- 3. Sub-pixel aliasing factor ----------------------------------
    // The full 3x3 average vs center pixel divergence indicates sub-pixel
    // aliasing (single-pixel features, thin lines, etc.).
    float lumaAvg = (lumaN + lumaS + lumaW + lumaE +
                     lumaNW + lumaNE + lumaSW + lumaSE) / 8.0;
    float subpixA = clamp(abs(lumaAvg - lumaM) / range, 0.0, 1.0);
    float subpixB = (-2.0 * subpixA + 3.0) * subpixA * subpixA;  // smoothstep
    float subpixC = subpixB * subpixB * FXAA_SUBPIX_QUALITY;

    // ----- 4. Edge orientation: horizontal or vertical? ------------------
    float edgeH = abs(-2.0 * lumaW + lumaNW + lumaSW) +
                  abs(-2.0 * lumaM + lumaN  + lumaS ) * 2.0 +
                  abs(-2.0 * lumaE + lumaNE + lumaSE);
    float edgeV = abs(-2.0 * lumaN + lumaNW + lumaNE) +
                  abs(-2.0 * lumaM + lumaW  + lumaE ) * 2.0 +
                  abs(-2.0 * lumaS + lumaSW + lumaSE);
    bool isHorizontal = (edgeH >= edgeV);

    // ----- 5. Pick steeper side (edge normal direction) ------------------
    float luma1 = isHorizontal ? lumaN : lumaW;
    float luma2 = isHorizontal ? lumaS : lumaE;
    float gradient1 = abs(luma1 - lumaM);
    float gradient2 = abs(luma2 - lumaM);
    bool is1Steeper = gradient1 >= gradient2;

    float gradientScaled = 0.25 * max(gradient1, gradient2);
    float stepLength = isHorizontal ? invSize.y : invSize.x;

    float lumaLocalAvg;
    if (is1Steeper) {
        stepLength = -stepLength;
        lumaLocalAvg = 0.5 * (luma1 + lumaM);
    } else {
        lumaLocalAvg = 0.5 * (luma2 + lumaM);
    }

    // ----- 6. Search along edge in both directions -----------------------
    // Start half a pixel into the edge normal direction.
    float2 currentUV = uv;
    if (isHorizontal) {
        currentUV.y += stepLength * 0.5;
    } else {
        currentUV.x += stepLength * 0.5;
    }

    float2 edgeStep = isHorizontal ? float2(invSize.x, 0.0) : float2(0.0, invSize.y);
    float2 uv1 = currentUV - edgeStep;
    float2 uv2 = currentUV + edgeStep;

    float lumaEnd1 = luma(input.sample(samp, uv1).rgb) - lumaLocalAvg;
    float lumaEnd2 = luma(input.sample(samp, uv2).rgb) - lumaLocalAvg;

    bool reached1 = abs(lumaEnd1) >= gradientScaled;
    bool reached2 = abs(lumaEnd2) >= gradientScaled;

    for (int i = 1; i < FXAA_SEARCH_STEPS && !(reached1 && reached2); i++) {
        if (!reached1) {
            uv1 -= edgeStep * SEARCH_STEPS[i];
            lumaEnd1 = luma(input.sample(samp, uv1).rgb) - lumaLocalAvg;
            reached1 = abs(lumaEnd1) >= gradientScaled;
        }
        if (!reached2) {
            uv2 += edgeStep * SEARCH_STEPS[i];
            lumaEnd2 = luma(input.sample(samp, uv2).rgb) - lumaLocalAvg;
            reached2 = abs(lumaEnd2) >= gradientScaled;
        }
    }

    // ----- 7. Calculate edge-based blend offset --------------------------
    float dist1 = isHorizontal ? (uv.x - uv1.x) : (uv.y - uv1.y);
    float dist2 = isHorizontal ? (uv2.x - uv.x) : (uv2.y - uv.y);
    float distMin = min(dist1, dist2);
    float edgeLength = dist1 + dist2;

    // Only apply offset if the nearer endpoint's direction is consistent.
    bool goodDirection = (dist1 < dist2)
        ? (lumaEnd1 < 0.0) != (lumaM - lumaLocalAvg < 0.0)
        : (lumaEnd2 < 0.0) != (lumaM - lumaLocalAvg < 0.0);

    float pixelOffset = goodDirection ? (-distMin / edgeLength + 0.5) : 0.0;

    // ----- 8. Final blend: max of edge-based and sub-pixel factor --------
    float finalOffset = max(pixelOffset, subpixC);

    float2 finalUV = uv;
    if (isHorizontal) {
        finalUV.y += finalOffset * stepLength;
    } else {
        finalUV.x += finalOffset * stepLength;
    }

    float3 result = input.sample(samp, finalUV).rgb;
    output.write(float4(result, 1.0), gid);
}
