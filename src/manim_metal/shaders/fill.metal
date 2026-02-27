// Stencil-then-cover fill shader for VMobject rendering.
//
// Pass 1 (stencil): Render triangle-fan triangles, toggling stencil bits.
//                    Fragment shader outputs nothing (color write mask = none).
// Pass 2 (cover):   Render a full-screen quad. Fragment shader outputs the
//                    fill color only where stencil != 0.

#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    float4x4 mvp;
    float4 color; // RGBA fill color, premultiplied alpha
};

struct VertexIn {
    float2 position [[attribute(0)]];
};

struct VertexOut {
    float4 position [[position]];
};

// ---- Pass 1: Stencil pass (triangle fan geometry) ----

vertex VertexOut fill_stencil_vertex(
    const device float2* vertices [[buffer(0)]],
    constant Uniforms& uniforms [[buffer(1)]],
    uint vid [[vertex_id]]
) {
    VertexOut out;
    float2 pos = vertices[vid];
    out.position = uniforms.mvp * float4(pos, 0.0, 1.0);
    return out;
}

fragment half4 fill_stencil_fragment(VertexOut in [[stage_in]]) {
    // This fragment shader is used with color write mask = none.
    // Only stencil operations matter.
    return half4(0.0);
}

// ---- Pass 2: Cover pass (full-screen quad or bounding quad) ----

vertex VertexOut fill_cover_vertex(
    const device float2* vertices [[buffer(0)]],
    constant Uniforms& uniforms [[buffer(1)]],
    uint vid [[vertex_id]]
) {
    VertexOut out;
    float2 pos = vertices[vid];
    out.position = uniforms.mvp * float4(pos, 0.0, 1.0);
    return out;
}

fragment half4 fill_cover_fragment(
    VertexOut in [[stage_in]],
    constant Uniforms& uniforms [[buffer(1)]]
) {
    return half4(uniforms.color);
}
