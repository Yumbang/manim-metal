// Stroke rendering shader.
//
// Strokes are pre-tessellated on the CPU into triangle quads (2 triangles
// per line segment). This shader simply transforms and colors them.

#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    float4x4 mvp;
    float4 color; // RGBA stroke color
};

vertex float4 stroke_vertex(
    const device float2* vertices [[buffer(0)]],
    constant Uniforms& uniforms [[buffer(1)]],
    uint vid [[vertex_id]]
) {
    float2 pos = vertices[vid];
    return uniforms.mvp * float4(pos, 0.0, 1.0);
}

fragment half4 stroke_fragment(
    float4 position [[position]],
    constant Uniforms& uniforms [[buffer(1)]]
) {
    return half4(uniforms.color);
}
