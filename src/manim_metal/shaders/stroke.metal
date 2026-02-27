// Stroke rendering shader.
//
// Strokes are pre-tessellated on the CPU into triangle quads (2 triangles
// per line segment). Each quad has 6 vertices in the order:
//   v0 = a+n, v1 = a-n, v2 = b+n, v3 = a-n, v4 = b-n, v5 = b+n
// where n is the XY-plane normal. Using vid % 6, the vertex shader derives
// an edge distance (-1 at -normal edge, +1 at +normal edge) that the
// fragment shader uses for smooth alpha falloff at stroke edges.

#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    float4x4 mvp;           // 64B — orthographic MVP for 2D
    float4   color;          // 16B — RGBA stroke color
    float3x3 rotation;       // 48B — camera rotation matrix (3D)
    float3   frame_center;   // 12B + 4B pad
    float2   frame_shape;    // 8B
    float    focal_distance; // 4B
    float    zoom;           // 4B
    uint     is_3d;          // 4B (0 = use mvp path, 1 = use camera path)
};

struct StrokeVertexOut {
    float4 position [[position]];
    float  edge_dist;
};

vertex StrokeVertexOut stroke_vertex(
    const device packed_float3* vertices [[buffer(0)]],
    constant Uniforms& uniforms [[buffer(1)]],
    uint vid [[vertex_id]]
) {
    float3 pos = vertices[vid];

    // Derive edge distance from vertex ordering within each quad.
    // Quad layout: v0(+n), v1(-n), v2(+n), v3(-n), v4(-n), v5(+n)
    uint local = vid % 6;
    float edge_dist = (local == 0 || local == 2 || local == 5) ? 1.0 : -1.0;

    StrokeVertexOut out;
    if (uniforms.is_3d) {
        float3 rotated = uniforms.rotation * (pos - uniforms.frame_center);
        float factor = uniforms.focal_distance / (uniforms.focal_distance - rotated.z);
        float2 ndc = rotated.xy * factor * uniforms.zoom * (2.0 / uniforms.frame_shape);
        out.position = float4(ndc, 0.5 - rotated.z * 0.001, 1.0);
    } else {
        out.position = uniforms.mvp * float4(pos, 1.0);
    }
    out.edge_dist = edge_dist;
    return out;
}

fragment half4 stroke_fragment(
    StrokeVertexOut in [[stage_in]],
    constant Uniforms& uniforms [[buffer(1)]]
) {
    float d = abs(in.edge_dist);
    // Adaptive smoothing: fwidth gives the screen-space rate of change of d.
    // For wide strokes (many pixels), fw is small → narrow fade.
    // For thin strokes (1-2 pixels), fw is large → broader fade.
    float fw = fwidth(d);
    float alpha = 1.0 - smoothstep(1.0 - fw, 1.0, d);
    float4 c = uniforms.color;
    // Output straight alpha — blending mode is SrcAlpha, OneMinusSrcAlpha
    return half4(half3(c.rgb), half(c.a * alpha));
}
