// Stencil-then-cover fill shader for VMobject rendering.
//
// Pass 1 (stencil): Render triangle-fan triangles, toggling stencil bits.
//                    Fragment shader outputs nothing (color write mask = none).
// Pass 2 (cover):   Render a full-screen quad. Fragment shader outputs the
//                    fill color only where stencil != 0.

#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    float4x4 mvp;           // 64B — orthographic MVP for 2D
    float4   color;          // 16B — RGBA fill color, premultiplied alpha
    float3x3 rotation;       // 48B — camera rotation matrix (3D)
    float3   frame_center;   // 12B + 4B pad
    float2   frame_shape;    // 8B
    float    focal_distance; // 4B
    float    zoom;           // 4B
    uint     is_3d;          // 4B (0 = use mvp path, 1 = use camera path)
};

struct VertexOut {
    float4 position [[position]];
};

// Shared helper: transform a 3D position using the appropriate path.
static inline float4 transform_position(float3 pos, constant Uniforms& u) {
    if (u.is_3d) {
        float3 rotated = u.rotation * (pos - u.frame_center);
        float factor = u.focal_distance / (u.focal_distance - rotated.z);
        float2 ndc = rotated.xy * factor * u.zoom * (2.0 / u.frame_shape);
        return float4(ndc, 0.5 - rotated.z * 0.001, 1.0);
    } else {
        return u.mvp * float4(pos, 1.0);
    }
}

// ---- Pass 1: Stencil pass (triangle fan geometry) ----

vertex VertexOut fill_stencil_vertex(
    const device packed_float3* vertices [[buffer(0)]],
    constant Uniforms& uniforms [[buffer(1)]],
    uint vid [[vertex_id]]
) {
    VertexOut out;
    out.position = transform_position(vertices[vid], uniforms);
    return out;
}

fragment half4 fill_stencil_fragment(VertexOut in [[stage_in]]) {
    // This fragment shader is used with color write mask = none.
    // Only stencil operations matter.
    return half4(0.0);
}

// ---- Pass 2: Cover pass (full-screen quad or bounding quad) ----

vertex VertexOut fill_cover_vertex(
    const device packed_float3* vertices [[buffer(0)]],
    constant Uniforms& uniforms [[buffer(1)]],
    uint vid [[vertex_id]]
) {
    VertexOut out;
    out.position = transform_position(vertices[vid], uniforms);
    return out;
}

fragment half4 fill_cover_fragment(
    VertexOut in [[stage_in]],
    constant Uniforms& uniforms [[buffer(1)]]
) {
    return half4(uniforms.color);
}
