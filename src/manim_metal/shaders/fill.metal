// Stencil-then-cover fill shader for VMobject rendering.
//
// Pass 1 (stencil): Render triangle-fan triangles, toggling stencil bits.
//                    Fragment shader outputs nothing (color write mask = none).
// Pass 2 (cover):   Render a full-screen quad. Fragment shader outputs the
//                    fill color only where stencil != 0.
// Pass 2 (lit):     Variant of cover pass with Blinn-Phong lighting for 3D
//                    surfaces that provide per-vertex normals (Phase 2).

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Shared definitions (inlined from lighting.h for runtime compilation).
// When the Python pipeline prepends lighting.h, these are skipped via guard.
// ---------------------------------------------------------------------------
#ifndef MANIM_METAL_LIGHTING_H

struct LitVertex {
    packed_float3 position;
    packed_float3 normal;
};

struct Uniforms {
    float4x4 mvp;              // offset 0,   64B
    float4   color;            // offset 64,  16B
    float3x3 rotation;         // offset 80,  48B
    float3   frame_center;     // offset 128, 16B
    float2   frame_shape;      // offset 144, 8B
    float    focal_distance;   // offset 152, 4B
    float    zoom;             // offset 156, 4B
    uint     is_3d;            // offset 160, 4B
    uint     use_lighting;     // offset 164, 4B
    float    ambient_strength; // offset 168, 4B
    float    diffuse_strength; // offset 172, 4B
    float3   light_position;   // offset 176, 16B
    float3   light_color;      // offset 192, 16B
    float    specular_strength;// offset 208, 4B
    float    shininess;        // offset 212, 4B
};

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

static inline float3 transform_normal(float3 n, constant Uniforms& u) {
    if (u.is_3d) {
        return normalize(u.rotation * n);
    } else {
        return float3(0.0, 0.0, 1.0);
    }
}

static inline float3 rotated_position(float3 pos, constant Uniforms& u) {
    if (u.is_3d) {
        return u.rotation * (pos - u.frame_center);
    } else {
        return pos;
    }
}

static inline float cairo_shading(float3 normal,
                                  float3 frag_pos,
                                  float3 light_pos,
                                  float  intensity,
                                  float  exponent) {
    float3 light_dir = normalize(light_pos - frag_pos);
    float n_dot_l = dot(normal, light_dir);
    float abs_ndl = abs(n_dot_l);
    float magnitude = intensity * pow(abs_ndl, exponent);
    return (n_dot_l >= 0.0) ? magnitude : -magnitude * 0.5;
}

#endif // MANIM_METAL_LIGHTING_H

// ===========================================================================
// Pass 1: Stencil pass (triangle fan geometry)
// Unchanged from Phase 1. Uses packed_float3 position-only vertices.
// ===========================================================================

struct VertexOut {
    float4 position [[position]];
};

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
    // Color write mask = none; only stencil operations matter.
    return half4(0.0);
}

// ===========================================================================
// Pass 2: Cover pass (unlit, original)
// Unchanged from Phase 1. Uses packed_float3 position-only vertices.
// ===========================================================================

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

// ===========================================================================
// Pass 2 (lit): Cover pass with Blinn-Phong lighting (Phase 2)
//
// Used for 3D surfaces (Sphere, Torus, ParametricSurface, etc.) that provide
// per-vertex normals via the LitVertex buffer format (24B per vertex).
//
// The stencil pass (Pass 1) still uses the position-only stencil shader above
// with a separate packed_float3 position buffer. Only the cover pass switches
// to the lit variant when use_lighting is set.
// ===========================================================================

struct LitFillVertexOut {
    float4 position [[position]];
    float3 world_normal;   // camera-space normal for lighting
    float3 world_pos;      // camera-space position for specular
};

vertex LitFillVertexOut fill_cover_lit_vertex(
    const device LitVertex* vertices [[buffer(0)]],
    constant Uniforms& uniforms [[buffer(1)]],
    uint vid [[vertex_id]]
) {
    float3 pos = float3(vertices[vid].position);
    float3 nrm = float3(vertices[vid].normal);

    LitFillVertexOut out;
    out.position     = transform_position(pos, uniforms);
    out.world_normal = transform_normal(nrm, uniforms);
    out.world_pos    = rotated_position(pos, uniforms);
    return out;
}

fragment half4 fill_cover_lit_fragment(
    LitFillVertexOut in [[stage_in]],
    constant Uniforms& uniforms [[buffer(1)]]
) {
    float3 normal = normalize(in.world_normal);

    // Transform light position to camera space for consistent lighting.
    // CPU passes light_position in world space; rotate it here.
    float3 light_pos_cam = uniforms.is_3d
        ? float3(uniforms.rotation * (uniforms.light_position - uniforms.frame_center))
        : uniforms.light_position;

    // Cairo-style additive shading: base color ± light value
    float light = cairo_shading(
        normal, in.world_pos, light_pos_cam,
        uniforms.diffuse_strength, uniforms.shininess
    );

    float3 lit_rgb = clamp(uniforms.color.rgb + float3(light), 0.0, 1.0);
    return half4(half3(lit_rgb), half(uniforms.color.a));
}
