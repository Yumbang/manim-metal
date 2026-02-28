// Stroke rendering shader.
//
// Strokes are pre-tessellated on the CPU into triangle quads (2 triangles
// per line segment). Each quad has 6 vertices in the order:
//   v0 = a+n, v1 = a-n, v2 = b+n, v3 = a-n, v4 = b-n, v5 = b+n
// where n is the XY-plane normal. Using vid % 6, the vertex shader derives
// an edge distance (-1 at -normal edge, +1 at +normal edge) that the
// fragment shader uses for smooth alpha falloff at stroke edges.
//
// Lit variant (Phase 2) adds Blinn-Phong shading for 3D surface outlines
// using per-vertex normals from the LitVertex buffer format.

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
// Unlit stroke (original, unchanged from Phase 1)
// ===========================================================================

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
    // For wide strokes (many pixels), fw is small -> narrow fade.
    // For thin strokes (1-2 pixels), fw is large -> broader fade.
    float fw = fwidth(d);
    float alpha = 1.0 - smoothstep(1.0 - fw, 1.0, d);
    float4 c = uniforms.color;
    // Output straight alpha -- blending mode is SrcAlpha, OneMinusSrcAlpha
    return half4(half3(c.rgb), half(c.a * alpha));
}

// ===========================================================================
// Lit stroke (Phase 2: Blinn-Phong shading for 3D surface edges)
//
// Uses LitVertex buffer (24B per vertex: packed_float3 position + packed_float3 normal).
// The per-vertex normal comes from the surface geometry (e.g., sphere normals),
// providing consistent lighting across fill and stroke of the same surface.
//
// Edge distance derivation is identical to the unlit variant (vid % 6 pattern).
// ===========================================================================

struct LitStrokeVertexOut {
    float4 position [[position]];
    float  edge_dist;
    float3 world_normal;   // camera-space normal for lighting
    float3 world_pos;      // camera-space position for specular
};

vertex LitStrokeVertexOut stroke_lit_vertex(
    const device LitVertex* vertices [[buffer(0)]],
    constant Uniforms& uniforms [[buffer(1)]],
    uint vid [[vertex_id]]
) {
    float3 pos = float3(vertices[vid].position);
    float3 nrm = float3(vertices[vid].normal);

    // Edge distance from quad vertex ordering (same pattern as unlit)
    uint local = vid % 6;
    float edge_dist = (local == 0 || local == 2 || local == 5) ? 1.0 : -1.0;

    LitStrokeVertexOut out;
    out.position     = transform_position(pos, uniforms);
    out.world_normal = transform_normal(nrm, uniforms);
    out.world_pos    = rotated_position(pos, uniforms);
    out.edge_dist    = edge_dist;
    return out;
}

fragment half4 stroke_lit_fragment(
    LitStrokeVertexOut in [[stage_in]],
    constant Uniforms& uniforms [[buffer(1)]]
) {
    // Edge AA (identical to unlit stroke)
    float d = abs(in.edge_dist);
    float fw = fwidth(d);
    float alpha = 1.0 - smoothstep(1.0 - fw, 1.0, d);

    float3 normal = normalize(in.world_normal);

    // Transform light position to camera space
    float3 light_pos_cam = uniforms.is_3d
        ? float3(uniforms.rotation * (uniforms.light_position - uniforms.frame_center))
        : uniforms.light_position;

    // Cairo-style additive shading
    float light = cairo_shading(
        normal, in.world_pos, light_pos_cam,
        uniforms.diffuse_strength, uniforms.shininess
    );

    // Combine lighting with object color and edge AA alpha
    float3 lit_rgb = clamp(uniforms.color.rgb + float3(light), 0.0, 1.0);
    return half4(half3(lit_rgb), half(uniforms.color.a * alpha));
}
