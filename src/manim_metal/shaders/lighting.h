// lighting.h -- Blinn-Phong lighting for manim-metal Phase 2
//
// Shared header for fill.metal and stroke.metal lit shader variants.
// Contains:
//   - LitVertex struct (position + normal, 24 bytes)
//   - Extended Uniforms struct with lighting parameters
//   - Helper functions: transform_position, transform_normal, rotated_position
//   - blinn_phong() Blinn-Phong lighting function
//
// IMPORTANT: Runtime shader compilation via newLibraryWithSource_ does not
// resolve #include directives. This header is designed to be concatenated
// with the .metal source files by the Python compilation pipeline:
//
//   combined = lighting_h_source + "\n" + shader_source
//   library = device.newLibraryWithSource_options_error_(combined, None, None)
//
// Each .metal file uses `#ifndef MANIM_METAL_LIGHTING_H` guards so that
// the definitions from this header take precedence when prepended, and
// the .metal files fall back to their own inline definitions when compiled
// standalone (e.g., via the xcrun metal CLI for validation).
//
// All existing unlit rendering paths remain unchanged. Lit variants are
// separate vertex/fragment function pairs that coexist alongside the originals.

#ifndef MANIM_METAL_LIGHTING_H
#define MANIM_METAL_LIGHTING_H

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// LitVertex: vertex buffer layout for lit geometry (surfaces with normals).
//
// Uses packed_float3 (12 bytes each, no padding) so the struct is exactly
// 24 bytes with no gaps. This matches the CPU-side numpy layout:
//   [px, py, pz, nx, ny, nz] as float32 = 24 bytes per vertex.
//
// Byte offsets:
//   position: 0..11  (3 x float, 12 bytes)
//   normal:   12..23 (3 x float, 12 bytes)
//   total:    24 bytes
// ---------------------------------------------------------------------------
struct LitVertex {
    packed_float3 position;  // 12 bytes -- world-space position
    packed_float3 normal;    // 12 bytes -- object-space normal (unit length)
};

// ---------------------------------------------------------------------------
// Uniforms: extended to include lighting parameters.
//
// The original fields (offsets 0..163) are unchanged. Lighting fields occupy
// the former padding region (offsets 164..215), keeping the struct at 256B.
//
// Byte offset map (float32 array index in parentheses):
//   0   [0]:   float4x4 mvp              (64B)
//   64  [16]:  float4   color            (16B)
//   80  [20]:  float3x3 rotation         (48B, columns padded to float4)
//   128 [32]:  float3   frame_center     (12B + 4B pad = 16B)
//   144 [36]:  float2   frame_shape      (8B)
//   152 [38]:  float    focal_distance   (4B)
//   156 [39]:  float    zoom             (4B)
//   160 [40]:  uint     is_3d            (4B) -- 0=2D mvp, 1=3D camera
//   164 [41]:  uint     use_lighting     (4B) -- 0=unlit, 1=Blinn-Phong
//   168 [42]:  float    ambient_strength (4B)
//   172 [43]:  float    diffuse_strength (4B)
//   176 [44]:  float3   light_position   (16B, 12B data + 4B align pad)
//   192 [48]:  float3   light_color      (16B, 12B data + 4B align pad)
//   208 [52]:  float    specular_strength(4B)
//   212 [53]:  float    shininess        (4B)
//   216 [54]:  --- padding to 256B ---   (40B)
//   Total: 256 bytes (no change in alignment requirement)
//
// Python-side packing (numpy float32 array, 64 elements):
//   buf[41] = use_lighting  (via .view(np.uint32)[41] = 1)
//   buf[42] = ambient_strength
//   buf[43] = diffuse_strength
//   buf[44:47] = light_position  (3 floats, pad at [47])
//   buf[48:51] = light_color     (3 floats, pad at [51])
//   buf[52] = specular_strength
//   buf[53] = shininess
// ---------------------------------------------------------------------------
struct Uniforms {
    float4x4 mvp;              // offset 0,   64B
    float4   color;            // offset 64,  16B
    float3x3 rotation;         // offset 80,  48B
    float3   frame_center;     // offset 128, 16B (12B + 4B pad)
    float2   frame_shape;      // offset 144, 8B
    float    focal_distance;   // offset 152, 4B
    float    zoom;             // offset 156, 4B
    uint     is_3d;            // offset 160, 4B
    uint     use_lighting;     // offset 164, 4B
    float    ambient_strength; // offset 168, 4B
    float    diffuse_strength; // offset 172, 4B
    float3   light_position;   // offset 176, 16B (float3 aligned to 16)
    float3   light_color;      // offset 192, 16B (float3 aligned to 16)
    float    specular_strength;// offset 208, 4B
    float    shininess;        // offset 212, 4B
    // implicit padding:       // offset 216..255 (40B)
};

// ---------------------------------------------------------------------------
// Shared position transform (same logic as the original shaders).
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Transform a normal from object space to camera (view) space.
//
// For rigid-body rotations (no non-uniform scale), the normal transform is
// the same rotation matrix. We do NOT translate normals (they are directions).
// The result is re-normalized to handle any floating-point drift.
// ---------------------------------------------------------------------------
static inline float3 transform_normal(float3 n, constant Uniforms& u) {
    if (u.is_3d) {
        return normalize(u.rotation * n);
    } else {
        // 2D: normals point straight out of the screen (+Z)
        return float3(0.0, 0.0, 1.0);
    }
}

// ---------------------------------------------------------------------------
// Compute the camera-space position of a vertex after rotation.
//
// Needed for specular calculations (view direction depends on fragment
// position). Returns the rotated position (before perspective division).
// ---------------------------------------------------------------------------
static inline float3 rotated_position(float3 pos, constant Uniforms& u) {
    if (u.is_3d) {
        return u.rotation * (pos - u.frame_center);
    } else {
        return pos;
    }
}

// ---------------------------------------------------------------------------
// Cairo-matching shading model.
//
// Manim CE (Cairo backend) uses an additive cubic shading formula:
//
//   to_sun   = normalize(light_pos - point)
//   light    = intensity * dot(normal, to_sun) ^ exponent
//   if light < 0: light *= 0.5           (asymmetric shadow)
//   color    = base_color + light        (additive, not multiplicative)
//
// Default: intensity = 0.5, exponent = 3.0 — matching Cairo's
//   ``0.5 * (n·L)^3`` formula in ``manim.utils.color.core.get_shaded_rgb``.
//
// Brightness range for default params:
//   Facing light (n·L=1):  +0.5   (color brightens)
//   Perpendicular (n·L=0):  0.0   (no change)
//   Facing away (n·L=-1): -0.25   (color darkens, halved)
//
// Parameters:
//   normal    -- unit surface normal in camera/world space
//   frag_pos  -- fragment position in camera/world space
//   light_pos -- light position in camera/world space
//   intensity -- light intensity coefficient (uniform: diffuse_strength)
//   exponent  -- falloff exponent (uniform: shininess). Cairo uses 3.
//
// Returns: scalar light value to ADD to the base color.
// ---------------------------------------------------------------------------
static inline float cairo_shading(float3 normal,
                                  float3 frag_pos,
                                  float3 light_pos,
                                  float  intensity,
                                  float  exponent) {
    float3 light_dir = normalize(light_pos - frag_pos);
    float n_dot_l = dot(normal, light_dir);

    // pow() requires non-negative base in Metal, so use abs + sign.
    float abs_ndl = abs(n_dot_l);
    float magnitude = intensity * pow(abs_ndl, exponent);

    // Asymmetric: shadow side (facing away from light) gets half intensity.
    return (n_dot_l >= 0.0) ? magnitude : -magnitude * 0.5;
}

// ---------------------------------------------------------------------------
// SHADER FUNCTION REFERENCE
//
// Existing (unlit, unchanged):
//   fill_stencil_vertex / fill_stencil_fragment   -- stencil pass
//   fill_cover_vertex / fill_cover_fragment        -- unlit cover pass
//   stroke_vertex / stroke_fragment                -- unlit stroke
//   blit_vertex / blit_fragment                    -- screen blit
//
// New (lit, Phase 2):
//   fill_cover_lit_vertex / fill_cover_lit_fragment -- lit cover pass
//   stroke_lit_vertex / stroke_lit_fragment         -- lit stroke
//
// STRUCT LAYOUTS
//
// LitVertex (vertex buffer, packed, 24 bytes):
//   Offset  Size  Field
//   0       12B   packed_float3 position
//   12      12B   packed_float3 normal
//   Total:  24B
//
// Uniforms (constant buffer, 256 bytes):
//   Offset  Size  Field                  numpy index
//   0       64B   float4x4 mvp           [0..15]
//   64      16B   float4   color         [16..19]
//   80      48B   float3x3 rotation      [20..31]
//   128     16B   float3   frame_center  [32..35]
//   144     8B    float2   frame_shape   [36..37]
//   152     4B    float    focal_distance[38]
//   156     4B    float    zoom          [39]
//   160     4B    uint     is_3d         [40] (uint32 view)
//   164     4B    uint     use_lighting  [41] (uint32 view)
//   168     4B    float    ambient_str   [42]
//   172     4B    float    diffuse_str   [43]
//   176     16B   float3   light_pos     [44..47]
//   192     16B   float3   light_color   [48..51]
//   208     4B    float    specular_str  [52]
//   212     4B    float    shininess     [53]
//   216     40B   padding                [54..63]
//   Total: 256B
// ---------------------------------------------------------------------------

#endif // MANIM_METAL_LIGHTING_H
