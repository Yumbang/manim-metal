// Blit / compositing shader.
//
// Draws a full-screen triangle and samples the render target texture.
// Used for final readback or compositing passes.

#include <metal_stdlib>
using namespace metal;

struct BlitVertexOut {
    float4 position [[position]];
    float2 texcoord;
};

// Full-screen triangle trick: 3 vertices, no vertex buffer needed.
// Vertex IDs 0, 1, 2 produce a triangle covering [-1,1] x [-1,1].
vertex BlitVertexOut blit_vertex(uint vid [[vertex_id]]) {
    BlitVertexOut out;
    // Generate full-screen triangle
    float2 pos;
    pos.x = (vid == 1) ? 3.0 : -1.0;
    pos.y = (vid == 2) ? 3.0 : -1.0;
    out.position = float4(pos, 0.0, 1.0);
    // Map to [0,1] UV, with y-flip for Metal's top-left origin
    out.texcoord = float2((pos.x + 1.0) * 0.5, (1.0 - pos.y) * 0.5);
    return out;
}

fragment half4 blit_fragment(
    BlitVertexOut in [[stage_in]],
    texture2d<half> tex [[texture(0)]],
    sampler samp [[sampler(0)]]
) {
    return tex.sample(samp, in.texcoord);
}
