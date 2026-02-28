/**
 * fast_encode.m — Native Metal draw-op encoder.
 *
 * Replaces thousands of pyobjc bridge crossings with a single native call.
 * Each draw op is 4 × int32 packed contiguously. The function iterates in C,
 * calling Metal API directly with state tracking to skip redundant changes.
 *
 * Build:
 *   clang -shared -fPIC -O2 -framework Metal -o fast_encode.dylib fast_encode.m
 */

#import <Metal/Metal.h>
#include <stdint.h>

typedef struct {
    int32_t kind;           // 0=fill_stencil, 1=fill_cover, 2=stroke,
                            // 3=fill_cover_lit, 4=stroke_lit
    int32_t vert_offset;    // byte offset into shared buffer
    int32_t vert_count;     // number of vertices
    int32_t uniform_offset; // byte offset for uniforms
} DrawOp;

// Draw op kinds — must match Python constants in metal_camera.py
enum {
    OP_FILL_STENCIL    = 0,
    OP_FILL_COVER      = 1,
    OP_STROKE          = 2,
    OP_FILL_COVER_LIT  = 3,
    OP_STROKE_LIT      = 4,
};

void encode_draw_ops(
    void *encoder_ptr,
    void *buf_ptr,
    const DrawOp *ops,
    int32_t n_ops,
    void *fill_stencil_pso_ptr,
    void *fill_cover_pso_ptr,
    void *stroke_pso_ptr,
    void *stencil_inc_dss_ptr,
    void *stencil_nz_dss_ptr,
    void *stencil_disabled_dss_ptr,
    void *fill_cover_lit_pso_ptr,
    void *stroke_lit_pso_ptr
) {
    id<MTLRenderCommandEncoder> enc = (__bridge id<MTLRenderCommandEncoder>)encoder_ptr;
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buf_ptr;

    id<MTLRenderPipelineState> fill_stencil_pso    = (__bridge id<MTLRenderPipelineState>)fill_stencil_pso_ptr;
    id<MTLRenderPipelineState> fill_cover_pso      = (__bridge id<MTLRenderPipelineState>)fill_cover_pso_ptr;
    id<MTLRenderPipelineState> stroke_pso          = (__bridge id<MTLRenderPipelineState>)stroke_pso_ptr;
    id<MTLRenderPipelineState> fill_cover_lit_pso  = (__bridge id<MTLRenderPipelineState>)fill_cover_lit_pso_ptr;
    id<MTLRenderPipelineState> stroke_lit_pso      = (__bridge id<MTLRenderPipelineState>)stroke_lit_pso_ptr;

    id<MTLDepthStencilState> stencil_inc_dss      = (__bridge id<MTLDepthStencilState>)stencil_inc_dss_ptr;
    id<MTLDepthStencilState> stencil_nz_dss       = (__bridge id<MTLDepthStencilState>)stencil_nz_dss_ptr;
    id<MTLDepthStencilState> stencil_disabled_dss = (__bridge id<MTLDepthStencilState>)stencil_disabled_dss_ptr;

    // State tracking — skip redundant calls
    id<MTLRenderPipelineState> cur_pso = nil;
    id<MTLDepthStencilState> cur_dss   = nil;
    uint32_t cur_stencil_ref           = UINT32_MAX;  // sentinel: not yet set

    for (int32_t i = 0; i < n_ops; i++) {
        const DrawOp *op = &ops[i];

        switch (op->kind) {
            case OP_FILL_STENCIL: {
                if (cur_pso != fill_stencil_pso) {
                    [enc setRenderPipelineState:fill_stencil_pso];
                    cur_pso = fill_stencil_pso;
                }
                if (cur_dss != stencil_inc_dss) {
                    [enc setDepthStencilState:stencil_inc_dss];
                    cur_dss = stencil_inc_dss;
                }
                if (cur_stencil_ref != 0) {
                    [enc setStencilReferenceValue:0];
                    cur_stencil_ref = 0;
                }
                [enc setVertexBufferOffset:(NSUInteger)op->vert_offset atIndex:0];
                [enc setVertexBufferOffset:(NSUInteger)op->uniform_offset atIndex:1];
                [enc drawPrimitives:MTLPrimitiveTypeTriangle
                        vertexStart:0
                        vertexCount:(NSUInteger)op->vert_count];
                break;
            }
            case OP_FILL_COVER: {
                if (cur_pso != fill_cover_pso) {
                    [enc setRenderPipelineState:fill_cover_pso];
                    cur_pso = fill_cover_pso;
                }
                if (cur_dss != stencil_nz_dss) {
                    [enc setDepthStencilState:stencil_nz_dss];
                    cur_dss = stencil_nz_dss;
                }
                if (cur_stencil_ref != 0) {
                    [enc setStencilReferenceValue:0];
                    cur_stencil_ref = 0;
                }
                [enc setVertexBufferOffset:(NSUInteger)op->vert_offset atIndex:0];
                [enc setVertexBufferOffset:(NSUInteger)op->uniform_offset atIndex:1];
                [enc setFragmentBufferOffset:(NSUInteger)op->uniform_offset atIndex:1];
                [enc drawPrimitives:MTLPrimitiveTypeTriangle
                        vertexStart:0
                        vertexCount:(NSUInteger)op->vert_count];
                break;
            }
            case OP_STROKE: {
                if (cur_pso != stroke_pso) {
                    [enc setRenderPipelineState:stroke_pso];
                    cur_pso = stroke_pso;
                }
                if (cur_dss != stencil_disabled_dss) {
                    [enc setDepthStencilState:stencil_disabled_dss];
                    cur_dss = stencil_disabled_dss;
                }
                [enc setVertexBufferOffset:(NSUInteger)op->vert_offset atIndex:0];
                [enc setVertexBufferOffset:(NSUInteger)op->uniform_offset atIndex:1];
                [enc setFragmentBufferOffset:(NSUInteger)op->uniform_offset atIndex:1];
                [enc drawPrimitives:MTLPrimitiveTypeTriangle
                        vertexStart:0
                        vertexCount:(NSUInteger)op->vert_count];
                break;
            }
            case OP_FILL_COVER_LIT: {
                if (cur_pso != fill_cover_lit_pso) {
                    [enc setRenderPipelineState:fill_cover_lit_pso];
                    cur_pso = fill_cover_lit_pso;
                }
                if (cur_dss != stencil_nz_dss) {
                    [enc setDepthStencilState:stencil_nz_dss];
                    cur_dss = stencil_nz_dss;
                }
                if (cur_stencil_ref != 0) {
                    [enc setStencilReferenceValue:0];
                    cur_stencil_ref = 0;
                }
                [enc setVertexBufferOffset:(NSUInteger)op->vert_offset atIndex:0];
                [enc setVertexBufferOffset:(NSUInteger)op->uniform_offset atIndex:1];
                [enc setFragmentBufferOffset:(NSUInteger)op->uniform_offset atIndex:1];
                [enc drawPrimitives:MTLPrimitiveTypeTriangle
                        vertexStart:0
                        vertexCount:(NSUInteger)op->vert_count];
                break;
            }
            case OP_STROKE_LIT: {
                if (cur_pso != stroke_lit_pso) {
                    [enc setRenderPipelineState:stroke_lit_pso];
                    cur_pso = stroke_lit_pso;
                }
                if (cur_dss != stencil_disabled_dss) {
                    [enc setDepthStencilState:stencil_disabled_dss];
                    cur_dss = stencil_disabled_dss;
                }
                [enc setVertexBufferOffset:(NSUInteger)op->vert_offset atIndex:0];
                [enc setVertexBufferOffset:(NSUInteger)op->uniform_offset atIndex:1];
                [enc setFragmentBufferOffset:(NSUInteger)op->uniform_offset atIndex:1];
                [enc drawPrimitives:MTLPrimitiveTypeTriangle
                        vertexStart:0
                        vertexCount:(NSUInteger)op->vert_count];
                break;
            }
        }
    }
}
