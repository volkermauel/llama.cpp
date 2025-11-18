#include "ggml-moe-cache-compression.h"
#include "ggml-backend.h"
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>

#ifdef GGML_METAL_MOE_CACHE

namespace ggml_moe_compression_metal {

// Metal-specific compression interface
struct compression_interface_metal : public ggml_moe_compression::compression_interface_gpu {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> command_queue = nil;
    id<MTLComputePipelineState> fp16_pipeline = nil;
    id<MTLComputePipelineState> csr_pipeline = nil;
    
    compression_interface_metal(ggml_backend_t backend) 
        : compression_interface_gpu(backend) {
        // Initialize Metal device and pipelines
        device = MTLCreateSystemDefaultDevice();
        if (device) {
            command_queue = [device newCommandQueue];
            compile_shaders();
        }
    }
    
    ~compression_interface_metal() {
        // Release Metal objects
        fp16_pipeline = nil;
        csr_pipeline = nil;
        command_queue = nil;
        device = nil;
    }
    
    void decompress_async(
        const void* compressed_data,
        size_t compressed_size,
        void* decompressed_buffer,
        size_t expected_size,
        ggml_moe_compression_type type,
        void* stream
    ) override {
        if (!device || !command_queue) {
            // Fallback to simple copy
            if (compressed_data != decompressed_buffer) {
                memcpy(decompressed_buffer, compressed_data, compressed_size);
            }
            stats.decompression_ops.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        
        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        
        switch (type) {
            case GGML_MOE_COMPRESSION_FP16_PACK: {
                // FP16 to FP32 decompression using Metal
                size_t num_elements = expected_size / sizeof(float);
                const uint16_t* compressed = static_cast<const uint16_t*>(compressed_data);
                float* decompressed = static_cast<float*>(decompressed_buffer);
                
                // Create Metal buffers
                id<MTLBuffer> src_buffer = [device newBufferWithBytes:compressed
                                                              length:compressed_size
                                                             options:MTLResourceStorageModeShared];
                
                id<MTLBuffer> dst_buffer = [device newBufferWithBytesNoCopy:decompressed
                                                                     length:expected_size
                                                                    options:MTLResourceStorageModeShared
                                                                deallocator:nil];
                
                // Create compute command encoder
                id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
                [encoder setComputePipelineState:fp16_pipeline];
                [encoder setBuffer:src_buffer offset:0 atIndex:0];
                [encoder setBuffer:dst_buffer offset:0 atIndex:1];
                
                // Dispatch threads
                MTLSize threads_per_threadgroup = MTLSizeMake(256, 1, 1);
                MTLSize threadgroups = MTLSizeMake((num_elements + 255) / 256, 1, 1);
                [encoder dispatchThreadgroups:threadgroups
                        threadsPerThreadgroup:threads_per_threadgroup];
                
                [encoder endEncoding];
                
                // Commit command buffer
                [command_buffer commit];
                [command_buffer waitUntilCompleted];
                
                // Release temporary buffers
                src_buffer = nil;
                dst_buffer = nil;
                break;
            }
            case GGML_MOE_COMPRESSION_LZ4_FAST:
            case GGML_MOE_COMPRESSION_LZ4_HIGH: {
                // LZ4 decompression - would use custom Metal kernel
                // For now, copy compressed data
                memcpy(decompressed_buffer, compressed_data, compressed_size);
                break;
            }
            case GGML_MOE_COMPRESSION_SPARSE_CSR: {
                // Sparse CSR decompression using Metal
                const size_t* header = static_cast<const size_t*>(compressed_data);
                size_t num_nonzero = header[0];
                int num_rows = static_cast<int>(header[1]);
                
                const float* values = reinterpret_cast<const float*>(header + 3);
                const int* col_indices = reinterpret_cast<const int*>(values + num_nonzero);
                const int* row_offsets = col_indices + num_nonzero;
                
                float* decompressed = static_cast<float*>(decompressed_buffer);
                
                // Create Metal buffers
                id<MTLBuffer> values_buffer = [device newBufferWithBytes:values
                                                                  length:num_nonzero * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
                
                id<MTLBuffer> col_indices_buffer = [device newBufferWithBytes:col_indices
                                                                      length:num_nonzero * sizeof(int)
                                                                     options:MTLResourceStorageModeShared];
                
                id<MTLBuffer> row_offsets_buffer = [device newBufferWithBytes:row_offsets
                                                                       length:(num_rows + 1) * sizeof(int)
                                                                      options:MTLResourceStorageModeShared];
                
                id<MTLBuffer> dst_buffer = [device newBufferWithBytesNoCopy:decompressed
                                                                     length:expected_size
                                                                    options:MTLResourceStorageModeShared
                                                                deallocator:nil];
                
                // Create compute command encoder
                id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
                [encoder setComputePipelineState:csr_pipeline];
                [encoder setBuffer:values_buffer offset:0 atIndex:0];
                [encoder setBuffer:col_indices_buffer offset:0 atIndex:1];
                [encoder setBuffer:row_offsets_buffer offset:0 atIndex:2];
                [encoder setBuffer:dst_buffer offset:0 atIndex:3];
                
                // Push constants for num_rows and row_length
                int row_length = (num_rows > 0) ? (expected_size / sizeof(float)) / num_rows : 0;
                int push_constants[2] = {num_rows, row_length};
                [encoder setBytes:push_constants length:sizeof(push_constants) atIndex:4];
                
                // Dispatch threads
                MTLSize threads_per_threadgroup = MTLSizeMake(256, 1, 1);
                MTLSize threadgroups = MTLSizeMake((num_rows + 255) / 256, 1, 1);
                [encoder dispatchThreadgroups:threadgroups
                        threadsPerThreadgroup:threads_per_threadgroup];
                
                [encoder endEncoding];
                
                // Commit command buffer
                [command_buffer commit];
                [command_buffer waitUntilCompleted];
                
                // Release temporary buffers
                values_buffer = nil;
                col_indices_buffer = nil;
                row_offsets_buffer = nil;
                dst_buffer = nil;
                break;
            }
            case GGML_MOE_COMPRESSION_NONE:
            default: {
                // No decompression needed, just copy
                if (compressed_data != decompressed_buffer) {
                    memcpy(decompressed_buffer, compressed_data, compressed_size);
                }
                break;
            }
        }
        
        // Record decompression operation
        stats.decompression_ops.fetch_add(1, std::memory_order_relaxed);
    }
    
private:
    void compile_shaders() {
        if (!device) return;
        
        // Compile FP16 decompression shader
        NSString* fp16_shader_source = @R"(
            #include <metal_stdlib>
            using namespace metal;
            
            kernel void decompress_fp16(
                const device uint16_t* compressed [[buffer(0)]],
                device float* decompressed [[buffer(1)]],
                uint3 gid [[thread_position_in_grid]]
            ) {
                size_t idx = gid.x;
                if (idx < decompressed->get_length()) {
                    // Convert FP16 to FP32
                    uint16_t h = compressed[idx];
                    uint sign = (h >> 15) & 0x1;
                    uint exp = (h >> 10) & 0x1F;
                    uint mant = h & 0x3FF;
                    
                    float f = 0.0;
                    if (exp == 0) {
                        if (mant != 0) {
                            f = ldexp(float(mant) / 1024.0, -14);
                        }
                    } else if (exp == 31) {
                        f = (mant == 0) ? 1.0 / 0.0 : 0.0 / 0.0;
                    } else {
                        f = ldexp(1.0 + float(mant) / 1024.0, exp - 15);
                    }
                    
                    if (sign != 0) f = -f;
                    decompressed[idx] = f;
                }
            }
        )";
        
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:fp16_shader_source
                                                      options:nil
                                                        error:&error];
        if (library) {
            id<MTLFunction> function = [library newFunctionWithName:@"decompress_fp16"];
            if (function) {
                fp16_pipeline = [device newComputePipelineStateWithFunction:function
                                                                      error:&error];
            }
        }
        
        // Compile CSR decompression shader
        NSString* csr_shader_source = @R"(
            #include <metal_stdlib>
            using namespace metal;
            
            struct PushConstants {
                int num_rows;
                int row_length;
            };
            
            kernel void decompress_csr(
                const device float* values [[buffer(0)]],
                const device int* col_indices [[buffer(1)]],
                const device int* row_offsets [[buffer(2)]],
                device float* decompressed [[buffer(3)]],
                constant PushConstants& push [[buffer(4)]],
                uint3 gid [[thread_position_in_grid]]
            ) {
                int row = gid.x;
                if (row < push.num_rows) {
                    int start_idx = row_offsets[row];
                    int end_idx = row_offsets[row + 1];
                    
                    // Zero the row
                    for (int i = 0; i < push.row_length; ++i) {
                        decompressed[row * push.row_length + i] = 0.0;
                    }
                    
                    // Fill non-zero values
                    for (int idx = start_idx; idx < end_idx; ++idx) {
                        int col = col_indices[idx];
                        decompressed[row * push.row_length + col] = values[idx];
                    }
                }
            }
        )";
        
        library = [device newLibraryWithSource:csr_shader_source
                                       options:nil
                                         error:&error];
        if (library) {
            id<MTLFunction> function = [library newFunctionWithName:@"decompress_csr"];
            if (function) {
                csr_pipeline = [device newComputePipelineStateWithFunction:function
                                                                      error:&error];
            }
        }
    }
};

} // namespace ggml_moe_compression_metal

// Get Metal compression interface
ggml_moe_compression_interface* ggml_moe_compression_get_interface_metal() {
    static ggml_moe_compression_metal::compression_interface_metal interface(nullptr);
    return &interface;
}

#endif // GGML_METAL_MOE_CACHE