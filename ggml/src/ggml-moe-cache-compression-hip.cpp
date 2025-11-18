#include "ggml-moe-cache-compression.h"
#include "ggml-cuda/common.cuh"  // Reuse CUDA structures
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifdef GGML_HIP_MOE_CACHE

namespace ggml_moe_compression_hip {

// HIP kernel for FP16 decompression (FP16 -> FP32)
__global__ void decompress_fp16_pack_kernel_hip(
    const uint16_t* __restrict__ compressed,
    float* __restrict__ decompressed,
    size_t num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        // Convert FP16 to FP32
        decompressed[idx] = __half2float(__ushort_as_half(compressed[idx]));
    }
}

// HIP kernel for sparse CSR decompression
__global__ void decompress_sparse_csr_kernel_hip(
    const float* __restrict__ values,
    const int* __restrict__ col_indices,
    const int* __restrict__ row_offsets,
    float* __restrict__ decompressed,
    int num_rows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        int start_idx = row_offsets[row];
        int end_idx = row_offsets[row + 1];
        
        // Zero the row first
        int row_length = row_offsets[row + 1] - row_offsets[row];
        for (int i = threadIdx.y; i < row_length; i += blockDim.y) {
            decompressed[row * row_length + i] = 0.0f;
        }
        __syncthreads();
        
        // Fill in non-zero values
        for (int idx = start_idx; idx < end_idx; ++idx) {
            int col = col_indices[idx];
            decompressed[row * row_length + col] = values[idx];
        }
    }
}

// HIP-specific compression interface
struct compression_interface_hip : public ggml_moe_compression::compression_interface_gpu {
    compression_interface_hip(ggml_backend_t backend) 
        : compression_interface_gpu(backend) {}
    
    void decompress_async(
        const void* compressed_data,
        size_t compressed_size,
        void* decompressed_buffer,
        size_t expected_size,
        ggml_moe_compression_type type,
        void* stream
    ) override {
        hipStream_t hip_stream = static_cast<hipStream_t>(stream);
        
        switch (type) {
            case GGML_MOE_COMPRESSION_FP16_PACK: {
                // FP16 to FP32 decompression
                size_t num_elements = expected_size / sizeof(float);
                const uint16_t* compressed = static_cast<const uint16_t*>(compressed_data);
                float* decompressed = static_cast<float*>(decompressed_buffer);
                
                int threads_per_block = 256;
                int blocks = (num_elements + threads_per_block - 1) / threads_per_block;
                
                hipLaunchKernelGGL(
                    decompress_fp16_pack_kernel_hip,
                    dim3(blocks), dim3(threads_per_block),
                    0, hip_stream,
                    compressed, decompressed, num_elements
                );
                break;
            }
            case GGML_MOE_COMPRESSION_LZ4_FAST:
            case GGML_MOE_COMPRESSION_LZ4_HIGH: {
                // LZ4 decompression - would use rocPRIM or custom HIP kernels
                // For now, copy compressed data (placeholder)
                hipMemcpyAsync(decompressed_buffer, compressed_data, compressed_size,
                              hipMemcpyDeviceToDevice, hip_stream);
                break;
            }
            case GGML_MOE_COMPRESSION_SPARSE_CSR: {
                // Sparse CSR decompression
                const size_t* header = static_cast<const size_t*>(compressed_data);
                size_t num_nonzero = header[0];
                int num_rows = static_cast<int>(header[1]);
                
                const float* values = reinterpret_cast<const float*>(header + 3);
                const int* col_indices = reinterpret_cast<const int*>(values + num_nonzero);
                const int* row_offsets = col_indices + num_nonzero;
                
                float* decompressed = static_cast<float*>(decompressed_buffer);
                
                int threads_per_block = 256;
                int blocks = (num_rows + threads_per_block - 1) / threads_per_block;
                
                hipLaunchKernelGGL(
                    decompress_sparse_csr_kernel_hip,
                    dim3(blocks), dim3(threads_per_block),
                    0, hip_stream,
                    values, col_indices, row_offsets, decompressed, num_rows
                );
                break;
            }
            case GGML_MOE_COMPRESSION_NONE:
            default: {
                // No decompression needed, just copy
                if (compressed_data != decompressed_buffer) {
                    hipMemcpyAsync(decompressed_buffer, compressed_data, compressed_size,
                                  hipMemcpyDeviceToDevice, hip_stream);
                }
                break;
            }
        }
        
        // Record decompression operation
        stats.decompression_ops.fetch_add(1, std::memory_order_relaxed);
    }
};

} // namespace ggml_moe_compression_hip

// Get HIP compression interface
ggml_moe_compression_interface* ggml_moe_compression_get_interface_hip() {
    static ggml_moe_compression_hip::compression_interface_hip interface(nullptr);
    return &interface;
}

#endif // GGML_HIP_MOE_CACHE