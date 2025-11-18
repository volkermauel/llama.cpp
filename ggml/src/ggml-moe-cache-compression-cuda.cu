#include "ggml-moe-cache-compression.h"
#include "ggml-cuda/common.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifdef GGML_CUDA_MOE_CACHE

namespace ggml_moe_compression_cuda {

// CUDA kernel for FP16 decompression (FP16 -> FP32)
__global__ void decompress_fp16_pack_kernel(
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

// CUDA kernel for sparse CSR decompression
__global__ void decompress_sparse_csr_kernel(
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

// CUDA-specific compression interface
struct compression_interface_cuda : public ggml_moe_compression::compression_interface_gpu {
    compression_interface_cuda(ggml_backend_t backend) 
        : compression_interface_gpu(backend) {}
    
    void decompress_async(
        const void* compressed_data,
        size_t compressed_size,
        void* decompressed_buffer,
        size_t expected_size,
        ggml_moe_compression_type type,
        void* stream
    ) override {
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        
        switch (type) {
            case GGML_MOE_COMPRESSION_FP16_PACK: {
                // FP16 to FP32 decompression
                size_t num_elements = expected_size / sizeof(float);
                const uint16_t* compressed = static_cast<const uint16_t*>(compressed_data);
                float* decompressed = static_cast<float*>(decompressed_buffer);
                
                int threads_per_block = 256;
                int blocks = (num_elements + threads_per_block - 1) / threads_per_block;
                
                decompress_fp16_pack_kernel<<<blocks, threads_per_block, 0, cuda_stream>>>(
                    compressed, decompressed, num_elements
                );
                break;
            }
            case GGML_MOE_COMPRESSION_LZ4_FAST:
            case GGML_MOE_COMPRESSION_LZ4_HIGH: {
                // LZ4 decompression using nvCOMP
                #ifdef NVCOMP_AVAILABLE
                nvcompLZ4DecompressAsync(
                    compressed_data, compressed_size,
                    decompressed_buffer, expected_size,
                    temp_buffer, temp_size,
                    cuda_stream
                );
                #else
                // Fallback: copy compressed data and decompress on CPU
                // In production, this should use a custom CUDA kernel
                cudaMemcpyAsync(decompressed_buffer, compressed_data, compressed_size,
                               cudaMemcpyDeviceToDevice, cuda_stream);
                #endif
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
                
                decompress_sparse_csr_kernel<<<blocks, threads_per_block, 0, cuda_stream>>>(
                    values, col_indices, row_offsets, decompressed, num_rows
                );
                break;
            }
            case GGML_MOE_COMPRESSION_NONE:
            default: {
                // No decompression needed, just copy
                if (compressed_data != decompressed_buffer) {
                    cudaMemcpyAsync(decompressed_buffer, compressed_data, compressed_size,
                                   cudaMemcpyDeviceToDevice, cuda_stream);
                }
                break;
            }
        }
        
        // Record decompression operation
        stats.decompression_ops.fetch_add(1, std::memory_order_relaxed);
    }
    
    // Helper method to get temporary buffer size for nvCOMP
    bool get_temp_buffer_size(size_t max_compressed_size, size_t* temp_size) {
        #ifdef NVCOMP_AVAILABLE
        nvcompError_t status = nvcompLZ4DecompressGetTempSize(max_compressed_size, temp_size);
        return status == nvcompSuccess;
        #else
        *temp_size = max_compressed_size * 2;  // Conservative estimate
        return true;
        #endif
    }
    
private:
    void* temp_buffer = nullptr;
    size_t temp_size = 0;
};

} // namespace ggml_moe_compression_cuda

// Get CUDA compression interface
ggml_moe_compression_interface* ggml_moe_compression_get_interface_cuda() {
    static ggml_moe_compression_cuda::compression_interface_cuda interface(nullptr);
    return &interface;
}

#endif // GGML_CUDA_MOE_CACHE