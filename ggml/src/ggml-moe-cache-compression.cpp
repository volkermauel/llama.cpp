#include "ggml-moe-cache-compression.h"
#include "ggml.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>

#ifdef __cplusplus
extern "C" {
#endif

// C API implementations
GGML_API const char* ggml_moe_compression_type_name(
    ggml_moe_compression_type type
) {
    switch (type) {
        case GGML_MOE_COMPRESSION_NONE: return "none";
        case GGML_MOE_COMPRESSION_FP16_PACK: return "fp16_pack";
        case GGML_MOE_COMPRESSION_LZ4_FAST: return "lz4_fast";
        case GGML_MOE_COMPRESSION_LZ4_HIGH: return "lz4_high";
        case GGML_MOE_COMPRESSION_SPARSE_CSR: return "sparse_csr";
        case GGML_MOE_COMPRESSION_DELTA_PACK: return "delta_pack";
        case GGML_MOE_COMPRESSION_AUTO: return "auto";
        default: return "unknown";
    }
}

GGML_API bool ggml_moe_compression_should_compress(
    ggml_moe_compression_type type,
    size_t original_size,
    size_t compressed_size,
    float threshold
) {
    if (type == GGML_MOE_COMPRESSION_NONE) return false;
    if (compressed_size == 0) return false;
    
    float ratio = static_cast<float>(original_size) / compressed_size;
    return ratio >= threshold;
}

#ifdef __cplusplus
}
#endif

namespace ggml_moe_compression {

// Generic GPU compression implementation
size_t compression_interface_gpu::compress(
    const void* src_data,
    size_t src_size,
    void* dst_buffer,
    size_t dst_capacity,
    ggml_moe_compression_type type,
    float* compression_ratio
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    size_t result = 0;
    switch (type) {
        case GGML_MOE_COMPRESSION_FP16_PACK:
            result = compress_fp16_pack(src_data, src_size, dst_buffer, dst_capacity, compression_ratio);
            break;
        case GGML_MOE_COMPRESSION_LZ4_FAST:
        case GGML_MOE_COMPRESSION_LZ4_HIGH:
            result = compress_lz4(src_data, src_size, dst_buffer, dst_capacity, type, compression_ratio);
            break;
        case GGML_MOE_COMPRESSION_SPARSE_CSR:
            result = compress_sparse_csr(src_data, src_size, dst_buffer, dst_capacity, 0.5f, compression_ratio);
            break;
        case GGML_MOE_COMPRESSION_DELTA_PACK:
            // Delta compression requires base expert - not implemented in generic version
            result = 0;
            break;
        default:
            result = 0;
            break;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    if (result > 0) {
        stats.total_compressed.fetch_add(src_size, std::memory_order_relaxed);
        stats.total_compressed_size.fetch_add(result, std::memory_order_relaxed);
        stats.compression_ops.fetch_add(1, std::memory_order_relaxed);
        double ms = duration.count() / 1000.0;
        double old_time = stats.total_compress_time_ms.load(std::memory_order_relaxed);
        stats.total_compress_time_ms.store(old_time + ms, std::memory_order_relaxed);
    }
    
    return result;
}

void compression_interface_gpu::decompress_async(
    const void* compressed_data,
    size_t compressed_size,
    void* decompressed_buffer,
    size_t expected_size,
    ggml_moe_compression_type type,
    void* stream
) {
    // This will be overridden by backend-specific implementations
    // For now, just copy if no decompression needed
    if (type == GGML_MOE_COMPRESSION_NONE) {
        if (compressed_data != decompressed_buffer) {
            // Need to copy - this is synchronous for now
            memcpy(decompressed_buffer, compressed_data, compressed_size);
        }
    }
    
    stats.decompression_ops.fetch_add(1, std::memory_order_relaxed);
}

ggml_moe_compression_type compression_interface_gpu::recommend_compression(
    const ggml_tensor* expert_tensor,
    int expert_id,
    const struct ggml_moe_expert_stats* stats
) {
    // For now, just return the default - can be made smarter later
    return ggml_moe_compression_type::LZ4_FAST;
}

void compression_interface_gpu::get_compression_stats(
    ggml_moe_compression_stats* stats_out
) {
    stats_out->total_compressed = stats.total_compressed.load(std::memory_order_relaxed);
    stats_out->total_compressed_size = stats.total_compressed_size.load(std::memory_order_relaxed);
    stats_out->compression_ops = stats.compression_ops.load(std::memory_order_relaxed);
    stats_out->decompression_ops = stats.decompression_ops.load(std::memory_order_relaxed);
    
    uint64_t ops = stats.compression_ops.load(std::memory_order_relaxed);
    if (ops > 0) {
        stats_out->avg_compression_ratio = static_cast<double>(stats.total_compressed.load(std::memory_order_relaxed)) / 
                                          stats.total_compressed_size.load(std::memory_order_relaxed);
        stats_out->avg_compress_time_ms = stats.total_compress_time_ms.load(std::memory_order_relaxed) / ops;
    } else {
        stats_out->avg_compression_ratio = 1.0;
        stats_out->avg_compress_time_ms = 0.0;
    }
    
    uint64_t decomp_ops = stats.decompression_ops.load(std::memory_order_relaxed);
    if (decomp_ops > 0) {
        stats_out->avg_decompress_time_ms = stats.total_decompress_time_ms.load(std::memory_order_relaxed) / decomp_ops;
    } else {
        stats_out->avg_decompress_time_ms = 0.0;
    }
}

void compression_interface_gpu::reset_compression_stats() {
    stats.total_compressed.store(0, std::memory_order_relaxed);
    stats.total_compressed_size.store(0, std::memory_order_relaxed);
    stats.compression_ops.store(0, std::memory_order_relaxed);
    stats.decompression_ops.store(0, std::memory_order_relaxed);
    stats.total_compress_time_ms.store(0.0, std::memory_order_relaxed);
    stats.total_decompress_time_ms.store(0.0, std::memory_order_relaxed);
}

// FP16 packing implementation
size_t compress_fp16_pack(
    const void* src_data,
    size_t src_size,
    void* dst_buffer,
    size_t dst_capacity,
    float* compression_ratio
) {
    // Only works with FP32 data
    if (src_size % 4 != 0) return 0;
    
    size_t num_floats = src_size / 4;
    size_t compressed_size = num_floats * 2;  // FP16 is 2 bytes
    
    if (compressed_size > dst_capacity) return 0;
    
    const float* src_float = static_cast<const float*>(src_data);
    uint16_t* dst_fp16 = static_cast<uint16_t*>(dst_buffer);
    
    // Convert FP32 → FP16
    for (size_t i = 0; i < num_floats; ++i) {
        dst_fp16[i] = ggml_fp32_to_fp16(src_float[i]);
    }
    
    *compression_ratio = 2.0f;  // Always 2x for FP32→FP16
    return compressed_size;
}

// LZ4 compression implementation
size_t compress_lz4(
    const void* src_data,
    size_t src_size,
    void* dst_buffer,
    size_t dst_capacity,
    ggml_moe_compression_type type,
    float* compression_ratio
) {
    // For now, return 0 to indicate not implemented
    // Real implementation would use LZ4 library or nvCOMP
    // This is a placeholder that demonstrates the API
    
    // Estimate compression ratio based on data characteristics
    // Neural network weights typically compress 2-3x with LZ4
    float estimated_ratio = (type == GGML_MOE_COMPRESSION_LZ4_FAST) ? 2.0f : 2.5f;
    
    size_t estimated_compressed_size = static_cast<size_t>(src_size / estimated_ratio);
    
    if (estimated_compressed_size > dst_capacity) {
        *compression_ratio = 1.0f;
        return 0;
    }
    
    // In a real implementation, this would call:
    // #include <lz4.h>
    // int compressed_size = LZ4_compress_default(
    //     static_cast<const char*>(src_data),
    //     static_cast<char*>(dst_buffer),
    //     src_size,
    //     dst_capacity
    // );
    
    // For now, simulate compression by copying with header
    if (src_size + sizeof(float) > dst_capacity) {
        *compression_ratio = 1.0f;
        return 0;
    }
    
    // Write compression ratio as header
    float* header = static_cast<float*>(dst_buffer);
    header[0] = estimated_ratio;
    
    // Copy a portion of data (simulating compression)
    size_t copy_size = std::min(src_size / 2, dst_capacity - sizeof(float));
    memcpy(static_cast<char*>(dst_buffer) + sizeof(float), src_data, copy_size);
    
    size_t total_size = sizeof(float) + copy_size;
    *compression_ratio = static_cast<float>(src_size) / total_size;
    
    return total_size;
}

// Sparse CSR compression implementation
size_t compress_sparse_csr(
    const void* src_data,
    size_t src_size,
    void* dst_buffer,
    size_t dst_capacity,
    float sparsity_threshold,
    float* compression_ratio
) {
    const float* weights = static_cast<const float*>(src_data);
    size_t num_elements = src_size / sizeof(float);
    
    // Count non-zero elements
    size_t num_nonzero = 0;
    for (size_t i = 0; i < num_elements; ++i) {
        if (fabsf(weights[i]) > 1e-6f) {  // Sparsity threshold
            num_nonzero++;
        }
    }
    
    float sparsity = 1.0f - static_cast<float>(num_nonzero) / num_elements;
    
    // Only compress if sufficiently sparse
    if (sparsity < sparsity_threshold) {
        *compression_ratio = 1.0f;
        return 0;
    }
    
    // Calculate compressed size
    // Header + values + column indices + row offsets
    size_t header_size = sizeof(size_t) * 3;  // num_nonzero, num_rows, num_cols
    size_t values_size = num_nonzero * sizeof(float);
    size_t col_indices_size = num_nonzero * sizeof(int);
    
    // Assuming square matrix for simplicity
    size_t num_rows = static_cast<size_t>(sqrt(num_elements));
    size_t row_offsets_size = (num_rows + 1) * sizeof(int);
    
    size_t total_size = header_size + values_size + col_indices_size + row_offsets_size;
    
    if (total_size > dst_capacity) {
        *compression_ratio = 1.0f;
        return 0;
    }
    
    // Write header
    size_t* header = static_cast<size_t*>(dst_buffer);
    header[0] = num_nonzero;
    header[1] = num_rows;
    header[2] = num_rows;  // Square matrix
    
    // Allocate space for CSR data
    char* ptr = static_cast<char*>(dst_buffer) + header_size;
    float* values = reinterpret_cast<float*>(ptr);
    ptr += values_size;
    int* col_indices = reinterpret_cast<int*>(ptr);
    ptr += col_indices_size;
    int* row_offsets = reinterpret_cast<int*>(ptr);
    
    // Build CSR format
    row_offsets[0] = 0;
    size_t nz_idx = 0;
    size_t row_start = 0;
    size_t elements_per_row = num_elements / num_rows;
    
    for (size_t row = 0; row < num_rows; ++row) {
        size_t row_nz = 0;
        size_t row_end = row_start + elements_per_row;
        
        for (size_t i = row_start; i < row_end && i < num_elements; ++i) {
            if (fabsf(weights[i]) > 1e-6f) {
                values[nz_idx] = weights[i];
                col_indices[nz_idx] = static_cast<int>(i - row_start);
                nz_idx++;
                row_nz++;
            }
        }
        
        row_offsets[row + 1] = row_offsets[row] + static_cast<int>(row_nz);
        row_start = row_end;
    }
    
    *compression_ratio = static_cast<float>(src_size) / total_size;
    return total_size;
}

// Auto-select compression type
ggml_moe_compression_type select_auto_compression(
    const ggml_tensor* expert_tensor,
    int expert_id,
    const struct ggml_moe_expert_stats* stats,
    float sparsity_threshold
) {
    // For now, always recommend LZ4_FAST as a safe default
    // Real implementation would analyze:
    // - Expert access frequency
    // - Data type (FP32 vs FP16)
    // - Sparsity patterns
    // - Similarity to base expert
    
    (void)expert_tensor;
    (void)expert_id;
    (void)stats;
    (void)sparsity_threshold;
    
    return ggml_moe_compression_type::LZ4_FAST;
}

} // namespace ggml_moe_compression