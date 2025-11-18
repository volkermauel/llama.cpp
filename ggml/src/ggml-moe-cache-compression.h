#pragma once

#include "ggml-moe-cache.h"
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Compression statistics
struct ggml_moe_compression_stats {
    uint64_t total_compressed;      // Total bytes before compression
    uint64_t total_compressed_size; // Total bytes after compression
    uint64_t compression_ops;       // Number of compression operations
    uint64_t decompression_ops;     // Number of decompression operations
    double avg_compression_ratio;   // Average compression ratio
    double avg_compress_time_ms;    // Average compression time
    double avg_decompress_time_ms;  // Average decompression time
};

// Backend-agnostic compression interface
struct ggml_moe_compression_interface {
    // Compress expert data on CPU/host
    virtual size_t compress(
        const void* src_data,
        size_t src_size,
        void* dst_buffer,
        size_t dst_capacity,
        ggml_moe_compression_type type,
        float* compression_ratio
    ) = 0;
    
    // Decompress expert data on GPU/device (async)
    virtual void decompress_async(
        const void* compressed_data,
        size_t compressed_size,
        void* decompressed_buffer,
        size_t expected_size,
        ggml_moe_compression_type type,
        void* stream
    ) = 0;
    
    // Get optimal compression type for expert
    virtual ggml_moe_compression_type recommend_compression(
        const ggml_tensor* expert_tensor,
        int expert_id,
        const struct ggml_moe_expert_stats* stats
    ) = 0;
    
    // Get compression statistics
    virtual void get_compression_stats(
        ggml_moe_compression_stats* stats
    ) = 0;
    
    // Reset compression statistics
    virtual void reset_compression_stats() = 0;
    
    virtual ~ggml_moe_compression_interface() = default;
};

// Get compression interface for backend
GGML_API ggml_moe_compression_interface* ggml_moe_compression_get_interface(
    ggml_backend_t backend
);

// Helper function to check if compression is beneficial
GGML_API bool ggml_moe_compression_should_compress(
    ggml_moe_compression_type type,
    size_t original_size,
    size_t compressed_size,
    float threshold
);

// Get compression type name for logging
GGML_API const char* ggml_moe_compression_type_name(
    ggml_moe_compression_type type
);

// Estimate compression ratio without actually compressing
GGML_API float ggml_moe_compression_estimate_ratio(
    const void* data,
    size_t size,
    ggml_moe_compression_type type
);

#ifdef __cplusplus
}
#endif

// C++ implementations
#ifdef __cplusplus

namespace ggml_moe_compression {

// Generic GPU compression implementation
struct compression_interface_gpu : public ggml_moe_compression_interface {
    ggml_backend_t backend;
    
    compression_interface_gpu(ggml_backend_t be) : backend(be) {}
    
    size_t compress(
        const void* src_data,
        size_t src_size,
        void* dst_buffer,
        size_t dst_capacity,
        ggml_moe_compression_type type,
        float* compression_ratio
    ) override;
    
    void decompress_async(
        const void* compressed_data,
        size_t compressed_size,
        void* decompressed_buffer,
        size_t expected_size,
        ggml_moe_compression_type type,
        void* stream
    ) override;
    
    ggml_moe_compression_type recommend_compression(
        const ggml_tensor* expert_tensor,
        int expert_id,
        const struct ggml_moe_expert_stats* stats
    ) override;
    
    void get_compression_stats(
        ggml_moe_compression_stats* stats
    ) override;
    
    void reset_compression_stats() override;
    
protected:
    // Statistics
    struct {
        std::atomic<uint64_t> total_compressed{0};
        std::atomic<uint64_t> total_compressed_size{0};
        std::atomic<uint64_t> compression_ops{0};
        std::atomic<uint64_t> decompression_ops{0};
        std::atomic<double> total_compress_time_ms{0};
        std::atomic<double> total_decompress_time_ms{0};
    } stats;
};

// FP16 packing implementation (zero-cost decompression)
size_t compress_fp16_pack(
    const void* src_data,
    size_t src_size,
    void* dst_buffer,
    size_t dst_capacity,
    float* compression_ratio
);

// LZ4 compression implementation
size_t compress_lz4(
    const void* src_data,
    size_t src_size,
    void* dst_buffer,
    size_t dst_capacity,
    ggml_moe_compression_type type,
    float* compression_ratio
);

// Sparse CSR compression implementation
size_t compress_sparse_csr(
    const void* src_data,
    size_t src_size,
    void* dst_buffer,
    size_t dst_capacity,
    float sparsity_threshold,
    float* compression_ratio
);

// Delta compression implementation
size_t compress_delta(
    const void* src_data,
    size_t src_size,
    void* dst_buffer,
    size_t dst_capacity,
    const void* base_data,
    float* compression_ratio
);

// Auto-select compression type
ggml_moe_compression_type select_auto_compression(
    const ggml_tensor* expert_tensor,
    int expert_id,
    const struct ggml_moe_expert_stats* stats,
    float sparsity_threshold
);

} // namespace ggml_moe_compression

#endif // __cplusplus