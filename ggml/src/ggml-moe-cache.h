#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <unordered_map>
#include <list>
#include <vector>
#include <mutex>
#include <memory>
#include <chrono>
#include <string>

// Forward declaration for ML prefetching
namespace ggml_moe_ml {
    class ml_prefetch_engine;
}

#ifdef __cplusplus
extern "C" {
#endif

// Compression types for lossless expert compression
enum ggml_moe_compression_type {
    GGML_MOE_COMPRESSION_NONE = 0,           // No compression (raw FP16/FP32)
    GGML_MOE_COMPRESSION_FP16_PACK = 1,      // FP32→FP16 packing (if applicable)
    GGML_MOE_COMPRESSION_LZ4_FAST = 2,       // LZ4 with low compression level
    GGML_MOE_COMPRESSION_LZ4_HIGH = 3,       // LZ4 with high compression level
    GGML_MOE_COMPRESSION_SPARSE_CSR = 4,     // CSR format for sparse experts
    GGML_MOE_COMPRESSION_DELTA_PACK = 5,     // Delta from base expert
    GGML_MOE_COMPRESSION_AUTO = 6,           // Automatically select based on expert
};

// Configuration for MoE cache
struct ggml_moe_cache_config {
    size_t max_cache_size;           // Maximum GPU memory for cache (bytes)
    size_t max_experts;              // Maximum number of experts to cache
    bool enable_prefetch;            // Enable expert pre-fetching
    int prefetch_depth;              // Number of experts to pre-fetch
    bool enable_stats;               // Enable statistics collection
    float eviction_threshold;        // Memory threshold to trigger eviction (0.0-1.0)
    
    // Compression settings
    bool enable_compression;                    // Master switch for compression
    ggml_moe_compression_type default_type;     // Default compression type
    bool enable_auto_selection;                 // Auto-select per expert
    float compression_threshold;                // Min ratio to enable (default: 1.5)
    bool enable_fp16_packing;                   // Enable FP32→FP16 packing
    int lz4_compression_level;                  // 1-16 (1=fast, 16=best)
    bool enable_sparse_detection;               // Auto-detect sparse experts
    float sparsity_threshold;                   // Min sparsity for CSR (default: 0.5)
    
    // ML prefetching settings
    bool enable_ml_prefetch;                    // Enable ML-enhanced prefetching
    std::string ml_model_cache_dir;             // Directory for ML model storage
    float ml_learning_rate;                     // Learning rate for online training
    bool ml_enable_persistence;                 // Save/load ML models
    bool ml_reset_on_startup;                   // Reset ML model on startup
    float ml_accuracy_threshold;                // Minimum accuracy to use ML predictions
};

// Statistics for cache monitoring
struct ggml_moe_cache_stats {
    uint64_t total_requests;         // Total expert requests
    uint64_t cache_hits;             // Successful cache hits
    uint64_t cache_misses;           // Cache misses requiring load
    uint64_t evictions;              // Number of evictions
    uint64_t prefetches;             // Number of pre-fetched experts
    uint64_t prefetch_hits;          // Pre-fetched experts that were used
    double hit_rate;                 // Cache hit rate (0.0-1.0)
    double prefetch_accuracy;        // Pre-fetch accuracy (0.0-1.0)
    size_t current_size;             // Current cache size (bytes)
    size_t peak_size;                // Peak cache size (bytes)
    double avg_load_time;            // Average expert load time (ms)
};

// Expert usage statistics
struct ggml_moe_expert_stats {
    uint64_t access_count;           // Total times expert was accessed
    std::chrono::steady_clock::time_point last_access_time;
    double access_frequency;         // Accesses per second
    bool is_cached;                  // Currently in GPU cache
};

// Forward declarations
struct ggml_moe_cache;
struct ggml_moe_prefetch_engine;

// Layer-aware expert key for cache indexing
struct ggml_moe_expert_key {
    int layer_id;    // Layer index (0 to n_layer-1)
    int expert_id;   // Expert index within the layer (0 to n_expert-1)
    
    bool operator==(const ggml_moe_expert_key& other) const {
        return layer_id == other.layer_id && expert_id == other.expert_id;
    }
    
    bool operator!=(const ggml_moe_expert_key& other) const {
        return !(*this == other);
    }
};

// Hash function for ggml_moe_expert_key
struct ggml_moe_expert_key_hash {
    size_t operator()(const ggml_moe_expert_key& key) const {
        // Combine layer_id and expert_id into a single hash
        // Assuming typical ranges: layer_id 0-80, expert_id 0-512
        // Use 10 bits for expert_id (0-1023) and remaining bits for layer_id
        return (static_cast<size_t>(key.layer_id) << 10) ^ static_cast<size_t>(key.expert_id);
    }
};

// Backend-agnostic cache interface
struct ggml_moe_cache_interface {
    // Initialize cache for a backend
    virtual ggml_moe_cache* create_cache(
        ggml_backend_t backend,
        const ggml_moe_cache_config* config,
        int num_layers,
        int num_experts_per_layer
    ) = 0;
    
    // Get expert from cache (async)
    virtual ggml_backend_buffer_t get_expert_async(
        ggml_moe_cache* cache,
        int layer_id,
        int expert_id,
        const ggml_tensor* expert_tensor,
        void* stream
    ) = 0;
    
    // Pre-fetch experts
    virtual void prefetch_experts_async(
        ggml_moe_cache* cache,
        int layer_id,
        const std::vector<int>& expert_ids,
        const ggml_tensor* expert_tensor,
        void* stream
    ) = 0;
    
    // Update LRU and statistics
    virtual void touch_expert(
        ggml_moe_cache* cache,
        int layer_id,
        int expert_id
    ) = 0;
    
    // Get cache statistics
    virtual ggml_moe_cache_stats get_stats(
        const ggml_moe_cache* cache
    ) = 0;
    
    // Reset cache statistics
    virtual void reset_stats(
        ggml_moe_cache* cache
    ) = 0;
    
    // Clear all cached experts
    virtual void clear_cache(
        ggml_moe_cache* cache
    ) = 0;
    
    // Destroy cache
    virtual void destroy_cache(
        ggml_moe_cache* cache
    ) = 0;
    
    virtual ~ggml_moe_cache_interface() = default;
};

// Main MoE cache structure
struct ggml_moe_cache {
    // Cache configuration
    ggml_moe_cache_config config;
    
    // Backend context
    ggml_backend_t backend;
    void* backend_context;  // Backend-specific context (e.g., cuda_ctx)
    
    // Cache storage: (layer_id, expert_id) -> GPU buffer mapping
    std::unordered_map<ggml_moe_expert_key, ggml_backend_buffer_t, ggml_moe_expert_key_hash> cache_map;
    
    // System RAM storage for experts not in GPU cache
    std::unordered_map<ggml_moe_expert_key, ggml_backend_buffer_t, ggml_moe_expert_key_hash> system_ram_storage;
    
    // LRU tracking: most recently used at front
    std::list<ggml_moe_expert_key> lru_list;
    std::unordered_map<ggml_moe_expert_key, std::list<ggml_moe_expert_key>::iterator, ggml_moe_expert_key_hash> lru_iter;
    
    // Expert statistics: (layer_id, expert_id) -> stats
    std::unordered_map<ggml_moe_expert_key, ggml_moe_expert_stats, ggml_moe_expert_key_hash> expert_stats;
    
    // Number of layers and experts per layer
    int num_layers;
    int num_experts_per_layer;
    
    // Cache statistics
    ggml_moe_cache_stats stats;
    
    // Pre-fetch engine
    std::unique_ptr<ggml_moe_prefetch_engine> prefetch_engine;
    
    // Thread safety
    mutable std::mutex cache_mutex;
    
    // Backend-specific implementation
    ggml_moe_cache_interface* impl;
    
    // Constructor
    ggml_moe_cache(
        ggml_backend_t backend,
        const ggml_moe_cache_config& config,
        int num_layers,
        int num_experts_per_layer,
        ggml_moe_cache_interface* impl
    );
    
    // Destructor
    ~ggml_moe_cache();

    // LRU management functions
    void update_lru(const ggml_moe_expert_key& key);
    void evict_lru();
    bool is_cache_full() const;
    size_t get_cache_size() const;
    size_t get_expert_size(const ggml_moe_expert_key& key) const;
    
    // Dynamic streaming functions
    ggml_backend_buffer_t get_expert_from_system_ram(const ggml_moe_expert_key& key);
    void stream_expert_to_gpu(const ggml_moe_expert_key& key);
    bool has_expert_in_system_ram(const ggml_moe_expert_key& key) const;
};

// Pre-fetch engine for predicting expert usage
struct ggml_moe_prefetch_engine {
    // Token history for pattern recognition (per layer)
    std::unordered_map<int, std::vector<int>> token_history_per_layer;
    static constexpr size_t HISTORY_SIZE = 1024;
    
    // Expert co-occurrence matrix (per layer)
    std::unordered_map<int, std::vector<std::vector<int>>> expert_cooccurrence_per_layer;
    
    // Recent expert usage for locality prediction (per layer)
    std::unordered_map<int, std::vector<int>> recent_experts_per_layer;
    static constexpr size_t RECENT_SIZE = 64;
    
    // Thread safety
    std::mutex engine_mutex;
    
    // Prediction algorithms (layer-aware)
    std::vector<std::pair<int, int>> predict_next_experts(
        int layer_id,
        const std::vector<int>& current_experts,
        int top_k = 3
    );
    
    // ML-enhanced prediction (layer-aware)
    std::vector<std::pair<int, int>> predict_next_experts_ml(
        int layer_id,
        const std::vector<int>& current_experts,
        const std::vector<int>& recent_tokens,
        int position,
        int top_k,
        class ggml_moe_ml::ml_prefetch_engine* ml_engine
    );
    
    // Update patterns based on actual usage (layer-aware)
    void update_patterns(
        int layer_id,
        const std::vector<int>& used_experts,
        const std::vector<int>& tokens
    );
    
    // Simple locality-based prediction (layer-aware)
    std::vector<std::pair<int, int>> predict_locality(
        int layer_id,
        const std::vector<int>& recent_experts
    );
    
    // Update recent expert list (layer-aware)
    void update_recent_experts(int layer_id, const std::vector<int>& experts);
    
    // Initialize co-occurrence matrix for a layer
    void init_cooccurrence(int layer_id, int num_experts);
};

// C API for external usage
GGML_API ggml_moe_cache* ggml_moe_cache_init(
    ggml_backend_t backend,
    const ggml_moe_cache_config* config,
    int num_layers,
    int num_experts_per_layer
);

GGML_API void ggml_moe_cache_free(
    ggml_moe_cache* cache
);

GGML_API ggml_backend_buffer_t ggml_moe_cache_get_expert(
    ggml_moe_cache* cache,
    int layer_id,
    int expert_id,
    const ggml_tensor* expert_tensor,
    void* stream
);

GGML_API void ggml_moe_cache_prefetch(
    ggml_moe_cache* cache,
    int layer_id,
    const int* expert_ids,
    int num_experts,
    const ggml_tensor* expert_tensor,
    void* stream
);

GGML_API void ggml_moe_cache_touch(
    ggml_moe_cache* cache,
    int layer_id,
    int expert_id
);

GGML_API ggml_moe_cache_stats ggml_moe_cache_get_stats(
    const ggml_moe_cache* cache
);

GGML_API void ggml_moe_cache_reset_stats(
    ggml_moe_cache* cache
);

GGML_API void ggml_moe_cache_clear(
    ggml_moe_cache* cache
);

// Initialize cache for a model
GGML_API ggml_moe_cache* ggml_moe_cache_init_for_model(
    ggml_backend_t backend,
    const ggml_tensor* expert_tensor,
    int num_layers
);

// Log cache statistics
GGML_API void ggml_moe_cache_log_stats(const ggml_moe_cache* cache);

// Reset cache for new inference session
GGML_API void ggml_moe_cache_reset_session(ggml_moe_cache* cache);

// Helper function to check if cache should be used for a tensor
GGML_API bool ggml_moe_should_use_cache(const ggml_tensor* tensor);

// Cached version of ggml_mul_mat_id for MoE operations
GGML_API void ggml_mul_mat_id_cached(
    ggml_tensor* dst,
    int layer_id,
    ggml_moe_cache* cache
);

// Get cache interface for backend
GGML_API ggml_moe_cache_interface* ggml_moe_cache_get_interface(
    ggml_backend_t backend
);

#ifdef __cplusplus
}
#endif