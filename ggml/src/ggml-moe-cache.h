#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <unordered_map>
#include <list>
#include <vector>
#include <mutex>
#include <memory>
#include <chrono>

#ifdef __cplusplus
extern "C" {
#endif

// Configuration for MoE cache
struct ggml_moe_cache_config {
    size_t max_cache_size;           // Maximum GPU memory for cache (bytes)
    size_t max_experts;              // Maximum number of experts to cache
    bool enable_prefetch;            // Enable expert pre-fetching
    int prefetch_depth;              // Number of experts to pre-fetch
    bool enable_stats;               // Enable statistics collection
    float eviction_threshold;        // Memory threshold to trigger eviction (0.0-1.0)
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

// Backend-agnostic cache interface
struct ggml_moe_cache_interface {
    // Initialize cache for a backend
    virtual ggml_moe_cache* create_cache(
        ggml_backend_t backend,
        const ggml_moe_cache_config& config,
        int num_experts
    ) = 0;
    
    // Get expert from cache (async)
    virtual ggml_backend_buffer_t get_expert_async(
        ggml_moe_cache* cache,
        int expert_id,
        const ggml_tensor* expert_tensor,
        void* stream
    ) = 0;
    
    // Pre-fetch experts
    virtual void prefetch_experts_async(
        ggml_moe_cache* cache,
        const std::vector<int>& expert_ids,
        const ggml_tensor* expert_tensor,
        void* stream
    ) = 0;
    
    // Update LRU and statistics
    virtual void touch_expert(
        ggml_moe_cache* cache,
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
    
    // Cache storage: expert_id -> GPU buffer mapping
    std::unordered_map<int, ggml_backend_buffer_t> cache_map;
    
    // LRU tracking: most recently used at front
    std::list<int> lru_list;
    std::unordered_map<int, std::list<int>::iterator> lru_iter;
    
    // Expert statistics
    std::vector<ggml_moe_expert_stats> expert_stats;
    int num_experts;
    
    // Cache statistics
    ggml_moe_cache_stats stats;
    
    // Pre-fetch engine
    std::unique_ptr<ggml_moe_prefetch_engine> prefetch_engine;
    
    // Thread safety
    std::mutex cache_mutex;
    
    // Backend-specific implementation
    ggml_moe_cache_interface* impl;
    
    // Constructor
    ggml_moe_cache(
        ggml_backend_t backend,
        const ggml_moe_cache_config& config,
        int num_experts,
        ggml_moe_cache_interface* impl
    );
    
    // Destructor
    ~ggml_moe_cache();
};

// Pre-fetch engine for predicting expert usage
struct ggml_moe_prefetch_engine {
    // Token history for pattern recognition
    std::vector<int> token_history;
    static constexpr size_t HISTORY_SIZE = 1024;
    
    // Expert co-occurrence matrix
    std::vector<std::vector<int>> expert_cooccurrence;
    
    // Recent expert usage for locality prediction
    std::vector<int> recent_experts;
    static constexpr size_t RECENT_SIZE = 64;
    
    // Thread safety
    std::mutex engine_mutex;
    
    // Prediction algorithms
    std::vector<int> predict_next_experts(
        const std::vector<int>& current_experts,
        int top_k = 3
    );
    
    // Update patterns based on actual usage
    void update_patterns(
        const std::vector<int>& used_experts,
        const std::vector<int>& tokens
    );
    
    // Simple locality-based prediction
    std::vector<int> predict_locality(
        const std::vector<int>& recent_experts
    );
    
    // Update recent expert list
    void update_recent_experts(const std::vector<int>& experts);
    
    // Initialize co-occurrence matrix
    void init_cooccurrence(int num_experts);
};

// C API for external usage
GGML_API ggml_moe_cache* ggml_moe_cache_init(
    ggml_backend_t backend,
    const ggml_moe_cache_config* config,
    int num_experts
);

GGML_API void ggml_moe_cache_free(
    ggml_moe_cache* cache
);

GGML_API ggml_backend_buffer_t ggml_moe_cache_get_expert(
    ggml_moe_cache* cache,
    int expert_id,
    const ggml_tensor* expert_tensor,
    void* stream
);

GGML_API void ggml_moe_cache_prefetch(
    ggml_moe_cache* cache,
    const int* expert_ids,
    int num_experts,
    const ggml_tensor* expert_tensor,
    void* stream
);

GGML_API void ggml_moe_cache_touch(
    ggml_moe_cache* cache,
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

// Get cache interface for backend
GGML_API ggml_moe_cache_interface* ggml_moe_cache_get_interface(
    ggml_backend_t backend
);

#ifdef __cplusplus
}
#endif