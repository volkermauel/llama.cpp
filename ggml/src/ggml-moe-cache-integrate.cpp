#include "ggml-moe-cache.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include <vector>
#include <algorithm>
#include <cstring>
#include <string>
#include <cinttypes>

// Integration of MoE cache with existing ggml operations

// Helper function to extract expert IDs from ggml tensor
static std::vector<int> extract_expert_ids_from_tensor(const ggml_tensor* ids_tensor) {
    std::vector<int> expert_ids;
    
    if (!ids_tensor || !ids_tensor->data) {
        return expert_ids;
    }
    
    // Assuming ids_tensor is 2D: [n_expert_used, n_tokens]
    int64_t n_expert_used = ids_tensor->ne[0];
    int64_t n_tokens = ids_tensor->ne[1];
    
    expert_ids.reserve(n_expert_used * n_tokens);
    
    // Extract expert IDs from tensor
    const int32_t* ids_data = (const int32_t*)ids_tensor->data;
    size_t nb1 = ids_tensor->nb[1] / sizeof(int32_t);  // stride between tokens
    
    for (int64_t i = 0; i < n_tokens; ++i) {
        for (int64_t j = 0; j < n_expert_used; ++j) {
            int expert_id = ids_data[i * nb1 + j];
            expert_ids.push_back(expert_id);
        }
    }
    
    // Remove duplicates
    std::sort(expert_ids.begin(), expert_ids.end());
    expert_ids.erase(std::unique(expert_ids.begin(), expert_ids.end()), expert_ids.end());
    
    return expert_ids;
}

// Cached version of ggml_mul_mat_id for MoE operations
void ggml_mul_mat_id_cached(
    ggml_tensor* dst,
    int layer_id,
    ggml_moe_cache* cache
) {
    if (!dst || !cache) {
        return;
    }
    
    const ggml_tensor* src0 = dst->src[0];  // Expert weights [n_experts, ...]
    const ggml_tensor* src1 = dst->src[1];  // Input tokens
    const ggml_tensor* ids = dst->src[2];   // Expert IDs
    
    if (!src0 || !src1 || !ids) {
        return;
    }
    
    // Extract expert IDs that will be used
    std::vector<int> current_expert_ids = extract_expert_ids_from_tensor(ids);
    
    if (current_expert_ids.empty()) {
        return;
    }
    
    // Get backend stream for async operations
    void* stream = nullptr; // Use default stream for now
    
    // Pre-fetch predicted experts for next iteration (layer-aware)
    if (cache->config.enable_prefetch && cache->prefetch_engine) {
        std::vector<std::pair<int, int>> next_experts = cache->prefetch_engine->predict_next_experts(
            layer_id,
            current_expert_ids,
            cache->config.prefetch_depth
        );
        
        if (!next_experts.empty()) {
            // Group predictions by layer for efficient prefetching
            std::unordered_map<int, std::vector<int>> experts_by_layer;
            for (const auto& prediction : next_experts) {
                int pred_layer_id = prediction.first;
                int pred_expert_id = prediction.second;
                experts_by_layer[pred_layer_id].push_back(pred_expert_id);
            }
            
            // Prefetch for each layer
            for (const auto& layer_experts : experts_by_layer) {
                int pred_layer_id = layer_experts.first;
                const std::vector<int>& expert_ids = layer_experts.second;
                cache->impl->prefetch_experts_async(
                    cache,
                    pred_layer_id,
                    expert_ids,
                    src0,
                    stream
                );
            }
        }
    }
    
    // Get current experts from cache (async)
    std::vector<ggml_backend_buffer_t> expert_buffers;
    expert_buffers.reserve(current_expert_ids.size());
    
    for (int expert_id : current_expert_ids) {
        auto buffer = cache->impl->get_expert_async(
            cache,
            layer_id,
            expert_id,
            src0,
            stream
        );
        
        if (buffer) {
            expert_buffers.push_back(buffer);
        }
    }
    
    // Wait for all expert transfers to complete
    // Backend-specific synchronization would go here
    
    // Update LRU and statistics for used experts
    for (int expert_id : current_expert_ids) {
        cache->impl->touch_expert(cache, layer_id, expert_id);
    }
    
    // Update prefetch engine with actual usage (layer-aware)
    if (cache->prefetch_engine) {
        cache->prefetch_engine->update_patterns(layer_id, current_expert_ids, {});
    }
    
    // Now perform the actual matrix multiplication with cached experts
    // This would call the existing ggml_cuda_mul_mat_id implementation
    // but with the cached expert buffers instead of the original CPU buffers
    
    // For now, we'll update the source tensor to point to cached buffers
    // In a full implementation, this would integrate with the existing CUDA kernels
    
    // Mark that we've used the cached experts
    for (size_t i = 0; i < current_expert_ids.size(); ++i) {
        int expert_id = current_expert_ids[i];
        if (i < expert_buffers.size() && expert_buffers[i]) {
            // Update statistics for successful cache usage
            ggml_moe_expert_key key{layer_id, expert_id};
            if (cache->cache_map.find(key) != cache->cache_map.end()) {
                cache->stats.prefetch_hits++;
            }
        }
    }
}

// Configuration from environment variables
static ggml_moe_cache_config get_cache_config_from_env() {
    ggml_moe_cache_config config = {};
    
    // Default values
    config.max_cache_size = 2ULL * 1024 * 1024 * 1024;  // 2GB default
    config.max_experts = 128;  // Default max experts
    config.enable_prefetch = true;
    config.prefetch_depth = 3;
    config.enable_stats = true;
    config.eviction_threshold = 0.9;  // Evict when 90% full
    
    // Read from environment variables
    const char* cache_size_env = getenv("GGML_MOE_CACHE_SIZE");
    if (cache_size_env) {
        config.max_cache_size = std::stoull(cache_size_env);
    }
    
    const char* max_experts_env = getenv("GGML_MOE_MAX_EXPERTS");
    if (max_experts_env) {
        config.max_experts = std::stoi(max_experts_env);
    }
    
    const char* enable_prefetch_env = getenv("GGML_MOE_ENABLE_PREFETCH");
    if (enable_prefetch_env) {
        config.enable_prefetch = (std::stoi(enable_prefetch_env) != 0);
    }
    
    const char* prefetch_depth_env = getenv("GGML_MOE_PREFETCH_DEPTH");
    if (prefetch_depth_env) {
        config.prefetch_depth = std::stoi(prefetch_depth_env);
    }
    
    const char* enable_stats_env = getenv("GGML_MOE_ENABLE_STATS");
    if (enable_stats_env) {
        config.enable_stats = (std::stoi(enable_stats_env) != 0);
    }
    
    const char* eviction_threshold_env = getenv("GGML_MOE_EVICTION_THRESHOLD");
    if (eviction_threshold_env) {
        config.eviction_threshold = std::stof(eviction_threshold_env);
    }
    
    return config;
}

// Initialize cache for a model
ggml_moe_cache* ggml_moe_cache_init_for_model(
    ggml_backend_t backend,
    const ggml_tensor* expert_tensor,
    int num_layers
) {
    if (!backend || !expert_tensor) {
        return nullptr;
    }
    
    // Get configuration from environment
    ggml_moe_cache_config config = get_cache_config_from_env();
    
    // Extract number of experts from tensor shape
    // Assuming expert_tensor is 3D: [dim0, dim1, num_experts]
    int num_experts = expert_tensor->ne[2];
    
    // Initialize cache with layer awareness
    ggml_moe_cache* cache = ggml_moe_cache_init(backend, &config, num_layers, num_experts);
    
    if (cache && cache->prefetch_engine) {
        // Initialize co-occurrence matrices for each layer
        for (int layer_id = 0; layer_id < num_layers; ++layer_id) {
            cache->prefetch_engine->init_cooccurrence(layer_id, num_experts);
        }
    }
    
    return cache;
}

// Log cache statistics
void ggml_moe_cache_log_stats(const ggml_moe_cache* cache) {
    if (!cache || !cache->config.enable_stats) {
        return;
    }
    
    ggml_moe_cache_stats stats;
    cache->impl->get_stats(cache, &stats);
    
    // Log cache statistics
    printf("MoE Cache Statistics:\n");
    printf("  Total Requests: %" PRIu64 "\n", stats.total_requests);
    printf("  Cache Hits: %" PRIu64 " (%.2f%%)\n",
           stats.cache_hits, stats.hit_rate * 100.0);
    printf("  Cache Misses: %" PRIu64 "\n", stats.cache_misses);
    printf("  Evictions: %" PRIu64 "\n", stats.evictions);
    printf("  Prefetches: %" PRIu64 "\n", stats.prefetches);
    printf("  Prefetch Accuracy: %.2f%%\n",
           stats.prefetch_accuracy * 100.0);
    printf("  Current Cache Size: %.2f MB\n",
           stats.current_size / (1024.0 * 1024.0));
    printf("  Peak Cache Size: %.2f MB\n",
           stats.peak_size / (1024.0 * 1024.0));
    printf("  Average Load Time: %.2f ms\n", stats.avg_load_time);
}

// Reset cache for new inference session
void ggml_moe_cache_reset_session(ggml_moe_cache* cache) {
    if (!cache) return;
    
    // Clear cached experts but keep statistics
    cache->impl->clear_cache(cache);
    
    // Reset prefetch engine state (layer-aware)
    if (cache->prefetch_engine) {
        cache->prefetch_engine->token_history_per_layer.clear();
        cache->prefetch_engine->recent_experts_per_layer.clear();
        cache->prefetch_engine->expert_cooccurrence_per_layer.clear();
    }
    
    // Reset timing statistics but keep counters
    cache->stats.avg_load_time = 0.0;
}

// Helper function to check if cache should be used for a tensor
bool ggml_moe_should_use_cache(const ggml_tensor* tensor) {
    if (!tensor) return false;
    
    // Check if this is an MoE weight tensor
    // Look for patterns in tensor name or dimensions
    const char* name = tensor->name;
    if (name && (
        strstr(name, "ffn_gate_exps") ||
        strstr(name, "ffn_up_exps") ||
        strstr(name, "ffn_down_exps") ||
        strstr(name, "expert") ||
        strstr(name, "moe")
    )) {
        return true;
    }
    
    // Check if tensor has expert-like dimensions
    // Typically MoE tensors are 3D with last dimension being number of experts
    // Check if tensor is 3D: ne[0], ne[1], ne[2] are valid, ne[3] should be 1 or 0
    if (tensor->ne[2] > 1 && tensor->ne[2] <= 512 &&
        (tensor->ne[3] == 0 || tensor->ne[3] == 1)) {
        return true;
    }
    
    return false;
}