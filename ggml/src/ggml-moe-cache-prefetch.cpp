#include "ggml-moe-cache.h"
#include "ggml-moe-cache-ml-prefetch.h"
#include "ggml.h"
#include <algorithm>
#include <set>
#include <cstring>

// Pre-fetch engine implementation

// Global ML prefetch engine (per-model)
static std::unordered_map<ggml_moe_cache*, std::unique_ptr<ggml_moe_ml::ml_prefetch_engine>> g_ml_engines;
static std::mutex g_ml_engines_mutex;

std::vector<int> ggml_moe_prefetch_engine::predict_next_experts(
    const std::vector<int>& current_experts,
    int top_k
) {
    std::lock_guard<std::mutex> lock(engine_mutex);
    
    std::vector<int> predictions;
    std::set<int> seen_experts(current_experts.begin(), current_experts.end());
    
    // Check if ML prefetching is enabled for this cache
    // This would need to be passed through the cache configuration
    // For now, we'll use the heuristic approach as fallback
    
    // Method 1: Co-occurrence based prediction
    for (int expert : current_experts) {
        if (expert >= 0 && expert < (int)expert_cooccurrence.size()) {
            // Find experts that frequently co-occur with current experts
            const auto& cooccur = expert_cooccurrence[expert];
            
            // Get top co-occurring experts
            std::vector<int> candidate_experts;
            for (int i = 0; i < (int)cooccur.size(); ++i) {
                if (cooccur[i] > 0 && seen_experts.find(i) == seen_experts.end()) {
                    candidate_experts.push_back(i);
                }
            }
            
            // Sort by co-occurrence frequency
            std::sort(candidate_experts.begin(), candidate_experts.end(),
                [&cooccur](int a, int b) {
                    return cooccur[a] > cooccur[b];
                });
            
            // Add top candidates
            for (int i = 0; i < std::min(top_k/2, (int)candidate_experts.size()); ++i) {
                predictions.push_back(candidate_experts[i]);
                seen_experts.insert(candidate_experts[i]);
            }
        }
    }
    
    // Method 2: Locality-based prediction
    std::vector<int> locality_preds = predict_locality(recent_experts);
    for (int expert : locality_preds) {
        if (seen_experts.find(expert) == seen_experts.end() && predictions.size() < (size_t)top_k) {
            predictions.push_back(expert);
            seen_experts.insert(expert);
        }
    }
    
    // Trim to requested size
    if (predictions.size() > (size_t)top_k) {
        predictions.resize(top_k);
    }
    
    return predictions;
}

// ML-enhanced prediction (new function)
std::vector<int> ggml_moe_prefetch_engine::predict_next_experts_ml(
    const std::vector<int>& current_experts,
    const std::vector<int>& recent_tokens,
    int layer_id,
    int position,
    int top_k,
    struct ggml_moe_ml::ml_prefetch_engine* ml_engine
) {
    if (!ml_engine || !ml_engine->is_ready()) {
        // Fall back to heuristic prediction
        return predict_next_experts(current_experts, top_k);
    }
    
    // Use ML model for prediction
    return ml_engine->predict_next_experts(
        current_experts,
        recent_tokens,
        layer_id,
        position,
        top_k
    );
}

void ggml_moe_prefetch_engine::update_patterns(
    const std::vector<int>& used_experts,
    const std::vector<int>& tokens
) {
    std::lock_guard<std::mutex> lock(engine_mutex);
    
    // Update co-occurrence matrix
    for (size_t i = 0; i < used_experts.size(); ++i) {
        int expert1 = used_experts[i];
        if (expert1 < 0 || expert1 >= (int)expert_cooccurrence.size()) continue;
        
        for (size_t j = i + 1; j < used_experts.size(); ++j) {
            int expert2 = used_experts[j];
            if (expert2 < 0 || expert2 >= (int)expert_cooccurrence.size()) continue;
            
            // Increment co-occurrence count
            expert_cooccurrence[expert1][expert2]++;
            expert_cooccurrence[expert2][expert1]++;
        }
    }
    
    // Update recent experts
    update_recent_experts(used_experts);
    
    // Update token history
    token_history.insert(token_history.end(), tokens.begin(), tokens.end());
    if (token_history.size() > HISTORY_SIZE) {
        token_history.erase(token_history.begin(), token_history.end() - HISTORY_SIZE);
    }
}

std::vector<int> ggml_moe_prefetch_engine::predict_locality(
    const std::vector<int>& recent_experts
) {
    std::vector<int> predictions;
    
    if (recent_experts.empty()) return predictions;
    
    // Simple locality: predict experts that are numerically close to recently used ones
    std::set<int> seen_experts(recent_experts.begin(), recent_experts.end());
    
    for (int expert : recent_experts) {
        // Check neighboring experts
        for (int offset = -2; offset <= 2; ++offset) {
            if (offset == 0) continue;
            
            int neighbor = expert + offset;
            if (neighbor >= 0 && neighbor < (int)expert_cooccurrence.size() &&
                seen_experts.find(neighbor) == seen_experts.end()) {
                predictions.push_back(neighbor);
                seen_experts.insert(neighbor);
            }
        }
    }
    
    return predictions;
}

void ggml_moe_prefetch_engine::update_recent_experts(const std::vector<int>& experts) {
    // Add experts to recent list, avoiding duplicates
    for (int expert : experts) {
        // Remove if already in list
        auto it = std::find(recent_experts.begin(), recent_experts.end(), expert);
        if (it != recent_experts.end()) {
            recent_experts.erase(it);
        }
        
        // Add to front
        recent_experts.insert(recent_experts.begin(), expert);
        
        // Trim to size
        if (recent_experts.size() > RECENT_SIZE) {
            recent_experts.resize(RECENT_SIZE);
        }
    }
}

void ggml_moe_prefetch_engine::init_cooccurrence(int num_experts) {
    std::lock_guard<std::mutex> lock(engine_mutex);
    
    expert_cooccurrence.assign(num_experts, std::vector<int>(num_experts, 0));
    token_history.clear();
    recent_experts.clear();
}

// C API implementation
ggml_moe_cache* ggml_moe_cache_init(
    ggml_backend_t backend,
    const ggml_moe_cache_config* config,
    int num_experts
) {
    if (!backend || !config || num_experts <= 0) {
        return nullptr;
    }
    
    // Get backend-specific interface
    ggml_moe_cache_interface* interface = ggml_moe_cache_get_interface(backend);
    if (!interface) {
        return nullptr;
    }
    
    // Create cache using backend interface
    return interface->create_cache(backend, config, num_experts);
}

void ggml_moe_cache_free(
    ggml_moe_cache* cache
) {
    if (!cache) return;
    
    // Use backend interface to destroy
    if (cache->impl) {
        cache->impl->destroy_cache(cache);
    }
}

ggml_backend_buffer_t ggml_moe_cache_get_expert(
    ggml_moe_cache* cache,
    int expert_id,
    const ggml_tensor* expert_tensor,
    void* stream
) {
    if (!cache || !cache->impl) return nullptr;
    
    return cache->impl->get_expert_async(cache, expert_id, expert_tensor, stream);
}

void ggml_moe_cache_prefetch(
    ggml_moe_cache* cache,
    const int* expert_ids,
    int num_experts,
    const ggml_tensor* expert_tensor,
    void* stream
) {
    if (!cache || !cache->impl || !expert_ids || num_experts <= 0) return;
    
    std::vector<int> expert_ids_vec(expert_ids, expert_ids + num_experts);
    cache->impl->prefetch_experts_async(cache, expert_ids_vec, expert_tensor, stream);
}

void ggml_moe_cache_touch(
    ggml_moe_cache* cache,
    int expert_id
) {
    if (!cache || !cache->impl) return;
    
    cache->impl->touch_expert(cache, expert_id);
}

ggml_moe_cache_stats ggml_moe_cache_get_stats(
    const ggml_moe_cache* cache
) {
    if (!cache || !cache->impl) return ggml_moe_cache_stats{};
    
    return cache->impl->get_stats(cache);
}

void ggml_moe_cache_reset_stats(
    ggml_moe_cache* cache
) {
    if (!cache || !cache->impl) return;
    
    cache->impl->reset_stats(cache);
}

void ggml_moe_cache_clear(
    ggml_moe_cache* cache
) {
    if (!cache || !cache->impl) return;
    
    cache->impl->clear_cache(cache);
}

// Constructor implementation
ggml_moe_cache::ggml_moe_cache(
    ggml_backend_t backend,
    const ggml_moe_cache_config& config,
    int num_experts,
    ggml_moe_cache_interface* impl
) : config(config), backend(backend), num_experts(num_experts), impl(impl) {
    // Initialize expert statistics
    expert_stats.resize(num_experts);
    for (auto& stats : expert_stats) {
        stats.access_count = 0;
        stats.access_frequency = 0.0;
        stats.is_cached = false;
        stats.access_frequency = 0.0;
        stats.is_cached = false;
    }
    
    // Initialize prefetch engine if enabled
    if (config.enable_prefetch) {
        prefetch_engine = std::make_unique<ggml_moe_prefetch_engine>();
        prefetch_engine->init_cooccurrence(num_experts);
    }
    
    // Initialize statistics
    memset(&stats, 0, sizeof(stats));
}

// Destructor implementation
ggml_moe_cache::~ggml_moe_cache() {
    // Clear cache before destruction
    if (impl) {
        impl->clear_cache(this);
    }
}