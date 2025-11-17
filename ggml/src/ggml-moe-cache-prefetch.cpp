#include "ggml-moe-cache.h"
#include <algorithm>
#include <set>
#include <cmath>

// Pre-fetching engine implementation

std::vector<int> ggml_moe_prefetch_engine::predict_next_experts(
    const std::vector<int>& current_experts,
    int top_k
) {
    std::lock_guard<std::mutex> lock(engine_mutex);
    
    if (current_experts.empty() || top_k <= 0) {
        return {};
    }
    
    // Combine predictions from different strategies
    std::unordered_map<int, double> expert_scores;
    
    // Strategy 1: Locality-based prediction (recently used experts)
    auto locality_predictions = predict_locality(current_experts);
    for (int expert : locality_predictions) {
        expert_scores[expert] += 1.0;
    }
    
    // Strategy 2: Co-occurrence based prediction
    if (!expert_cooccurrence.empty()) {
        for (int current_expert : current_experts) {
            if (current_expert >= 0 && current_expert < (int)expert_cooccurrence.size()) {
                const auto& cooccurrence = expert_cooccurrence[current_expert];
                for (size_t i = 0; i < cooccurrence.size(); ++i) {
                    if (cooccurrence[i] > 0) {
                        // Weight by co-occurrence frequency
                        double score = std::log(1.0 + cooccurrence[i]);
                        expert_scores[i] += score;
                    }
                }
            }
        }
    }
    
    // Strategy 3: Simple temporal locality (recent experts)
    for (int expert : recent_experts) {
        expert_scores[expert] += 0.5;  // Lower weight for temporal locality
    }
    
    // Remove experts that are already in current batch
    for (int expert : current_experts) {
        expert_scores.erase(expert);
    }
    
    // Sort by score and return top_k predictions
    std::vector<std::pair<int, double>> scored_experts(
        expert_scores.begin(), expert_scores.end()
    );
    
    std::sort(scored_experts.begin(), scored_experts.end(),
        [](const auto& a, const auto& b) {
            return a.second > b.second;
        });
    
    std::vector<int> predictions;
    for (size_t i = 0; i < std::min(size_t(top_k), scored_experts.size()); ++i) {
        predictions.push_back(scored_experts[i].first);
    }
    
    return predictions;
}

void ggml_moe_prefetch_engine::update_patterns(
    const std::vector<int>& used_experts,
    const std::vector<int>& tokens
) {
    std::lock_guard<std::mutex> lock(engine_mutex);
    
    // Update recent experts list
    update_recent_experts(used_experts);
    
    // Update co-occurrence matrix
    if (!expert_cooccurrence.empty()) {
        for (size_t i = 0; i < used_experts.size(); ++i) {
            int expert1 = used_experts[i];
            if (expert1 >= 0 && expert1 < (int)expert_cooccurrence.size()) {
                for (size_t j = i + 1; j < used_experts.size(); ++j) {
                    int expert2 = used_experts[j];
                    if (expert2 >= 0 && expert2 < (int)expert_cooccurrence.size()) {
                        // Increment co-occurrence count
                        expert_cooccurrence[expert1][expert2]++;
                        expert_cooccurrence[expert2][expert1]++;
                    }
                }
            }
        }
    }
    
    // Update token history (if tokens are provided)
    if (!tokens.empty()) {
        token_history.insert(token_history.end(), tokens.begin(), tokens.end());
        if (token_history.size() > HISTORY_SIZE) {
            token_history.erase(
                token_history.begin(),
                token_history.begin() + (token_history.size() - HISTORY_SIZE)
            );
        }
    }
}

std::vector<int> ggml_moe_prefetch_engine::predict_locality(
    const std::vector<int>& recent_experts
) {
    std::set<int> predictions;
    
    // Simple locality: predict experts with similar IDs
    for (int expert : recent_experts) {
        // Predict nearby expert IDs (with wrap-around for small expert counts)
        for (int offset = -2; offset <= 2; ++offset) {
            if (offset == 0) continue;
            
            int nearby_expert = expert + offset;
            if (nearby_expert >= 0 && nearby_expert < (int)expert_cooccurrence.size()) {
                predictions.insert(nearby_expert);
            }
        }
    }
    
    return std::vector<int>(predictions.begin(), predictions.end());
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
    return interface->create_cache(backend, *config, num_experts);
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
        stats.last_access_time = std::chrono::steady_clock::now();
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