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

std::vector<std::pair<int, int>> ggml_moe_prefetch_engine::predict_next_experts(
    int layer_id,
    const std::vector<int>& current_experts,
    int top_k
) {
    std::lock_guard<std::mutex> lock(engine_mutex);
    
    std::vector<std::pair<int, int>> predictions;
    std::set<int> seen_experts(current_experts.begin(), current_experts.end());
    
    // Get co-occurrence matrix for this layer
    auto& cooccurrence = expert_cooccurrence_per_layer[layer_id];
    auto& recent_experts = recent_experts_per_layer[layer_id];
    
    // Method 1: Co-occurrence based prediction
    for (int expert : current_experts) {
        if (expert >= 0 && expert < (int)cooccurrence.size()) {
            // Find experts that frequently co-occur with current experts
            const auto& cooccur = cooccurrence[expert];
            
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
            
            // Add top candidates (same layer)
            for (int i = 0; i < std::min(top_k/2, (int)candidate_experts.size()); ++i) {
                predictions.push_back({layer_id, candidate_experts[i]});
                seen_experts.insert(candidate_experts[i]);
            }
        }
    }
    
    // Method 2: Locality-based prediction
    std::vector<std::pair<int, int>> locality_preds = predict_locality(layer_id, recent_experts);
    for (const auto& [pred_layer_id, expert_id] : locality_preds) {
        if (seen_experts.find(expert_id) == seen_experts.end() && predictions.size() < (size_t)top_k) {
            predictions.push_back({pred_layer_id, expert_id});
            seen_experts.insert(expert_id);
        }
    }
    
    // Trim to requested size
    if (predictions.size() > (size_t)top_k) {
        predictions.resize(top_k);
    }
    
    return predictions;
}

// ML-enhanced prediction (layer-aware)
std::vector<std::pair<int, int>> ggml_moe_prefetch_engine::predict_next_experts_ml(
    int layer_id,
    const std::vector<int>& current_experts,
    const std::vector<int>& recent_tokens,
    int position,
    int top_k,
    class ggml_moe_ml::ml_prefetch_engine* ml_engine
) {
    if (!ml_engine || !ml_engine->is_ready()) {
        // Fall back to heuristic prediction
        return predict_next_experts(layer_id, current_experts, top_k);
    }
    
    // Use ML model for prediction
    std::vector<int> predicted_experts = ml_engine->predict_next_experts(
        current_experts,
        recent_tokens,
        layer_id,
        position,
        top_k
    );
    
    // Convert to layer-expert pairs
    std::vector<std::pair<int, int>> predictions;
    for (int expert_id : predicted_experts) {
        predictions.push_back({layer_id + 1, expert_id}); // Predict next layer
    }
    
    return predictions;
}

void ggml_moe_prefetch_engine::update_patterns(
    int layer_id,
    const std::vector<int>& used_experts,
    const std::vector<int>& tokens
) {
    std::lock_guard<std::mutex> lock(engine_mutex);
    
    // Update co-occurrence matrix for this layer
    auto& cooccurrence = expert_cooccurrence_per_layer[layer_id];
    for (size_t i = 0; i < used_experts.size(); ++i) {
        int expert1 = used_experts[i];
        if (expert1 < 0 || expert1 >= (int)cooccurrence.size()) continue;
        
        for (size_t j = i + 1; j < used_experts.size(); ++j) {
            int expert2 = used_experts[j];
            if (expert2 < 0 || expert2 >= (int)cooccurrence.size()) continue;
            
            // Increment co-occurrence count
            cooccurrence[expert1][expert2]++;
            cooccurrence[expert2][expert1]++;
        }
    }
    
    // Update recent experts for this layer
    update_recent_experts(layer_id, used_experts);
    
    // Update token history for this layer
    auto& token_history = token_history_per_layer[layer_id];
    token_history.insert(token_history.end(), tokens.begin(), tokens.end());
    if (token_history.size() > HISTORY_SIZE) {
        token_history.erase(token_history.begin(), token_history.end() - HISTORY_SIZE);
    }
}

std::vector<std::pair<int, int>> ggml_moe_prefetch_engine::predict_locality(
    int layer_id,
    const std::vector<int>& recent_experts
) {
    std::vector<std::pair<int, int>> predictions;
    
    if (recent_experts.empty()) return predictions;
    
    // Simple locality: predict experts that are numerically close to recently used ones
    std::set<int> seen_experts(recent_experts.begin(), recent_experts.end());
    
    for (int expert : recent_experts) {
        // Check neighboring experts in the same layer
        for (int offset = -2; offset <= 2; ++offset) {
            if (offset == 0) continue;
            
            int neighbor = expert + offset;
            auto& cooccurrence = expert_cooccurrence_per_layer[layer_id];
            if (neighbor >= 0 && neighbor < (int)cooccurrence.size() &&
                seen_experts.find(neighbor) == seen_experts.end()) {
                predictions.push_back({layer_id, neighbor});
                seen_experts.insert(neighbor);
            }
        }
    }
    
    return predictions;
}

void ggml_moe_prefetch_engine::update_recent_experts(int layer_id, const std::vector<int>& experts) {
    auto& recent_experts = recent_experts_per_layer[layer_id];
    
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

void ggml_moe_prefetch_engine::init_cooccurrence(int layer_id, int num_experts) {
    std::lock_guard<std::mutex> lock(engine_mutex);
    
    expert_cooccurrence_per_layer[layer_id] = std::vector<std::vector<int>>(num_experts, std::vector<int>(num_experts, 0));
    token_history_per_layer[layer_id].clear();
    recent_experts_per_layer[layer_id].clear();
}

// C API implementations are provided in ggml-moe-cache-backend.cpp

// Constructor implementation
// Constructor implementation
ggml_moe_cache::ggml_moe_cache(
    ggml_backend_t backend,
    const ggml_moe_cache_config& config,
    int num_layers,
    int num_experts_per_layer,
    ggml_moe_cache_interface* impl
) : config(config),
    backend(backend),
    backend_context(nullptr),
    num_layers(num_layers),
    num_experts_per_layer(num_experts_per_layer),
    stats(),
    prefetch_engine(nullptr),
    expert_source(nullptr),
    impl(impl) {
    // Initialize prefetch engine if enabled
    if (config.enable_prefetch) {
        prefetch_engine = std::make_unique<ggml_moe_prefetch_engine>();
        // Initialize co-occurrence matrices for each layer
        for (int layer_id = 0; layer_id < num_layers; ++layer_id) {
            prefetch_engine->init_cooccurrence(layer_id, num_experts_per_layer);
        }
    }
    
    // Initialize statistics
    stats = ggml_moe_cache_stats{};
}
// Destructor implementation
ggml_moe_cache::~ggml_moe_cache() {
    // Clear cache before destruction
    if (impl) {
        impl->clear_cache(this);
    }
}
