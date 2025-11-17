#pragma once

#include "llama.h"
#include <cstdint>
#include <string>

// MoE cache parameters for command-line configuration
struct llama_moe_cache_params {
    // Number of experts to keep offloaded to GPU (-1 for auto based on VRAM)
    int32_t n_gpu_experts = -1;
    
    // Target VRAM allocation for experts (bytes, 0 for auto)
    size_t vram_budget = 0;
    
    // Enable/disable MoE caching
    bool enable_cache = true;
    
    // Enable/disable expert pre-fetching
    bool enable_prefetch = true;
    
    // Pre-fetch depth (number of experts to pre-fetch)
    int32_t prefetch_depth = 3;
    
    // Enable statistics collection
    bool enable_stats = false;
    
    // Eviction threshold (0.0-1.0, when to start evicting)
    float eviction_threshold = 0.9f;
    
    // Force all experts to system memory (no GPU offloading)
    bool force_cpu = false;
    
    // Override mmap setting for experts
    bool experts_no_mmap = false;
    
    // Parse from command line arguments
    static llama_moe_cache_params parse_from_args(int argc, char ** argv);
    
    // Parse from string (for API usage)
    static llama_moe_cache_params parse_from_string(const std::string& params);
    
    // Convert to string for logging
    std::string to_string() const;
};

// Add MoE cache parameters to existing llama_context_params
struct llama_context_params_moe : public llama_context_params {
    llama_moe_cache_params moe_cache_params;
};

// C API extensions
GGML_API struct llama_context_params llama_context_default_params_with_moe(void);
GGML_API void llama_context_params_set_moe_cache(struct llama_context_params* params, 
                                                 const llama_moe_cache_params* moe_params);