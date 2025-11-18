#pragma once

#include "llama.h"
#include "ggml-moe-cache.h"  // For ggml_moe_compression_type enum
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
    
    // Compression settings
    ggml_moe_compression_type compression_type = GGML_MOE_COMPRESSION_NONE;
    float compression_threshold = 1.5f;
    int32_t lz4_compression_level = 4;
    bool enable_auto_selection = false;
    bool enable_fp16_packing = false;
    float sparsity_threshold = 0.5f;
    
    // ML prefetching settings
    bool enable_ml_prefetch = false;
    std::string ml_model_cache_dir;
    float ml_learning_rate = 0.01f;
    bool ml_enable_persistence = true;
    bool ml_reset_on_startup = false;
    float ml_accuracy_threshold = 0.7f;
    
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