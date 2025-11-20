#pragma once

#include "llama.h"
#include "../ggml/src/ggml-moe-cache.h"
#include <mutex>
#include <string>

// Debug configuration structure
struct llama_moe_debug_config {
    bool enable_prefetch_logging;
    bool enable_eviction_logging;
    bool enable_statistics_logging;
    bool enable_transfer_logging;
    bool enable_layer_assignment_logging;
    bool enable_expert_lifecycle_logging;
    bool enable_performance_logging;
    bool enable_warning_logging;
    
    // Constructor with sensible defaults
    llama_moe_debug_config() :
        enable_prefetch_logging(false),
        enable_eviction_logging(false),
        enable_statistics_logging(false),
        enable_transfer_logging(false),
        enable_layer_assignment_logging(false),
        enable_expert_lifecycle_logging(false),
        enable_performance_logging(false),
        enable_warning_logging(false) {}
};

// Debug logging functions
void llama_moe_log_prefetch(int layer_id, int expert_id, size_t size_mb, const char* source, const char* destination);
void llama_moe_log_eviction(int layer_id, int expert_id, const char* reason);
void llama_moe_log_cache_stats(const ggml_moe_cache* cache);
void llama_moe_log_layer_assignment(int layer_id, const char* device_type, const char* layer_type);
void llama_moe_log_transfer(const char* direction, size_t size_mb, const char* details);
void llama_moe_log_expert_lifecycle(int layer_id, int expert_id, const char* operation, const char* status);
void llama_moe_log_error(int layer_id, int expert_id, const char* operation, int error_code);
void llama_moe_log_warning(int layer_id, int expert_id, const char* message);
void llama_moe_log_performance_metrics(const ggml_moe_cache* cache);

// Phase 5: Post-prompt statistics reporting
void llama_report_moe_cache_stats(struct llama_context* ctx);

// Helper functions
const char* llama_moe_format_expert_key(int layer_id, int expert_id);
const char* llama_moe_format_size(size_t bytes);
const char* llama_moe_get_timestamp();

// Phase 5: Enhanced statistics reporting
void llama_moe_log_cache_stats_phase5(const ggml_moe_cache* cache);

// Global debug configuration
extern llama_moe_debug_config g_moe_debug_config;
extern std::mutex g_moe_debug_mutex;

// Macro for conditional logging
#define LLAMA_MOE_LOG_IF_ENABLED(category, ...) \
    do { \
        if (g_moe_debug_config.enable_##category##_logging) { \
            std::lock_guard<std::mutex> lock(g_moe_debug_mutex); \
            LLAMA_LOG_INFO(__VA_ARGS__); \
        } \
    } while (0)