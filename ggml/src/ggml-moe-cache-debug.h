#pragma once

#include "ggml.h"
#include <string>
#include <cstdarg>

#ifdef __GNUC__
#    if defined(__MINGW32__) && !defined(__clang__)
#        define GGML_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#    else
#        define GGML_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#    endif
#else
#    define GGML_ATTRIBUTE_FORMAT(...)
#endif

// Forward declaration
struct ggml_moe_cache;

// Debug logging functions - moved from src to ggml to avoid circular dependencies
void ggml_moe_log_prefetch(int layer_id, int expert_id, size_t size_mb, const char* source, const char* destination);
void ggml_moe_log_eviction(int layer_id, int expert_id, const char* reason);
void ggml_moe_log_cache_stats(const ggml_moe_cache* cache);
void ggml_moe_log_layer_assignment(int layer_id, const char* device_type, const char* layer_type);
void ggml_moe_log_transfer(const char* direction, size_t size_mb, const char* details);
void ggml_moe_log_expert_lifecycle(int layer_id, int expert_id, const char* operation, const char* status);
void ggml_moe_log_error(int layer_id, int expert_id, const char* operation, int error_code);
void ggml_moe_log_warning(int layer_id, int expert_id, const char* message);
void ggml_moe_log_performance_metrics(const ggml_moe_cache* cache);

// Phase 5: Enhanced statistics reporting
void ggml_moe_log_cache_stats_phase5(const ggml_moe_cache* cache);

// Helper function for formatting expert keys
std::string ggml_moe_format_expert_key(int layer_id, int expert_id);

// Format function - moved from src to ggml to avoid circular dependencies
GGML_ATTRIBUTE_FORMAT(1, 2)
std::string ggml_format(const char * fmt, ...);

// Compatibility macros for existing code
#define llama_moe_log_prefetch ggml_moe_log_prefetch
#define llama_moe_log_eviction ggml_moe_log_eviction
#define llama_moe_log_cache_stats ggml_moe_log_cache_stats
#define llama_moe_log_layer_assignment ggml_moe_log_layer_assignment
#define llama_moe_log_transfer ggml_moe_log_transfer
#define llama_moe_log_expert_lifecycle ggml_moe_log_expert_lifecycle
#define llama_moe_log_error ggml_moe_log_error
#define llama_moe_log_warning ggml_moe_log_warning
#define llama_moe_log_performance_metrics ggml_moe_log_performance_metrics
#define llama_moe_log_cache_stats_phase5 ggml_moe_log_cache_stats_phase5
#define llama_moe_format_expert_key ggml_moe_format_expert_key
#define format ggml_format