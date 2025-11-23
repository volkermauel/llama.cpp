#include "ggml-moe-cache-debug.h"
#include "ggml-moe-cache.h"
#include <mutex>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <cinttypes>

// Debug configuration
static struct {
    bool enable_debug_logging = false;
    bool enable_statistics_logging = false;
    bool enable_performance_logging = false;
    bool enable_eviction_logging = false;
    bool enable_transfer_logging = false;
    bool enable_layer_assignment_logging = false;
    bool enable_expert_lifecycle_logging = false;
    bool enable_warning_logging = true;
    bool enable_error_logging = true;
} g_moe_debug_config;

static std::mutex g_moe_debug_mutex;

// Debug logging macro
#define GGML_MOE_LOG_IF_ENABLED(category, ...) \
    do { \
        if (g_moe_debug_config.enable_##category##_logging) { \
            std::lock_guard<std::mutex> lock(g_moe_debug_mutex); \
            printf(__VA_ARGS__); \
            fflush(stdout); \
        } \
    } while(0)

// Format function implementation - moved from src/llama-impl.cpp
std::string ggml_format(const char * fmt, ...) {
    va_list ap;
    va_list ap_copy;
    va_start(ap, fmt);
    va_copy(ap_copy, ap);
    
    int size = vsnprintf(NULL, 0, fmt, ap);
    std::string result;
    if (size >= 0) {
        result.resize(size);
        vsnprintf(&result[0], size + 1, fmt, ap_copy);
    }
    
    va_end(ap_copy);
    va_end(ap);
    return result;
}

// Helper function for formatting expert keys
const char* ggml_moe_format_expert_key(int layer_id, int expert_id) {
    static thread_local char buffer[64];
    if (expert_id >= 0) {
        snprintf(buffer, sizeof(buffer), "L%d_E%d", layer_id, expert_id);
    } else {
        snprintf(buffer, sizeof(buffer), "L%d_Special", layer_id);
    }
    return buffer;
}

// Debug logging function implementations
void ggml_moe_log_prefetch(int layer_id, int expert_id, size_t size_mb,
                           const char* source, const char* destination) {
    GGML_MOE_LOG_IF_ENABLED(debug, "[MoE Cache] Prefetch expert %s: %zu MB from %s to %s\n",
                            ggml_moe_format_expert_key(layer_id, expert_id),
                            size_mb, source, destination);
}

void ggml_moe_log_eviction(int layer_id, int expert_id, const char* reason) {
    GGML_MOE_LOG_IF_ENABLED(eviction, "[MoE Cache] Evicting expert %s (Reason: %s)\n",
                            ggml_moe_format_expert_key(layer_id, expert_id), reason);
}

void ggml_moe_log_cache_stats(const ggml_moe_cache* cache) {
    if (!cache || !g_moe_debug_config.enable_statistics_logging) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(g_moe_debug_mutex);
    printf("[MoE Cache] Statistics:\n");
    printf("  Current size: %zu MB\n", cache->stats.current_size / (1024 * 1024));
    printf("  Peak size: %zu MB\n", cache->stats.peak_size / (1024 * 1024));
    printf("  Hit rate: %.2f%%\n", cache->stats.hit_rate * 100);
    printf("  Eviction count: %" PRIu64 "\n", cache->stats.evictions);
}

void ggml_moe_log_layer_assignment(int layer_id, const char* device_type, const char* layer_type) {
    GGML_MOE_LOG_IF_ENABLED(layer_assignment, "[MoE Cache] Layer %d (%s) assigned to %s\n",
                            layer_id, layer_type, device_type);
}

void ggml_moe_log_transfer(const char* direction, size_t size_mb, const char* details) {
    GGML_MOE_LOG_IF_ENABLED(transfer, "[MoE Cache] Transfer %s: %zu MB - %s\n",
                            direction, size_mb, details);
}

void ggml_moe_log_expert_lifecycle(int layer_id, int expert_id, const char* operation, const char* status) {
    GGML_MOE_LOG_IF_ENABLED(expert_lifecycle, "[MoE Cache] Expert %s %s: %s\n",
                            ggml_moe_format_expert_key(layer_id, expert_id),
                            operation, status);
}

void ggml_moe_log_error(int layer_id, int expert_id, const char* operation, int error_code) {
    std::lock_guard<std::mutex> lock(g_moe_debug_mutex);
    printf("[MoE Cache] ERROR for expert %s during %s: code %d\n",
           ggml_moe_format_expert_key(layer_id, expert_id),
           operation, error_code);
    fflush(stdout);
}

void ggml_moe_log_warning(int layer_id, int expert_id, const char* message) {
    GGML_MOE_LOG_IF_ENABLED(warning, "[MoE Cache] WARNING for expert %s: %s\n",
                            ggml_moe_format_expert_key(layer_id, expert_id), message);
}
void ggml_moe_log_performance_metrics(const ggml_moe_cache* cache) {
    if (!cache || !g_moe_debug_config.enable_performance_logging) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(g_moe_debug_mutex);
    printf("[MoE Cache] Performance Metrics:\n");
    printf("  Average load time: %.2f ms\n", cache->stats.avg_load_time);
    printf("  Average transfer time: %.2f ms\n", cache->stats.avg_transfer_time_ms);
    printf("  Prefetch accuracy: %.2f%%\n", cache->stats.prefetch_accuracy * 100);
}

// Phase 5: Enhanced statistics reporting with Phase 5 metrics
void ggml_moe_log_cache_stats_phase5(const ggml_moe_cache* cache) {
    if (!cache || !g_moe_debug_config.enable_statistics_logging) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(g_moe_debug_mutex);
    printf("=== MoE Cache Phase 5 Statistics ===\n");
    printf("Cache Capacity: %zu MB\n", cache->config.max_cache_size / (1024 * 1024));
    printf("Current Usage: %zu MB (%.1f%%)\n", 
           cache->stats.current_size / (1024 * 1024),
           (double)cache->stats.current_size / cache->config.max_cache_size * 100);
    printf("Total Hits: %" PRIu64 "\n", cache->stats.cache_hits);
    printf("Total Misses: %" PRIu64 "\n", cache->stats.cache_misses);
    printf("Hit Rate: %.2f%%\n", cache->stats.hit_rate * 100);
    printf("Eviction Count: %" PRIu64 "\n", cache->stats.evictions);
    printf("Prefetch Hits: %" PRIu64 "\n", cache->stats.prefetch_hits);
    printf("Prefetch Accuracy: %.2f%%\n", cache->stats.prefetch_accuracy * 100);
    printf("Average Load Time: %.2f ms\n", cache->stats.avg_load_time);
    printf("Average Transfer Time: %.2f ms\n", cache->stats.avg_transfer_time_ms);
    printf("Async Operations Completed: %" PRIu64 "\n", cache->stats.async_operations_completed);
    printf("====================================\n");
}