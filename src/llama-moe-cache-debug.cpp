#include "llama-moe-cache-debug.h"
#include <cstdio>
#include <cstring>
#include <ctime>
#include <sstream>
#include <iomanip>

// Global debug configuration instance
llama_moe_debug_config g_moe_debug_config;
std::mutex g_moe_debug_mutex;

// Constructor for debug configuration
llama_moe_debug_config::llama_moe_debug_config() 
    : enable_prefetch_logging(false),
      enable_eviction_logging(false),
      enable_statistics_logging(false),
      enable_transfer_logging(false),
      enable_layer_assignment_logging(false),
      enable_expert_lifecycle_logging(false),
      enable_performance_logging(false) {
}

// Helper function implementations
const char* llama_moe_format_expert_key(int layer_id, int expert_id) {
    static thread_local char buffer[64];
    snprintf(buffer, sizeof(buffer), "L%d.E%d", layer_id, expert_id);
    return buffer;
}

const char* llama_moe_format_size(size_t bytes) {
    static thread_local char buffer[64];
    const char* units[] = {"B", "KB", "MB", "GB"};
    int unit_index = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024.0 && unit_index < 3) {
        size /= 1024.0;
        unit_index++;
    }
    
    snprintf(buffer, sizeof(buffer), "%.2f %s", size, units[unit_index]);
    return buffer;
}

const char* llama_moe_get_timestamp() {
    static thread_local char buffer[64];
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::tm tm;
#ifdef _WIN32
    localtime_s(&tm, &time_t); // Windows version
#else
    localtime_r(&time_t, &tm); // POSIX version
#endif
    
    snprintf(buffer, sizeof(buffer), "%04d-%02d-%02d %02d:%02d:%02d.%03d",
             tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
             tm.tm_hour, tm.tm_min, tm.tm_sec, static_cast<int>(ms.count()));
    return buffer;
}

// Debug logging function implementations
void llama_moe_log_prefetch(int layer_id, int expert_id, size_t size_mb, 
                           const char* source, const char* destination) {
    LLAMA_MOE_LOG_IF_ENABLED(prefetch, "[MoE Cache] Prefetching expert %s (%s) from %s to %s\n",
                             llama_moe_format_expert_key(layer_id, expert_id),
                             llama_moe_format_size(size_mb * 1024 * 1024), // Convert MB to bytes
                             source, destination);
}

void llama_moe_log_eviction(int layer_id, int expert_id, const char* reason) {
    LLAMA_MOE_LOG_IF_ENABLED(eviction, "[MoE Cache] Evicting expert %s (Reason: %s)\n",
                             llama_moe_format_expert_key(layer_id, expert_id), reason);
}

void llama_moe_log_cache_stats(const ggml_moe_cache* cache) {
    if (!cache || !g_moe_debug_config.enable_statistics_logging) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(g_moe_debug_mutex);
    
    // Get current statistics
    ggml_moe_cache_stats stats = ggml_moe_cache_get_stats(cache);
    
    // Calculate derived metrics
    double hit_rate = (stats.total_requests > 0) 
        ? (static_cast<double>(stats.cache_hits) / stats.total_requests * 100.0)
        : 0.0;
        
    double prefetch_accuracy = (stats.prefetches > 0)
        ? (static_cast<double>(stats.prefetch_hits) / stats.prefetches * 100.0)
        : 0.0;
    
    // Log comprehensive statistics
    LLAMA_LOG_INFO("=== MoE Cache Statistics ===\n");
    LLAMA_LOG_INFO("Timestamp: %s\n", llama_moe_get_timestamp());
    LLAMA_LOG_INFO("Total Requests: %llu\n", static_cast<unsigned long long>(stats.total_requests));
    LLAMA_LOG_INFO("Cache Hits: %llu (%.1f%%)\n", static_cast<unsigned long long>(stats.cache_hits), hit_rate);
    LLAMA_LOG_INFO("Cache Misses: %llu (%.1f%%)\n", static_cast<unsigned long long>(stats.cache_misses), 100.0 - hit_rate);
    LLAMA_LOG_INFO("Evictions: %llu\n", static_cast<unsigned long long>(stats.evictions));
    LLAMA_LOG_INFO("Prefetches: %llu\n", static_cast<unsigned long long>(stats.prefetches));
    LLAMA_LOG_INFO("Prefetch Hits: %llu (%.1f%%)\n", static_cast<unsigned long long>(stats.prefetch_hits), prefetch_accuracy);
    LLAMA_LOG_INFO("Current Cache Size: %s\n", llama_moe_format_size(stats.current_size));
    LLAMA_LOG_INFO("Peak Cache Size: %s\n", llama_moe_format_size(stats.peak_size));
    LLAMA_LOG_INFO("Average Load Time: %.2f ms\n", stats.avg_load_time);
    
    // Log transfer statistics if available
    if (stats.total_transfers_ram_to_vram > 0) {
        LLAMA_LOG_INFO("RAM → VRAM Transfers: %llu (%s total)\n",
                       static_cast<unsigned long long>(stats.total_transfers_ram_to_vram),
                       llama_moe_format_size(stats.total_transfers_ram_to_vram));
    }
    
    if (stats.total_transfers_vram_to_ram > 0) {
        LLAMA_LOG_INFO("VRAM → RAM Transfers: %llu (%s total)\n",
                       static_cast<unsigned long long>(stats.total_transfers_vram_to_ram),
                       llama_moe_format_size(stats.total_transfers_vram_to_ram));
    }
    
    if (stats.total_transfers_gpu_to_gpu > 0) {
        LLAMA_LOG_INFO("GPU → GPU Transfers: %llu (%s total)\n",
                       static_cast<unsigned long long>(stats.total_transfers_gpu_to_gpu),
                       llama_moe_format_size(stats.total_transfers_gpu_to_gpu));
    }
    
    LLAMA_LOG_INFO("============================\n");
}

// Phase 5: Enhanced statistics reporting with Phase 5 metrics
void llama_moe_log_cache_stats_phase5(const ggml_moe_cache* cache) {
    if (!cache || !g_moe_debug_config.enable_statistics_logging) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(g_moe_debug_mutex);
    
    // Get current statistics
    ggml_moe_cache_stats stats = ggml_moe_cache_get_stats(cache);
    
    // Calculate derived metrics
    double hit_rate = (stats.total_requests > 0)
        ? (static_cast<double>(stats.cache_hits) / stats.total_requests * 100.0)
        : 0.0;
        
    double prefetch_accuracy = (stats.prefetches > 0)
        ? (static_cast<double>(stats.prefetch_hits) / stats.prefetches * 100.0)
        : 0.0;
    
    // Log comprehensive statistics
    LLAMA_LOG_INFO("=== MoE Cache Statistics ===\n");
    LLAMA_LOG_INFO("Timestamp: %s\n", llama_moe_get_timestamp());
    LLAMA_LOG_INFO("Total Requests: %llu\n", static_cast<unsigned long long>(stats.total_requests));
    LLAMA_LOG_INFO("Cache Hits: %llu (%.1f%%)\n", static_cast<unsigned long long>(stats.cache_hits), hit_rate);
    LLAMA_LOG_INFO("Cache Misses: %llu (%.1f%%)\n", static_cast<unsigned long long>(stats.cache_misses), 100.0 - hit_rate);
    LLAMA_LOG_INFO("Evictions: %llu\n", static_cast<unsigned long long>(stats.evictions));
    LLAMA_LOG_INFO("Prefetches: %llu\n", static_cast<unsigned long long>(stats.prefetches));
    LLAMA_LOG_INFO("Prefetch Hits: %llu (%.1f%%)\n", static_cast<unsigned long long>(stats.prefetch_hits), prefetch_accuracy);
    LLAMA_LOG_INFO("Current Cache Size: %s\n", llama_moe_format_size(stats.current_size));
    LLAMA_LOG_INFO("Peak Cache Size: %s\n", llama_moe_format_size(stats.peak_size));
    LLAMA_LOG_INFO("Average Load Time: %.2f ms\n", stats.avg_load_time);
    
    // Log transfer statistics if available
    if (stats.total_transfers_ram_to_vram > 0) {
        LLAMA_LOG_INFO("RAM → VRAM Transfers: %llu (%s total)\n",
                       static_cast<unsigned long long>(stats.total_transfers_ram_to_vram),
                       llama_moe_format_size(stats.total_transfers_ram_to_vram));
    }
    
    if (stats.total_transfers_vram_to_ram > 0) {
        LLAMA_LOG_INFO("VRAM → RAM Transfers: %llu (%s total)\n",
                       static_cast<unsigned long long>(stats.total_transfers_vram_to_ram),
                       llama_moe_format_size(stats.total_transfers_vram_to_ram));
    }
    
    if (stats.total_transfers_gpu_to_gpu > 0) {
        LLAMA_LOG_INFO("GPU → GPU Transfers: %llu (%s total)\n",
                       static_cast<unsigned long long>(stats.total_transfers_gpu_to_gpu),
                       llama_moe_format_size(stats.total_transfers_gpu_to_gpu));
    }
    
    // Phase 5 enhanced metrics
    if (stats.async_operations_completed > 0) {
        LLAMA_LOG_INFO("Async Operations Completed: %llu\n", static_cast<unsigned long long>(stats.async_operations_completed));
    }
    
    if (stats.gpu_to_gpu_transfers_attempted > 0) {
        double gpu_transfer_success_rate = (stats.gpu_to_gpu_transfers_attempted > 0) ?
            (static_cast<double>(stats.gpu_to_gpu_transfers_successful) /
             stats.gpu_to_gpu_transfers_attempted * 100.0) : 0.0;
        LLAMA_LOG_INFO("GPU-to-GPU Transfers: %llu attempted, %llu succeeded (%.1f%%)\n",
                       static_cast<unsigned long long>(stats.gpu_to_gpu_transfers_attempted),
                       static_cast<unsigned long long>(stats.gpu_to_gpu_transfers_successful),
                       gpu_transfer_success_rate);
    }
    
    if (stats.pinned_memory_used > 0) {
        LLAMA_LOG_INFO("Pinned Memory Used: %s\n", llama_moe_format_size(stats.pinned_memory_used));
    }
    
    if (stats.cpu_involvement_reduction_percent > 0) {
        LLAMA_LOG_INFO("CPU Involvement Reduction: %.1f%%\n",
                       stats.cpu_involvement_reduction_percent);
    }
    
    // Performance metrics
    if (stats.tokens_per_second > 0) {
        LLAMA_LOG_INFO("Tokens/Second: %.2f\n", stats.tokens_per_second);
    }
    
    if (stats.inference_latency_ms > 0) {
        LLAMA_LOG_INFO("Inference Latency: %.2f ms\n", stats.inference_latency_ms);
    }
    
    if (stats.total_tokens_generated > 0) {
        LLAMA_LOG_INFO("Total Tokens Generated: %llu\n", static_cast<unsigned long long>(stats.total_tokens_generated));
    }
    
    LLAMA_LOG_INFO("============================\n");
}

void llama_moe_log_layer_assignment(int layer_id, const char* device_type, const char* layer_type) {
    LLAMA_MOE_LOG_IF_ENABLED(layer_assignment, "[MoE Cache] Layer %d (%s) assigned to %s\n",
                             layer_id, layer_type, device_type);
}

void llama_moe_log_transfer(const char* direction, size_t size_mb, const char* details) {
    LLAMA_MOE_LOG_IF_ENABLED(transfer, "[MoE Cache] Transfer %s: %s - %s\n",
                             direction, llama_moe_format_size(size_mb * 1024 * 1024), details);
}

void llama_moe_log_expert_lifecycle(int layer_id, int expert_id, const char* operation, const char* status) {
    LLAMA_MOE_LOG_IF_ENABLED(expert_lifecycle, "[MoE Cache] Expert %s %s: %s\n",
                             llama_moe_format_expert_key(layer_id, expert_id), operation, status);
}

void llama_moe_log_error(int layer_id, int expert_id, const char* operation, int error_code) {
    std::lock_guard<std::mutex> lock(g_moe_debug_mutex);
    LLAMA_LOG_ERROR("[MoE Cache] ERROR in %s for expert %s: error_code=%d\n",
                    operation, llama_moe_format_expert_key(layer_id, expert_id), error_code);
}

void llama_moe_log_warning(int layer_id, int expert_id, const char* message) {
    LLAMA_MOE_LOG_IF_ENABLED(warning, "[MoE Cache] WARNING for expert %s: %s\n",
                             llama_moe_format_expert_key(layer_id, expert_id), message);
}

void llama_moe_log_performance_metrics(const ggml_moe_cache* cache) {
    if (!cache || !g_moe_debug_config.enable_performance_logging) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(g_moe_debug_mutex);
    
    ggml_moe_cache_stats stats = ggml_moe_cache_get_stats(cache);
    
    // Calculate performance metrics
    double throughput_mbps = (stats.current_size > 0 && stats.avg_load_time > 0)
        ? (static_cast<double>(stats.current_size) / (1024.0 * 1024.0)) / (stats.avg_load_time / 1000.0)
        : 0.0;
    
    double efficiency = (stats.total_requests > 0)
        ? (static_cast<double>(stats.cache_hits + stats.prefetch_hits) / stats.total_requests)
        : 0.0;
    
    LLAMA_LOG_INFO("=== MoE Cache Performance Metrics ===\n");
    LLAMA_LOG_INFO("Cache Efficiency: %.1f%%\n", efficiency * 100.0);
    LLAMA_LOG_INFO("Data Throughput: %.2f MB/s\n", throughput_mbps);
    LLAMA_LOG_INFO("Average Load Time: %.2f ms\n", stats.avg_load_time);
    LLAMA_LOG_INFO("Current VRAM Usage: %s\n", llama_moe_format_size(stats.current_size));
    
    if (stats.peak_size > 0) {
        double utilization = static_cast<double>(stats.current_size) / stats.peak_size * 100.0;
        LLAMA_LOG_INFO("VRAM Utilization: %.1f%% of peak\n", utilization);
    }
    
    LLAMA_LOG_INFO("=====================================\n");
}
// Phase 5: Post-prompt statistics reporting function
void llama_report_moe_cache_stats(struct llama_context* ctx) {
    if (!ctx || !ctx->model.moe_cache) {
        return;
    }
    
    // Get current statistics
    ggml_moe_cache_stats stats = ggml_moe_cache_get_stats(ctx->model.moe_cache);
    
    // Update with context-specific performance metrics
    stats.tokens_per_second = ctx->tokens_per_second;
    stats.total_tokens_generated = ctx->generated_tokens;
    
    // Report comprehensive statistics using Phase 5 enhanced reporting
    llama_moe_log_cache_stats_phase5(ctx->model.moe_cache);
    
    // Additional prompt-specific reporting
    LLAMA_LOG_INFO("=== Prompt Completion Summary ===\n");
    LLAMA_LOG_INFO("Tokens generated: %d\n", ctx->generated_tokens);
    LLAMA_LOG_INFO("Average tokens/sec: %.2f\n", ctx->tokens_per_second);
    LLAMA_LOG_INFO("==================================\n");
}