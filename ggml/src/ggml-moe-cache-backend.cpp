#include "ggml-moe-cache.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml.h"
#include <cstring>

// Backend detection functions (simplified versions)
static bool ggml_backend_is_cuda(ggml_backend_t backend) {
    return backend != nullptr && strstr(ggml_backend_name(backend), "CUDA") != nullptr;
}

static bool ggml_backend_is_hip(ggml_backend_t backend) {
    return backend != nullptr && strstr(ggml_backend_name(backend), "HIP") != nullptr;
}

// Forward declarations for backend-specific interfaces
extern "C" {
    const ggml_moe_cache_interface* ggml_moe_cache_get_interface_cuda();
    const ggml_moe_cache_interface* ggml_moe_cache_get_interface_hip();
}

// Forward declaration for GPU interface function
ggml_moe_cache_interface* ggml_moe_cache_get_interface_gpu();

#include <algorithm>
#include <vector>
#include <queue>
#include <mutex>
#include <chrono>
#include <cstring>
#include <string>

// Unified backend implementation for all GPU backends
// This file provides implementations for Vulkan, SYCL, Metal, and other GPU backends

// Generic GPU cache implementation that works with any backend
struct ggml_moe_cache_gpu : public ggml_moe_cache {
    // Backend-specific stream/context
    void* compute_stream;
    
    // Pinned memory pool for async transfers
    std::queue<void*> pinned_buffer_pool;
    std::mutex buffer_pool_mutex;
    size_t pinned_buffer_size;
    
    // Backend buffer type for GPU allocations
    ggml_backend_buffer_type_t gpu_buft;
    
    // Expert source tensor for size calculations
    const ggml_tensor* expert_source;
    
    // Constructor
    ggml_moe_cache_gpu(
        ggml_backend_t backend,
        const ggml_moe_cache_config& config,
        int num_experts,
        ggml_moe_cache_interface* impl,
        void* stream
    ) : ggml_moe_cache(backend, config, num_experts, impl), compute_stream(stream), expert_source(nullptr) {
        // Get GPU buffer type for this backend
        gpu_buft = ggml_backend_get_default_buffer_type(backend);
        
        // Initialize pinned buffer pool
        pinned_buffer_size = 64 * 1024 * 1024;  // 64MB default
        int num_buffers = 4;
        
        // Allocate pinned host memory using backend's host buffer type
        ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(
            ggml_backend_get_device(backend)
        );
        
        for (int i = 0; i < num_buffers; ++i) {
            void* pinned_buffer = nullptr;
            if (host_buft) {
                ggml_backend_buffer_t host_buffer = ggml_backend_buft_alloc_buffer(
                    host_buft, pinned_buffer_size
                );
                if (host_buffer) {
                    pinned_buffer = ggml_backend_buffer_get_base(host_buffer);
                    // Store buffer handle for later freeing
                    *(void**)pinned_buffer = host_buffer;  // Store handle at start
                }
            }
            if (pinned_buffer) {
                pinned_buffer_pool.push(pinned_buffer);
            }
        }
    }
    
    // Destructor
    ~ggml_moe_cache_gpu() {
        // Free pinned buffers
        while (!pinned_buffer_pool.empty()) {
            void* buffer = pinned_buffer_pool.front();
            pinned_buffer_pool.pop();
            if (buffer) {
                // Retrieve buffer handle and free it
                ggml_backend_buffer_t host_buffer = *(ggml_backend_buffer_t*)buffer;
                if (host_buffer) {
                    ggml_backend_buffer_free(host_buffer);
                }
            }
        }
    }
    
    // Get or create pinned buffer
    void* acquire_pinned_buffer(size_t size) {
        std::lock_guard<std::mutex> lock(buffer_pool_mutex);
        
        if (size > pinned_buffer_size) {
            // Allocate temporary larger buffer using backend host memory
            ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(
                ggml_backend_get_device(backend)
            );
            if (host_buft) {
                ggml_backend_buffer_t host_buffer = ggml_backend_buft_alloc_buffer(
                    host_buft, size
                );
                if (host_buffer) {
                    void* pinned_buffer = ggml_backend_buffer_get_base(host_buffer);
                    *(void**)pinned_buffer = host_buffer;  // Store handle
                    return pinned_buffer;
                }
            }
            return nullptr;
        }
        
        if (!pinned_buffer_pool.empty()) {
            void* buffer = pinned_buffer_pool.front();
            pinned_buffer_pool.pop();
            return buffer;
        }
        
        // Allocate new buffer if pool is empty
        ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(
            ggml_backend_get_device(backend)
        );
        if (host_buft) {
            ggml_backend_buffer_t host_buffer = ggml_backend_buft_alloc_buffer(
                host_buft, pinned_buffer_size
            );
            if (host_buffer) {
                void* pinned_buffer = ggml_backend_buffer_get_base(host_buffer);
                *(void**)pinned_buffer = host_buffer;  // Store handle
                return pinned_buffer;
            }
        }
        return nullptr;
    }
    
    // Return pinned buffer to pool
    void release_pinned_buffer(void* buffer) {
        std::lock_guard<std::mutex> lock(buffer_pool_mutex);
        
        if (buffer) {
            // Check if this is a temporary oversized buffer
            ggml_backend_buffer_t host_buffer = *(ggml_backend_buffer_t*)buffer;
            size_t buffer_size = ggml_backend_buffer_get_size(host_buffer);
            
            if (buffer_size > pinned_buffer_size) {
                // Free temporary buffer
                ggml_backend_buffer_free(host_buffer);
                return;
            }
            
            // Return to pool
            pinned_buffer_pool.push(buffer);
        }
    }
    // Calculate expert size in bytes
    size_t get_expert_size(const ggml_tensor* expert_tensor, int expert_id) {
        (void)expert_id; // Mark as unused
        size_t expert_bytes = ggml_nbytes(expert_tensor) / expert_tensor->ne[2];
        return expert_bytes;
    }
    
    // Calculate priority score combining recency and frequency
    double calculate_priority(int expert_id) {
        auto& stats = expert_stats[expert_id];
        auto time_since_access = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - stats.last_access_time).count();
        
        // Combine frequency and recency (lower score = better candidate for eviction)
        // Add small epsilon to avoid division by zero
        double freq_factor = stats.access_frequency + 0.01;
        double time_factor = time_since_access + 1.0; // +1 to avoid zero
        
        // Priority score: time since access divided by access frequency
        // Experts accessed frequently get lower scores (kept longer)
        // Experts not accessed recently get higher scores (evicted sooner)
        return time_factor / freq_factor;
    }
    
    // Evict experts to make room for new ones (Frequency-Enhanced LRU)
    void evict_experts(size_t required_space) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        
        size_t available_space = config.max_cache_size - stats.current_size;
        if (available_space >= required_space) {
            return;  // Enough space available
        }
        
        size_t space_to_free = required_space - available_space;
        size_t freed_space = 0;
        
        // Build list of eviction candidates with priority scores
        std::vector<std::pair<int, double>> eviction_candidates;
        for (int expert_id : lru_list) {
            // Skip if expert is currently being used (accessed within last 100ms)
            if (expert_stats[expert_id].access_count > 0 &&
                expert_stats[expert_id].last_access_time > std::chrono::steady_clock::now() - std::chrono::milliseconds(100)) {
                continue;
            }
            
            // Calculate priority score for this expert
            double priority = calculate_priority(expert_id);
            eviction_candidates.push_back({expert_id, priority});
        }
        
        // Sort by priority score (higher score = better eviction candidate)
        std::sort(eviction_candidates.begin(), eviction_candidates.end(),
            [](const auto& a, const auto& b) {
                return a.second > b.second; // Higher priority first
            });
        
        // Evict experts based on priority until we have enough space
        for (const auto& [expert_id, priority] : eviction_candidates) {
            if (freed_space >= space_to_free) {
                break; // Enough space freed
            }
            
            // Find and free the expert buffer
            auto it = cache_map.find(expert_id);
            if (it != cache_map.end()) {
                size_t expert_size = get_expert_size(expert_source, expert_id);
                
                // Free the GPU buffer
                ggml_backend_buffer_free(it->second);
                
                // Update statistics
                freed_space += expert_size;
                stats.current_size -= expert_size;
                stats.evictions++;
                
                // Update expert status
                expert_stats[expert_id].is_cached = false;
                
                // Remove from cache map and LRU list
                cache_map.erase(it);
                lru_iter.erase(expert_id);
                
                // Remove from LRU list
                auto lru_it = std::find(lru_list.begin(), lru_list.end(), expert_id);
                if (lru_it != lru_list.end()) {
                    lru_list.erase(lru_it);
                }
            }
        }
    }
    
    // Load expert from CPU to GPU memory
    ggml_backend_buffer_t load_expert_async(
        int expert_id,
        const ggml_tensor* expert_tensor,
        void* compute_stream
    ) {
        (void)compute_stream; // Mark as unused
        // Calculate expert size and offset
        size_t expert_size = get_expert_size(expert_tensor, expert_id);
        size_t expert_offset = expert_id * expert_size;
        
        // Evict experts if necessary
        evict_experts(expert_size);
        
        // Check if we have enough space after eviction
        if (stats.current_size + expert_size > config.max_cache_size) {
            return nullptr;  // Not enough space even after eviction
        }
        
        // Allocate GPU buffer for expert
        ggml_backend_buffer_t gpu_buffer = ggml_backend_buft_alloc_buffer(gpu_buft, expert_size);
        
        if (!gpu_buffer) {
            return nullptr;
        }
        
        // Get pinned buffer for async transfer
        void* pinned_buffer = acquire_pinned_buffer(expert_size);
        
        if (!pinned_buffer) {
            ggml_backend_buffer_free(gpu_buffer);
            return nullptr;
        }
        
        // Copy expert data to pinned buffer first
        const char* cpu_data = (const char*)expert_tensor->data + expert_offset;
        memcpy(pinned_buffer, cpu_data, expert_size);
        
        // Initiate transfer from pinned buffer to GPU
        char* gpu_data = (char*)ggml_backend_buffer_get_base(gpu_buffer);
        
        // For now, use synchronous copy as fallback
        // In a full implementation, this would use backend-specific async copy
        memcpy(gpu_data, pinned_buffer, expert_size);
        
        // Release pinned buffer
        release_pinned_buffer(pinned_buffer);
        
        // Update cache statistics
        stats.current_size += expert_size;
        stats.peak_size = std::max(stats.peak_size, stats.current_size);
        
        return gpu_buffer;
    }
};

// Generic GPU backend cache interface implementation
struct ggml_moe_cache_interface_gpu : public ggml_moe_cache_interface {
    ggml_moe_cache* create_cache(
        ggml_backend_t backend,
        const ggml_moe_cache_config* config,
        int num_experts
    ) override {
        if (!config) return nullptr;
        // Check if this is a GPU backend
        ggml_backend_dev_t device = ggml_backend_get_device(backend);
        if (!device) return nullptr;
        
        enum ggml_backend_dev_type dev_type = ggml_backend_dev_type(device);
        if (dev_type != GGML_BACKEND_DEVICE_TYPE_GPU && 
            dev_type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            return nullptr;
        }
        
        // Create cache with null stream (will use default)
        return new ggml_moe_cache_gpu(backend, *config, num_experts, this, nullptr);
    }
    
    ggml_backend_buffer_t get_expert_async(
        ggml_moe_cache* cache,
        int expert_id,
        const ggml_tensor* expert_tensor,
        void* stream
    ) override {
        if (!cache || !expert_tensor) {
            return nullptr;
        }
        
        ggml_moe_cache_gpu* gpu_cache = static_cast<ggml_moe_cache_gpu*>(cache);
        
        std::lock_guard<std::mutex> lock(cache->cache_mutex);
        
        // Get current time for statistics
        auto now = std::chrono::steady_clock::now();
        
        // Update statistics
        cache->stats.total_requests++;
        cache->expert_stats[expert_id].access_count++;
        cache->expert_stats[expert_id].last_access_time = now;
        
        // Update access frequency (exponential moving average)
        auto& expert_stat = cache->expert_stats[expert_id];
        double time_delta = std::chrono::duration_cast<std::chrono::seconds>(
            now - expert_stat.last_access_time).count();
        
        if (time_delta > 0) {
            // Update frequency: accesses per second (exponential decay)
            double alpha = 0.1; // Smoothing factor
            double current_freq = 1.0 / time_delta;
            expert_stat.access_frequency = (1.0 - alpha) * expert_stat.access_frequency + alpha * current_freq;
        } else {
            expert_stat.access_frequency += 1.0; // Multiple accesses in same second
        }
        
        // Check if expert is already cached
        auto it = cache->cache_map.find(expert_id);
        if (it != cache->cache_map.end()) {
            // Cache hit
            cache->stats.cache_hits++;
            
            // Update LRU: move to front
            auto lru_it = cache->lru_iter[expert_id];
            cache->lru_list.erase(lru_it);
            cache->lru_list.push_front(expert_id);
            cache->lru_iter[expert_id] = cache->lru_list.begin();
            
            return it->second;
        }
        
        // Cache miss - need to load expert
        cache->stats.cache_misses++;
        
        // Load expert asynchronously
        ggml_backend_buffer_t gpu_buffer = gpu_cache->load_expert_async(
            expert_id,
            expert_tensor,
            stream
        );
        
        if (gpu_buffer) {
            // Add to cache
            cache->cache_map[expert_id] = gpu_buffer;
            cache->lru_list.push_front(expert_id);
            cache->lru_iter[expert_id] = cache->lru_list.begin();
            cache->expert_stats[expert_id].is_cached = true;
        }
        
        return gpu_buffer;
    }
    
    void prefetch_experts_async(
        ggml_moe_cache* cache,
        const std::vector<int>& expert_ids,
        const ggml_tensor* expert_tensor,
        void* stream
    ) override {
        if (!cache || !cache->prefetch_engine || !expert_tensor) {
            return;
        }
        
        ggml_moe_cache_gpu* gpu_cache = static_cast<ggml_moe_cache_gpu*>(cache);
        
        // Start async prefetch for each expert
        for (int expert_id : expert_ids) {
            // Skip if already cached
            if (cache->cache_map.find(expert_id) != cache->cache_map.end()) {
                continue;
            }
            
            // Load expert asynchronously (this will evict if needed)
            ggml_backend_buffer_t gpu_buffer = gpu_cache->load_expert_async(
                expert_id,
                expert_tensor,
                stream
            );
            
            if (gpu_buffer) {
                std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(cache->cache_mutex));
                cache->cache_map[expert_id] = gpu_buffer;
                cache->lru_list.push_front(expert_id);
                cache->lru_iter[expert_id] = cache->lru_list.begin();
                cache->expert_stats[expert_id].is_cached = true;
                cache->stats.prefetches++;
            }
        }
    }
    
    void touch_expert(
        ggml_moe_cache* cache,
        int expert_id
    ) override {
        if (!cache) return;
        
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(cache->cache_mutex));
        
        // Update access statistics
        auto now = std::chrono::steady_clock::now();
        auto& stats = cache->expert_stats[expert_id];
        stats.access_count++;
        stats.last_access_time = now;
        
        // Update access frequency (exponential moving average)
        double time_delta = std::chrono::duration_cast<std::chrono::seconds>(
            now - stats.last_access_time).count();
        
        if (time_delta > 0) {
            double alpha = 0.1; // Smoothing factor
            double current_freq = 1.0 / time_delta;
            stats.access_frequency = (1.0 - alpha) * stats.access_frequency + alpha * current_freq;
        } else {
            stats.access_frequency += 1.0;
        }
        
        // Update LRU if expert is cached
        auto it = cache->lru_iter.find(expert_id);
        if (it != cache->lru_iter.end()) {
            cache->lru_list.erase(it->second);
            cache->lru_list.push_front(expert_id);
            it->second = cache->lru_list.begin();
        }
        
        // Update global statistics
        if (cache->stats.total_requests > 0) {
            cache->stats.hit_rate = (double)cache->stats.cache_hits / cache->stats.total_requests;
        }
    }
    
    ggml_moe_cache_stats get_stats(
        const ggml_moe_cache* cache
    ) override {
        if (!cache) return ggml_moe_cache_stats{};
        
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(cache->cache_mutex));
        ggml_moe_cache_stats stats = cache->stats;
        
        // Calculate derived statistics
        if (stats.total_requests > 0) {
            stats.hit_rate = (double)stats.cache_hits / stats.total_requests;
        }
        
        if (stats.prefetches > 0) {
            stats.prefetch_accuracy = (double)stats.prefetch_hits / stats.prefetches;
        }
        
        stats.current_size = cache->stats.current_size;
        
        return stats;
    }
    
    void reset_stats(
        ggml_moe_cache* cache
    ) override {
        if (!cache) return;
        
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(cache->cache_mutex));
        memset(&cache->stats, 0, sizeof(cache->stats));
    }
    
    void clear_cache(
        ggml_moe_cache* cache
    ) override {
        if (!cache) return;
        
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(cache->cache_mutex));
        
        // Free all cached buffers
        for (auto& [expert_id, buffer] : cache->cache_map) {
            if (buffer) {
                ggml_backend_buffer_free(buffer);
            }
            cache->expert_stats[expert_id].is_cached = false;
        }
        
        cache->cache_map.clear();
        cache->lru_list.clear();
        cache->lru_iter.clear();
        cache->stats.current_size = 0;
    }
    
    void destroy_cache(
        ggml_moe_cache* cache
    ) override {
        if (!cache) return;
        // Forward declaration for the GPU interface function
        static ggml_moe_cache_interface* ggml_moe_cache_get_interface_gpu_impl();
        
        // Get generic GPU cache interface
        ggml_moe_cache_interface* ggml_moe_cache_get_interface_gpu() {
            return ggml_moe_cache_get_interface_gpu_impl();
        }
        
        // Implementation
        static ggml_moe_cache_interface* ggml_moe_cache_get_interface_gpu_impl() {
        #ifdef GGML_GPU_MOE_CACHE
            static ggml_moe_cache_interface_gpu interface;
            return &interface;
        #else
            return nullptr;
        #endif
        }
        
        // Update backend detection to include generic GPU backend
        ggml_moe_cache_interface* ggml_moe_cache_get_interface(ggml_backend_t backend) {
            if (ggml_backend_is_cuda(backend)) {
                return const_cast<ggml_moe_cache_interface*>(ggml_moe_cache_get_interface_cuda());
            }
            if (ggml_backend_is_hip(backend)) {
                return const_cast<ggml_moe_cache_interface*>(ggml_moe_cache_get_interface_hip());
            }
            
        #ifdef GGML_GPU_MOE_CACHE
            // Check if this is a GPU backend (Vulkan, SYCL, Metal, etc.)
            ggml_backend_dev_t device = ggml_backend_get_device(backend);
            if (device) {
                enum ggml_backend_dev_type dev_type = ggml_backend_dev_type(device);
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU ||
                    dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                    return ggml_moe_cache_get_interface_gpu();
                }
            }
        #endif
            
            return nullptr;
        }
    }
    
    return nullptr;
}