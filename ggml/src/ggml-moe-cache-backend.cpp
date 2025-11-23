#include "ggml-moe-cache.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml.h"
#include "ggml-moe-cache-debug.h"
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
#ifdef GGML_CUDA_MOE_CACHE
    const ggml_moe_cache_interface* ggml_moe_cache_get_interface_cuda();
#endif
#ifdef GGML_HIP_MOE_CACHE
    const ggml_moe_cache_interface* ggml_moe_cache_get_interface_hip();
#endif
}

// Forward declaration for GPU interface function
static ggml_moe_cache_interface* ggml_moe_cache_get_interface_gpu();

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
    
    // Constructor
    // Constructor
    ggml_moe_cache_gpu(
        ggml_backend_t backend,
        const ggml_moe_cache_config& config,
        int num_layers,
        int num_experts_per_layer,
        ggml_moe_cache_interface* impl,
        void* stream
    ) : ggml_moe_cache(backend, config, num_layers, num_experts_per_layer, impl), compute_stream(stream) {
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
    size_t get_expert_size(const ggml_tensor* expert_tensor, int layer_id, int expert_id) {
        (void)layer_id; // Mark as unused for now
        (void)expert_id; // Mark as unused for now
        size_t expert_bytes = ggml_nbytes(expert_tensor) / expert_tensor->ne[2];
        return expert_bytes;
    }
    
    // Layer-aware prefetch engine implementation
    std::vector<std::pair<int, int>> predict_next_experts(
        int layer_id,
        const std::vector<int>& current_experts,
        int top_k
    ) {
        std::vector<std::pair<int, int>> predictions;
        
        // Simple implementation: predict same experts in next layer
        // In a real implementation, this would use historical patterns
        for (int expert_id : current_experts) {
            if ((int)predictions.size() >= top_k) break;
            // Predict same expert in next layer (if exists)
            if (layer_id + 1 < num_layers) {
                predictions.push_back({layer_id + 1, expert_id});
            }
        }
        
        return predictions;
    }
    
    void update_patterns(
        int layer_id,
        const std::vector<int>& used_experts,
        const std::vector<int>& tokens
    ) {
        (void)tokens; // Mark as intentionally unused
        if (!prefetch_engine) return;
        
        // Update recent experts for this layer
        update_recent_experts(layer_id, used_experts);
        
        // Update co-occurrence matrix for this layer
        auto& cooccurrence = prefetch_engine->expert_cooccurrence_per_layer[layer_id];
        for (size_t i = 0; i < used_experts.size(); ++i) {
            for (size_t j = i + 1; j < used_experts.size(); ++j) {
                int expert_i = used_experts[i];
                int expert_j = used_experts[j];
                if (expert_i < (int)cooccurrence.size() && expert_j < (int)cooccurrence.size()) {
                    cooccurrence[expert_i][expert_j]++;
                    cooccurrence[expert_j][expert_i]++;
                }
            }
        }
    }
    
    std::vector<std::pair<int, int>> predict_locality(
        int layer_id,
        const std::vector<int>& recent_experts
    ) {
        std::vector<std::pair<int, int>> predictions;
        
        // Simple locality-based prediction: predict recently used experts
        for (int expert_id : recent_experts) {
            predictions.push_back({layer_id, expert_id});
        }
        
        return predictions;
    }
    
    void update_recent_experts(int layer_id, const std::vector<int>& experts) {
        if (!prefetch_engine) return;
        
        auto& recent_experts = prefetch_engine->recent_experts_per_layer[layer_id];
        
        for (int expert_id : experts) {
            // Add to recent list, remove duplicates
            auto it = std::find(recent_experts.begin(), recent_experts.end(), expert_id);
            if (it != recent_experts.end()) {
                recent_experts.erase(it);
            }
            recent_experts.push_back(expert_id);
            
            // Keep only recent entries
            if (recent_experts.size() > prefetch_engine->RECENT_SIZE) {
                recent_experts.erase(recent_experts.begin());
            }
        }
    }
    
    void init_cooccurrence(int layer_id, int num_experts) {
        if (!prefetch_engine) return;
        
        prefetch_engine->expert_cooccurrence_per_layer[layer_id] =
            std::vector<std::vector<int>>(num_experts, std::vector<int>(num_experts, 0));
    }
    
    // Calculate priority score combining recency and frequency
    double calculate_priority(const ggml_moe_expert_key& key) {
        auto& expert_stat = expert_stats[key];
        auto time_since_access = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - expert_stat.last_access_time).count();
        
        // Combine frequency and recency (lower score = better candidate for eviction)
        // Add small epsilon to avoid division by zero
        double freq_factor = expert_stat.access_frequency + 0.01;
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
        std::vector<std::pair<ggml_moe_expert_key, double>> eviction_candidates;
        for (const auto& key : lru_list) {
            // Skip if expert is currently being used (accessed within last 100ms)
            if (expert_stats[key].access_count > 0 &&
                expert_stats[key].last_access_time > std::chrono::steady_clock::now() - std::chrono::milliseconds(100)) {
                continue;
            }
            
            // Calculate priority score for this expert
            double priority = calculate_priority(key);
            eviction_candidates.push_back({key, priority});
        }
        
        // Sort by priority score (higher score = better eviction candidate)
        std::sort(eviction_candidates.begin(), eviction_candidates.end(),
            [](const std::pair<ggml_moe_expert_key, double>& a, const std::pair<ggml_moe_expert_key, double>& b) {
                return a.second > b.second; // Higher priority first
            });
        
        // Evict experts based on priority until we have enough space
        for (const std::pair<ggml_moe_expert_key, double>& candidate : eviction_candidates) {
            const ggml_moe_expert_key& key = candidate.first;
            if (freed_space >= space_to_free) {
                break; // Enough space freed
            }
            
            // Find and free the expert buffer
            auto it = cache_map.find(key);
            if (it != cache_map.end()) {
                size_t expert_size = get_expert_size(expert_source, key.layer_id, key.expert_id);
                
                // Log eviction event
                ggml_moe_log_eviction(key.layer_id, key.expert_id, "Cache full - LRU eviction");
                ggml_moe_log_expert_lifecycle(key.layer_id, key.expert_id, "Evicting", "Making room for new expert");
                
                // Free the GPU buffer
                ggml_backend_buffer_free(it->second);
                
                // Update statistics
                freed_space += expert_size;
                stats.current_size -= expert_size;
                stats.evictions++;
                stats.total_transfers_vram_to_ram++;
                
                // Update expert status
                expert_stats[key].is_cached = false;
                
                // Remove from cache map and LRU list
                cache_map.erase(it);
                lru_iter.erase(key);
                
                // Remove from LRU list
                auto lru_it = std::find(lru_list.begin(), lru_list.end(), key);
                if (lru_it != lru_list.end()) {
                    lru_list.erase(lru_it);
                }
                
                // Log transfer operation
                ggml_moe_log_transfer("VRAM->RAM", expert_size,
                                      ggml_moe_format_expert_key(key.layer_id, key.expert_id));
            }
        }
    }
    
    // Load expert from CPU to GPU memory
    ggml_backend_buffer_t load_expert_async(
        int layer_id,
        int expert_id,
        const ggml_tensor* expert_tensor,
        void* compute_stream
    ) {
        (void)compute_stream; // Mark as unused
        // Calculate expert size and offset
        size_t expert_size = get_expert_size(expert_tensor, layer_id, expert_id);
        size_t expert_offset = expert_id * expert_size;
        
        // Log expert lifecycle event
        ggml_moe_log_expert_lifecycle(layer_id, expert_id, "Loading", "Starting async load");
        
        // Evict experts if necessary
        evict_experts(expert_size);
        
        // Check if we have enough space after eviction
        if (stats.current_size + expert_size > config.max_cache_size) {
            ggml_moe_log_error(layer_id, expert_id, "Not enough space even after eviction", -1);
            return nullptr;  // Not enough space even after eviction
        }
        
        // Allocate GPU buffer for expert
        ggml_backend_buffer_t gpu_buffer = ggml_backend_buft_alloc_buffer(gpu_buft, expert_size);
        
        if (!gpu_buffer) {
            ggml_moe_log_error(layer_id, expert_id, "Failed to allocate GPU buffer", -2);
            return nullptr;
        }
        
        // Get pinned buffer for async transfer
        void* pinned_buffer = acquire_pinned_buffer(expert_size);
        
        if (!pinned_buffer) {
            ggml_backend_buffer_free(gpu_buffer);
            ggml_moe_log_error(layer_id, expert_id, "Failed to acquire pinned buffer", -3);
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
        stats.total_transfers_ram_to_vram++;
        
        // Log transfer operation
        ggml_moe_log_transfer("RAM->VRAM", expert_size,
                              ggml_moe_format_expert_key(layer_id, expert_id));
        
        // Log expert lifecycle event
        ggml_moe_log_expert_lifecycle(layer_id, expert_id, "Loaded", "Successfully loaded to GPU");
        
        return gpu_buffer;
    }
};

// Generic GPU backend cache interface implementation
struct ggml_moe_cache_interface_gpu : public ggml_moe_cache_interface {
    ggml_moe_cache* create_cache(
        ggml_backend_t backend,
        const ggml_moe_cache_config* config,
        int num_layers,
        int num_experts_per_layer
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
        return new ggml_moe_cache_gpu(backend, *config, num_layers, num_experts_per_layer, this, nullptr);
    }
    
    ggml_backend_buffer_t get_expert_async(
        ggml_moe_cache* cache,
        int layer_id,
        int expert_id,
        const ggml_tensor* expert_tensor,
        void* stream
    ) override {
        if (!cache || !expert_tensor) {
            return nullptr;
        }
        
        ggml_moe_cache_gpu* gpu_cache = static_cast<ggml_moe_cache_gpu*>(cache);
        
        std::lock_guard<std::mutex> lock(cache->cache_mutex);
        
        // Create composite key
        ggml_moe_expert_key key{layer_id, expert_id};
        
        // Get current time for statistics
        auto now = std::chrono::steady_clock::now();
        
        // Update statistics
        cache->stats.total_requests++;
        auto& expert_stat = cache->expert_stats[key];
        expert_stat.access_count++;
        expert_stat.last_access_time = now;
        
        // Update access frequency (exponential moving average)
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
        auto it = cache->cache_map.find(key);
        if (it != cache->cache_map.end()) {
            // Cache hit
            cache->stats.cache_hits++;
            
            // Update LRU: move to front
            auto lru_it = cache->lru_iter[key];
            cache->lru_list.erase(lru_it);
            cache->lru_list.push_front(key);
            cache->lru_iter[key] = cache->lru_list.begin();
            
            return it->second;
        }
        
        // Cache miss - need to load expert
        cache->stats.cache_misses++;
        
        // Load expert asynchronously
        ggml_backend_buffer_t gpu_buffer = gpu_cache->load_expert_async(
            layer_id,
            expert_id,
            expert_tensor,
            stream
        );
        
        if (gpu_buffer) {
            // Add to cache
            cache->cache_map[key] = gpu_buffer;
            cache->lru_list.push_front(key);
            cache->lru_iter[key] = cache->lru_list.begin();
            cache->expert_stats[key].is_cached = true;
        }
        
        return gpu_buffer;
    }
    
    void prefetch_experts_async(
        ggml_moe_cache* cache,
        int layer_id,
        const std::vector<int>& expert_ids,
        const ggml_tensor* expert_tensor,
        void* stream
    ) override {
        if (!cache || !cache->prefetch_engine || !expert_tensor) {
            return;
        }
        
        ggml_moe_cache_gpu* gpu_cache = static_cast<ggml_moe_cache_gpu*>(cache);
        
        // Log prefetch operation
        ggml_moe_log_prefetch(layer_id, -1, 0, "RAM", "VRAM");
        
        // Start async prefetch for each expert
        for (int expert_id : expert_ids) {
            // Create composite key
            ggml_moe_expert_key key{layer_id, expert_id};
            
            // Skip if already cached
            if (cache->cache_map.find(key) != cache->cache_map.end()) {
                continue;
            }
            
            // Log individual expert prefetch
            size_t expert_size = gpu_cache->get_expert_size(expert_tensor, layer_id, expert_id);
            (void)expert_size; // Mark as intentionally unused
            ggml_moe_log_expert_lifecycle(layer_id, expert_id, "Prefetching", "Async prefetch started");
            
            // Load expert asynchronously (this will evict if needed)
            ggml_backend_buffer_t gpu_buffer = gpu_cache->load_expert_async(
                layer_id,
                expert_id,
                expert_tensor,
                stream
            );
            
            if (gpu_buffer) {
                std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(cache->cache_mutex));
                cache->cache_map[key] = gpu_buffer;
                cache->lru_list.push_front(key);
                cache->lru_iter[key] = cache->lru_list.begin();
                cache->expert_stats[key].is_cached = true;
                cache->stats.prefetches++;
                
                ggml_moe_log_expert_lifecycle(layer_id, expert_id, "Prefetched", "Successfully loaded to cache");
            } else {
                ggml_moe_log_warning(layer_id, expert_id, "Prefetch failed - could not load expert");
            }
        }
    }
    
    void touch_expert(
        ggml_moe_cache* cache,
        int layer_id,
        int expert_id
    ) override {
        if (!cache) return;
        
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(cache->cache_mutex));
        
        // Create composite key
        ggml_moe_expert_key key{layer_id, expert_id};
        
        // Update access statistics
        auto now = std::chrono::steady_clock::now();
        auto& stats = cache->expert_stats[key];
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
        auto it = cache->lru_iter.find(key);
        if (it != cache->lru_iter.end()) {
            cache->lru_list.erase(it->second);
            cache->lru_list.push_front(key);
            it->second = cache->lru_list.begin();
        }
        
        // Update global statistics
        if (cache->stats.total_requests > 0) {
            cache->stats.hit_rate = (double)cache->stats.cache_hits / cache->stats.total_requests;
        }
    }
    
    // Overlap computation and transfer
    void overlap_computation_and_transfer(
        ggml_moe_cache* cache,
        int layer_id,
        const std::vector<int>& expert_ids,
        void* compute_stream,
        void* transfer_stream
    ) {
        (void)compute_stream; // Mark as unused for now
        (void)transfer_stream; // Mark as unused for now
        
        if (!cache || expert_ids.empty()) return;
        
        ggml_moe_cache_gpu* gpu_cache = static_cast<ggml_moe_cache_gpu*>(cache);
        
        // Start async prefetch for experts on transfer stream
        for (int expert_id : expert_ids) {
            // Create composite key
            ggml_moe_expert_key key{layer_id, expert_id};
            
            // Skip if already cached
            if (cache->cache_map.find(key) != cache->cache_map.end()) {
                continue;
            }
            
            // Log stream overlap operation
            ggml_moe_log_expert_lifecycle(layer_id, expert_id, "Stream Overlap", "Starting async prefetch on transfer stream");
            
            // Load expert asynchronously (this will evict if needed)
            // In a full implementation, this would use the transfer_stream
            ggml_backend_buffer_t gpu_buffer = gpu_cache->load_expert_async(
                layer_id,
                expert_id,
                gpu_cache->expert_source,
                compute_stream // Using compute_stream as fallback
            );
            
            if (gpu_buffer) {
                std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(cache->cache_mutex));
                cache->cache_map[key] = gpu_buffer;
                cache->lru_list.push_front(key);
                cache->lru_iter[key] = cache->lru_list.begin();
                cache->expert_stats[key].is_cached = true;
                cache->stats.prefetches++;
                
                ggml_moe_log_expert_lifecycle(layer_id, expert_id, "Stream Overlap", "Successfully prefetched on transfer stream");
            }
        }
        
        // Continue computation on compute_stream (handled by caller)
        // The overlap is achieved by having transfers happen concurrently
    }
    
    void get_stats(
        const ggml_moe_cache* cache,
        ggml_moe_cache_stats* stats
    ) override {
        if (!cache || !stats) return;
        
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(cache->cache_mutex));
        *stats = cache->stats;
        
        // Calculate derived statistics
        if (stats->total_requests > 0) {
            stats->hit_rate = (double)stats->cache_hits / stats->total_requests;
        }
        
        if (stats->prefetches > 0) {
            stats->prefetch_accuracy = (double)stats->prefetch_hits / stats->prefetches;
        }
        
        stats->current_size = cache->stats.current_size;
    }
    
    void reset_stats(
        ggml_moe_cache* cache
    ) override {
        if (!cache) return;
        
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(cache->cache_mutex));
        cache->stats = ggml_moe_cache_stats{};
    }
    
    void clear_cache(
        ggml_moe_cache* cache
    ) override {
        if (!cache) return;
        
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(cache->cache_mutex));
        
        // Free all cached buffers
        for (auto it = cache->cache_map.begin(); it != cache->cache_map.end(); ++it) {
            const ggml_moe_expert_key& key = it->first;
            ggml_backend_buffer_t buffer = it->second;
            if (buffer) {
                ggml_backend_buffer_free(buffer);
            }
            cache->expert_stats[key].is_cached = false;
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
        
        // Free all cached buffers
        for (auto it = cache->cache_map.begin(); it != cache->cache_map.end(); ++it) {
            const ggml_moe_expert_key& key = it->first;
            ggml_backend_buffer_t buffer = it->second;
            if (buffer) {
                ggml_backend_buffer_free(buffer);
            }
            cache->expert_stats[key].is_cached = false;
        }
        
        cache->cache_map.clear();
        cache->lru_list.clear();
        cache->lru_iter.clear();
        cache->stats.current_size = 0;
        
        // Delete the cache object
        delete cache;
    }
};

// Forward declaration for the GPU interface function implementation
static ggml_moe_cache_interface* ggml_moe_cache_get_interface_gpu_impl();

// Get generic GPU cache interface
static ggml_moe_cache_interface* ggml_moe_cache_get_interface_gpu() {
    return ggml_moe_cache_get_interface_gpu_impl();
}

// Implementation of GPU interface function
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
    // Mark backend as potentially unused if no MOE cache backends are enabled
    (void)backend;
#ifdef GGML_CUDA_MOE_CACHE
    if (ggml_backend_is_cuda(backend)) {
        return const_cast<ggml_moe_cache_interface*>(ggml_moe_cache_get_interface_cuda());
    }
#endif
#ifdef GGML_HIP_MOE_CACHE
    if (ggml_backend_is_hip(backend)) {
        return const_cast<ggml_moe_cache_interface*>(ggml_moe_cache_get_interface_hip());
    }
#endif
    
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

// C API implementations
extern "C" {

GGML_API ggml_moe_cache* ggml_moe_cache_init(
    ggml_backend_t backend,
    const ggml_moe_cache_config* config,
    int num_layers,
    int num_experts_per_layer
) {
    if (!backend || !config) return nullptr;
    
    ggml_moe_cache_interface* interface = ggml_moe_cache_get_interface(backend);
    if (!interface) return nullptr;
    
    return interface->create_cache(backend, config, num_layers, num_experts_per_layer);
}

GGML_API void ggml_moe_cache_free(
    ggml_moe_cache* cache
) {
    if (!cache || !cache->impl) return;
    cache->impl->destroy_cache(cache);
}

GGML_API ggml_backend_buffer_t ggml_moe_cache_get_expert(
    ggml_moe_cache* cache,
    int layer_id,
    int expert_id,
    const ggml_tensor* expert_tensor,
    void* stream
) {
    if (!cache || !cache->impl) return nullptr;
    return cache->impl->get_expert_async(cache, layer_id, expert_id, expert_tensor, stream);
}

GGML_API void ggml_moe_cache_prefetch(
    ggml_moe_cache* cache,
    int layer_id,
    const int* expert_ids,
    int num_experts,
    const ggml_tensor* expert_tensor,
    void* stream
) {
    if (!cache || !cache->impl || !expert_ids) return;
    
    std::vector<int> expert_ids_vec(expert_ids, expert_ids + num_experts);
    cache->impl->prefetch_experts_async(cache, layer_id, expert_ids_vec, expert_tensor, stream);
}

GGML_API void ggml_moe_cache_touch(
    ggml_moe_cache* cache,
    int layer_id,
    int expert_id
) {
    if (!cache || !cache->impl) return;
    cache->impl->touch_expert(cache, layer_id, expert_id);
}

GGML_API void ggml_moe_cache_get_stats(
    const ggml_moe_cache* cache,
    ggml_moe_cache_stats* stats
) {
    if (!cache || !cache->impl || !stats) return;
    cache->impl->get_stats(cache, stats);
}

GGML_API void ggml_moe_cache_reset_stats(
    ggml_moe_cache* cache
) {
    if (!cache || !cache->impl) return;
    cache->impl->reset_stats(cache);
}

GGML_API void ggml_moe_cache_clear(
    ggml_moe_cache* cache
) {
    if (!cache || !cache->impl) return;
    cache->impl->clear_cache(cache);
}

// Phase 3: Async operation implementations
GGML_API ggml_backend_buffer_t ggml_moe_cache_get_expert_async(
    ggml_moe_cache* cache,
    int layer_id,
    int expert_id,
    const ggml_tensor* expert_tensor,
    void* stream
) {
    if (!cache || !cache->impl) return nullptr;
    return cache->impl->get_expert_async(cache, layer_id, expert_id, expert_tensor, stream);
}

GGML_API void ggml_moe_cache_prefetch_async(
    ggml_moe_cache* cache,
    int layer_id,
    const int* expert_ids,
    int num_experts,
    const ggml_tensor* expert_tensor,
    void* stream
) {
    if (!cache || !cache->impl || !expert_ids) return;
    
    std::vector<int> expert_ids_vec(expert_ids, expert_ids + num_experts);
    cache->impl->prefetch_experts_async(cache, layer_id, expert_ids_vec, expert_tensor, stream);
}

GGML_API bool ggml_moe_cache_try_gpu_to_gpu_transfer(
    ggml_moe_cache* cache,
    int src_layer_id,
    int dst_layer_id,
    int expert_id,
    void* stream
) {
    (void)cache;
    (void)src_layer_id;
    (void)dst_layer_id;
    (void)expert_id;
    (void)stream;
    
    // For now, return false to indicate GPU-to-GPU transfer is not supported
    // In a full implementation, this would detect if both source and destination
    // are GPU buffers and use direct GPU copy (cudaMemcpyDeviceToDevice)
    return false;
}

GGML_API ggml_backend_buffer_t ggml_moe_cache_allocate_pinned_buffer(
    ggml_moe_cache* cache,
    size_t size
) {
    if (!cache) return nullptr;
    
    // This is a generic implementation - backend-specific implementations
    // would use their respective pinned memory allocation functions
    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(
        ggml_backend_get_device(cache->backend)
    );
    
    if (!host_buft) return nullptr;
    
    ggml_backend_buffer_t host_buffer = ggml_backend_buft_alloc_buffer(
        host_buft, size
    );
    
    return host_buffer;
}

GGML_API void ggml_moe_cache_release_pinned_buffer(
    ggml_moe_cache* cache,
    void* buffer
) {
    (void)cache;
    
    if (buffer) {
        // Retrieve buffer handle and free it
        ggml_backend_buffer_t host_buffer = *(ggml_backend_buffer_t*)buffer;
        if (host_buffer) {
            ggml_backend_buffer_free(host_buffer);
        }
    }
}

GGML_API void ggml_moe_cache_overlap_computation_and_transfer(
    ggml_moe_cache* cache,
    int layer_id,
    const int* expert_ids,
    int num_experts,
    void* compute_stream,
    void* transfer_stream
) {
    (void)compute_stream; // Mark as intentionally unused
    if (!cache || !cache->impl || !expert_ids) return;
    
    std::vector<int> expert_ids_vec(expert_ids, expert_ids + num_experts);
    
    // Call the interface's overlap function if available
    // For now, we'll use the prefetch as a fallback
    cache->impl->prefetch_experts_async(cache, layer_id, expert_ids_vec, nullptr, transfer_stream);
}

} // extern "C"
