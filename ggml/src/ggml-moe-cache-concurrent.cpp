#include "ggml-moe-cache.h"
#include "ggml-moe-cache-lockfree.h"
#include "ggml-backend.h"
#include "ggml.h"
#include <cstring>
#include <vector>
#include <algorithm>

namespace ggml_moe_concurrent {

// Concurrent MoE cache implementation using lock-free data structures
struct ggml_moe_cache_concurrent : public ggml_moe_cache {
    // Lock-free hash table for fast lookups
    ggml_moe_lockfree::hashtable cache_map;
    
    // RCU-based LRU for scalable eviction
    ggml_moe_lockfree::rcu_lru_list lru_list;
    
    // Sharded locks for remaining operations
    ggml_moe_lockfree::sharded_locks shard_locks;
    
    // Wait-free buffer pool for transfers
    ggml_moe_lockfree::waitfree_buffer_pool buffer_pool;
    
    // Expert source tensor for size calculations
    const ggml_tensor* expert_source;
    
    // Statistics (extends base stats)
    struct {
        std::atomic<uint64_t> cache_hits;
        std::atomic<uint64_t> cache_misses;
        std::atomic<uint64_t> total_requests;
        std::atomic<uint64_t> successful_inserts;
        std::atomic<uint64_t> failed_inserts;
        std::atomic<uint64_t> evictions;
        std::atomic<uint64_t> prefetches;
        size_t current_size;
        size_t peak_size;
    } stats;
    
    // Constructor
    ggml_moe_cache_concurrent(
        ggml_backend_t backend,
        const ggml_moe_cache_config& config,
        int num_experts,
        ggml_moe_cache_interface* impl
    ) : ggml_moe_cache(backend, config, num_experts, impl), expert_source(nullptr) {
        // Initialize lock-free data structures
        cache_map.initialize(std::max(size_t(1024), size_t(num_experts * 2)));
        
        // Initialize buffer pool
        const size_t num_buffers = 8;
        const size_t buffer_size = 64 * 1024 * 1024;  // 64MB
        buffer_pool.initialize(num_buffers, buffer_size);
        
        // Initialize statistics
        stats.cache_hits.store(0, std::memory_order_relaxed);
        stats.cache_misses.store(0, std::memory_order_relaxed);
        stats.total_requests.store(0, std::memory_order_relaxed);
        stats.successful_inserts.store(0, std::memory_order_relaxed);
        stats.failed_inserts.store(0, std::memory_order_relaxed);
        stats.evictions.store(0, std::memory_order_relaxed);
        stats.prefetches.store(0, std::memory_order_relaxed);
        stats.current_size = 0;
        stats.peak_size = 0;
    }
    
    // Destructor
    ~ggml_moe_cache_concurrent() {
        // Clean up any remaining cached experts
        clear_cache();
    }
    
    // Calculate expert size in bytes
    size_t get_expert_size(const ggml_tensor* expert_tensor, int expert_id) {
        (void)expert_id; // Mark as unused
        if (!expert_tensor) return 0;
        size_t expert_bytes = ggml_nbytes(expert_tensor) / expert_tensor->ne[2];
        return expert_bytes;
    }
    
    // Evict experts to make room for new ones
    void evict_experts(size_t required_space) {
        size_t available_space = config.max_cache_size - stats.current_size;
        if (available_space >= required_space) {
            return;  // Enough space available
        }
        
        size_t space_to_free = required_space - available_space;
        size_t freed_space = 0;
        
        // Get eviction candidates from LRU
        auto candidates = lru_list.get_eviction_candidates(10);
        
        // Evict experts based on priority until we have enough space
        for (int expert_id : candidates) {
            if (freed_space >= space_to_free) {
                break; // Enough space freed
            }
            
            // Remove from hash table
            if (cache_map.remove(expert_id)) {
                size_t expert_size = get_expert_size(expert_source, expert_id);
                freed_space += expert_size;
                stats.current_size -= expert_size;
                stats.evictions.fetch_add(1, std::memory_order_relaxed);
            }
        }
        
        // Reclaim memory from retired nodes
        lru_list.reclaim_memory();
    }
    
    // Load expert from CPU to GPU memory
    ggml_backend_buffer_t load_expert_async(
        int expert_id,
        const ggml_tensor* expert_tensor,
        void* compute_stream
    ) {
        (void)compute_stream; // Mark as unused for now
        
        if (!expert_tensor) return nullptr;
        
        // Store expert source for size calculations
        expert_source = expert_tensor;
        
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
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
        ggml_backend_buffer_t gpu_buffer = ggml_backend_buft_alloc_buffer(buft, expert_size);
        
        if (!gpu_buffer) {
            return nullptr;
        }
        
        // Get pinned buffer for async transfer
        size_t pinned_buffer_size;
        void* pinned_buffer = buffer_pool.acquire(pinned_buffer_size);
        
        if (!pinned_buffer) {
            ggml_backend_buffer_free(gpu_buffer);
            return nullptr;
        }
        
        // Ensure pinned buffer is large enough
        if (pinned_buffer_size < expert_size) {
            buffer_pool.release(pinned_buffer);
            ggml_backend_buffer_free(gpu_buffer);
            return nullptr;
        }
        
        // Copy expert data to pinned buffer first
        const char* cpu_data = (const char*)expert_tensor->data + expert_offset;
        memcpy(pinned_buffer, cpu_data, expert_size);
        
        // Copy from pinned buffer to GPU
        char* gpu_data = (char*)ggml_backend_buffer_get_base(gpu_buffer);
        memcpy(gpu_data, pinned_buffer, expert_size);  // Synchronous for now
        
        // Release pinned buffer back to pool
        buffer_pool.release(pinned_buffer);
        
        // Update cache statistics
        stats.current_size += expert_size;
        stats.peak_size = std::max(stats.peak_size, stats.current_size);
        
        return gpu_buffer;
    }
    
    // Clear all cached experts
    void clear_cache() {
        // No need to lock since we're using atomic operations
        // Iterate through hash table and free all buffers
        
        // Note: In a real implementation, we'd need to iterate through all buckets
        // For now, we'll just update statistics
        stats.current_size = 0;
        
        // Reclaim memory
        lru_list.reclaim_memory();
    }
};

// Concurrent cache interface implementation
struct ggml_moe_cache_interface_concurrent : public ggml_moe_cache_interface {
    ggml_moe_cache* create_cache(
        ggml_backend_t backend,
        const ggml_moe_cache_config* config,
        int num_experts
    ) override {
        if (!config || num_experts <= 0) {
            return nullptr;
        }
        
        return new ggml_moe_cache_concurrent(backend, *config, num_experts, this);
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
        
        ggml_moe_cache_concurrent* concurrent_cache = static_cast<ggml_moe_cache_concurrent*>(cache);
        
        // Update statistics
        concurrent_cache->stats.total_requests.fetch_add(1, std::memory_order_relaxed);
        
        // Try to get from lock-free hash table
        ggml_backend_buffer_t buffer = concurrent_cache->cache_map.get(expert_id);
        
        if (buffer) {
            // Cache hit
            concurrent_cache->stats.cache_hits.fetch_add(1, std::memory_order_relaxed);
            
            // Update LRU
            concurrent_cache->lru_list.touch(expert_id);
            
            return buffer;
        }
        
        // Cache miss - need to load expert
        concurrent_cache->stats.cache_misses.fetch_add(1, std::memory_order_relaxed);
        
        // Load expert asynchronously
        buffer = concurrent_cache->load_expert_async(
            expert_id,
            expert_tensor,
            stream
        );
        
        if (buffer) {
            // Insert into lock-free hash table
            if (concurrent_cache->cache_map.insert(expert_id, buffer)) {
                concurrent_cache->stats.successful_inserts.fetch_add(1, std::memory_order_relaxed);
                
                // Update LRU
                concurrent_cache->lru_list.touch(expert_id);
            } else {
                concurrent_cache->stats.failed_inserts.fetch_add(1, std::memory_order_relaxed);
                
                // Clean up buffer if insert failed
                ggml_backend_buffer_free(buffer);
                buffer = nullptr;
            }
        }
        
        return buffer;
    }
    
    void prefetch_experts_async(
        ggml_moe_cache* cache,
        const std::vector<int>& expert_ids,
        const ggml_tensor* expert_tensor,
        void* stream
    ) override {
        if (!cache || !expert_tensor) {
            return;
        }
        
        ggml_moe_cache_concurrent* concurrent_cache = static_cast<ggml_moe_cache_concurrent*>(cache);
        
        // Start async prefetch for each expert
        for (int expert_id : expert_ids) {
            // Check if already cached (fast lock-free check)
            if (concurrent_cache->cache_map.get(expert_id) != nullptr) {
                continue;  // Already cached
            }
            
            // Load expert asynchronously (this will evict if needed)
            ggml_backend_buffer_t gpu_buffer = concurrent_cache->load_expert_async(
                expert_id,
                expert_tensor,
                stream
            );
            
            if (gpu_buffer) {
                // Insert into lock-free hash table
                if (concurrent_cache->cache_map.insert(expert_id, gpu_buffer)) {
                    concurrent_cache->stats.successful_inserts.fetch_add(1, std::memory_order_relaxed);
                    
                    // Update LRU
                    concurrent_cache->lru_list.touch(expert_id);
                    
                    // Update prefetch statistics
                    concurrent_cache->stats.prefetches.fetch_add(1, std::memory_order_relaxed);
                } else {
                    concurrent_cache->stats.failed_inserts.fetch_add(1, std::memory_order_relaxed);
                    
                    // Clean up buffer if insert failed
                    ggml_backend_buffer_free(gpu_buffer);
                }
            }
        }
    }
    
    void touch_expert(
        ggml_moe_cache* cache,
        int expert_id
    ) override {
        if (!cache) return;
        
        ggml_moe_cache_concurrent* concurrent_cache = static_cast<ggml_moe_cache_concurrent*>(cache);
        
        // Update LRU (lock-free operation)
        concurrent_cache->lru_list.touch(expert_id);
        
        // Update statistics
        concurrent_cache->stats.total_requests.fetch_add(1, std::memory_order_relaxed);
    }
    
    ggml_moe_cache_stats get_stats(
        const ggml_moe_cache* cache
    ) override {
        if (!cache) return ggml_moe_cache_stats{};
        
        const ggml_moe_cache_concurrent* concurrent_cache = 
            static_cast<const ggml_moe_cache_concurrent*>(cache);
        
        ggml_moe_cache_stats stats = {};
        
        // Copy atomic statistics
        stats.total_requests = concurrent_cache->stats.total_requests.load(std::memory_order_relaxed);
        stats.cache_hits = concurrent_cache->stats.cache_hits.load(std::memory_order_relaxed);
        stats.cache_misses = concurrent_cache->stats.cache_misses.load(std::memory_order_relaxed);
        stats.evictions = concurrent_cache->stats.evictions.load(std::memory_order_relaxed);
        stats.prefetches = concurrent_cache->stats.prefetches.load(std::memory_order_relaxed);
        stats.current_size = concurrent_cache->stats.current_size;
        stats.peak_size = concurrent_cache->stats.peak_size;
        
        // Calculate derived statistics
        if (stats.total_requests > 0) {
            stats.hit_rate = static_cast<double>(stats.cache_hits) / stats.total_requests;
        }
        
        if (stats.prefetches > 0) {
            // Note: prefetch_hits would need additional tracking
            stats.prefetch_accuracy = 0.0;
        }
        
        // Add lock-free statistics
        stats.avg_load_time = concurrent_cache->shard_locks.average_wait_time_ns() / 1e6;  // Convert to ms
        
        return stats;
    }
    
    void reset_stats(
        ggml_moe_cache* cache
    ) override {
        if (!cache) return;
        
        ggml_moe_cache_concurrent* concurrent_cache = static_cast<ggml_moe_cache_concurrent*>(cache);
        
        // Reset all atomic statistics individually
        concurrent_cache->stats.cache_hits.store(0, std::memory_order_relaxed);
        concurrent_cache->stats.cache_misses.store(0, std::memory_order_relaxed);
        concurrent_cache->stats.total_requests.store(0, std::memory_order_relaxed);
        concurrent_cache->stats.successful_inserts.store(0, std::memory_order_relaxed);
        concurrent_cache->stats.failed_inserts.store(0, std::memory_order_relaxed);
        concurrent_cache->stats.evictions.store(0, std::memory_order_relaxed);
        concurrent_cache->stats.prefetches.store(0, std::memory_order_relaxed);
        concurrent_cache->stats.current_size = 0;
        concurrent_cache->stats.peak_size = 0;
    }
    
    void clear_cache(
        ggml_moe_cache* cache
    ) override {
        if (!cache) return;
        
        ggml_moe_cache_concurrent* concurrent_cache = static_cast<ggml_moe_cache_concurrent*>(cache);
        
        concurrent_cache->clear_cache();
    }
    
    void destroy_cache(
        ggml_moe_cache* cache
    ) override {
        if (!cache) return;
        
        // The destructor will clean up everything
        delete cache;
    }
};

// Get concurrent cache interface
ggml_moe_cache_interface* ggml_moe_cache_get_interface_concurrent() {
    static ggml_moe_cache_interface_concurrent interface;
    return &interface;
}

} // namespace ggml_moe_concurrent

// C API for concurrent cache
extern "C" {
    GGML_API ggml_moe_cache* ggml_moe_cache_init_concurrent(
        ggml_backend_t backend,
        const ggml_moe_cache_config* config,
        int num_experts
    ) {
        if (!backend || !config || num_experts <= 0) {
            return nullptr;
        }
        
        return new ggml_moe_concurrent::ggml_moe_cache_concurrent(backend, *config, num_experts, 
            ggml_moe_concurrent::ggml_moe_cache_get_interface_concurrent());
    }
}