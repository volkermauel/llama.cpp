#include "ggml-moe-cache.h"
#include "ggml-cuda/common.cuh"
#include "ggml-backend-impl.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <queue>

// CUDA-specific MoE cache implementation
struct ggml_moe_cache_cuda : public ggml_moe_cache {
    // CUDA-specific members
    cudaStream_t transfer_stream;  // Dedicated stream for async transfers
    std::queue<void*> pinned_buffer_pool;  // Pool of pinned memory buffers
    std::mutex buffer_pool_mutex;
    size_t pinned_buffer_size;
    
    // Expert data stored in CPU memory (original location)
    const ggml_tensor* expert_source;
    
    // Constructor
    ggml_moe_cache_cuda(
        ggml_backend_t backend,
        const ggml_moe_cache_config& config,
        int num_experts,
        ggml_moe_cache_interface* impl
    ) : ggml_moe_cache(backend, config, num_experts, impl) {
        // Create dedicated transfer stream
        cudaStreamCreateWithFlags(&transfer_stream, cudaStreamNonBlocking);
        
        // Initialize pinned buffer pool
        pinned_buffer_size = 64 * 1024 * 1024;  // 64MB default
        int num_buffers = 4;
        for (int i = 0; i < num_buffers; ++i) {
            void* pinned_buffer;
            cudaMallocHost(&pinned_buffer, pinned_buffer_size);
            pinned_buffer_pool.push(pinned_buffer);
        }
    }
    
    // Destructor
    ~ggml_moe_cache_cuda() {
        // Free pinned buffers
        while (!pinned_buffer_pool.empty()) {
            void* buffer = pinned_buffer_pool.front();
            pinned_buffer_pool.pop();
            cudaFreeHost(buffer);
        }
        
        // Destroy transfer stream
        cudaStreamDestroy(transfer_stream);
    }
    
    // Get or create pinned buffer
    void* acquire_pinned_buffer(size_t size) {
        std::lock_guard<std::mutex> lock(buffer_pool_mutex);
        
        if (size > pinned_buffer_size) {
            // Allocate temporary larger buffer
            void* temp_buffer;
            cudaMallocHost(&temp_buffer, size);
            return temp_buffer;
        }
        
        if (!pinned_buffer_pool.empty()) {
            void* buffer = pinned_buffer_pool.front();
            pinned_buffer_pool.pop();
            return buffer;
        }
        
        // Allocate new buffer if pool is empty
        void* new_buffer;
        cudaMallocHost(&new_buffer, pinned_buffer_size);
        return new_buffer;
    }
    
    // Return pinned buffer to pool
    void release_pinned_buffer(void* buffer) {
        std::lock_guard<std::mutex> lock(buffer_pool_mutex);
        
        // Check if this is a temporary oversized buffer
        if (buffer && buffer != pinned_buffer_pool.front()) {
            // Assume it's a temporary buffer and free it
            cudaFreeHost(buffer);
            return;
        }
        
        if (buffer) {
            pinned_buffer_pool.push(buffer);
        }
    }
    
    // Calculate expert size in bytes
    size_t get_expert_size(const ggml_tensor* expert_tensor, int expert_id) {
        // For expert matrices, calculate slice size
        // Assuming expert_tensor is 3D: [dim0, dim1, num_experts]
        size_t expert_bytes = ggml_nbytes(expert_tensor) / expert_tensor->ne[2];
        return expert_bytes;
    }
    
    // Evict experts to make room for new ones
    void evict_experts(size_t required_space) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        
        size_t available_space = config.max_cache_size - stats.current_size;
        if (available_space >= required_space) {
            return;  // Enough space available
        }
        
        // Calculate how much space we need to free
        size_t space_to_free = required_space - available_space;
        size_t freed_space = 0;
        
        // Evict from LRU list until we have enough space
        while (!lru_list.empty() && freed_space < space_to_free) {
            int expert_id = lru_list.back();
            
            // Skip if expert is currently being used
            if (expert_stats[expert_id].access_count > 0 && 
                expert_stats[expert_id].last_access > std::chrono::steady_clock::now() - std::chrono::milliseconds(100)) {
                lru_list.pop_back();
                continue;
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
            }
            
            lru_list.pop_back();
        }
    }
    
    // Load expert from CPU to GPU memory
    ggml_backend_buffer_t load_expert_async(
        int expert_id,
        const ggml_tensor* expert_tensor,
        cudaStream_t compute_stream
    ) {
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
        void* pinned_buffer = acquire_pinned_buffer(expert_size);
        
        // Copy expert data to pinned buffer first
        const char* cpu_data = (const char*)expert_tensor->data + expert_offset;
        memcpy(pinned_buffer, cpu_data, expert_size);
        
        // Initiate async transfer from pinned buffer to GPU
        char* gpu_data = (char*)ggml_backend_buffer_get_base(gpu_buffer);
        cudaMemcpyAsync(
            gpu_data,
            pinned_buffer,
            expert_size,
            cudaMemcpyHostToDevice,
            transfer_stream
        );
        
        // Wait for transfer to complete before releasing pinned buffer
        cudaStreamSynchronize(transfer_stream);
        release_pinned_buffer(pinned_buffer);
        
        // Update cache statistics
        stats.current_size += expert_size;
        stats.peak_size = std::max(stats.peak_size, stats.current_size);
        
        return gpu_buffer;
    }
};

// CUDA backend cache interface implementation
struct ggml_moe_cache_interface_cuda : public ggml_moe_cache_interface {
    ggml_moe_cache* create_cache(
        ggml_backend_t backend,
        const ggml_moe_cache_config* config,
        int num_experts
    ) override {
        if (!ggml_backend_is_cuda(backend) || !config || num_experts <= 0) {
            return nullptr;
        }
        
        return new ggml_moe_cache_cuda(backend, *config, num_experts, this);
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
        
        ggml_moe_cache_cuda* cuda_cache = static_cast<ggml_moe_cache_cuda*>(cache);
        cudaStream_t compute_stream = static_cast<cudaStream_t>(stream);
        
        std::lock_guard<std::mutex> lock(cache->cache_mutex);
        
        // Update statistics
        cache->stats.total_requests++;
        cache->expert_stats[expert_id].access_count++;
        cache->expert_stats[expert_id].last_access_time = std::chrono::steady_clock::now();
        
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
        ggml_backend_buffer_t gpu_buffer = cuda_cache->load_expert_async(
            expert_id,
            expert_tensor,
            compute_stream
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
        
        ggml_moe_cache_cuda* cuda_cache = static_cast<ggml_moe_cache_cuda*>(cache);
        cudaStream_t compute_stream = static_cast<cudaStream_t>(stream);
        
        // Start async prefetch for each expert
        for (int expert_id : expert_ids) {
            // Skip if already cached
            if (cache->cache_map.find(expert_id) != cache->cache_map.end()) {
                continue;
            }
            
            // Load expert asynchronously (this will evict if needed)
            ggml_backend_buffer_t gpu_buffer = cuda_cache->load_expert_async(
                expert_id,
                expert_tensor,
                compute_stream
            );
            
            if (gpu_buffer) {
                std::lock_guard<std::mutex> lock(cache->cache_mutex);
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
        
        std::lock_guard<std::mutex> lock(cache->cache_mutex);
        
        // Update access statistics
        auto now = std::chrono::steady_clock::now();
        auto& stats = cache->expert_stats[expert_id];
        stats.access_count++;
        stats.last_access_time = now;
        
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
        
        std::lock_guard<std::mutex> lock(cache->cache_mutex);
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
        
        std::lock_guard<std::mutex> lock(cache->cache_mutex);
        memset(&cache->stats, 0, sizeof(cache->stats));
    }
    
    void clear_cache(
        ggml_moe_cache* cache
    ) override {
        if (!cache) return;
        
        std::lock_guard<std::mutex> lock(cache->cache_mutex);
        
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
        delete cache;
    }
};

// Get CUDA cache interface
ggml_moe_cache_interface* ggml_moe_cache_get_interface_cuda() {
    static ggml_moe_cache_interface_cuda interface;
    return &interface;
}

// Backend detection and interface selection
ggml_moe_cache_interface* ggml_moe_cache_get_interface(ggml_backend_t backend) {
    if (ggml_backend_is_cuda(backend)) {
        return ggml_moe_cache_get_interface_cuda();
    }
    
    // Add other backend interfaces here
    // else if (ggml_backend_is_hip(backend)) {
    //     return ggml_moe_cache_get_interface_hip();
    // }
    
    return nullptr;
}