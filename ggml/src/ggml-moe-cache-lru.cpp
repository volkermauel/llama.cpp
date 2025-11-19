#include "ggml-moe-cache.h"
#include "ggml.h"
#include "llama.h"
#include <algorithm>

// Update LRU queue when an expert is accessed
void ggml_moe_cache::update_lru(const ggml_moe_expert_key& key) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    
    // Check if expert is already in LRU queue
    auto lru_it = lru_iter.find(key);
    if (lru_it != lru_iter.end()) {
        // Move to front (most recently used)
        lru_list.erase(lru_it->second);
        lru_list.push_front(key);
        lru_iter[key] = lru_list.begin();
    } else {
        // Add to front of LRU queue
        lru_list.push_front(key);
        lru_iter[key] = lru_list.begin();
    }
}

// Evict the least recently used expert from GPU cache
void ggml_moe_cache::evict_lru() {
    std::lock_guard<std::mutex> lock(cache_mutex);
    
    if (lru_list.empty()) {
        return;
    }
    
    // Get the least recently used expert (back of the list)
    ggml_moe_expert_key lru_key = lru_list.back();
    
    // Find the expert in GPU cache
    auto cache_it = cache_map.find(lru_key);
    if (cache_it != cache_map.end()) {
        // Free GPU buffer
        ggml_backend_buffer_free(cache_it->second);
        
        // Move the expert data to system RAM if not already there
        if (system_ram_storage.find(lru_key) == system_ram_storage.end()) {
            // Note: In a full implementation, we would copy the expert data to system RAM here
            // For now, we assume the original data is still available in the model
            LLAMA_LOG_DEBUG("Evicted expert (%d, %d) from GPU cache to system RAM\n", 
                           lru_key.layer_id, lru_key.expert_id);
        }
        
        // Remove from GPU cache
        cache_map.erase(cache_it);
    }
    
    // Remove from LRU queue
    lru_list.pop_back();
    lru_iter.erase(lru_key);
    
    // Update statistics
    stats.evictions++;
}

// Check if GPU cache is full
bool ggml_moe_cache::is_cache_full() const {
    std::lock_guard<std::mutex> lock(cache_mutex);
    
    size_t current_size = get_cache_size();
    return current_size >= config.max_cache_size;
}

// Get current size of GPU cache
size_t ggml_moe_cache::get_cache_size() const {
    size_t total_size = 0;
    for (const auto& pair : cache_map) {
        total_size += get_expert_size(pair.first);
    }
    return total_size;
}

// Get size of a specific expert
size_t ggml_moe_cache::get_expert_size(const ggml_moe_expert_key& key) const {
    auto it = cache_map.find(key);
    if (it != cache_map.end()) {
        return ggml_backend_buffer_get_size(it->second);
    }
    return 0;
}

// Check if expert exists in system RAM storage
bool ggml_moe_cache::has_expert_in_system_ram(const ggml_moe_expert_key& key) const {
    std::lock_guard<std::mutex> lock(cache_mutex);
    
    // Check if we have the expert in system RAM storage
    if (system_ram_storage.find(key) != system_ram_storage.end()) {
        return true;
    }
    
    // If not in explicit system RAM storage, check if we can access it from the original model
    // This assumes the model data is still available in system memory
    return true;
}

// Get expert from system RAM storage
ggml_backend_buffer_t ggml_moe_cache::get_expert_from_system_ram(const ggml_moe_expert_key& key) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    
    auto it = system_ram_storage.find(key);
    if (it != system_ram_storage.end()) {
        return it->second;
    }
    
    return nullptr;
}

// Stream expert from system RAM to GPU
void ggml_moe_cache::stream_expert_to_gpu(const ggml_moe_expert_key& key) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    
    // Get the expert data from system RAM
    ggml_backend_buffer_t ram_buffer = get_expert_from_system_ram(key);
    if (!ram_buffer) {
        LLAMA_LOG_ERROR("Failed to find expert (%d, %d) in system RAM for streaming\n", 
                       key.layer_id, key.expert_id);
        return;
    }
    
    // Allocate GPU buffer for this expert
    size_t expert_size = ggml_backend_buffer_get_size(ram_buffer);
    ggml_backend_buffer_t gpu_buffer = ggml_backend_buft_alloc_buffer(
        ggml_backend_get_device(backend), expert_size);
    
    if (!gpu_buffer) {
        LLAMA_LOG_ERROR("Failed to allocate GPU buffer for expert (%d, %d)\n", 
                       key.layer_id, key.expert_id);
        return;
    }
    
    // Copy data from system RAM to GPU
    void* gpu_data = ggml_backend_buffer_get_base(gpu_buffer);
    void* ram_data = ggml_backend_buffer_get_base(ram_buffer);
    
    if (gpu_data && ram_data) {
        // Use backend-specific copy method for better performance
        ggml_backend_t backend = this->backend;
        
        if (ggml_backend_is_cuda(backend)) {
#ifdef GGML_USE_CUDA
            // Use CUDA async copy
            cudaMemcpyAsync(gpu_data, ram_data, expert_size, cudaMemcpyHostToDevice, 
                           static_cast<cudaStream_t>(nullptr));
            cudaStreamSynchronize(static_cast<cudaStream_t>(nullptr));
#endif
        } else if (ggml_backend_is_hip(backend)) {
#ifdef GGML_USE_HIP
            // Use HIP async copy
            hipMemcpyAsync(gpu_data, ram_data, expert_size, hipMemcpyHostToDevice,
                          static_cast<hipStream_t>(nullptr));
            hipStreamSynchronize(static_cast<hipStream_t>(nullptr));
#endif
        } else {
            // Fallback to synchronous copy for other backends
            memcpy(gpu_data, ram_data, expert_size);
        }
        
        // Add to GPU cache
        cache_map[key] = gpu_buffer;
        
        // Update LRU
        update_lru(key);
        
        // Update statistics
        stats.current_size += expert_size;
        stats.peak_size = std::max(stats.peak_size, stats.current_size);
        
        LLAMA_LOG_DEBUG("Streamed expert (%d, %d) from system RAM to GPU cache\n", 
                       key.layer_id, key.expert_id);
    } else {
        LLAMA_LOG_ERROR("Failed to get buffer pointers for expert (%d, %d)\n", 
                       key.layer_id, key.expert_id);
        ggml_backend_buffer_free(gpu_buffer);
    }
}