# MoE Cache Architecture Design for llama.cpp

## Overview

This document describes the design of a GPU-accelerated Mixture of Experts (MoE) cache system for llama.cpp. The cache automatically offloads frequently used experts to GPU memory while evicting less used ones, significantly improving inference speed for large MoE models.

## Problem Statement

Large MoE models (e.g., Mixtral 8x7B, DeepSeek-V2, Qwen-MoE) have many expert weights that don't fit in GPU memory simultaneously. Current implementations either:
- Load all experts into GPU memory (requires large GPU)
- Keep experts in CPU memory (slow inference)
- Use manual offloading (inflexible, suboptimal)

## Solution: Intelligent MoE Cache

An automatic caching system that:
1. Keeps recently/frequently used experts in GPU memory
2. Evicts less used experts automatically (LRU policy)
3. Pre-fetches experts predicted to be needed soon
4. Works across all GPU backends (CUDA, HIP, etc.)
5. Provides minimal overhead and maximum performance

## Architecture Components

### 1. Expert Cache Manager (`ggml_moe_cache`)

Central component managing the entire cache system:

```cpp
struct ggml_moe_cache {
    // Cache storage: expert_id -> GPU buffer mapping
    std::unordered_map<int, ggml_backend_buffer_t> cache_map;
    
    // LRU tracking: most recently used at front
    std::list<int> lru_list;
    std::unordered_map<int, std::list<int>::iterator> lru_iter;
    
    // Usage statistics per expert
    struct expert_stats {
        uint64_t access_count;
        double last_access_time;
        double access_frequency;
    };
    std::vector<expert_stats> expert_stats;
    
    // Configuration
    size_t max_cache_size;  // Maximum GPU memory for cache
    size_t current_size;    // Currently used GPU memory
    
    // Pre-fetching state
    std::vector<int> predicted_experts;
    std::mutex prefetch_mutex;
};
```

### 2. Backend Integration

The cache integrates with existing GPU backends through a unified interface:

```cpp
// Backend-agnostic cache interface
struct ggml_moe_cache_interface {
    // Initialize cache for a backend
    virtual ggml_moe_cache* create_cache(
        ggml_backend_t backend, 
        size_t max_size
    ) = 0;
    
    // Get expert from cache (async)
    virtual ggml_backend_buffer_t get_expert_async(
        ggml_moe_cache* cache,
        int expert_id,
        const ggml_tensor* expert_tensor,
        cudaStream_t stream
    ) = 0;
    
    // Pre-fetch experts
    virtual void prefetch_experts_async(
        ggml_moe_cache* cache,
        const std::vector<int>& expert_ids,
        const ggml_tensor* expert_tensor,
        cudaStream_t stream
    ) = 0;
    
    // Update LRU and statistics
    virtual void touch_expert(
        ggml_moe_cache* cache,
        int expert_id
    ) = 0;
    
    // Get cache statistics
    virtual ggml_moe_cache_stats get_stats(
        const ggml_moe_cache* cache
    ) = 0;
};
```

### 3. CUDA Backend Implementation

CUDA-specific implementation leveraging async memory operations:

```cpp
struct ggml_moe_cache_cuda : ggml_moe_cache {
    ggml_backend_cuda_context* cuda_ctx;
    cudaStream_t transfer_stream;  // Dedicated stream for async transfers
    
    // Pinned memory buffers for async transfers
    std::queue<void*> pinned_buffers;
    std::mutex buffer_mutex;
    
    // Expert data stored in CPU memory (original location)
    const ggml_tensor* expert_source;
    
    // Implementation of virtual methods
    ggml_backend_buffer_t get_expert_async(
        int expert_id,
        const ggml_tensor* expert_tensor,
        cudaStream_t compute_stream
    ) override;
    
    void evict_experts(size_t required_space);
    void* acquire_pinned_buffer(size_t size);
    void release_pinned_buffer(void* buffer);
};
```

### 4. Pre-fetching Engine

Predicts which experts will be needed based on token patterns:

```cpp
struct ggml_moe_prefetch_engine {
    // Token history for pattern recognition
    std::vector<int> token_history;
    static constexpr size_t HISTORY_SIZE = 1024;
    
    // Expert co-occurrence matrix
    std::vector<std::vector<int>> expert_cooccurrence;
    
    // Prediction algorithms
    std::vector<int> predict_next_experts(
        const std::vector<int>& current_experts,
        int top_k = 3
    );
    
    // Update patterns based on actual usage
    void update_patterns(
        const std::vector<int>& used_experts,
        const std::vector<int>& tokens
    );
    
    // Simple locality-based prediction
    std::vector<int> predict_locality(
        const std::vector<int>& recent_experts
    );
};
```

### 5. Integration with ggml_mul_mat_id

The cache hooks into the existing MoE operation:

```cpp
// Modified ggml_cuda_mul_mat_id with caching
void ggml_cuda_mul_mat_id_cached(
    ggml_backend_cuda_context& ctx,
    ggml_tensor* dst,
    ggml_moe_cache* cache
) {
    // Extract expert IDs from the operation
    const ggml_tensor* ids = dst->src[2];
    std::vector<int> expert_ids = extract_expert_ids(ids);
    
    // Pre-fetch predicted experts for next iteration
    std::vector<int> next_experts = cache->prefetch_engine.predict_next_experts(expert_ids);
    cache->prefetch_experts_async(next_experts, dst->src[0], ctx.stream());
    
    // Get current experts from cache (async)
    std::vector<ggml_backend_buffer_t> expert_buffers;
    for (int expert_id : expert_ids) {
        auto buffer = cache->get_expert_async(
            expert_id, 
            dst->src[0], 
            ctx.stream()
        );
        expert_buffers.push_back(buffer);
    }
    
    // Wait for experts to be ready and perform computation
    // ... existing mul_mat_id logic with cached buffers
    
    // Update LRU and statistics
    for (int expert_id : expert_ids) {
        cache->touch_expert(expert_id);
    }
}
```

## Data Flow

### Expert Access Flow
1. **Request**: MoE layer requests expert by ID
2. **Cache Check**: Look up expert in cache map
3. **Cache Hit**: Return GPU buffer immediately, update LRU
4. **Cache Miss**: 
   - Allocate GPU memory
   - Initiate async transfer from CPU to GPU
   - If needed, evict LRU experts first
   - Return buffer when ready
5. **Update Stats**: Record access for future predictions

### Pre-fetching Flow
1. **Pattern Analysis**: Analyze recent token/expert patterns
2. **Prediction**: Use co-occurrence and locality to predict next experts
3. **Async Transfer**: Start transferring predicted experts in background
4. **Validation**: Update prediction model based on actual usage

## Memory Management

### GPU Memory Allocation
- **Fixed Pool**: Pre-allocate GPU memory pool for experts
- **Dynamic Growth**: Allow cache to grow within configured limits
- **Eviction Policy**: LRU with frequency-based weighting

### CPU-GPU Transfers
- **Async Transfers**: Use dedicated CUDA streams for overlap
- **Pinned Memory**: Use pinned host memory for faster transfers
- **Batch Transfers**: Group multiple expert transfers when possible

## Configuration

Users can configure the cache through environment variables or API:

```cpp
// Environment variables
GGML_MOE_CACHE_SIZE=4GB          // Max GPU memory for cache
GGML_MOE_CACHE_POLICY=LRU        // Eviction policy
GGML_MOE_PREFETCH=true           // Enable pre-fetching
GGML_MOE_PREFETCH_DEPTH=3        // Number of experts to pre-fetch
GGML_MOE_STATS=true              // Enable statistics

// Runtime API
ggml_moe_cache_set_max_size(cache, 4ULL * 1024 * 1024 * 1024);
ggml_moe_cache_set_policy(cache, GGML_MOE_CACHE_POLICY_LRU);
ggml_moe_cache_enable_prefetch(cache, true);
```

## Performance Considerations

### Optimization Strategies
1. **Stream Overlap**: Use separate streams for compute and transfers
2. **Batch Operations**: Group expert operations when possible
3. **Pinned Memory**: Reuse pinned memory buffers to reduce allocation overhead
4. **Prediction Accuracy**: Continuously improve expert prediction algorithms

### Expected Performance
- **Cache Hit Rate**: 80-95% for typical inference patterns
- **Memory Savings**: 60-80% reduction in GPU memory usage
- **Speedup**: 2-5x faster inference vs CPU-only experts
- **Overhead**: <5% overhead for cache management

## Implementation Phases

### Phase 1: Basic LRU Cache
- Implement core cache with LRU eviction
- Integrate with CUDA backend
- Basic statistics collection

### Phase 2: Pre-fetching
- Add prediction engine
- Implement async pre-fetching
- Tune prediction algorithms

### Phase 3: Multi-Backend Support
- HIP backend implementation
- Other GPU backend support
- Unified interface

### Phase 4: Advanced Features
- Adaptive cache sizing
- Expert quantization in cache
- Distributed cache for multi-GPU

## Testing Strategy

1. **Unit Tests**: Cache operations, eviction, statistics
2. **Integration Tests**: End-to-end with MoE models
3. **Performance Tests**: Benchmark vs baseline
4. **Model Tests**: Mixtral, DeepSeek, Qwen, GLM-4-MoE
5. **Stress Tests**: High batch sizes, long sequences

## Future Enhancements

- **Multi-GPU Cache**: Distribute experts across multiple GPUs
- **Compression**: Compress cached experts to fit more in memory
- **Persistent Cache**: Save cache state between runs
- **Adaptive Sizing**: Automatically adjust cache size based on usage
- **Expert Fusion**: Combine similar experts in cache

## Conclusion

This MoE cache architecture provides a robust, efficient solution for running large MoE models on limited GPU memory. The combination of LRU eviction, intelligent pre-fetching, and backend-agnostic design ensures optimal performance across different hardware configurations.