# MoE Cache Improvement Design Document

## Executive Summary

This document outlines the design for improving the MoE (Mixture of Experts) caching system in llama.cpp to reduce CPU involvement and improve token generation throughput. The key improvements include separating layer offloading from expert caching, adding comprehensive debug logging, and optimizing data transfer paths.

## Current Issues

1. **Confused Layer Management**: The `-ngl` parameter currently includes both regular transformer layers AND MoE expert layers, leading to inefficient VRAM usage
2. **High CPU Involvement**: Excessive CPU activity during inference due to unnecessary data copying between system RAM and VRAM
3. **Lack of Visibility**: No debug output to track cache operations, making optimization difficult
4. **Suboptimal Data Paths**: CPU intervention in expert loading/unloading operations

## Proposed Architecture

### 1. Layer Separation Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    Model Architecture                        │
├─────────────────────────────────────────────────────────────┤
│  Input Layer (Embeddings)                                   │
│  ├─ Always on GPU (compute intensive)                      │
│  ├─ Managed by -ngl parameter                              │
├─────────────────────────────────────────────────────────────┤
│  Regular Transformer Layers (non-MoE)                       │
│  ├─ Counted by -ngl parameter                              │
│  ├─ Example: -ngl 20 = 20 regular layers on GPU            │
├─────────────────────────────────────────────────────────────┤
│  MoE Layers (with expert layers)                           │
│  ├─ NOT counted by -ngl parameter                          │
│  ├─ Managed by --moe-gpu-experts parameter                 │
│  ├─ Example: --moe-gpu-experts 50 = 50 experts cached     │
├─────────────────────────────────────────────────────────────┤
│  Output Layer                                               │
│  ├─ Always on GPU (compute intensive)                      │
│  ├─ Managed by -ngl parameter                              │
└─────────────────────────────────────────────────────────────┘
```

### 2. Parameter Redefinition

- **`-ngl N`**: Number of regular transformer layers to offload to GPU (excluding MoE expert layers)
- **`--moe-gpu-experts M`**: Number of MoE experts to cache in GPU VRAM
- **Total GPU VRAM usage**: Determined by both parameters independently

### 3. Cache Architecture Improvements

```cpp
// Enhanced cache structure with better separation
struct ggml_moe_cache {
    // Regular layer cache (managed by -ngl)
    std::unordered_map<int, ggml_backend_buffer_t> regular_layer_cache;
    
    // Expert cache (managed by --moe-gpu-experts)
    std::unordered_map<ggml_moe_expert_key, ggml_backend_buffer_t, ggml_moe_expert_key_hash> expert_cache;
    
    // Separate LRU tracking for better eviction policies
    std::list<int> regular_layer_lru;
    std::list<ggml_moe_expert_key> expert_lru;
    
    // Statistics with per-layer tracking
    ggml_moe_cache_stats stats;
    std::unordered_map<int, ggml_moe_layer_stats> layer_stats;
};
```

### 4. Debug Logging System

#### 4.1 Prefetch Logging
```cpp
// Log when experts are prefetched from system RAM to VRAM
void log_prefetch_operation(int layer_id, int expert_id, size_t data_size) {
    // Format: "[MoE Cache] Prefetching expert L<layer_id>.E<expert_id> (X.Y MB) from RAM to VRAM"
}
```

#### 4.2 Eviction Logging
```cpp
// Log when experts are evicted from VRAM
void log_eviction_operation(int layer_id, int expert_id, const std::string& reason) {
    // Format: "[MoE Cache] Evicting expert L<layer_id>.E<expert_id> (Reason: <reason>)"
}
```

#### 4.3 Statistics Reporting
```cpp
// Report after each prompt completion
void report_cache_statistics(const ggml_moe_cache* cache) {
    // Format:
    // "=== MoE Cache Statistics ==="
    // "Total Requests: X, Cache Hits: Y (Z%), Cache Misses: W"
    // "Prefetches: A, Prefetch Hits: B (C%)"
    // "Evictions: D, Current Cache Size: E MB"
    // "Layer-wise breakdown..."
}
```

### 5. CPU Involvement Reduction Strategies

#### 5.1 Direct GPU Transfers
- Implement GPU-to-GPU expert copying when source and destination are both GPU buffers
- Use async DMA operations for system RAM ↔ VRAM transfers
- Pin system RAM buffers for faster transfers

#### 5.2 Async Operations
```cpp
// Async expert loading
void async_load_expert(int layer_id, int expert_id, cudaStream_t stream) {
    // Launch async copy operation
    // Return immediately without CPU blocking
}

// Async prefetch operations
void async_prefetch_experts(int layer_id, const std::vector<int>& expert_ids) {
    // Launch multiple async operations
    // Overlap with computation
}
```

#### 5.3 Zero-Copy Operations
- Use unified memory where supported
- Implement zero-copy expert access patterns
- Minimize CPU cache invalidation

### 6. Input/Output Layer Optimization

#### 6.1 Embedding Layer
```cpp
// Always place embedding layer on GPU
void ensure_embedding_layer_on_gpu(llama_model& model) {
    if (model.embedding_layer.device != GPU) {
        // Force GPU placement for compute-intensive operations
        offload_tensor_to_gpu(model.embedding_layer);
    }
}
```

#### 6.2 Output Layer
```cpp
// Always place output layer on GPU
void ensure_output_layer_on_gpu(llama_model& model) {
    if (model.output_layer.device != GPU) {
        // Force GPU placement for compute-intensive operations
        offload_tensor_to_gpu(model.output_layer);
    }
}
```

## Implementation Phases

### Phase 1: Debug Logging Infrastructure
1. Add logging macros for cache operations
2. Implement statistics collection
3. Add post-prompt reporting

### Phase 2: Layer Separation
1. Modify `-ngl` parameter handling to exclude MoE experts
2. Implement separate expert counting with `--moe-gpu-experts`
3. Update layer assignment logic in `llama-model.cpp`

### Phase 3: CPU Optimization
1. Implement async expert loading
2. Add direct GPU-to-GPU transfer paths
3. Optimize data transfer pipelines

### Phase 4: Input/Output Layer Guarantees
1. Force GPU placement for embedding layer
2. Force GPU placement for output layer
3. Add validation checks

## Performance Metrics

### Primary Metrics
- **Tokens per second**: Target 20-30% improvement
- **CPU utilization**: Target 50% reduction during inference
- **Cache hit rate**: Target >85% for expert cache

### Secondary Metrics
- **VRAM usage efficiency**: Better distribution between layers and experts
- **Data transfer overhead**: Reduce CPU involvement in transfers by 70%
- **Latency**: Reduce inference latency for MoE models

## Testing Strategy

### Unit Tests
- Cache hit/miss tracking accuracy
- LRU eviction correctness
- Async operation completion

### Integration Tests
- End-to-end inference with various `-ngl` and `--moe-gpu-experts` combinations
- Performance regression testing
- Memory usage validation

### Benchmarking
- Compare tokens/second before and after changes
- Measure CPU utilization during inference
- Profile data transfer patterns

## Risk Mitigation

1. **Backward Compatibility**: Maintain existing parameter behavior as fallback
2. **Graceful Degradation**: If GPU memory is insufficient, fall back to CPU
3. **Error Handling**: Comprehensive error handling for async operations
4. **Performance Monitoring**: Continuous monitoring to detect regressions

## Future Enhancements

1. **ML-Enhanced Prefetching**: Implement the existing ML prefetching framework
2. **Dynamic Cache Sizing**: Automatically adjust cache size based on VRAM availability
3. **Multi-GPU Support**: Distribute experts across multiple GPUs
4. **Compression**: Implement expert compression for better VRAM utilization

## Conclusion

This design addresses the core issues with the current MoE caching system by:
1. Clearly separating layer offloading from expert caching
2. Adding comprehensive debug visibility
3. Optimizing data transfer paths to reduce CPU involvement
4. Ensuring compute-intensive layers remain on GPU

The implementation will result in significant performance improvements for MoE models while maintaining backward compatibility and providing better observability into cache operations.