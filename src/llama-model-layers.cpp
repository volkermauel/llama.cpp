// Layer management and tensor creation
// Extracted from src/llama-model.cpp
// Contains: Tensor creation functions, layer organization, device assignment

#include "llama-model.h"
#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-model-loader.h"
#include "llama-moe-cache-params.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"
#include "llama-memory.h"
#include "ggml-cpp.h"
#include "models/models.h"
#include <algorithm>
#include <map>
#include <vector>
#include <string>

// External function declarations
extern const char * llama_arch_name(llm_arch arch);
extern llama_rope_scaling_type llama_rope_scaling_type_from_string(const std::string & name);

// Shared helpers are implemented in llama-model.cpp to avoid duplicate symbols.

// Force GPU placement for compute-intensive layers
void llama_model::ensure_embedding_layer_on_gpu() {
    if (pimpl->dev_input.dev && ggml_backend_dev_type(pimpl->dev_input.dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
        // Already on GPU
        return;
    }
    
    // Find first GPU device
    for (auto & dev : pimpl->devices) {
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            pimpl->dev_input = {dev, &pimpl->gpu_buft_list[dev]};
            LLAMA_LOG_INFO("%s: embedding layer moved to GPU\n", __func__);
            return;
        }
    }
}

void llama_model::ensure_output_layer_on_gpu() {
    if (pimpl->dev_output.dev && ggml_backend_dev_type(pimpl->dev_output.dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
        // Already on GPU
        return;
    }
    
    // Find first GPU device
    for (auto & dev : pimpl->devices) {
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            pimpl->dev_output = {dev, &pimpl->gpu_buft_list[dev]};
            LLAMA_LOG_INFO("%s: output layer moved to GPU\n", __func__);
            return;
        }
    }
}

bool llama_model::validate_critical_layers_on_gpu() const {
    bool valid = true;
    
    // Check embedding layer
    if (!pimpl->dev_input.dev || ggml_backend_dev_type(pimpl->dev_input.dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        LLAMA_LOG_WARN("%s: embedding layer not on GPU\n", __func__);
        valid = false;
    }
    
    // Check output layer
    if (!pimpl->dev_output.dev || ggml_backend_dev_type(pimpl->dev_output.dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        LLAMA_LOG_WARN("%s: output layer not on GPU\n", __func__);
        valid = false;
    }
    
    // Check at least some layers are on GPU
    bool has_gpu_layers = false;
    for (const auto & layer_dev : pimpl->dev_layer) {
        if (layer_dev.dev && ggml_backend_dev_type(layer_dev.dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            has_gpu_layers = true;
            break;
        }
    }
    
    if (!has_gpu_layers) {
        LLAMA_LOG_WARN("%s: no layers assigned to GPU\n", __func__);
        valid = false;
    }
    
    return valid;
}

