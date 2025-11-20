// Utility functions for llama-model
// Extracted from src/llama-model.cpp
// Contains: Device management, logging, buffer placement, common helpers

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
#include "ggml-cpp.h"
#include "models/models.h"
#include <algorithm>
#include <map>
#include <vector>
#include <string>

// Forward declarations
static const char * llm_type_name(llm_type type);
static const char * llama_expert_gating_func_name(llama_expert_gating_func_type type);
static std::string llama_rope_scaling_type_name(llama_rope_scaling_type rope_scaling_type);
static llama_rope_scaling_type llama_rope_scaling_type_from_string(const std::string & name);
static bool weight_buft_supported(const llama_hparams & hparams, ggml_tensor * w, ggml_op op, ggml_backend_buffer_type_t buft, ggml_backend_dev_t dev);
static ggml_backend_buffer_type_t select_weight_buft(const llama_hparams & hparams, ggml_tensor * tensor, ggml_op op, const std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>> & buft_list);

// Device management functions
ggml_backend_dev_t llama_model::dev_layer(int il) const {
    if (il < 0 || (size_t)il >= pimpl->dev_layer.size()) {
        return nullptr;
    }
    return pimpl->dev_layer[il].dev;
}

ggml_backend_dev_t llama_model::dev_output() const {
    return pimpl->dev_output.dev;
}

ggml_backend_buffer_type_t llama_model::select_buft(int il) const {
    if (il < 0 || (size_t)il >= pimpl->dev_layer.size()) {
        return nullptr;
    }
    return pimpl->dev_layer[il].buft_list->front().second;
}

// Device placement logging
static void llama_moe_log_layer_assignment(int il, ggml_backend_dev_t dev, const char * layer_name) {
    if (dev) {
        LLAMA_LOG_INFO("%s: layer %d (%s) assigned to %s\n", __func__, il, layer_name, ggml_backend_dev_name(dev));
    } else {
        LLAMA_LOG_INFO("%s: layer %d (%s) assigned to CPU\n", __func__, il, layer_name);
    }
}

// Force GPU placement for compute-intensive layers
void llama_model::ensure_embedding_layer_on_gpu() {
    if (pimpl->dev_input.dev && pimpl->dev_input.dev->type == GGML_BACKEND_DEVICE_TYPE_GPU) {
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
    if (pimpl->dev_output.dev && pimpl->dev_output.dev->type == GGML_BACKEND_DEVICE_TYPE_GPU) {
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
    if (!pimpl->dev_input.dev || pimpl->dev_input.dev->type != GGML_BACKEND_DEVICE_TYPE_GPU) {
        LLAMA_LOG_WARN("%s: embedding layer not on GPU\n", __func__);
        valid = false;
    }
    
    // Check output layer
    if (!pimpl->dev_output.dev || pimpl->dev_output.dev->type != GGML_BACKEND_DEVICE_TYPE_GPU) {
        LLAMA_LOG_WARN("%s: output layer not on GPU\n", __func__);
        valid = false;
    }
    
    // Check at least some layers are on GPU
    bool has_gpu_layers = false;
    for (const auto & layer_dev : pimpl->dev_layer) {
        if (layer_dev.dev && layer_dev.dev->type == GGML_BACKEND_DEVICE_TYPE_GPU) {
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

// Tensor lookup
const struct ggml_tensor * llama_model::get_tensor(const char * name) const {
    for (const auto & cb : pimpl->ctxs_bufs) {
        ggml_tensor * tensor = ggml_get_tensor(cb.first.get(), name);
        if (tensor) {
            return tensor;
        }
    }
    return nullptr;
}

// RoPE frequency calculations
float llama_model::get_rope_freq_base(const llama_cparams & cparams, int il) const {
    if (il < 0 || (size_t)il >= hparams.n_layer) {
        return hparams.rope_freq_base_train;
    }

    const bool is_swa_layer = hparams.swa_layers[il];
    const float rope_freq_base  = is_swa_layer ? hparams.rope_freq_base_train_swa  : hparams.rope_freq_base_train;
    const float rope_freq_scale = is_swa_layer ? hparams.rope_freq_scale_train_swa : hparams.rope_freq_scale_train;

    if (cparams.rope_freq_base != 0.0f) {
        return cparams.rope_freq_base;
    }

    return rope_freq_base;
}

float llama_model::get_rope_freq_scale(const llama_cparams & cparams, int il) const {
    if (il < 0 || (size_t)il >= hparams.n_layer) {
        return hparams.rope_freq_scale_train;
    }

    const bool is_swa_layer = hparams.swa_layers[il];
    const float rope_freq_scale = is_swa_layer ? hparams.rope_freq_scale_train_swa : hparams.rope_freq_scale_train;

    if (cparams.rope_freq_scale != 0.0f) {
        return cparams.rope_freq_scale;
    }

    return rope_freq_scale;
}

ggml_tensor * llama_model::get_rope_factors(const llama_cparams & cparams, int il) const {
    if (il < 0 || (size_t)il >= layers.size()) {
        return nullptr;
    }

    const bool is_swa_layer = hparams.swa_layers[il];
    if (is_swa_layer) {
        return layers[il].rope_long;
    }

    return layers[il].rope_short;
}

// Memory creation
llama_memory_i * llama_model::create_memory(const llama_memory_params & mparams, const llama_cparams & cparams) const {
    // Dispatch to appropriate memory implementation based on architecture
    if (llm_arch_is_recurrent(arch)) {
        return new llama_memory_recurrent(mparams, cparams, hparams);
    } else if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        return new llama_memory_hybrid(mparams, cparams, hparams);
    } else {
        return new llama_memory_kv_cache(mparams, cparams, hparams);
    }
}

// Graph building
ggml_cgraph * llama_model::build_graph(const llm_graph_params & params) const {
    // Graph building implementation will be architecture-specific
    // This is a placeholder that should be overridden by architecture-specific implementations
    GGML_ABORT("build_graph not implemented for architecture %s", llama_arch_name(arch));
    return nullptr;
}

// Internal tensor map access
const std::vector<std::pair<std::string, ggml_tensor *>> & llama_internal_get_tensor_map(const llama_model * model) {
    static std::vector<std::pair<std::string, ggml_tensor *>> tensor_map;
    tensor_map.clear();
    
    for (const auto & cb : model->pimpl->ctxs_bufs) {
        ggml_context * ctx = cb.first.get();
        for (ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor)) {
            tensor_map.emplace_back(tensor->name, tensor);
        }
    }
    
    return tensor_map;
}