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

// Forward declarations - remove conflicting static declarations
static const char * llama_expert_gating_func_name(llama_expert_gating_func_type type);
static bool weight_buft_supported(const llama_hparams & hparams, ggml_tensor * w, ggml_op op, ggml_backend_buffer_type_t buft, ggml_backend_dev_t dev);
static ggml_backend_buffer_type_t select_weight_buft(const llama_hparams & hparams, ggml_tensor * tensor, ggml_op op, const std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>> & buft_list);

// External function declarations
extern const char * llama_arch_name(llm_arch arch);
extern llama_rope_scaling_type llama_rope_scaling_type_from_string(const std::string & name);

// Add missing function implementations
const char * llm_type_name(llm_type type) {
    switch (type) {
        case LLM_TYPE_14M: return "14M";
        case LLM_TYPE_17M: return "17M";
        case LLM_TYPE_22M: return "22M";
        case LLM_TYPE_33M: return "33M";
        case LLM_TYPE_60M: return "60M";
        case LLM_TYPE_70M: return "70M";
        case LLM_TYPE_80M: return "80M";
        case LLM_TYPE_109M: return "109M";
        case LLM_TYPE_137M: return "137M";
        case LLM_TYPE_140M: return "140M";
        case LLM_TYPE_160M: return "160M";
        case LLM_TYPE_190M: return "190M";
        case LLM_TYPE_220M: return "220M";
        case LLM_TYPE_250M: return "250M";
        case LLM_TYPE_256M: return "256M";
        case LLM_TYPE_270M: return "270M";
        case LLM_TYPE_335M: return "335M";
        case LLM_TYPE_350M: return "350M";
        case LLM_TYPE_360M: return "360M";
        case LLM_TYPE_410M: return "410M";
        case LLM_TYPE_450M: return "450M";
        case LLM_TYPE_475M: return "475M";
        case LLM_TYPE_558M: return "558M";
        case LLM_TYPE_700M: return "700M";
        case LLM_TYPE_770M: return "770M";
        case LLM_TYPE_780M: return "780M";
        case LLM_TYPE_950M: return "950M";
        case LLM_TYPE_0_3B: return "0.3B";
        case LLM_TYPE_0_5B: return "0.5B";
        case LLM_TYPE_0_6B: return "0.6B";
        case LLM_TYPE_1B: return "1B";
        case LLM_TYPE_1_2B: return "1.2B";
        case LLM_TYPE_1_3B: return "1.3B";
        case LLM_TYPE_1_4B: return "1.4B";
        case LLM_TYPE_1_5B: return "1.5B";
        case LLM_TYPE_1_6B: return "1.6B";
        case LLM_TYPE_1_7B: return "1.7B";
        case LLM_TYPE_1_8B: return "1.8B";
        case LLM_TYPE_2B: return "2B";
        case LLM_TYPE_2_6B: return "2.6B";
        case LLM_TYPE_2_8B: return "2.8B";
        case LLM_TYPE_2_9B: return "2.9B";
        case LLM_TYPE_3B: return "3B";
        case LLM_TYPE_4B: return "4B";
        case LLM_TYPE_6B: return "6B";
        case LLM_TYPE_6_9B: return "6.9B";
        case LLM_TYPE_7B: return "7B";
        case LLM_TYPE_8B: return "8B";
        case LLM_TYPE_9B: return "9B";
        case LLM_TYPE_11B: return "11B";
        case LLM_TYPE_12B: return "12B";
        case LLM_TYPE_13B: return "13B";
        case LLM_TYPE_14B: return "14B";
        case LLM_TYPE_15B: return "15B";
        case LLM_TYPE_16B: return "16B";
        case LLM_TYPE_20B: return "20B";
        case LLM_TYPE_26B: return "26B";
        case LLM_TYPE_27B: return "27B";
        case LLM_TYPE_30B: return "30B";
        case LLM_TYPE_32B: return "32B";
        case LLM_TYPE_34B: return "34B";
        case LLM_TYPE_35B: return "35B";
        case LLM_TYPE_36B: return "36B";
        case LLM_TYPE_40B: return "40B";
        case LLM_TYPE_65B: return "65B";
        case LLM_TYPE_70B: return "70B";
        case LLM_TYPE_120B: return "120B";
        case LLM_TYPE_142B: return "142B";
        case LLM_TYPE_236B: return "236B";
        case LLM_TYPE_290B: return "290B";
        case LLM_TYPE_314B: return "314B";
        case LLM_TYPE_405B: return "405B";
        case LLM_TYPE_671B: return "671B";
        case LLM_TYPE_SMALL: return "Small";
        case LLM_TYPE_MEDIUM: return "Medium";
        case LLM_TYPE_LARGE: return "Large";
        case LLM_TYPE_XL: return "XL";
        case LLM_TYPE_A1_7B: return "A1.7B";
        case LLM_TYPE_A2_7B: return "A2.7B";
        case LLM_TYPE_8x7B: return "8x7B";
        case LLM_TYPE_8x22B: return "8x22B";
        case LLM_TYPE_16x12B: return "16x12B";
        case LLM_TYPE_16x3_8B: return "16x3.8B";
        case LLM_TYPE_10B_128x3_66B: return "10B_128x3.66B";
        case LLM_TYPE_57B_A14B: return "57B_A14B";
        case LLM_TYPE_17B_16E: return "17B_16E";
        case LLM_TYPE_17B_128E: return "17B_128E";
        case LLM_TYPE_A13B: return "A13B";
        case LLM_TYPE_7B_A1B: return "7B_A1B";
        case LLM_TYPE_8B_A1B: return "8B_A1B";
        case LLM_TYPE_16B_A1B: return "16B_A1B";
        case LLM_TYPE_21B_A3B: return "21B_A3B";
        case LLM_TYPE_30B_A3B: return "30B_A3B";
        case LLM_TYPE_100B_A6B: return "100B_A6B";
        case LLM_TYPE_106B_A12B: return "106B_A12B";
        case LLM_TYPE_230B_A10B: return "230B_A10B";
        case LLM_TYPE_235B_A22B: return "235B_A22B";
        case LLM_TYPE_300B_A47B: return "300B_A47B";
        case LLM_TYPE_355B_A32B: return "355B_A32B";
        case LLM_TYPE_E2B: return "E2B";
        case LLM_TYPE_E4B: return "E4B";
        default: return "Unknown";
    }
}

// llama_rope_scaling_type_name is now defined in llama-model-base.cpp only

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
    // rope_freq_scale is calculated in get_rope_freq_scale, not needed here

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
    const float rope_freq_scale_val = is_swa_layer ? hparams.rope_freq_scale_train_swa : hparams.rope_freq_scale_train;

    if (cparams.rope_freq_scale != 0.0f) {
        return cparams.rope_freq_scale;
    }

    return rope_freq_scale_val;
}

ggml_tensor * llama_model::get_rope_factors(const llama_cparams & cparams, int il) const {
    // Suppress unused parameter warning
    (void)cparams;
    
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
    // Suppress unused parameter warning
    (void)cparams;
    
    // Dispatch to appropriate memory implementation based on architecture
    if (llm_arch_is_recurrent(arch)) {
        // For recurrent architectures, create llama_memory_recurrent
        // Extract parameters from mparams and model
        return new llama_memory_recurrent(
            *this,                    // const llama_model & model
            mparams.type_k,           // ggml_type type_r
            mparams.type_v,           // ggml_type type_s
            true,                     // bool offload
            hparams.n_layer,          // uint32_t mem_size (using n_layer as approximation)
            cparams.n_seq_max,        // uint32_t n_seq_max
            nullptr                   // const layer_filter_cb & filter
        );
    } else if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        // For architectures with SWA, create llama_memory_hybrid
        return new llama_memory_hybrid(
            *this,                    // const llama_model & model
            // Attention parameters
            mparams.type_k,           // ggml_type type_k
            mparams.type_v,           // ggml_type type_v
            true,                     // bool v_trans
            hparams.n_layer,          // uint32_t kv_size
            0,                        // uint32_t n_pad
            hparams.n_swa,            // uint32_t n_swa
            hparams.swa_type,         // llama_swa_type swa_type
            // Recurrent parameters
            mparams.type_k,           // ggml_type type_r
            mparams.type_v,           // ggml_type type_s
            hparams.n_layer,          // uint32_t rs_size
            // Common parameters
            cparams.n_seq_max,        // uint32_t n_seq_max
            true,                     // bool offload
            false,                    // bool unified
            nullptr,                  // const layer_filter_cb & filter_attn
            nullptr                   // const layer_filter_cb & filter_recr
        );
    } else {
        // Default to KV cache for standard attention-based architectures
        return new llama_kv_cache(
            *this,                    // const llama_model & model
            mparams.type_k,           // ggml_type type_k
            mparams.type_v,           // ggml_type type_v
            true,                     // bool v_trans
            true,                     // bool offload
            false,                    // bool unified
            hparams.n_layer,          // uint32_t kv_size
            cparams.n_seq_max,        // uint32_t n_seq_max
            0,                        // uint32_t n_pad
            0,                        // uint32_t n_swa
            LLAMA_SWA_TYPE_NONE,      // llama_swa_type swa_type
            nullptr,                  // const layer_filter_cb & filter
            nullptr                   // const layer_reuse_cb & reuse
        );
    }
}

// Graph building
ggml_cgraph * llama_model::build_graph(const llm_graph_params & params) const {
    // Suppress unused parameter warning
    (void)params;
    
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