#include "llama-moe-cache-params.h"
#include "llama.h"
#include "llama-impl.h"
#include "ggml.h"
#include <cstring>
#include <cstdlib>

// Define GGML_UNUSED if not already defined
#ifndef GGML_UNUSED
#define GGML_UNUSED(x) (void)(x)
#endif

// Implementation of the C API extensions for MoE cache parameters
// Note: These functions are declared in the header but should not have LLAMA_API here
// as they are not part of the main llama.h API

// Export helper wrappers so the symbols are available when building shared libs
LLAMA_API struct llama_context_params llama_context_default_params_with_moe(void) {
    // Start with default parameters
    struct llama_context_params params = llama_context_default_params();
    return params;
}

LLAMA_API void llama_context_params_set_moe_cache(struct llama_context_params* params,
                                                 const llama_moe_cache_params* moe_params) {
    // This function is deprecated and should not be used.
    // MoE cache parameters should be set directly in llama_model_params.
    // Keeping for backward compatibility but doing nothing.
    GGML_UNUSED(params);
    GGML_UNUSED(moe_params);
}

LLAMA_API void llama_model_params_set_moe_cache(struct llama_model_params* params,
                                               const llama_moe_cache_params* moe_params) {
    if (!params || !moe_params) {
        LLAMA_LOG_ERROR("%s: null parameters provided\n", __func__);
        return;
    }
    
    // Allocate memory for MoE cache parameters if not already allocated
    if (!params->moe_cache_params) {
        params->moe_cache_params = (llama_moe_cache_params*)malloc(sizeof(llama_moe_cache_params));
        if (!params->moe_cache_params) {
            LLAMA_LOG_ERROR("%s: failed to allocate memory for moe_cache_params\n", __func__);
            return;
        }
    }
    
    // Copy the provided parameters
    *params->moe_cache_params = *moe_params;
    
    LLAMA_LOG_INFO("%s: MoE cache parameters set - n_gpu_experts: %d, enable_cache: %s, enable_prefetch: %s\n",
                   __func__,
                   moe_params->n_gpu_experts,
                   moe_params->enable_cache ? "true" : "false",
                   moe_params->enable_prefetch ? "true" : "false");
}
