// MoE (Mixture of Experts) family architectures
// Extracted from src/llama-model.cpp
// Contains: LLM_ARCH_MIXTRAL, LLM_ARCH_DBRX, LLM_ARCH_QWEN2MOE, LLM_ARCH_QWEN3MOE,
//           LLM_ARCH_QWEN3VLMOE, LLM_ARCH_DEEPSEEK2_MOE, LLM_ARCH_OLMOE, LLM_ARCH_PHIMOE,
//           LLM_ARCH_GLM4_MOE, LLM_ARCH_BAILINGMOE, LLM_ARCH_BAILINGMOE2, LLM_ARCH_OPENAI_MOE,
//           LLM_ARCH_LFM2MOE, LLM_ARCH_GROVEMOE, LLM_ARCH_NOMIC_BERT_MOE, LLM_ARCH_AFMOE,
//           LLM_ARCH_LLAMA4, LLM_ARCH_GRANITE_MOE, LLM_ARCH_GRANITE_HYBRID

#include "llama-model.h"
#include "llama-arch.h"
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
#include <cmath>
#include <map>
#include <stdexcept>

// Forward declarations
void llama_load_hparams_moe_family(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch, uint32_t n_vocab);
static const char * llama_expert_gating_func_name(llama_expert_gating_func_type type);
// llama_rope_scaling_type_name is now defined in llama-arch.cpp
// llama_rope_scaling_type_from_string is now defined in llama-arch.cpp
const char * llama_arch_name(llm_arch arch);

// Mixtral architecture hyperparameter loading
static void llama_load_hparams_mixtral(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);

    switch (hparams.n_layer) {
        case 32: type = LLM_TYPE_8x7B; break;
        case 56: type = LLM_TYPE_8x22B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// DBRX architecture hyperparameter loading
static void llama_load_hparams_dbrx(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
    ml.get_key(LLM_KV_ATTENTION_CLAMP_KQV,     hparams.f_clamp_kqv);

    switch (hparams.n_layer) {
        case 40: type = LLM_TYPE_16x12B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Qwen2MoE architecture hyperparameter loading
static void llama_load_hparams_qwen2moe(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    switch (hparams.n_layer) {
        case 24: type = LLM_TYPE_A2_7B; break;
        case 28: type = LLM_TYPE_57B_A14B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Qwen3MoE architecture hyperparameter loading
static void llama_load_hparams_qwen3moe(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp, false);

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    switch (hparams.n_layer) {
        case 48: type = LLM_TYPE_30B_A3B; break;
        case 94: type = LLM_TYPE_235B_A22B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Qwen3VLMoE architecture hyperparameter loading
static void llama_load_hparams_qwen3vlmoe(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_NUM_DEEPSTACK_LAYERS, hparams.n_deepstack_layers, false);
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, true);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    switch (hparams.n_layer) {
        case 48: type = LLM_TYPE_30B_A3B; break;
        case 94: type = LLM_TYPE_235B_A22B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// DeepSeek2 MoE architecture hyperparameter loading
static void llama_load_hparams_deepseek2_moe(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa, false);

    // Set up interleaved sliding window attention (ISWA)
    // Pattern: 3 sliding - 1 full (global_attn_every_n_layers = 4)
    if (hparams.n_swa > 0) {
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        hparams.set_swa_pattern(4);
    } else {
        hparams.swa_type = LLAMA_SWA_TYPE_NONE;
    }

    // Default to sigmoid if not set
    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
    }

    switch (hparams.n_layer) {
        case 56: type = LLM_TYPE_6B; break;
        case 32: type = LLM_TYPE_26B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// OLMoE architecture hyperparameter loading
static void llama_load_hparams_olmoe(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
    ml.get_key(LLM_KV_ATTENTION_CLAMP_KQV,     hparams.f_clamp_kqv, false);

    switch (hparams.n_layer) {
        case 16: type = LLM_TYPE_1B; break;
        case 32: type = LLM_TYPE_7B; break;
        case 80: type = LLM_TYPE_70B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// PhiMoE architecture hyperparameter loading
static void llama_load_hparams_phimoe(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer) {
        case 32: type = LLM_TYPE_16x3_8B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// GLM4 MoE architecture hyperparameter loading
static void llama_load_hparams_glm4_moe(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);

    // Default to sigmoid if not set
    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
    }

    switch (hparams.n_layer) {
        case 48: type = LLM_TYPE_106B_A12B; break;
        case 64: type = LLM_TYPE_355B_A32B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Bailing MoE architecture hyperparameter loading
static void llama_load_hparams_bailingmoe(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);

    // Default to sigmoid if not set
    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
    }

    switch (hparams.n_layer) {
        case 56: type = LLM_TYPE_6B; break;
        case 32: type = LLM_TYPE_26B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Bailing MoE v2 architecture hyperparameter loading
static void llama_load_hparams_bailingmoe2(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);

    // Default to sigmoid if not set
    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
    }

    switch (hparams.n_layer) {
        case 56: type = LLM_TYPE_6B; break;
        case 32: type = LLM_TYPE_26B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// OpenAI MoE architecture hyperparameter loading
static void llama_load_hparams_openai_moe(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp, false);

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    switch (hparams.n_layer) {
        case 48: type = LLM_TYPE_30B_A3B; break;
        case 94: type = LLM_TYPE_235B_A22B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// LFM2 MoE architecture hyperparameter loading
static void llama_load_hparams_lfm2moe(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp, false);

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    switch (hparams.n_layer) {
        case 48: type = LLM_TYPE_8B_A1B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Grove MoE architecture hyperparameter loading
static void llama_load_hparams_grovernoe(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp, false);

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    switch (hparams.n_layer) {
        case 48: type = LLM_TYPE_30B_A3B; break;
        case 94: type = LLM_TYPE_235B_A22B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Dispatch function for MoE family architectures
void llama_load_hparams_moe_family(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch, uint32_t n_vocab) {
    GGML_UNUSED(n_vocab);
    switch (arch) {
        case LLM_ARCH_MIXTRAL:      llama_load_hparams_mixtral(ml, hparams, type, arch); break;
        case LLM_ARCH_DBRX:         llama_load_hparams_dbrx(ml, hparams, type, arch); break;
        case LLM_ARCH_QWEN2MOE:     llama_load_hparams_qwen2moe(ml, hparams, type, arch); break;
        case LLM_ARCH_QWEN3MOE:     llama_load_hparams_qwen3moe(ml, hparams, type, arch); break;
        case LLM_ARCH_QWEN3VLMOE:   llama_load_hparams_qwen3vlmoe(ml, hparams, type, arch); break;
        case LLM_ARCH_DEEPSEEK2:    llama_load_hparams_deepseek2_moe(ml, hparams, type, arch); break;
        case LLM_ARCH_OLMOE:        llama_load_hparams_olmoe(ml, hparams, type, arch); break;
        case LLM_ARCH_PHIMOE:       llama_load_hparams_phimoe(ml, hparams, type, arch); break;
        case LLM_ARCH_GLM4_MOE:     llama_load_hparams_glm4_moe(ml, hparams, type, arch); break;
        case LLM_ARCH_BAILINGMOE:   llama_load_hparams_bailingmoe(ml, hparams, type, arch); break;
        case LLM_ARCH_BAILINGMOE2:  llama_load_hparams_bailingmoe2(ml, hparams, type, arch); break;
        case LLM_ARCH_OPENAI_MOE:   llama_load_hparams_openai_moe(ml, hparams, type, arch); break;
        case LLM_ARCH_LFM2MOE:      llama_load_hparams_lfm2moe(ml, hparams, type, arch); break;
        case LLM_ARCH_GROVEMOE:     llama_load_hparams_grovernoe(ml, hparams, type, arch); break;
        case LLM_ARCH_NOMIC_BERT_MOE: // Handled in BERT family
        case LLM_ARCH_AFMOE:         // Handled in LLaMA family
        case LLM_ARCH_LLAMA4:        // Handled in LLaMA family
        case LLM_ARCH_GRANITE_MOE:   // Handled in LLaMA family
        case LLM_ARCH_GRANITE_HYBRID: // Handled in LLaMA family
            break;
        default:
            throw std::runtime_error(format("Architecture %s not handled by MoE family loader", llama_arch_name(arch)));
    }
}
