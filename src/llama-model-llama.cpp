// LLaMA family architectures
// Extracted from src/llama-model.cpp
// Contains: LLM_ARCH_LLAMA, LLM_ARCH_LLAMA4, LLM_ARCH_DECI, LLM_ARCH_FALCON, 
//           LLM_ARCH_BAICHUAN, LLM_ARCH_GROK, LLM_ARCH_ARCEE, LLM_ARCH_AFMOE

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
#include <cassert>
#include <cmath>
#include <map>
#include <stdexcept>

// Forward declarations for functions used in LLaMA architectures
const char * llama_arch_name(llm_arch arch);

// LLaMA architecture hyperparameter loading
static void llama_load_hparams_llama(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch, uint32_t n_vocab) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    if (hparams.n_expert == 8) {
        switch (hparams.n_layer) {
            case 32: type = LLM_TYPE_8x7B; break;
            case 56: type = LLM_TYPE_8x22B; break;
            default: type = LLM_TYPE_UNKNOWN;
        }
    } else {
        switch (hparams.n_layer) {
            case 16: type = LLM_TYPE_1B; break; // Llama 3.2 1B
            case 22: type = LLM_TYPE_1B; break;
            case 26: type = LLM_TYPE_3B; break;
            case 28: type = LLM_TYPE_3B; break; // Llama 3.2 3B
            case 30: type = LLM_TYPE_256M; break; // smoldocling 256M
            // granite uses a vocab with len 49152
            case 32: type = n_vocab == 49152 ? LLM_TYPE_3B : (n_vocab < 40000 ? LLM_TYPE_7B : LLM_TYPE_8B); break;
            case 36: type = LLM_TYPE_8B; break; // granite
            case 40: type = LLM_TYPE_13B; break;
            case 48: type = LLM_TYPE_34B; break;
            case 60: type = LLM_TYPE_30B; break;
            case 80: type = hparams.n_head() == hparams.n_head_kv() ? LLM_TYPE_65B : LLM_TYPE_70B; break;
            default: type = LLM_TYPE_UNKNOWN;
        }
    }
}

// LLaMA4 architecture hyperparameter loading
static void llama_load_hparams_llama4(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,   hparams.n_moe_layer_step);

    const bool found_swa = ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
    if (found_swa && hparams.n_swa == 0) {
        hparams.swa_type             = LLAMA_SWA_TYPE_NONE;
        hparams.n_no_rope_layer_step = hparams.n_layer; // always use rope
    } else {
        hparams.swa_type      = LLAMA_SWA_TYPE_CHUNKED;
        hparams.n_swa         = 8192;
        hparams.set_swa_pattern(4);   // pattern: 3 chunked - 1 full
    }

    switch (hparams.n_expert) {
        case 0: {
            // MobileLLM (no MoE)
            switch (hparams.n_embd) {
                case 2048: type = LLM_TYPE_140M; break;
                case 4096: type = LLM_TYPE_360M; break;
                case 6144: type = LLM_TYPE_950M; break;
                default:   type = LLM_TYPE_UNKNOWN;
            }
        } break;
        case 16:  type = LLM_TYPE_17B_16E; break;
        case 128: type = LLM_TYPE_17B_128E; break;
        default:  type = LLM_TYPE_UNKNOWN;
    }

    hparams.use_kq_norm = type != LLM_TYPE_17B_128E;
}

// Arcee architecture hyperparameter loading
static void llama_load_hparams_arcee(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    // Arcee uses the same structure as Llama
    switch (hparams.n_layer) {
        case 36: type = LLM_TYPE_4B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// AFMoE architecture hyperparameter loading
static void llama_load_hparams_afmoe(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
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

// Deci architecture hyperparameter loading
static void llama_load_hparams_deci(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    switch (hparams.n_layer) {
        case 32: type = LLM_TYPE_7B; break;
        case 80: type = LLM_TYPE_70B; break;
        case 162: type = LLM_TYPE_405B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Grok architecture hyperparameter loading
static void llama_load_hparams_grok(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    // defaults for old GGUFs
    hparams.yarn_beta_fast = 8.0f;
    hparams.f_logit_scale = 0.5773502691896257f;
    hparams.f_embedding_scale = 78.38367176906169f;
    hparams.f_attn_out_scale = 0.08838834764831845f;
    hparams.f_attn_logit_softcapping = 30.0f;
    hparams.f_router_logit_softcapping = 30.0f;
    // no final_logit_softcapping in grok-1
    hparams.f_final_logit_softcapping = 0.0f;

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,  hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,   hparams.n_ff_exp, false);
    ml.get_key(LLM_KV_LOGIT_SCALE,                  hparams.f_logit_scale, false);
    ml.get_key(LLM_KV_EMBEDDING_SCALE,              hparams.f_embedding_scale, false);
    ml.get_key(LLM_KV_ATTENTION_OUTPUT_SCALE,       hparams.f_attn_out_scale, false);
    ml.get_key(LLM_KV_ATTN_LOGIT_SOFTCAPPING,       hparams.f_attn_logit_softcapping, false);
    ml.get_key(LLM_KV_ROUTER_LOGIT_SOFTCAPPING,     hparams.f_router_logit_softcapping, false);
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,      hparams.f_final_logit_softcapping, false);

    ml.get_key(LLM_KV_ATTENTION_TEMPERATURE_LENGTH,  hparams.attn_temp_length, false);
    ml.get_key(LLM_KV_ROPE_SCALING_YARN_EXT_FACTOR,  hparams.yarn_ext_factor, false);
    ml.get_key(LLM_KV_ROPE_SCALING_YARN_ATTN_FACTOR, hparams.yarn_attn_factor, false);
    ml.get_key(LLM_KV_ROPE_SCALING_YARN_BETA_FAST,   hparams.yarn_beta_fast, false);
    ml.get_key(LLM_KV_ROPE_SCALING_YARN_BETA_SLOW,   hparams.yarn_beta_slow, false);

    switch (hparams.n_layer) {
        case 64: type = LLM_TYPE_314B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Falcon architecture hyperparameter loading
static void llama_load_hparams_falcon(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

    switch (hparams.n_layer) {
        case 32: type = LLM_TYPE_7B; break;
        case 60: type = LLM_TYPE_40B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Baichuan architecture hyperparameter loading
static void llama_load_hparams_baichuan(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    switch (hparams.n_layer) {
        case 32: type = LLM_TYPE_7B; break;
        case 40: type = LLM_TYPE_13B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }

    if (type == LLM_TYPE_13B) {
        // TODO: become GGUF KV parameter
        hparams.f_max_alibi_bias = 8.0f;
    }
}

// MiniCPM architecture hyperparameter loading
static void llama_load_hparams_minicpm(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    // Backward-compatible defaults for older MiniCPM GGUFs
    hparams.f_embedding_scale = 12.0f;
    hparams.f_residual_scale  = 1.4f / sqrtf(float(hparams.n_layer));
    hparams.f_logit_scale     = hparams.n_embd ? (256.0f / float(hparams.n_embd)) : 1.0f;

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    // Optional KV reads, override defaults if present in newer GGUF exports
    ml.get_key(LLM_KV_EMBEDDING_SCALE, hparams.f_embedding_scale, /*required=*/false);
    ml.get_key(LLM_KV_RESIDUAL_SCALE, hparams.f_residual_scale, /*required=*/false);
    ml.get_key(LLM_KV_LOGIT_SCALE, hparams.f_logit_scale, /*required=*/false);

    // MiniCPM uses rope by default, unlike Granite which uses it as a switch
    hparams.rope_finetuned = true;

    switch (hparams.n_layer) {
        case 52: type = LLM_TYPE_1B; break;
        case 40: type = LLM_TYPE_2B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// MiniCPM3 architecture hyperparameter loading
static void llama_load_hparams_minicpm3(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,      hparams.n_lora_kv);

    switch (hparams.n_layer) {
        case 62: type = LLM_TYPE_4B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Granite architecture hyperparameter loading
static void llama_load_hparams_granite(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch, uint32_t n_vocab) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer) {
        case 32: type = n_vocab == 49152 ? LLM_TYPE_3B : (n_vocab < 40000 ? LLM_TYPE_7B : LLM_TYPE_8B); break;
        case 36: type = LLM_TYPE_8B; break; // granite
        case 40: type = LLM_TYPE_13B; break;
        case 48: type = LLM_TYPE_34B; break;
        case 52: type = LLM_TYPE_20B; break; // granite
        case 60: type = LLM_TYPE_30B; break;
        case 88: type = LLM_TYPE_34B; break; // granite
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Dispatch function for LLaMA family architectures
void llama_load_hparams_llama_family(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch, uint32_t n_vocab) {
    switch (arch) {
        case LLM_ARCH_LLAMA:     llama_load_hparams_llama(ml, hparams, type, arch, n_vocab); break;
        case LLM_ARCH_LLAMA4:    llama_load_hparams_llama4(ml, hparams, type, arch); break;
        case LLM_ARCH_ARCEE:     llama_load_hparams_arcee(ml, hparams, type, arch); break;
        case LLM_ARCH_AFMOE:     llama_load_hparams_afmoe(ml, hparams, type, arch); break;
        case LLM_ARCH_DECI:      llama_load_hparams_deci(ml, hparams, type, arch); break;
        case LLM_ARCH_GROK:      llama_load_hparams_grok(ml, hparams, type, arch); break;
        case LLM_ARCH_FALCON:    llama_load_hparams_falcon(ml, hparams, type, arch); break;
        case LLM_ARCH_BAICHUAN:  llama_load_hparams_baichuan(ml, hparams, type, arch); break;
        case LLM_ARCH_MINICPM:   llama_load_hparams_minicpm(ml, hparams, type, arch); break;
        case LLM_ARCH_MINICPM3:  llama_load_hparams_minicpm3(ml, hparams, type, arch); break;
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_GRANITE_HYBRID:
            llama_load_hparams_granite(ml, hparams, type, arch, n_vocab);
            break;
        default:
            throw std::runtime_error(format("Architecture %s not handled by LLaMA family loader", llama_arch_name(arch)));
    }
}