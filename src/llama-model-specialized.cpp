// Specialized architectures
// Extracted from src/llama-model.cpp
// Contains: LLM_ARCH_MAMBA, LLM_ARCH_MAMBA2, LLM_ARCH_JAMBA, LLM_ARCH_T5,
//           LLM_ARCH_CHATGLM, LLM_ARCH_GLM4, LLM_ARCH_RWKV6, LLM_ARCH_RWKV7,
//           LLM_ARCH_PLAMO2, LLM_ARCH_XVERSE, LLM_ARCH_COMMAND_R, LLM_ARCH_COHERE2,
//           LLM_ARCH_OLMO, LLM_ARCH_OLMO2, LLM_ARCH_SEED_OSS, LLM_ARCH_PLM,
//           LLM_ARCH_BITNET, LLM_ARCH_INTERNLM2, LLM_ARCH_PHI2, LLM_ARCH_PHI3,
//           LLM_ARCH_PLAMO, LLM_ARCH_DREAM, LLM_ARCH_LLADA, LLM_ARCH_LLADA_MOE,
//           LLM_ARCH_QWEN, LLM_ARCH_QWEN2, LLM_ARCH_QWEN2VL, LLM_ARCH_QWEN3,
//           LLM_ARCH_QWEN3VL, LLM_ARCH_GEMMA, LLM_ARCH_GEMMA2, LLM_ARCH_GEMMA3,
//           LLM_ARCH_GEMMA3N, LLM_ARCH_GEMMA_EMBEDDING, LLM_ARCH_STARCODER2,
//           LLM_ARCH_JAIS, LLM_ARCH_NEMOTRON, LLM_ARCH_NEMOTRON_H, LLM_ARCH_EXAONE,
//           LLM_ARCH_EXAONE4, LLM_ARCH_CHAMELEON, LLM_ARCH_WAVTOKENIZER_DEC,
//           LLM_ARCH_BAILINGMOE, LLM_ARCH_BAILINGMOE2, LLM_ARCH_DOTS1,
//           LLM_ARCH_ERNIE4_5, LLM_ARCH_ERNIE4_5_MOE, LLM_ARCH_FALCON_H1,
//           LLM_ARCH_HUNYUAN_MOE, LLM_ARCH_HUNYUAN_DENSE, LLM_ARCH_SMOLLM3,
//           LLM_ARCH_OPENAI_MOE, LLM_ARCH_LFM2, LLM_ARCH_LFM2MOE,
//           LLM_ARCH_SMALLTHINKER, LLM_ARCH_GROVEMOE, LLM_ARCH_APERTUS,
//           LLM_ARCH_MINIMAX_M2, LLM_ARCH_COGVLM, LLM_ARCH_PANGU_EMBED,
//           LLM_ARCH_ARCEE, LLM_ARCH_AFMOE

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
const char * llama_arch_name(llm_arch arch);
void llama_load_hparams_specialized(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch, uint32_t n_vocab);

// Mamba architecture hyperparameter loading
static void llama_load_hparams_mamba(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_DT_B_C_RMS,     hparams.ssm_dt_b_c_rms, false);

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer) {
        case 24:
            switch (hparams.n_embd) {
                case 768: type = LLM_TYPE_SMALL; break;
                default: type = LLM_TYPE_UNKNOWN;
            } break;
        case 48:
            switch (hparams.n_embd) {
                case 1024: type = LLM_TYPE_MEDIUM; break;
                case 1536: type = LLM_TYPE_LARGE; break;
                case 2048: type = LLM_TYPE_XL; break;
                default:   type = LLM_TYPE_UNKNOWN;
            } break;
        case 64:
            switch (hparams.n_embd) {
                case 2560: type = LLM_TYPE_3B; break;
                default: type = LLM_TYPE_UNKNOWN;
            } break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Mamba2 architecture hyperparameter loading
static void llama_load_hparams_mamba2(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer) {
        case 24:
            switch (hparams.n_embd) {
                case 768: type = LLM_TYPE_SMALL; break;
                default: type = LLM_TYPE_UNKNOWN;
            } break;
        case 48:
            switch (hparams.n_embd) {
                case 1024: type = LLM_TYPE_MEDIUM; break;
                case 1536: type = LLM_TYPE_LARGE; break;
                case 2048: type = LLM_TYPE_XL; break;
                default: type = LLM_TYPE_UNKNOWN;
            } break;
        case 64:
            switch (hparams.n_embd) {
                case 2560: type = LLM_TYPE_3B; break;
                case 4096: type = LLM_TYPE_7B; break;
                default: type = LLM_TYPE_UNKNOWN;
            } break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Jamba architecture hyperparameter loading
static void llama_load_hparams_jamba(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
        hparams.recurrent_layer_arr[i] = hparams.n_head_kv(i) == 0;
    }

    switch (hparams.n_layer) {
        // TODO: Jamba layers are a bit heterogenous, so naming this is hard.
        case 12: // 900M  8x???M
        case 32: // 51B  16x?B
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// T5 architecture hyperparameter loading
static void llama_load_hparams_t5(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_CAUSAL,            hparams.causal_attn);
    ml.get_key(LLM_KV_ATTENTION_CLAMP_KQV,         hparams.f_clamp_kqv, false);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH,        hparams.n_embd_head_k, false);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH,      hparams.n_embd_head_v, false);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT,  hparams.n_rel_attn_bkts, false);
    ml.get_key(LLM_KV_DECODER_START_TOKEN_ID,      hparams.dec_start_token_id, false);
    ml.get_key(LLM_KV_POOLING_TYPE,                hparams.pooling_type, false);

    switch (hparams.n_layer) {
        case 24: type = LLM_TYPE_3B; break;
        case 48: type = LLM_TYPE_11B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// ChatGLM architecture hyperparameter loading
static void llama_load_hparams_chatglm(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_CLAMP_KQV,         hparams.f_clamp_kqv, false);
    ml.get_key(LLM_KV_ATTENTION_MAX_ALIBI_BIAS,    hparams.f_max_alibi_bias);

    switch (hparams.n_layer) {
        case 28: type = LLM_TYPE_6B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// GLM4 architecture hyperparameter loading
static void llama_load_hparams_glm4(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_CLAMP_KQV,         hparams.f_clamp_kqv, false);
    ml.get_key(LLM_KV_ATTENTION_MAX_ALIBI_BIAS,    hparams.f_max_alibi_bias);

    switch (hparams.n_layer) {
        case 30: type = LLM_TYPE_9B; break;
        case 40: type = LLM_TYPE_9B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// RWKV6 architecture hyperparameter loading
static void llama_load_hparams_rwkv6(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

    switch (hparams.n_layer) {
        case 24: type = LLM_TYPE_3B; break;
        case 32: type = LLM_TYPE_7B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// RWKV7 architecture hyperparameter loading
static void llama_load_hparams_rwkv7(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

    switch (hparams.n_layer) {
        case 24: type = LLM_TYPE_1B; break;
        case 32: type = LLM_TYPE_3B; break;
        case 64: type = LLM_TYPE_7B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Plamo2 architecture hyperparameter loading
static void llama_load_hparams_plamo2(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    // Load Mamba SSM parameters
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
        hparams.recurrent_layer_arr[i] = hparams.n_head_kv(i) == 0;
    }

    switch (hparams.n_layer) {
        case 16: type = LLM_TYPE_1B; break;
        case 32:
            if (hparams.n_embd == 2048) {
                type = LLM_TYPE_2B;
            } else if (hparams.n_embd == 4096) {
                type = LLM_TYPE_8B;
            }
            break;
        default: type = LLM_TYPE_UNKNOWN;
    }

    // Load attention parameters
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH,   hparams.n_embd_head_k, false);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH, hparams.n_embd_head_v, false);
}

// XVERSE architecture hyperparameter loading
static void llama_load_hparams_xverse(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    switch (hparams.n_layer) {
        case 32: type = LLM_TYPE_7B; break;
        case 40: type = LLM_TYPE_13B; break;
        case 80: type = LLM_TYPE_65B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Command-R architecture hyperparameter loading
static void llama_load_hparams_command_r(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_LOGIT_SCALE,             hparams.f_logit_scale);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
    switch (hparams.n_layer) {
        case 40: type = LLM_TYPE_35B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Cohere2 architecture hyperparameter loading
static void llama_load_hparams_cohere2(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.set_swa_pattern(4);

    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
    ml.get_key(LLM_KV_LOGIT_SCALE,              hparams.f_logit_scale);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,  hparams.f_norm_eps);
    switch (hparams.n_layer) {
        case 32: type = LLM_TYPE_8B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// OLMo architecture hyperparameter loading
static void llama_load_hparams_olmo(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
    ml.get_key(LLM_KV_ATTENTION_CLAMP_KQV,     hparams.f_clamp_kqv, false);

    switch (hparams.n_layer) {
        case 22: type = LLM_TYPE_1B; break;
        case 32: type = LLM_TYPE_7B; break;
        case 80: type = LLM_TYPE_70B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// OLMo2 architecture hyperparameter loading
static void llama_load_hparams_olmo2(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    const bool found_swa = ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
    if (found_swa && hparams.n_swa > 0) {
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        hparams.set_swa_pattern(4);
    } else {
        hparams.swa_type = LLAMA_SWA_TYPE_NONE;
    }

    switch (hparams.n_layer) {
        case 16: type = LLM_TYPE_1B; break;
        case 32: type = LLM_TYPE_7B; break;
        case 40: type = LLM_TYPE_13B; break;
        case 64: type = LLM_TYPE_32B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Seed OSS architecture hyperparameter loading
static void llama_load_hparams_seed_oss(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    switch (hparams.n_layer) {
        case 64: type = LLM_TYPE_36B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// PLM architecture hyperparameter loading
static void llama_load_hparams_plm(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
    ml.get_key(LLM_KV_ATTENTION_CLAMP_KQV,     hparams.f_clamp_kqv, false);

    switch (hparams.n_layer) {
        case 48: type = LLM_TYPE_1B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// BitNet architecture hyperparameter loading
static void llama_load_hparams_bitnet(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    switch (hparams.n_layer) {
        case 26: type = LLM_TYPE_3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// InternLM2 architecture hyperparameter loading
static void llama_load_hparams_internlm2(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    switch (hparams.n_layer) {
        case 32: type = LLM_TYPE_7B; break;
        case 48: type = LLM_TYPE_20B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Phi2 architecture hyperparameter loading
static void llama_load_hparams_phi2(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

    switch (hparams.n_layer) {
        case 24: type = LLM_TYPE_1B; break;
        case 32: type = LLM_TYPE_3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Phi3 architecture hyperparameter loading
static void llama_load_hparams_phi3(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer) {
        case 24: type = LLM_TYPE_1B; break;
        case 32: type = LLM_TYPE_3B; break;
        case 40: type = LLM_TYPE_14B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }

    const bool found_swa = ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);

    if (found_swa && hparams.n_swa > 0) {
        LLAMA_LOG_WARN("%s: Phi SWA is currently disabled - results might be suboptimal for some models (see %s)\n",
                __func__, "https://github.com/ggml-org/llama.cpp/pull/13676");

        // TODO: fix conversion scripts to correctly populate `n_swa` and `n_swa_pattern`
        hparams.swa_type = LLAMA_SWA_TYPE_NONE;

        hparams.n_swa         = 0;
        hparams.set_swa_pattern(1);
    }
}

// Plamo architecture hyperparameter loading
static void llama_load_hparams_plamo(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer) {
        case 40: type = LLM_TYPE_13B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Dream architecture hyperparameter loading
static void llama_load_hparams_dream(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    // Dream models are primarily 7B with 28 layers
    switch (hparams.n_layer) {
        case 28:
            type = LLM_TYPE_7B;
            break;
        default:
            type = LLM_TYPE_UNKNOWN;
    }
    // Set non-causal attention for diffusion models
    hparams.causal_attn = false;
}

// LLaDA architecture hyperparameter loading
static void llama_load_hparams_llada(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    // LLaDA-8B has 32 layers, similar to LLaMA but for diffusion
    switch (hparams.n_layer) {
        case 32:
            type = LLM_TYPE_8B;
            break;
        default:
            type = LLM_TYPE_UNKNOWN;
    }
    // Set non-causal attention for diffusion models
    hparams.causal_attn = false;
}

// LLaDA MoE architecture hyperparameter loading
static void llama_load_hparams_llada_moe(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp, false);

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    // diffusion language model uses non-causal attention
    hparams.causal_attn = false;
    switch (hparams.n_layer) {
        case 16: type = LLM_TYPE_A1_7B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Qwen architecture hyperparameter loading
static void llama_load_hparams_qwen(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer) {
        case 32: type = LLM_TYPE_7B; break;
        case 40: type = LLM_TYPE_13B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Qwen2 architecture hyperparameter loading
static void llama_load_hparams_qwen2(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    ml.get_key(LLM_KV_POOLING_TYPE, hparams.pooling_type, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    switch (hparams.n_layer) {
        case 24: type = hparams.n_embd == 1024 ? LLM_TYPE_0_5B : LLM_TYPE_1B; break;
        case 28: type = hparams.n_embd == 1536 ? LLM_TYPE_1_5B : LLM_TYPE_7B; break;
        case 32: type = LLM_TYPE_7B; break;
        case 36: type = LLM_TYPE_3B; break;
        case 40: type = hparams.n_head() == 20 ? LLM_TYPE_4B : LLM_TYPE_13B; break;
        case 48: type = LLM_TYPE_14B; break;
        case 64: type = LLM_TYPE_32B; break;
        case 80: type = LLM_TYPE_70B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Qwen2VL architecture hyperparameter loading
static void llama_load_hparams_qwen2vl(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, true);
    ml.get_key(LLM_KV_POOLING_TYPE, hparams.pooling_type, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    switch (hparams.n_layer) {
        case 24: type = hparams.n_embd == 1024 ? LLM_TYPE_0_5B : LLM_TYPE_1B; break;
        case 28: type = hparams.n_embd == 1536 ? LLM_TYPE_1_5B : LLM_TYPE_7B; break;
        case 32: type = LLM_TYPE_7B; break;
        case 36: type = LLM_TYPE_3B; break;
        case 40: type = hparams.n_head() == 20 ? LLM_TYPE_4B : LLM_TYPE_13B; break;
        case 48: type = LLM_TYPE_14B; break;
        case 64: type = LLM_TYPE_32B; break;
        case 80: type = LLM_TYPE_70B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Qwen3 architecture hyperparameter loading
static void llama_load_hparams_qwen3(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    ml.get_key(LLM_KV_POOLING_TYPE, hparams.pooling_type, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    switch (hparams.n_layer) {
        case 28: type = hparams.n_embd == 1024 ? LLM_TYPE_0_6B : LLM_TYPE_1_7B; break;
        case 36: type = hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_8B; break;
        case 40: type = LLM_TYPE_14B; break;
        case 64: type = LLM_TYPE_32B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Qwen3VL architecture hyperparameter loading
static void llama_load_hparams_qwen3vl(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    ml.get_key(LLM_KV_NUM_DEEPSTACK_LAYERS, hparams.n_deepstack_layers, false);
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, true);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    switch (hparams.n_layer) {
        case 28: type = LLM_TYPE_1_7B; break;
        case 36: type = hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_8B; break;
        case 64: type = LLM_TYPE_32B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Gemma architecture hyperparameter loading
static void llama_load_hparams_gemma(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer) {
        case 18: type = LLM_TYPE_2B; break;
        case 28: type = LLM_TYPE_7B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Gemma2 architecture hyperparameter loading
static void llama_load_hparams_gemma2(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.set_swa_pattern(2);
    hparams.attn_soft_cap = true;

    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTN_LOGIT_SOFTCAPPING,      hparams.f_attn_logit_softcapping, false);
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     hparams.f_final_logit_softcapping, false);

    switch (hparams.n_layer) {
        case 26: type = LLM_TYPE_2B; break;
        case 42: type = LLM_TYPE_9B; break;
        case 46: type = LLM_TYPE_27B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }

    // ref: https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/config.py#L173
    hparams.f_attention_scale = type == LLM_TYPE_27B
        ? 1.0f / std::sqrt(float(hparams.n_embd / hparams.n_head(0)))
        : 1.0f / std::sqrt(float(hparams.n_embd_head_k));
}

// Gemma3 architecture hyperparameter loading
static void llama_load_hparams_gemma3(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.set_swa_pattern(6);

    hparams.rope_freq_base_train_swa  = 10000.0f;
    hparams.rope_freq_scale_train_swa = 1.0f;

    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer) {
        case 18: type = LLM_TYPE_270M; break;
        case 26: type = LLM_TYPE_1B; break;
        case 34: type = LLM_TYPE_4B; break;
        case 48: type = LLM_TYPE_12B; break;
        case 62: type = LLM_TYPE_27B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }

    // ref: https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/config.py#L289
    hparams.f_attention_scale = type == LLM_TYPE_27B
        ? 1.0f / std::sqrt(float(hparams.n_embd / hparams.n_head(0)))
        : 1.0f / std::sqrt(float(hparams.n_embd_head_k));
}

// Gemma3N architecture hyperparameter loading
static void llama_load_hparams_gemma3n(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.set_swa_pattern(5);

    hparams.n_layer_kv_from_start     = 20;
    hparams.rope_freq_base_train_swa  = 10000.0f;
    hparams.rope_freq_scale_train_swa = 1.0f;
    hparams.f_attention_scale         = 1.0f;

    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer) {
        case 30: type = LLM_TYPE_E2B; break;
        case 35: type = LLM_TYPE_E4B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Gemma Embedding architecture hyperparameter loading
static void llama_load_hparams_gemma_embedding(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    hparams.swa_type = LLAMA_SWA_TYPE_SYMMETRIC;
    hparams.set_swa_pattern(6);

    hparams.causal_attn = false; // embeddings do not use causal attention
    hparams.rope_freq_base_train_swa = 10000.0f;
    hparams.rope_freq_scale_train_swa = 1.0f;

    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_POOLING_TYPE, hparams.pooling_type);

    //applied only if model converted with --sentence-transformers-dense-modules
    ml.get_key(LLM_KV_DENSE_2_FEAT_IN, hparams.dense_2_feat_in, false);
    ml.get_key(LLM_KV_DENSE_2_FEAT_OUT, hparams.dense_2_feat_out, false);
    ml.get_key(LLM_KV_DENSE_3_FEAT_IN, hparams.dense_3_feat_in, false);
    ml.get_key(LLM_KV_DENSE_3_FEAT_OUT, hparams.dense_3_feat_out, false);

    GGML_ASSERT((hparams.dense_2_feat_in == 0 || hparams.dense_2_feat_in == hparams.n_embd) && "dense_2_feat_in must be equal to n_embd");
    GGML_ASSERT((hparams.dense_3_feat_out == 0 || hparams.dense_3_feat_out == hparams.n_embd) && "dense_3_feat_out must be equal to n_embd");

    switch (hparams.n_layer) {
        case 24: type = LLM_TYPE_0_3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
    hparams.f_attention_scale = 1.0f / std::sqrt(float(hparams.n_embd_head_k));
}

// StarCoder2 architecture hyperparameter loading
static void llama_load_hparams_starcoder2(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
    switch (hparams.n_layer) {
        case 30: type = LLM_TYPE_3B; break;
        case 32: type = LLM_TYPE_7B; break;
        case 40: type = LLM_TYPE_15B; break;
        case 52: type = LLM_TYPE_20B; break; // granite
        case 88: type = LLM_TYPE_34B; break; // granite
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// JAIS architecture hyperparameter loading
static void llama_load_hparams_jais(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
    ml.get_key(LLM_KV_ATTENTION_CLAMP_KQV,     hparams.f_clamp_kqv, false);
    ml.get_key(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, hparams.f_max_alibi_bias);

    switch (hparams.n_layer) {
        case 30: type = LLM_TYPE_13B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Nemotron architecture hyperparameter loading
static void llama_load_hparams_nemotron(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
    ml.get_key(LLM_KV_ATTENTION_CLAMP_KQV,     hparams.f_clamp_kqv, false);
    ml.get_key(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, hparams.f_max_alibi_bias);

    switch (hparams.n_layer) {
        case 32: type = LLM_TYPE_8B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Nemotron-H architecture hyperparameter loading
static void llama_load_hparams_nemotron_h(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
        hparams.recurrent_layer_arr[i] = hparams.n_head_kv(i) == 0;
    }

    switch (hparams.n_layer) {
        case 32: type = LLM_TYPE_8B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Exaone architecture hyperparameter loading
static void llama_load_hparams_exaone(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

    switch (hparams.n_layer) {
        case 32: type = LLM_TYPE_8B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Exaone4 architecture hyperparameter loading
static void llama_load_hparams_exaone4(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer) {
        case 32: type = LLM_TYPE_8B; break;
        case 40: type = LLM_TYPE_16B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Chameleon architecture hyperparameter loading
static void llama_load_hparams_chameleon(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer) {
        case 32: type = LLM_TYPE_7B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// WaveTokenizer architecture hyperparameter loading
static void llama_load_hparams_wavtokenizer_dec(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(type);
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_FEATURES_LENGTH, hparams.n_embd_features);

    ml.get_key(LLM_KV_POSNET_EMBEDDING_LENGTH, hparams.posnet.n_embd);
    ml.get_key(LLM_KV_POSNET_BLOCK_COUNT,      hparams.posnet.n_layer);

    ml.get_key(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, hparams.convnext.n_embd);
    ml.get_key(LLM_KV_CONVNEXT_BLOCK_COUNT,      hparams.convnext.n_layer);
}

// Dots1 architecture hyperparameter loading
static void llama_load_hparams_dots1(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer) {
        case 32: type = LLM_TYPE_7B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Ernie4.5 architecture hyperparameter loading
static void llama_load_hparams_ernie45(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    switch (hparams.n_layer) {
        case 56: type = LLM_TYPE_21B_A3B; break;
        case 64: type = LLM_TYPE_30B_A3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Ernie4.5 MoE architecture hyperparameter loading
static void llama_load_hparams_ernie45_moe(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    if (arch == LLM_ARCH_ERNIE4_5_MOE) {
        ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
    }
    switch (hparams.n_layer) {
        case 56: type = LLM_TYPE_100B_A6B; break;
        case 64: type = LLM_TYPE_300B_A47B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Falcon H1 architecture hyperparameter loading
static void llama_load_hparams_falcon_h1(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
        hparams.recurrent_layer_arr[i] = true; // All layers are recurrent
    }

    switch (hparams.n_layer) {
        case 80: type = LLM_TYPE_7B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Hunyuan MoE architecture hyperparameter loading
static void llama_load_hparams_hunyuan_moe(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
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

// Hunyuan Dense architecture hyperparameter loading
static void llama_load_hparams_hunyuan_dense(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer) {
        case 56: type = LLM_TYPE_6B; break;
        case 32: type = LLM_TYPE_26B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// SmolLM3 architecture hyperparameter loading
static void llama_load_hparams_smollm3(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer) {
        case 30: type = LLM_TYPE_1B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// LFM2 architecture hyperparameter loading
static void llama_load_hparams_lfm2(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer) {
        case 48: type = LLM_TYPE_8B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// SmallThinker architecture hyperparameter loading
static void llama_load_hparams_smallthinker(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer) {
        case 28: type = LLM_TYPE_1B; break;
        case 40: type = LLM_TYPE_3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Apertus architecture hyperparameter loading
static void llama_load_hparams_apertus(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer) {
        case 32: type = LLM_TYPE_7B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Minimax M2 architecture hyperparameter loading
static void llama_load_hparams_minimax_m2(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp, false);

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    switch (hparams.n_layer) {
        case 60: type = LLM_TYPE_230B_A10B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// CogVLM architecture hyperparameter loading
static void llama_load_hparams_cogvlm(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer) {
        case 40: type = LLM_TYPE_1B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Pangu Embed architecture hyperparameter loading
static void llama_load_hparams_pangu_embed(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_POOLING_TYPE, hparams.pooling_type);

    switch (hparams.n_layer) {
        case 48: type = LLM_TYPE_7B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}


// Main dispatch function for specialized architectures
void llama_load_hparams_specialized(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch, uint32_t n_vocab) {
    GGML_UNUSED(n_vocab);

    switch (arch) {
        case LLM_ARCH_MAMBA:          llama_load_hparams_mamba(ml, hparams, type, arch); break;
        case LLM_ARCH_MAMBA2:         llama_load_hparams_mamba2(ml, hparams, type, arch); break;
        case LLM_ARCH_JAMBA:          llama_load_hparams_jamba(ml, hparams, type, arch); break;
        case LLM_ARCH_T5:             llama_load_hparams_t5(ml, hparams, type, arch); break;
        case LLM_ARCH_T5ENCODER:      llama_load_hparams_t5(ml, hparams, type, arch); break;
        case LLM_ARCH_CHATGLM:        llama_load_hparams_chatglm(ml, hparams, type, arch); break;
        case LLM_ARCH_GLM4:           llama_load_hparams_glm4(ml, hparams, type, arch); break;
        case LLM_ARCH_RWKV6:          llama_load_hparams_rwkv6(ml, hparams, type, arch); break;
        case LLM_ARCH_RWKV6QWEN2:     llama_load_hparams_rwkv6(ml, hparams, type, arch); break;
        case LLM_ARCH_RWKV7:          llama_load_hparams_rwkv7(ml, hparams, type, arch); break;
        case LLM_ARCH_ARWKV7:         llama_load_hparams_rwkv7(ml, hparams, type, arch); break;
        case LLM_ARCH_PLAMO2:         llama_load_hparams_plamo2(ml, hparams, type, arch); break;
        case LLM_ARCH_XVERSE:         llama_load_hparams_xverse(ml, hparams, type, arch); break;
        case LLM_ARCH_COMMAND_R:      llama_load_hparams_command_r(ml, hparams, type, arch); break;
        case LLM_ARCH_COHERE2:        llama_load_hparams_cohere2(ml, hparams, type, arch); break;
        case LLM_ARCH_OLMO:           llama_load_hparams_olmo(ml, hparams, type, arch); break;
        case LLM_ARCH_OLMO2:          llama_load_hparams_olmo2(ml, hparams, type, arch); break;
        case LLM_ARCH_SEED_OSS:       llama_load_hparams_seed_oss(ml, hparams, type, arch); break;
        case LLM_ARCH_PLM:            llama_load_hparams_plm(ml, hparams, type, arch); break;
        case LLM_ARCH_BITNET:         llama_load_hparams_bitnet(ml, hparams, type, arch); break;
        case LLM_ARCH_INTERNLM2:      llama_load_hparams_internlm2(ml, hparams, type, arch); break;
        case LLM_ARCH_PHI2:           llama_load_hparams_phi2(ml, hparams, type, arch); break;
        case LLM_ARCH_PHI3:           llama_load_hparams_phi3(ml, hparams, type, arch); break;
        case LLM_ARCH_PLAMO:          llama_load_hparams_plamo(ml, hparams, type, arch); break;
        case LLM_ARCH_DREAM:          llama_load_hparams_dream(ml, hparams, type, arch); break;
        case LLM_ARCH_LLADA:          llama_load_hparams_llada(ml, hparams, type, arch); break;
        case LLM_ARCH_LLADA_MOE:      llama_load_hparams_llada_moe(ml, hparams, type, arch); break;
        case LLM_ARCH_QWEN:           llama_load_hparams_qwen(ml, hparams, type, arch); break;
        case LLM_ARCH_QWEN2:          llama_load_hparams_qwen2(ml, hparams, type, arch); break;
        case LLM_ARCH_QWEN2VL:        llama_load_hparams_qwen2vl(ml, hparams, type, arch); break;
        case LLM_ARCH_QWEN3:          llama_load_hparams_qwen3(ml, hparams, type, arch); break;
        case LLM_ARCH_QWEN3VL:        llama_load_hparams_qwen3vl(ml, hparams, type, arch); break;
        case LLM_ARCH_GEMMA:          llama_load_hparams_gemma(ml, hparams, type, arch); break;
        case LLM_ARCH_GEMMA2:         llama_load_hparams_gemma2(ml, hparams, type, arch); break;
        case LLM_ARCH_GEMMA3:         llama_load_hparams_gemma3(ml, hparams, type, arch); break;
        case LLM_ARCH_GEMMA3N:        llama_load_hparams_gemma3n(ml, hparams, type, arch); break;
        case LLM_ARCH_GEMMA_EMBEDDING: llama_load_hparams_gemma_embedding(ml, hparams, type, arch); break;
        case LLM_ARCH_STARCODER2:     llama_load_hparams_starcoder2(ml, hparams, type, arch); break;
        case LLM_ARCH_JAIS:           llama_load_hparams_jais(ml, hparams, type, arch); break;
        case LLM_ARCH_NEMOTRON:       llama_load_hparams_nemotron(ml, hparams, type, arch); break;
        case LLM_ARCH_NEMOTRON_H:     llama_load_hparams_nemotron_h(ml, hparams, type, arch); break;
        case LLM_ARCH_EXAONE:         llama_load_hparams_exaone(ml, hparams, type, arch); break;
        case LLM_ARCH_EXAONE4:        llama_load_hparams_exaone4(ml, hparams, type, arch); break;
        case LLM_ARCH_CHAMELEON:      llama_load_hparams_chameleon(ml, hparams, type, arch); break;
        case LLM_ARCH_WAVTOKENIZER_DEC: llama_load_hparams_wavtokenizer_dec(ml, hparams, type, arch); break;
        case LLM_ARCH_DOTS1:          llama_load_hparams_dots1(ml, hparams, type, arch); break;
        case LLM_ARCH_ERNIE4_5:       llama_load_hparams_ernie45(ml, hparams, type, arch); break;
        case LLM_ARCH_ERNIE4_5_MOE:   llama_load_hparams_ernie45_moe(ml, hparams, type, arch); break;
        case LLM_ARCH_FALCON_H1:      llama_load_hparams_falcon_h1(ml, hparams, type, arch); break;
        case LLM_ARCH_HUNYUAN_MOE:    llama_load_hparams_hunyuan_moe(ml, hparams, type, arch); break;
        case LLM_ARCH_HUNYUAN_DENSE:  llama_load_hparams_hunyuan_dense(ml, hparams, type, arch); break;
        case LLM_ARCH_SMOLLM3:        llama_load_hparams_smollm3(ml, hparams, type, arch); break;
        case LLM_ARCH_OPENAI_MOE:     // Handled in MoE family
        case LLM_ARCH_LFM2:           llama_load_hparams_lfm2(ml, hparams, type, arch); break;
        case LLM_ARCH_LFM2MOE:        // Handled in MoE family
        case LLM_ARCH_SMALLTHINKER:   llama_load_hparams_smallthinker(ml, hparams, type, arch); break;
        case LLM_ARCH_GROVEMOE:       // Handled in MoE family
        case LLM_ARCH_APERTUS:        llama_load_hparams_apertus(ml, hparams, type, arch); break;
        case LLM_ARCH_MINIMAX_M2:     llama_load_hparams_minimax_m2(ml, hparams, type, arch); break;
        case LLM_ARCH_COGVLM:         llama_load_hparams_cogvlm(ml, hparams, type, arch); break;
        case LLM_ARCH_PANGU_EMBED:    llama_load_hparams_pangu_embed(ml, hparams, type, arch); break;
        case LLM_ARCH_ARCEE:          // Handled in LLaMA family
        case LLM_ARCH_AFMOE:          // Handled in LLaMA family
            break;
        default:
            throw std::runtime_error(format("Architecture %s not handled by specialized loader", llama_arch_name(arch)));
    }
}
