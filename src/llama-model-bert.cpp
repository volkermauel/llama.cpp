// BERT/encoder family architectures
// Extracted from src/llama-model.cpp
// Contains: LLM_ARCH_BERT, LLM_ARCH_JINA_BERT_V2, LLM_ARCH_JINA_BERT_V3, 
//           LLM_ARCH_NOMIC_BERT, LLM_ARCH_NOMIC_BERT_MOE, LLM_ARCH_NEO_BERT

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
static const char * llm_type_name(llm_type type);
static const char * llama_expert_gating_func_name(llama_expert_gating_func_type type);
static std::string llama_rope_scaling_type_name(llama_rope_scaling_type rope_scaling_type);
static llama_rope_scaling_type llama_rope_scaling_type_from_string(const std::string & name);
const char * llama_arch_name(llm_arch arch);

// BERT architecture hyperparameter loading
static void llama_load_hparams_bert(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,    hparams.f_norm_eps);
    ml.get_key(LLM_KV_ATTENTION_CAUSAL,           hparams.causal_attn);
    ml.get_key(LLM_KV_POOLING_TYPE,               hparams.pooling_type, false);

    switch (hparams.n_layer) {
        case 3:
            type = LLM_TYPE_17M; break; // bge-micro
        case 6:
            type = LLM_TYPE_22M; break; // MiniLM-L6
        case 12:
            switch (hparams.n_embd) {
                case 384: type = LLM_TYPE_33M; break; // MiniLM-L12, bge-small
                case 768: type = LLM_TYPE_109M; break; // bge-base
                default: type = LLM_TYPE_UNKNOWN;
            } break;
        case 24:
            type = LLM_TYPE_335M; break; // bge-large
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Jina BERT v2 architecture hyperparameter loading
static void llama_load_hparams_jina_bert_v2(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,    hparams.f_norm_eps);
    ml.get_key(LLM_KV_ATTENTION_CAUSAL,           hparams.causal_attn);
    ml.get_key(LLM_KV_POOLING_TYPE,               hparams.pooling_type, false);
    hparams.f_max_alibi_bias = 8.0f;

    switch (hparams.n_layer) {
        case 4:  type = LLM_TYPE_33M;  break; // jina-embeddings-small
        case 12: type = LLM_TYPE_137M; break; // jina-embeddings-base
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Jina BERT v3 architecture hyperparameter loading
static void llama_load_hparams_jina_bert_v3(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,    hparams.f_norm_eps);
    ml.get_key(LLM_KV_ATTENTION_CAUSAL,           hparams.causal_attn);
    ml.get_key(LLM_KV_POOLING_TYPE,               hparams.pooling_type, false);

    switch (hparams.n_layer) {
        case 24:
            type = LLM_TYPE_558M; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Nomic BERT architecture hyperparameter loading
static void llama_load_hparams_nomic_bert(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,    hparams.f_norm_eps);
    ml.get_key(LLM_KV_ATTENTION_CAUSAL,           hparams.causal_attn);
    ml.get_key(LLM_KV_POOLING_TYPE,               hparams.pooling_type);
    ml.get_key(LLM_KV_MOE_EVERY_N_LAYERS,         hparams.moe_every_n_layers, 0);

    if (hparams.n_layer == 12 && hparams.n_embd == 768) {
        if (arch == LLM_ARCH_NOMIC_BERT) {
            type = LLM_TYPE_137M;
        } else if (arch == LLM_ARCH_NOMIC_BERT_MOE && hparams.moe_every_n_layers == 2) {
            type = LLM_TYPE_475M;
        }
    }
}

// Neo BERT architecture hyperparameter loading
static void llama_load_hparams_neo_bert(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_CAUSAL,            hparams.causal_attn);
    ml.get_key(LLM_KV_POOLING_TYPE,                hparams.pooling_type);

    if (hparams.n_layer == 28) {
        type = LLM_TYPE_250M;
    }
}

// Dispatch function for BERT family architectures
void llama_load_hparams_bert_family(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_BERT:           llama_load_hparams_bert(ml, hparams, type, arch); break;
        case LLM_ARCH_JINA_BERT_V2:   llama_load_hparams_jina_bert_v2(ml, hparams, type, arch); break;
        case LLM_ARCH_JINA_BERT_V3:   llama_load_hparams_jina_bert_v3(ml, hparams, type, arch); break;
        case LLM_ARCH_NOMIC_BERT:
        case LLM_ARCH_NOMIC_BERT_MOE: llama_load_hparams_nomic_bert(ml, hparams, type, arch); break;
        case LLM_ARCH_NEO_BERT:       llama_load_hparams_neo_bert(ml, hparams, type, arch); break;
        default:
            throw std::runtime_error(format("Architecture %s not handled by BERT family loader", llama_arch_name(arch)));
    }
}