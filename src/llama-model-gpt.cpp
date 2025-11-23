// GPT family architectures
// Extracted from src/llama-model.cpp
// Contains: LLM_ARCH_GPT2, LLM_ARCH_GPTNEOX, LLM_ARCH_STARCODER, LLM_ARCH_MPT, 
//           LLM_ARCH_STABLELM, LLM_ARCH_REFACT, LLM_ARCH_CODESHELL, LLM_ARCH_ORION

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
#include <cmath>
#include <map>
#include <stdexcept>

// Forward declarations
const char * llama_arch_name(llm_arch arch);

// GPT-2 architecture hyperparameter loading
static void llama_load_hparams_gpt2(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
    switch (hparams.n_layer) {
        case 12: type = LLM_TYPE_SMALL; break;
        case 24: type = LLM_TYPE_MEDIUM; break;
        case 36: type = LLM_TYPE_LARGE; break;
        case 48: type = LLM_TYPE_XL; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// GPT-NeoX architecture hyperparameter loading
static void llama_load_hparams_gptneox(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
    switch (hparams.n_layer) {
        case 32: type = LLM_TYPE_1B; break;
        case 36: type = LLM_TYPE_3B; break;
        case 40: type = LLM_TYPE_6B; break;
        case 44: type = LLM_TYPE_12B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// StarCoder architecture hyperparameter loading
static void llama_load_hparams_starcoder(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
    switch (hparams.n_layer) {
        case 24: type = LLM_TYPE_1B; break;
        case 36: type = LLM_TYPE_3B; break;
        case 42: type = LLM_TYPE_7B; break;
        case 40: type = LLM_TYPE_15B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// MPT architecture hyperparameter loading
static void llama_load_hparams_mpt(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,  hparams.f_norm_eps);
    ml.get_key(LLM_KV_ATTENTION_CLAMP_KQV,      hparams.f_clamp_kqv, false);
    ml.get_key(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, hparams.f_max_alibi_bias);

    switch (hparams.n_layer) {
        case 32: type = LLM_TYPE_7B; break;
        case 48: type = LLM_TYPE_30B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// StableLM architecture hyperparameter loading
static void llama_load_hparams_stablelm(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

    switch (hparams.n_layer) {
        case 24: type = LLM_TYPE_1B; break;
        case 32: type = LLM_TYPE_3B; break;
        case 40: type = LLM_TYPE_12B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Refact architecture hyperparameter loading
static void llama_load_hparams_refact(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    switch (hparams.n_layer) {
        case 32: type = LLM_TYPE_1B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }

    // TODO: become GGUF KV parameter
    hparams.f_max_alibi_bias = 8.0f;
}

// CodeShell architecture hyperparameter loading
static void llama_load_hparams_codeshell(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
    switch (hparams.n_layer) {
        case 42: type = LLM_TYPE_7B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Orion architecture hyperparameter loading
static void llama_load_hparams_orion(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    GGML_UNUSED(arch);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

    switch (hparams.n_layer) {
        case 40: type = LLM_TYPE_14B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// Dispatch function for GPT family architectures
void llama_load_hparams_gpt_family(llama_model_loader & ml, llama_hparams & hparams, llm_type & type, llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_GPT2:      llama_load_hparams_gpt2(ml, hparams, type, arch); break;
        case LLM_ARCH_GPTNEOX:   llama_load_hparams_gptneox(ml, hparams, type, arch); break;
        case LLM_ARCH_STARCODER: llama_load_hparams_starcoder(ml, hparams, type, arch); break;
        case LLM_ARCH_MPT:       llama_load_hparams_mpt(ml, hparams, type, arch); break;
        case LLM_ARCH_STABLELM:  llama_load_hparams_stablelm(ml, hparams, type, arch); break;
        case LLM_ARCH_REFACT:    llama_load_hparams_refact(ml, hparams, type, arch); break;
        case LLM_ARCH_CODESHELL: llama_load_hparams_codeshell(ml, hparams, type, arch); break;
        case LLM_ARCH_ORION:     llama_load_hparams_orion(ml, hparams, type, arch); break;
        default:
            throw std::runtime_error(format("Architecture %s not handled by GPT family loader", llama_arch_name(arch)));
    }
}