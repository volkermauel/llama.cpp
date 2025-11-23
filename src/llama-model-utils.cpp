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

// Forward declarations - these are now defined in llama-model-base.cpp
extern const char * llm_type_name(llm_type type);
extern const char * llama_expert_gating_func_name(llama_expert_gating_func_type type);
extern std::string llama_rope_scaling_type_name(llama_rope_scaling_type rope_scaling_type);
extern llama_rope_scaling_type llama_rope_scaling_type_from_string(const std::string & name);
extern bool weight_buft_supported(const llama_hparams & hparams, ggml_tensor * w, ggml_op op, ggml_backend_buffer_type_t buft, ggml_backend_dev_t dev);
extern ggml_backend_buffer_type_t select_weight_buft(const llama_hparams & hparams, ggml_tensor * tensor, ggml_op op, const std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>> & buft_list);

// Utility functions for llama-model
// This file contains utility functions that don't have duplicate implementations elsewhere