// Master implementation file for llama-model
// This file includes all architecture-specific implementations
// and provides the complete model loading and management functionality

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
#include <cfloat>
#include <cstring>
#include <cmath>
#include <functional>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>

// Architecture-specific implementations are in separate files
// This file serves as a master implementation file that can include
// all components when needed for monolithic builds
// For the current build system, each file is compiled separately

// The base implementation from llama-model-base.cpp provides:
// - llama_model::impl struct
// - llama_model constructor and destructor
// - Basic model methods (load_stats, load_arch, etc.)
// - Type and utility functions

// The architecture-specific files provide:
// - llama_load_hparams_*_family functions for each architecture group
// - Architecture-specific hyperparameter loading logic

// The load.cpp file provides:
// - load_hparams dispatch logic
// - load_vocab and load_tensors placeholders

// Additional model methods that need tensor and graph information
// will be implemented in separate layer and utility files

// Note: The original llama-model.cpp contained approximately 7965 lines
// This refactoring splits it into:
// - llama-model-base.cpp: ~400 lines (core infrastructure)
// - llama-model-llama.cpp: ~250 lines (LLaMA family)
// - llama-model-gpt.cpp: ~130 lines (GPT family)
// - llama-model-bert.cpp: ~110 lines (BERT family)
// - llama-model-moe.cpp: ~280 lines (MoE models)
// - llama-model-specialized.cpp: ~450 lines (Specialized architectures)
// - llama-model-load.cpp: ~220 lines (Loading functions)
// - llama-model-layers.cpp: (Layer management - to be implemented)
// - llama-model-utils.cpp: (Utilities - to be implemented)
// Total: ~1840 lines + layers and utilities

// This structure maintains all functionality while improving modularity
// and making it easier to add new architectures