// Test compilation of the refactored llama-model files
// This file includes the main implementation to verify all components compile together

#define LLAMA_API_INTERNAL

#include "llama-model-impl.cpp"

// Simple test function to verify compilation
void test_llama_model_compilation() {
    // This will fail to link but should compile
    // The important thing is that all headers and implementations are consistent
    
    // Test that we can reference key types and functions
    llama_model_params params;
    params.n_gpu_layers = 0;
    
    // Test architecture enum is available
    llm_arch arch = LLM_ARCH_LLAMA;
    (void)arch; // Suppress unused variable warning
    
    // Test type enum is available
    llm_type type = LLM_TYPE_7B;
    (void)type;
    
    // Test that we can call the type name function
    const char * type_name = llm_type_name(LLM_TYPE_7B);
    (void)type_name;
    
    // Test rope scaling type
    llama_rope_scaling_type rope_type = LLAMA_ROPE_SCALING_TYPE_NONE;
    (void)rope_type;
    
    // Test expert gating function type
    llama_expert_gating_func_type gating_type = LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX;
    (void)gating_type;
}

// Test that layer structures are complete
void test_layer_structures() {
    llama_layer layer;
    (void)layer;
    
    llama_layer_posnet posnet;
    (void)posnet;
    
    llama_layer_convnext convnext;
    (void)convnext;
    
    llama_layer_shortconv shortconv;
    (void)shortconv;
    
    llama_layer_nextn nextn;
    (void)nextn;
}

// Test model structure
void test_model_structure() {
    llama_hparams hparams;
    (void)hparams;
    
    llama_vocab vocab;
    (void)vocab;
}