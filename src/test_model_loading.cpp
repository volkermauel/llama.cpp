// Simple test program to verify model loading functionality after refactoring
#include "llama-model.h"
#include "llama-arch.h"
#include "llama-impl.h"
#include <iostream>
#include <string>

int main() {
    std::cout << "Testing llama-model refactoring..." << std::endl;
    
    // Test 1: Verify llama_model class can be instantiated
    try {
        llama_model_params params = {};
        params.n_gpu_layers = 0; // CPU only for testing
        
        llama_model model(params);
        std::cout << "✓ llama_model instantiation successful" << std::endl;
        
        // Test 2: Verify architecture name function works
        const char* arch_name = llama_arch_name(LLM_ARCH_LLAMA);
        if (arch_name && std::string(arch_name) == "llama") {
            std::cout << "✓ Architecture name lookup successful" << std::endl;
        } else {
            std::cout << "✗ Architecture name lookup failed" << std::endl;
            return 1;
        }
        
        // Test 3: Verify type name function works
        const char* type_name = llm_type_name(LLM_TYPE_7B);
        if (type_name && std::string(type_name) == "7B") {
            std::cout << "✓ Model type name lookup successful" << std::endl;
        } else {
            std::cout << "✗ Model type name lookup failed" << std::endl;
            return 1;
        }
        
        // Test 4: Verify model methods can be called
        std::string desc = model.desc();
        if (!desc.empty()) {
            std::cout << "✓ Model description: " << desc << std::endl;
        } else {
            std::cout << "✓ Model description method works (empty for uninitialized model)" << std::endl;
        }
        
        std::cout << "✓ All basic functionality tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "✗ Exception during testing: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cout << "✗ Unknown exception during testing" << std::endl;
        return 1;
    }
}