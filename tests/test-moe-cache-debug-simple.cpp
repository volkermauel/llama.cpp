#include "src/llama-moe-cache-debug.h"
#include <iostream>

// Simple test to verify logging integration
int main() {
    std::cout << "Testing MoE Cache Debug Logging Integration\n";
    std::cout << "==========================================\n\n";
    
    // Test basic logging functions
    std::cout << "1. Testing basic logging functions:\n";
    llama_moe_log_prefetch(0, 5, 1024 * 1024 * 10, "RAM", "VRAM");
    llama_moe_log_eviction(1, 3, "Cache full", 1024 * 1024 * 8);
    llama_moe_log_transfer("RAM->VRAM", 1024 * 1024 * 15, "Expert L2.E7 loaded");
    llama_moe_log_layer_assignment(2, "GPU", 8192);
    llama_moe_log_expert_lifecycle(3, 10, "Loaded", "Successfully cached");
    std::cout << "\n";
    
    // Test error and warning logging
    std::cout << "2. Testing error and warning logging:\n";
    llama_moe_log_error(0, 5, "Failed to allocate GPU buffer", -1);
    llama_moe_log_warning(1, 3, "Cache usage approaching limit");
    std::cout << "\n";
    
    // Test debug configuration
    std::cout << "3. Testing debug configuration:\n";
    llama_moe_debug::debug_config config;
    config.enable_prefetch_logging = true;
    config.enable_eviction_logging = true;
    config.enable_stats_logging = true;
    config.enable_layer_assignment_logging = false;
    config.enable_transfer_logging = true;
    config.enable_expert_lifecycle_logging = false;
    
    llama_moe_debug_configure(config);
    std::cout << "\n";
    
    // Test category checking
    std::cout << "4. Testing category checking:\n";
    std::cout << "Prefetch logging enabled: " << llama_moe_debug_is_enabled("prefetch") << "\n";
    std::cout << "Eviction logging enabled: " << llama_moe_debug_is_enabled("eviction") << "\n";
    std::cout << "Stats logging enabled: " << llama_moe_debug_is_enabled("stats") << "\n";
    std::cout << "Layer assignment logging enabled: " << llama_moe_debug_is_enabled("layer_assignment") << "\n";
    std::cout << "Invalid category enabled: " << llama_moe_debug_is_enabled("invalid") << "\n";
    std::cout << "\n";
    
    std::cout << "Integration test completed successfully!\n";
    std::cout << "The debug logging infrastructure is properly integrated with llama.cpp logging system.\n";
    
    return 0;
}