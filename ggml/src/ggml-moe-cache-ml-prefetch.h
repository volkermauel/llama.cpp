#pragma once

#include "ggml-moe-cache.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <atomic>
#include <memory>
#include <mutex>
#include <fstream>

namespace ggml_moe_ml {

// Forward declarations
struct ml_predictor;
struct online_learner;
struct feature_extractor;
struct pattern_analyzer;
struct persistence_manager;

// Configuration for ML prefetching
struct ml_prefetch_config {
    bool enabled = false;
    std::string model_cache_dir;  // Directory for model storage
    float learning_rate = 0.01f;
    float momentum = 0.9f;
    size_t max_history_size = 10000;
    bool enable_persistence = true;
    bool reset_on_startup = false;
    float accuracy_threshold = 0.7f;
    size_t min_training_samples = 100;
    size_t save_interval_samples = 1000;
    
    // Model architecture
    size_t hidden_size = 128;
    size_t feature_size = 256;
    
    // Prediction parameters
    size_t top_k_predictions = 5;
    float prediction_confidence_threshold = 0.5f;
};

// Statistics for ML prefetching
struct ml_prefetch_stats {
    uint64_t total_predictions = 0;
    uint64_t correct_predictions = 0;
    uint64_t training_samples = 0;
    uint64_t model_updates = 0;
    uint64_t persistence_saves = 0;
    uint64_t persistence_loads = 0;
    double current_accuracy = 0.0;
    double average_prediction_time_ms = 0.0;
    double average_learning_time_ms = 0.0;
};

// Model identifier based on model configuration
struct model_identifier {
    std::string model_path;      // Original model path (e.g., "XXX/yyy:zzz")
    std::string model_hash;      // Hash of model configuration
    int num_experts;             // Number of experts in the model
    
    // Generate unique string for this model
    std::string to_string() const;
    
    // Generate filename-safe version
    std::string to_filename() const;
    
    bool operator==(const model_identifier& other) const;
};

// Hash function for model_identifier
struct model_identifier_hash {
    size_t operator()(const model_identifier& id) const;
};

// Main ML prefetch engine
class ml_prefetch_engine {
public:
    // Constructor
    explicit ml_prefetch_engine(const ml_prefetch_config& config);
    
    // Destructor
    ~ml_prefetch_engine();
    
    // Initialize for a specific model
    bool initialize(const model_identifier& model_id, int num_experts);
    
    // Predict next experts based on current context
    std::vector<int> predict_next_experts(
        const std::vector<int>& current_experts,
        const std::vector<int>& recent_tokens,
        int layer_id,
        int position,
        size_t top_k = 5
    );
    
    // Update model with actual expert usage (online learning)
    void update_with_actual(
        const std::vector<int>& predicted_experts,
        const std::vector<int>& actual_experts,
        const std::vector<int>& context_tokens,
        int layer_id,
        int position
    );
    
    // Get current prediction confidence
    float get_prediction_confidence() const;
    
    // Get statistics
    ml_prefetch_stats get_stats() const;
    
    // Save model to disk
    bool save_model();
    
    // Load model from disk
    bool load_model();
    
    // Reset learned model
    void reset_model();
    
    // Check if model is ready for predictions
    bool is_ready() const;
    
    // Get model identifier
    const model_identifier& get_model_id() const { return model_id_; }
    
private:
    // Configuration
    ml_prefetch_config config_;
    
    // Model identifier
    model_identifier model_id_;
    
    // Core components
    std::unique_ptr<ml_predictor> predictor_;
    std::unique_ptr<online_learner> learner_;
    std::unique_ptr<feature_extractor> feature_extractor_;
    std::unique_ptr<pattern_analyzer> pattern_analyzer_;
    std::unique_ptr<persistence_manager> persistence_manager_;
    
    // Internal atomic statistics for thread-safe updates
    struct {
        std::atomic<uint64_t> total_predictions{0};
        std::atomic<uint64_t> correct_predictions{0};
        std::atomic<uint64_t> training_samples{0};
        std::atomic<uint64_t> model_updates{0};
        std::atomic<uint64_t> persistence_saves{0};
        std::atomic<uint64_t> persistence_loads{0};
        std::atomic<double> current_accuracy{0.0};
        std::atomic<double> average_prediction_time_ms{0.0};
        std::atomic<double> average_learning_time_ms{0.0};
    } stats_;
    
    // Thread safety
    mutable std::mutex engine_mutex_;
    
    // State
    std::atomic<bool> is_initialized_{false};
    std::atomic<bool> is_ready_{false};
    
    // Helper methods
    std::vector<float> extract_features(
        const std::vector<int>& current_experts,
        const std::vector<int>& recent_tokens,
        int layer_id,
        int position
    );
    
    void update_statistics(
        const std::vector<int>& predicted,
        const std::vector<int>& actual
    );
    
    std::string get_model_filepath() const;
};

// Neural network predictor
struct ml_predictor {
    // Model parameters
    int num_experts;
    int feature_size;
    int hidden_size;
    
    // Weights (stored as float for computation, can be quantized for storage)
    std::vector<float> w1;  // [feature_size, hidden_size]
    std::vector<float> b1;  // [hidden_size]
    std::vector<float> w2;  // [hidden_size, num_experts]
    std::vector<float> b2;  // [num_experts]
    
    // Forward pass
    std::vector<float> predict(const std::vector<float>& features);
    
    // Training step
    void train_step(
        const std::vector<float>& features,
        const std::vector<int>& actual_experts,
        float learning_rate,
        float momentum
    );
    
    // Initialize weights
    void initialize_weights();
    
    // Get model size in bytes
    size_t get_model_size() const;
};

// Online learner with momentum
struct online_learner {
    float learning_rate;
    float momentum;
    size_t training_samples;
    
    // Velocity for momentum
    std::vector<float> w1_velocity;
    std::vector<float> b1_velocity;
    std::vector<float> w2_velocity;
    std::vector<float> b2_velocity;
    
    // Update model parameters
    void update_parameters(
        ml_predictor& predictor,
        const std::vector<float>& feature_gradients,
        const std::vector<float>& output_gradients
    );
    
    // Compute gradients
    void compute_gradients(
        const ml_predictor& predictor,
        const std::vector<float>& features,
        const std::vector<int>& actual_experts,
        std::vector<float>& feature_gradients,
        std::vector<float>& output_gradients
    );
};

// Feature extraction
struct feature_extractor {
    // Token embedding cache (simple hash-based)
    std::unordered_map<int, std::vector<float>> token_embedding_cache;
    
    // Extract features from current state
    std::vector<float> extract_features(
        const std::vector<int>& current_experts,
        const std::vector<int>& recent_tokens,
        int layer_id,
        int position,
        int num_experts,
        int feature_size
    );
    
private:
    // Helper methods
    std::vector<float> embed_tokens(const std::vector<int>& tokens);
    std::vector<float> encode_experts(const std::vector<int>& experts, int num_experts);
    std::vector<float> encode_position(int position, int layer_id, int feature_size);
};

// Pattern analyzer for additional signals
struct pattern_analyzer {
    // Pattern databases
    std::vector<std::vector<int>> sequential_patterns;
    std::vector<std::vector<int>> cooccurrence_patterns;
    
    // Pattern mining
    void analyze_history(const std::vector<std::pair<std::vector<int>, std::vector<int>>>& history);
    
    // Find matching patterns
    std::vector<int> find_matching_patterns(const std::vector<int>& context);
    
    // Update with new sample
    void update_patterns(const std::vector<int>& experts, const std::vector<int>& next_experts);
};

// Persistence manager
struct persistence_manager {
    // Save model to file
    static bool save_model(
        const std::string& filepath,
        const ml_predictor& predictor,
        const pattern_analyzer& analyzer,
        const ml_prefetch_stats& stats,
        const ml_prefetch_config& config
    );
    
    // Load model from file
    static bool load_model(
        const std::string& filepath,
        ml_predictor& predictor,
        pattern_analyzer& analyzer,
        ml_prefetch_stats& stats,
        ml_prefetch_config& config
    );
    
    // Check if file exists
    static bool model_exists(const std::string& filepath);
    
    // Generate model path from model identifier
    static std::string generate_model_path(
        const std::string& cache_dir,
        const model_identifier& model_id
    );
};

// Factory function to create ML prefetch engine
std::unique_ptr<ml_prefetch_engine> create_ml_prefetch_engine(
    const ml_prefetch_config& config,
    const model_identifier& model_id,
    int num_experts
);

// Utility functions
namespace ml_prefetch_utils {

// Generate model hash from model configuration
std::string generate_model_hash(const std::string& model_path, int num_experts);

// Create model identifier
model_identifier create_model_identifier(
    const std::string& model_path,
    int num_experts
);

// Validate model compatibility
bool is_model_compatible(
    const model_identifier& saved_model,
    const model_identifier& current_model
);

} // namespace ml_prefetch_utils

} // namespace ggml_moe_ml

// C API for integration
extern "C" {

// Create ML prefetch engine
GGML_API void* ggml_moe_ml_prefetch_create(
    const char* model_path,
    int num_experts,
    const char* cache_dir,
    bool enable_learning
);

// Destroy ML prefetch engine
GGML_API void ggml_moe_ml_prefetch_destroy(void* engine);

// Predict next experts
GGML_API int ggml_moe_ml_prefetch_predict(
    void* engine,
    const int* current_experts,
    int num_current_experts,
    const int* recent_tokens,
    int num_recent_tokens,
    int layer_id,
    int position,
    int* predicted_experts,
    int max_predictions
);

// Update with actual experts (learning)
GGML_API void ggml_moe_ml_prefetch_update(
    void* engine,
    const int* predicted_experts,
    int num_predicted,
    const int* actual_experts,
    int num_actual,
    const int* context_tokens,
    int num_context_tokens,
    int layer_id,
    int position
);

// Save model
GGML_API bool ggml_moe_ml_prefetch_save(void* engine);

// Load model
GGML_API bool ggml_moe_ml_prefetch_load(void* engine);

// Get statistics
GGML_API void ggml_moe_ml_prefetch_get_stats(
    void* engine,
    ggml_moe_ml::ml_prefetch_stats* stats
);

// Reset model
GGML_API void ggml_moe_ml_prefetch_reset(void* engine);

} // extern "C"