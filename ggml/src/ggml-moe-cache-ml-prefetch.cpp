#include "ggml-moe-cache-ml-prefetch.h"
#include "ggml-moe-cache.h"
#include "ggml.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <set>

// Platform-specific headers for filesystem utilities
#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#include <pwd.h>
#endif
#include <cstdlib>

// Local filesystem utility functions to avoid dependency on common library
namespace {

// Create directory and all parent directories if they don't exist
bool fs_create_directory_with_parents(const std::string & path) {
#ifdef _WIN32
    int ret = _mkdir(path.c_str());
    if (ret == 0) return true;
    if (errno == EEXIST) return true;
    
    // Try creating parent directories
    size_t pos = path.find_last_of("/\\");
    if (pos != std::string::npos && pos > 0) {
        std::string parent = path.substr(0, pos);
        if (fs_create_directory_with_parents(parent)) {
            ret = _mkdir(path.c_str());
            return ret == 0 || errno == EEXIST;
        }
    }
    return false;
#else
    // Unix-like systems (Linux, macOS)
    int ret = mkdir(path.c_str(), 0755);
    if (ret == 0) return true;
    if (errno == EEXIST) return true;
    
    // Try creating parent directories
    size_t pos = path.find_last_of('/');
    if (pos != std::string::npos && pos > 0) {
        std::string parent = path.substr(0, pos);
        if (fs_create_directory_with_parents(parent)) {
            ret = mkdir(path.c_str(), 0755);
            return ret == 0 || errno == EEXIST;
        }
    }
    return false;
#endif
}

// Get the system cache directory
std::string fs_get_cache_directory() {
    std::string cache_directory;
    
    // Check environment variable first
    const char* xdg_cache_home = getenv("XDG_CACHE_HOME");
    if (xdg_cache_home && xdg_cache_home[0] != '\0') {
        cache_directory = xdg_cache_home;
        cache_directory += "/llama.cpp";
        return cache_directory;
    }
    
    // Check HOME environment variable
    const char* home_dir = getenv("HOME");
    if (home_dir && home_dir[0] != '\0') {
        cache_directory = home_dir;
        cache_directory += "/.cache/llama.cpp";
        return cache_directory;
    }
    
    // Fallback: try to get home directory from passwd
    struct passwd* pw = getpwuid(getuid());
    if (pw && pw->pw_dir) {
        cache_directory = pw->pw_dir;
        cache_directory += "/.cache/llama.cpp";
        return cache_directory;
    }
    
    // Last resort: use /tmp
    cache_directory = "/tmp/llama.cpp.cache";
    return cache_directory;
}

} // anonymous namespace

namespace ggml_moe_ml {

// Model identifier implementation
std::string model_identifier::to_string() const {
    std::ostringstream ss;
    ss << "Model[path=" << model_path << ",hash=" << model_hash 
       << ",experts=" << num_experts << "]";
    return ss.str();
}

std::string model_identifier::to_filename() const {
    std::string filename = model_hash;
    // Replace characters that might cause issues in filenames
    std::replace(filename.begin(), filename.end(), '/', '_');
    std::replace(filename.begin(), filename.end(), '\\', '_');
    std::replace(filename.begin(), filename.end(), ':', '_');
    return filename + "_e" + std::to_string(num_experts) + ".mlm";
}

bool model_identifier::operator==(const model_identifier& other) const {
    return model_path == other.model_path && 
           model_hash == other.model_hash && 
           num_experts == other.num_experts;
}

size_t model_identifier_hash::operator()(const model_identifier& id) const {
    size_t h1 = std::hash<std::string>{}(id.model_path);
    size_t h2 = std::hash<std::string>{}(id.model_hash);
    size_t h3 = std::hash<int>{}(id.num_experts);
    return h1 ^ (h2 << 1) ^ (h3 << 2);
}

// ML predictor implementation
std::vector<float> ml_predictor::predict(const std::vector<float>& features) {
    if (features.size() != static_cast<size_t>(feature_size)) {
        return std::vector<float>(num_experts, 0.0f);
    }

    // Forward pass: features -> hidden layer
    std::vector<float> hidden(hidden_size, 0.0f);
    
    // Matrix multiplication: hidden = features * w1 + b1
    for (int i = 0; i < hidden_size; ++i) {
        float sum = b1[i];
        for (int j = 0; j < feature_size; ++j) {
            sum += features[j] * w1[j * hidden_size + i];
        }
        // ReLU activation
        hidden[i] = std::max(0.0f, sum);
    }

    // Forward pass: hidden -> output layer
    std::vector<float> output(num_experts, 0.0f);
    for (int i = 0; i < num_experts; ++i) {
        float sum = b2[i];
        for (int j = 0; j < hidden_size; ++j) {
            sum += hidden[j] * w2[j * num_experts + i];
        }
        output[i] = sum;
    }

    // Apply softmax to get probabilities
    float max_val = *std::max_element(output.begin(), output.end());
    float sum_exp = 0.0f;
    
    for (float& val : output) {
        val = std::exp(val - max_val);
        sum_exp += val;
    }
    
    for (float& val : output) {
        val /= sum_exp;
    }

    return output;
}

void ml_predictor::initialize_weights() {
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    // Initialize w1: [feature_size, hidden_size]
    w1.resize(feature_size * hidden_size);
    for (float& weight : w1) {
        weight = dist(rng);
    }

    // Initialize b1: [hidden_size]
    b1.resize(hidden_size, 0.0f);

    // Initialize w2: [hidden_size, num_experts]
    w2.resize(hidden_size * num_experts);
    for (float& weight : w2) {
        weight = dist(rng);
    }

    // Initialize b2: [num_experts]
    b2.resize(num_experts, 0.0f);
}

size_t ml_predictor::get_model_size() const {
    return (w1.size() + b1.size() + w2.size() + b2.size()) * sizeof(float);
}

// Feature extractor implementation
std::vector<float> feature_extractor::extract_features(
    const std::vector<int>& current_experts,
    const std::vector<int>& recent_tokens,
    int layer_id,
    int position,
    int num_experts,
    int feature_size
) {
    std::vector<float> features(feature_size, 0.0f);
    int idx = 0;

    // 1. One-hot encode current experts (first num_experts features)
    for (int expert_id : current_experts) {
        if (expert_id >= 0 && expert_id < num_experts && idx < feature_size) {
            features[idx + expert_id] = 1.0f;
        }
    }
    idx += num_experts;

    // 2. Embed recent tokens (next 128 features)
    auto token_embeddings = embed_tokens(recent_tokens);
    int token_embed_size = std::min(128, feature_size - idx);
    for (int i = 0; i < token_embed_size && i < (int)token_embeddings.size(); ++i) {
        features[idx + i] = token_embeddings[i];
    }
    idx += token_embed_size;

    // 3. Position and layer encoding (next 64 features)
    auto pos_enc = encode_position(position, layer_id, std::min(64, feature_size - idx));
    for (int i = 0; i < (int)pos_enc.size(); ++i) {
        features[idx + i] = pos_enc[i];
    }
    idx += pos_enc.size();

    // 4. Fill remaining features with zeros or additional encodings
    // Could add expert frequency, timing info, etc.

    return features;
}

std::vector<float> feature_extractor::embed_tokens(const std::vector<int>& tokens) {
    std::vector<float> embeddings;
    embeddings.reserve(tokens.size() * 4); // Rough estimate

    // Simple hash-based embedding - in practice, could use actual token embeddings
    for (int token : tokens) {
        // Create a simple deterministic embedding from token ID
        float embed_1 = std::sin(token * 0.01f);
        float embed_2 = std::cos(token * 0.01f);
        float embed_3 = std::sin(token * 0.001f);
        float embed_4 = std::cos(token * 0.001f);
        
        embeddings.push_back(embed_1);
        embeddings.push_back(embed_2);
        embeddings.push_back(embed_3);
        embeddings.push_back(embed_4);
    }

    // Average pool if too many tokens
    if (embeddings.size() > 128) {
        std::vector<float> pooled(128, 0.0f);
        for (size_t i = 0; i < embeddings.size(); ++i) {
            pooled[i % 128] += embeddings[i];
        }
        for (float& val : pooled) {
            val /= (embeddings.size() / 128 + 1);
        }
        return pooled;
    }

    return embeddings;
}

std::vector<float> feature_extractor::encode_position(int position, int layer_id, int size) {
    std::vector<float> encoding(size, 0.0f);
    
    // Sinusoidal position encoding
    for (int i = 0; i < size; i += 2) {
        float pos_val = position * 0.001f;
        float layer_val = layer_id * 0.01f;
        
        encoding[i] = std::sin(pos_val + layer_val);
        if (i + 1 < size) {
            encoding[i + 1] = std::cos(pos_val + layer_val);
        }
    }
    
    return encoding;
}

// Pattern analyzer implementation
void pattern_analyzer::analyze_history(
    const std::vector<std::pair<std::vector<int>, std::vector<int>>>& history
) {
    // Clear existing patterns
    sequential_patterns.clear();
    cooccurrence_patterns.clear();

    // Analyze sequential patterns
    std::unordered_map<int, std::unordered_map<int, int>> seq_counts;
    
    for (const auto& [experts, next_experts] : history) {
        for (int expert : experts) {
            for (int next_expert : next_experts) {
                seq_counts[expert][next_expert]++;
            }
        }
    }

    // Convert to patterns
    for (const auto& [expert, next_map] : seq_counts) {
        std::vector<int> pattern;
        pattern.push_back(expert);
        
        // Find most common next experts
        std::vector<std::pair<int, int>> next_counts;
        for (const auto& [next_expert, count] : next_map) {
            next_counts.push_back({next_expert, count});
        }
        
        std::sort(next_counts.begin(), next_counts.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Add top next experts to pattern
        for (int i = 0; i < std::min(3, (int)next_counts.size()); ++i) {
            pattern.push_back(next_counts[i].first);
        }
        
        if (pattern.size() > 1) {
            sequential_patterns.push_back(pattern);
        }
    }
}

std::vector<int> pattern_analyzer::find_matching_patterns(const std::vector<int>& context) {
    std::vector<int> predictions;
    std::set<int> seen_experts(context.begin(), context.end());

    // Find matching sequential patterns
    for (const auto& pattern : sequential_patterns) {
        if (pattern.empty()) continue;
        
        // Check if first element of pattern is in context
        if (std::find(context.begin(), context.end(), pattern[0]) != context.end()) {
            // Add remaining pattern elements as predictions
            for (size_t i = 1; i < pattern.size(); ++i) {
                int expert = pattern[i];
                if (seen_experts.find(expert) == seen_experts.end()) {
                    predictions.push_back(expert);
                    seen_experts.insert(expert);
                }
            }
        }
    }

    return predictions;
}

void pattern_analyzer::update_patterns(
    const std::vector<int>& experts,
    const std::vector<int>& next_experts
) {
    // Add to sequential patterns (simplified)
    if (!experts.empty() && !next_experts.empty()) {
        std::vector<int> pattern;
        pattern.push_back(experts[0]); // Use first expert as context
        pattern.insert(pattern.end(), next_experts.begin(), next_experts.end());
        
        if (pattern.size() > 1) {
            sequential_patterns.push_back(pattern);
            
            // Keep only recent patterns to avoid unbounded growth
            if (sequential_patterns.size() > 1000) {
                sequential_patterns.erase(sequential_patterns.begin(),
                                        sequential_patterns.begin() + 100);
            }
        }
    }
}

// Persistence manager implementation
bool persistence_manager::save_model(
    const std::string& filepath,
    const ml_predictor& predictor,
    const pattern_analyzer& analyzer,
    const ml_prefetch_stats& stats,
    const ml_prefetch_config& config
) {
    // Suppress unused parameter warning
    (void)analyzer;
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        return false;
    }

    // Write header
    const char* header = "MLMOE_v1";
    file.write(header, 8);

    // Write model metadata
    file.write(reinterpret_cast<const char*>(&predictor.num_experts), sizeof(int));
    file.write(reinterpret_cast<const char*>(&predictor.feature_size), sizeof(int));
    file.write(reinterpret_cast<const char*>(&predictor.hidden_size), sizeof(int));

    // Write weights
    size_t w1_size = predictor.w1.size();
    file.write(reinterpret_cast<const char*>(&w1_size), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(predictor.w1.data()), w1_size * sizeof(float));

    size_t b1_size = predictor.b1.size();
    file.write(reinterpret_cast<const char*>(&b1_size), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(predictor.b1.data()), b1_size * sizeof(float));

    size_t w2_size = predictor.w2.size();
    file.write(reinterpret_cast<const char*>(&w2_size), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(predictor.w2.data()), w2_size * sizeof(float));

    size_t b2_size = predictor.b2.size();
    file.write(reinterpret_cast<const char*>(&b2_size), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(predictor.b2.data()), b2_size * sizeof(float));

    // Write statistics
    file.write(reinterpret_cast<const char*>(&stats.training_samples), sizeof(uint64_t));
    file.write(reinterpret_cast<const char*>(&stats.current_accuracy), sizeof(double));

    // Write configuration
    file.write(reinterpret_cast<const char*>(&config.learning_rate), sizeof(float));

    file.close();
    return true;
}

bool persistence_manager::load_model(
    const std::string& filepath,
    ml_predictor& predictor,
    pattern_analyzer& analyzer,
    ml_prefetch_stats& stats,
    ml_prefetch_config& config
) {
    // Suppress unused parameter warning
    (void)analyzer;
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        return false;
    }

    // Read header
    char header[9];
    file.read(header, 8);
    header[8] = '\0';

    if (std::string(header) != "MLMOE_v1") {
        return false; // Invalid file format
    }

    // Read model metadata
    file.read(reinterpret_cast<char*>(&predictor.num_experts), sizeof(int));
    file.read(reinterpret_cast<char*>(&predictor.feature_size), sizeof(int));
    file.read(reinterpret_cast<char*>(&predictor.hidden_size), sizeof(int));

    // Read weights
    size_t w1_size;
    file.read(reinterpret_cast<char*>(&w1_size), sizeof(size_t));
    predictor.w1.resize(w1_size);
    file.read(reinterpret_cast<char*>(predictor.w1.data()), w1_size * sizeof(float));

    size_t b1_size;
    file.read(reinterpret_cast<char*>(&b1_size), sizeof(size_t));
    predictor.b1.resize(b1_size);
    file.read(reinterpret_cast<char*>(predictor.b1.data()), b1_size * sizeof(float));

    size_t w2_size;
    file.read(reinterpret_cast<char*>(&w2_size), sizeof(size_t));
    predictor.w2.resize(w2_size);
    file.read(reinterpret_cast<char*>(predictor.w2.data()), w2_size * sizeof(float));

    size_t b2_size;
    file.read(reinterpret_cast<char*>(&b2_size), sizeof(size_t));
    predictor.b2.resize(b2_size);
    file.read(reinterpret_cast<char*>(predictor.b2.data()), b2_size * sizeof(float));

    // Read statistics
    file.read(reinterpret_cast<char*>(&stats.training_samples), sizeof(uint64_t));
    file.read(reinterpret_cast<char*>(&stats.current_accuracy), sizeof(double));

    // Read configuration
    file.read(reinterpret_cast<char*>(&config.learning_rate), sizeof(float));

    file.close();
    return true;
}

bool persistence_manager::model_exists(const std::string& filepath) {
    std::ifstream file(filepath);
    return file.good();
}

std::string persistence_manager::generate_model_path(
    const std::string& cache_dir,
    const model_identifier& model_id
) {
    // Create cache directory if it doesn't exist
    fs_create_directory_with_parents(cache_dir);
    
    // Generate filename from model ID
    std::string filename = model_id.to_filename();
    
    // Combine directory and filename
    if (cache_dir.empty() || cache_dir.back() == '/') {
        return cache_dir + filename;
    } else {
        return cache_dir + "/" + filename;
    }
}

// ML prefetch engine implementation
ml_prefetch_engine::ml_prefetch_engine(const ml_prefetch_config& config) 
    : config_(config), is_initialized_(false), is_ready_(false) {
}

ml_prefetch_engine::~ml_prefetch_engine() {
    // Save model on destruction if persistence is enabled
    if (is_initialized_ && config_.enable_persistence) {
        save_model();
    }
}

bool ml_prefetch_engine::initialize(const model_identifier& model_id, int num_experts) {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    
    model_id_ = model_id;
    
    // Create components
    predictor_ = std::make_unique<ml_predictor>();
    predictor_->num_experts = num_experts;
    predictor_->feature_size = config_.feature_size;
    predictor_->hidden_size = config_.hidden_size;
    predictor_->initialize_weights();
    
    learner_ = std::make_unique<online_learner>();
    learner_->learning_rate = config_.learning_rate;
    learner_->momentum = config_.momentum;
    learner_->training_samples = 0;
    
    feature_extractor_ = std::make_unique<feature_extractor>();
    pattern_analyzer_ = std::make_unique<pattern_analyzer>();
    persistence_manager_ = std::make_unique<persistence_manager>();
    
    // Try to load existing model
    if (config_.enable_persistence && !config_.reset_on_startup) {
        std::string model_path = get_model_filepath();
        if (persistence_manager_->model_exists(model_path)) {
            if (load_model()) {
                is_ready_ = stats_.training_samples >= config_.min_training_samples &&
                           stats_.current_accuracy >= config_.accuracy_threshold;
            }
        }
    }
    
    is_initialized_ = true;
    return true;
}

std::vector<int> ml_prefetch_engine::predict_next_experts(
    const std::vector<int>& current_experts,
    const std::vector<int>& recent_tokens,
    int layer_id,
    int position,
    size_t top_k
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::lock_guard<std::mutex> lock(engine_mutex_);
    
    if (!is_initialized_ || !predictor_) {
        return std::vector<int>();
    }

    // Extract features
    std::vector<float> features = extract_features(
        current_experts, recent_tokens, layer_id, position
    );

    // Get predictions from ML model
    std::vector<float> probabilities = predictor_->predict(features);

    // Get top-k predictions
    std::vector<int> predicted_experts;
    std::set<int> seen_experts(current_experts.begin(), current_experts.end());
    
    // Create list of expert IDs with their probabilities
    std::vector<std::pair<int, float>> expert_probs;
    for (int i = 0; i < predictor_->num_experts; ++i) {
        expert_probs.push_back({i, probabilities[i]});
    }
    
    // Sort by probability
    std::sort(expert_probs.begin(), expert_probs.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Select top-k unique experts
    for (const auto& [expert_id, prob] : expert_probs) {
        if (seen_experts.find(expert_id) == seen_experts.end()) {
            predicted_experts.push_back(expert_id);
            seen_experts.insert(expert_id);
            
            if (predicted_experts.size() >= top_k) {
                break;
            }
        }
    }

    // Update statistics
    stats_.total_predictions++;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double ms = duration.count() / 1000.0;
    
    double old_time = stats_.average_prediction_time_ms.load();
    stats_.average_prediction_time_ms.store(
        (old_time * (stats_.total_predictions - 1) + ms) / stats_.total_predictions
    );

    return predicted_experts;
}

void ml_prefetch_engine::update_with_actual(
    const std::vector<int>& predicted_experts,
    const std::vector<int>& actual_experts,
    const std::vector<int>& context_tokens,
    int layer_id,
    int position
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::lock_guard<std::mutex> lock(engine_mutex_);
    
    if (!is_initialized_ || !learner_ || !predictor_) {
        return;
    }

    // Extract features (reuse the same feature extraction as prediction)
    std::vector<float> features = extract_features(
        predicted_experts, context_tokens, layer_id, position
    );

    // Create target distribution (one-hot encoding of actual experts)
    std::vector<float> target_distribution(predictor_->num_experts, 0.0f);
    for (int expert : actual_experts) {
        if (expert >= 0 && expert < predictor_->num_experts) {
            target_distribution[expert] = 1.0f / actual_experts.size();
        }
    }

    // Simple gradient descent update
    // Forward pass
    std::vector<float> hidden(predictor_->hidden_size, 0.0f);
    for (int i = 0; i < predictor_->hidden_size; ++i) {
        float sum = predictor_->b1[i];
        for (int j = 0; j < predictor_->feature_size; ++j) {
            sum += features[j] * predictor_->w1[j * predictor_->hidden_size + i];
        }
        hidden[i] = std::max(0.0f, sum); // ReLU
    }

    std::vector<float> output(predictor_->num_experts, 0.0f);
    for (int i = 0; i < predictor_->num_experts; ++i) {
        float sum = predictor_->b2[i];
        for (int j = 0; j < predictor_->hidden_size; ++j) {
            sum += hidden[j] * predictor_->w2[j * predictor_->num_experts + i];
        }
        output[i] = sum;
    }

    // Compute gradients (simplified cross-entropy gradient)
    std::vector<float> output_grad(predictor_->num_experts, 0.0f);
    for (int i = 0; i < predictor_->num_experts; ++i) {
        float prob = std::exp(output[i]);
        output_grad[i] = prob - target_distribution[i]; // Simplified gradient
    }

    // Backpropagate to hidden layer
    std::vector<float> hidden_grad(predictor_->hidden_size, 0.0f);
    for (int i = 0; i < predictor_->hidden_size; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < predictor_->num_experts; ++j) {
            sum += output_grad[j] * predictor_->w2[i * predictor_->num_experts + j];
        }
        hidden_grad[i] = sum * (hidden[i] > 0 ? 1.0f : 0.0f); // ReLU derivative
    }

    // Update weights with momentum
    if (learner_->w1_velocity.empty()) {
        learner_->w1_velocity.resize(predictor_->w1.size(), 0.0f);
        learner_->b1_velocity.resize(predictor_->b1.size(), 0.0f);
        learner_->w2_velocity.resize(predictor_->w2.size(), 0.0f);
        learner_->b2_velocity.resize(predictor_->b2.size(), 0.0f);
    }

    // Update w2
    for (int i = 0; i < predictor_->hidden_size; ++i) {
        for (int j = 0; j < predictor_->num_experts; ++j) {
            int idx = i * predictor_->num_experts + j;
            float grad = output_grad[j] * hidden[i];
            learner_->w2_velocity[idx] = learner_->momentum * learner_->w2_velocity[idx] - 
                                         learner_->learning_rate * grad;
            predictor_->w2[idx] += learner_->w2_velocity[idx];
        }
    }

    // Update b2
    for (int i = 0; i < predictor_->num_experts; ++i) {
        learner_->b2_velocity[i] = learner_->momentum * learner_->b2_velocity[i] - 
                                   learner_->learning_rate * output_grad[i];
        predictor_->b2[i] += learner_->b2_velocity[i];
    }

    // Update w1
    for (int i = 0; i < predictor_->feature_size; ++i) {
        for (int j = 0; j < predictor_->hidden_size; ++j) {
            int idx = i * predictor_->hidden_size + j;
            float grad = hidden_grad[j] * features[i];
            learner_->w1_velocity[idx] = learner_->momentum * learner_->w1_velocity[idx] - 
                                         learner_->learning_rate * grad;
            predictor_->w1[idx] += learner_->w1_velocity[idx];
        }
    }

    // Update b1
    for (int i = 0; i < predictor_->hidden_size; ++i) {
        learner_->b1_velocity[i] = learner_->momentum * learner_->b1_velocity[i] - 
                                   learner_->learning_rate * hidden_grad[i];
        predictor_->b1[i] += learner_->b1_velocity[i];
    }

    // Update statistics
    learner_->training_samples++;
    stats_.training_samples = learner_->training_samples;
    stats_.model_updates++;

    // Calculate accuracy
    int correct = 0;
    for (int predicted : predicted_experts) {
        if (std::find(actual_experts.begin(), actual_experts.end(), predicted) != actual_experts.end()) {
            correct++;
        }
    }
    
    if (!predicted_experts.empty()) {
        float accuracy = static_cast<float>(correct) / predicted_experts.size();
        stats_.correct_predictions += correct;
        
        // Update running accuracy
        float current_acc = stats_.current_accuracy.load();
        float new_acc = (current_acc * (stats_.training_samples - 1) + accuracy) / stats_.training_samples;
        stats_.current_accuracy = new_acc;
    }

    // Update pattern analyzer
    if (pattern_analyzer_) {
        pattern_analyzer_->update_patterns(predicted_experts, actual_experts);
    }

    // Check if model is ready
    is_ready_ = stats_.training_samples >= config_.min_training_samples &&
                stats_.current_accuracy.load() >= config_.accuracy_threshold;

    // Auto-save periodically
    if (config_.enable_persistence && 
        stats_.training_samples % config_.save_interval_samples == 0) {
        save_model();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double ms = duration.count() / 1000.0;
    
    double old_time = stats_.average_learning_time_ms.load();
    stats_.average_learning_time_ms.store(
        (old_time * (stats_.training_samples - 1) + ms) / stats_.training_samples
    );
}

float ml_prefetch_engine::get_prediction_confidence() const {
    return stats_.current_accuracy.load();
}

void ml_prefetch_engine::get_stats(ml_prefetch_stats& stats) const {
    // Manually copy each atomic value to the output parameter
    stats.total_predictions = stats_.total_predictions.load();
    stats.correct_predictions = stats_.correct_predictions.load();
    stats.training_samples = stats_.training_samples.load();
    stats.model_updates = stats_.model_updates.load();
    stats.persistence_saves = stats_.persistence_saves.load();
    stats.persistence_loads = stats_.persistence_loads.load();
    stats.current_accuracy = stats_.current_accuracy.load();
    stats.average_prediction_time_ms = stats_.average_prediction_time_ms.load();
    stats.average_learning_time_ms = stats_.average_learning_time_ms.load();
}

bool ml_prefetch_engine::save_model() {
    if (!persistence_manager_ || !predictor_) {
        return false;
    }

    std::string filepath = get_model_filepath();
    bool success = persistence_manager_->save_model(
        filepath, *predictor_, *pattern_analyzer_, stats_, config_
    );

    if (success) {
        stats_.persistence_saves++;
    }

    return success;
}

bool ml_prefetch_engine::load_model() {
    if (!persistence_manager_ || !predictor_) {
        return false;
    }

    std::string filepath = get_model_filepath();
    bool success = persistence_manager_->load_model(
        filepath, *predictor_, *pattern_analyzer_, stats_, config_
    );

    if (success) {
        stats_.persistence_loads++;
        if (learner_) {
            learner_->training_samples = stats_.training_samples;
        }
    }

    return success;
}

void ml_prefetch_engine::reset_model() {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    
    if (predictor_) {
        predictor_->initialize_weights();
    }
    
    if (learner_) {
        learner_->training_samples = 0;
        learner_->w1_velocity.clear();
        learner_->b1_velocity.clear();
        learner_->w2_velocity.clear();
        learner_->b2_velocity.clear();
    }
    
    if (pattern_analyzer_) {
        pattern_analyzer_->sequential_patterns.clear();
        pattern_analyzer_->cooccurrence_patterns.clear();
    }
    
    // Reset statistics
    stats_.total_predictions = 0;
    stats_.correct_predictions = 0;
    stats_.training_samples = 0;
    stats_.model_updates = 0;
    stats_.persistence_saves = 0;
    stats_.persistence_loads = 0;
    stats_.current_accuracy = 0.0;
    stats_.average_prediction_time_ms = 0.0;
    stats_.average_learning_time_ms = 0.0;
    
    is_ready_ = false;
}

bool ml_prefetch_engine::is_ready() const {
    return is_ready_.load();
}

std::vector<float> ml_prefetch_engine::extract_features(
    const std::vector<int>& current_experts,
    const std::vector<int>& recent_tokens,
    int layer_id,
    int position
) {
    if (!feature_extractor_) {
        return std::vector<float>(config_.feature_size, 0.0f);
    }

    return feature_extractor_->extract_features(
        current_experts, recent_tokens, layer_id, position,
        predictor_ ? predictor_->num_experts : 8,
        config_.feature_size
    );
}

void ml_prefetch_engine::update_statistics(
    const std::vector<int>& predicted,
    const std::vector<int>& actual
) {
    // Suppress unused parameter warnings
    (void)predicted;
    (void)actual;
    // This is called from update_with_actual, so we don't need to do much here
    // Statistics are updated in the main update method
}

std::string ml_prefetch_engine::get_model_filepath() const {
    if (!persistence_manager_) {
        return "";
    }

    std::string cache_dir = config_.model_cache_dir;
    if (cache_dir.empty()) {
        cache_dir = fs_get_cache_directory() + "/moe_ml";
    }

    return persistence_manager_->generate_model_path(cache_dir, model_id_);
}

// Factory function
std::unique_ptr<ml_prefetch_engine> create_ml_prefetch_engine(
    const ml_prefetch_config& config,
    const model_identifier& model_id,
    int num_experts
) {
    auto engine = std::make_unique<ml_prefetch_engine>(config);
    if (engine->initialize(model_id, num_experts)) {
        return engine;
    }
    return nullptr;
}

// Utility functions
namespace ml_prefetch_utils {

std::string generate_model_hash(const std::string& model_path, int num_experts) {
    // Simple hash combining model path and number of experts
    std::hash<std::string> hasher;
    size_t hash1 = hasher(model_path);
    size_t hash2 = std::hash<int>{}(num_experts);
    
    // Combine hashes
    size_t combined = hash1 ^ (hash2 + 0x9e3779b97f4a7c15ULL + (hash1 << 6) + (hash1 >> 2));
    
    // Convert to hex string
    std::ostringstream ss;
    ss << std::hex << std::setw(16) << std::setfill('0') << combined;
    return ss.str();
}

model_identifier create_model_identifier(
    const std::string& model_path,
    int num_experts
) {
    model_identifier id;
    id.model_path = model_path;
    id.model_hash = generate_model_hash(model_path, num_experts);
    id.num_experts = num_experts;
    return id;
}

bool is_model_compatible(
    const model_identifier& saved_model,
    const model_identifier& current_model
) {
    return saved_model.model_hash == current_model.model_hash &&
           saved_model.num_experts == current_model.num_experts;
}

} // namespace ml_prefetch_utils

} // namespace ggml_moe_ml

// C API implementation
extern "C" {

void* ggml_moe_ml_prefetch_create(
    const char* model_path,
    int num_experts,
    const char* cache_dir,
    bool enable_learning
) {
    if (!model_path || num_experts <= 0) {
        return nullptr;
    }

    ggml_moe_ml::ml_prefetch_config config;
    config.enabled = enable_learning;
    if (cache_dir) {
        config.model_cache_dir = cache_dir;
    }

    auto model_id = ggml_moe_ml::ml_prefetch_utils::create_model_identifier(
        std::string(model_path), num_experts
    );

    auto engine = ggml_moe_ml::create_ml_prefetch_engine(config, model_id, num_experts);
    return engine.release();
}

void ggml_moe_ml_prefetch_destroy(void* engine) {
    delete static_cast<ggml_moe_ml::ml_prefetch_engine*>(engine);
}

int ggml_moe_ml_prefetch_predict(
    void* engine,
    const int* current_experts,
    int num_current_experts,
    const int* recent_tokens,
    int num_recent_tokens,
    int layer_id,
    int position,
    int* predicted_experts,
    int max_predictions
) {
    if (!engine || !current_experts || !predicted_experts || max_predictions <= 0) {
        return 0;
    }

    auto* ml_engine = static_cast<ggml_moe_ml::ml_prefetch_engine*>(engine);
    
    std::vector<int> current_experts_vec(current_experts, current_experts + num_current_experts);
    std::vector<int> recent_tokens_vec;
    if (recent_tokens && num_recent_tokens > 0) {
        recent_tokens_vec.assign(recent_tokens, recent_tokens + num_recent_tokens);
    }

    auto predictions = ml_engine->predict_next_experts(
        current_experts_vec, recent_tokens_vec, layer_id, position, max_predictions
    );

    int num_predictions = std::min(max_predictions, (int)predictions.size());
    for (int i = 0; i < num_predictions; ++i) {
        predicted_experts[i] = predictions[i];
    }

    return num_predictions;
}

void ggml_moe_ml_prefetch_update(
    void* engine,
    const int* predicted_experts,
    int num_predicted,
    const int* actual_experts,
    int num_actual,
    const int* context_tokens,
    int num_context_tokens,
    int layer_id,
    int position
) {
    if (!engine || !predicted_experts || !actual_experts) {
        return;
    }

    auto* ml_engine = static_cast<ggml_moe_ml::ml_prefetch_engine*>(engine);
    
    std::vector<int> predicted_vec(predicted_experts, predicted_experts + num_predicted);
    std::vector<int> actual_vec(actual_experts, actual_experts + num_actual);
    std::vector<int> context_tokens_vec;
    if (context_tokens && num_context_tokens > 0) {
        context_tokens_vec.assign(context_tokens, context_tokens + num_context_tokens);
    }

    ml_engine->update_with_actual(
        predicted_vec, actual_vec, context_tokens_vec, layer_id, position
    );
}

bool ggml_moe_ml_prefetch_save(void* engine) {
    if (!engine) return false;
    auto* ml_engine = static_cast<ggml_moe_ml::ml_prefetch_engine*>(engine);
    return ml_engine->save_model();
}

bool ggml_moe_ml_prefetch_load(void* engine) {
    if (!engine) return false;
    auto* ml_engine = static_cast<ggml_moe_ml::ml_prefetch_engine*>(engine);
    return ml_engine->load_model();
}

void ggml_moe_ml_prefetch_get_stats(
    void* engine,
    ggml_moe_ml::ml_prefetch_stats* stats
) {
    if (!engine || !stats) return;
    auto* ml_engine = static_cast<ggml_moe_ml::ml_prefetch_engine*>(engine);
    
    // Get stats from engine (fills the provided stats object)
    ml_engine->get_stats(*stats);
}

void ggml_moe_ml_prefetch_reset(void* engine) {
    if (!engine) return;
    auto* ml_engine = static_cast<ggml_moe_ml::ml_prefetch_engine*>(engine);
    ml_engine->reset_model();
}

} // extern "C"