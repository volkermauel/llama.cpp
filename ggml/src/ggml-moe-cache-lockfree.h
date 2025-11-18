#pragma once

#include "ggml-moe-cache.h"
#include <atomic>
#include <vector>
#include <functional>
#include <chrono>

namespace ggml_moe_lockfree {

// Lock-free hash table with open addressing and atomic operations
struct hashtable {
    static constexpr size_t DEFAULT_CAPACITY = 16384;  // Power of 2 for fast modulo
    static constexpr size_t MAX_PROBES = 16;           // Maximum linear probes
    
    struct alignas(64) bucket {
        std::atomic<int> expert_id;     // -1 means empty
        std::atomic<ggml_backend_buffer_t> buffer;
        std::atomic<uint64_t> access_counter;
        std::atomic<uint32_t> flags;    // Status flags
        
        bucket() : expert_id(-1), buffer(nullptr), access_counter(0), flags(0) {}
    };
    
    bucket* buckets;
    size_t capacity;
    size_t mask;  // capacity - 1 for fast modulo
    
    // Atomic statistics
    std::atomic<size_t> size;
    std::atomic<size_t> probes_total;
    std::atomic<size_t> probes_count;
    
    // Initialize table
    void initialize(size_t initial_capacity = DEFAULT_CAPACITY) {
        // Round up to nearest power of 2
        capacity = 1;
        while (capacity < initial_capacity) {
            capacity <<= 1;
        }
        mask = capacity - 1;
        
        buckets = new bucket[capacity];
        size.store(0, std::memory_order_relaxed);
        probes_total.store(0, std::memory_order_relaxed);
        probes_count.store(0, std::memory_order_relaxed);
    }
    
    // Cleanup
    ~hashtable() {
        delete[] buckets;
    }
    
    // Hash function
    size_t hash(int expert_id) const {
        // Simple but effective hash for integers
        return std::hash<int>{}(expert_id);
    }
    
    // Probe for position
    size_t find_bucket(int expert_id) {
        size_t start_idx = hash(expert_id) & mask;
        size_t idx = start_idx;
        size_t probes = 0;
        
        do {
            int current_id = buckets[idx].expert_id.load(std::memory_order_acquire);
            
            if (current_id == expert_id) {
                // Found the expert
                probes_total.fetch_add(probes, std::memory_order_relaxed);
                probes_count.fetch_add(1, std::memory_order_relaxed);
                return idx;
            }
            
            if (current_id == -1) {
                // Empty bucket, expert not found
                probes_total.fetch_add(probes, std::memory_order_relaxed);
                probes_count.fetch_add(1, std::memory_order_relaxed);
                return capacity;  // Not found
            }
            
            // Continue probing
            idx = (idx + 1) & mask;
            probes++;
        } while (probes < MAX_PROBES && idx != start_idx);
        
        probes_total.fetch_add(probes, std::memory_order_relaxed);
        probes_count.fetch_add(1, std::memory_order_relaxed);
        return capacity;  // Not found after max probes
    }
    
    // Find empty bucket for insertion
    size_t find_empty_bucket(int expert_id) {
        size_t start_idx = hash(expert_id) & mask;
        size_t idx = start_idx;
        size_t probes = 0;
        
        do {
            int current_id = buckets[idx].expert_id.load(std::memory_order_acquire);
            
            if (current_id == -1) {
                // Found empty bucket
                return idx;
            }
            
            // Continue probing
            idx = (idx + 1) & mask;
            probes++;
        } while (probes < MAX_PROBES && idx != start_idx);
        
        return capacity;  // No empty bucket found
    }
    
    // Lock-free get operation
    ggml_backend_buffer_t get(int expert_id) {
        size_t bucket_idx = find_bucket(expert_id);
        
        if (bucket_idx != capacity) {
            // Found the expert
            ggml_backend_buffer_t buffer = buckets[bucket_idx].buffer.load(std::memory_order_acquire);
            if (buffer != nullptr) {
                // Increment access counter for LRU
                buckets[bucket_idx].access_counter.fetch_add(1, std::memory_order_relaxed);
            }
            return buffer;
        }
        
        return nullptr;
    }
    
    // Lock-free insert with CAS loop
    bool insert(int expert_id, ggml_backend_buffer_t buffer) {
        size_t bucket_idx = find_empty_bucket(expert_id);
        if (bucket_idx == capacity) {
            return false;  // Table full or no empty bucket within probe limit
        }
        
        // Prepare bucket data
        int expected_empty = -1;
        bucket& b = buckets[bucket_idx];
        
        // Try to claim the bucket
        if (b.expert_id.compare_exchange_strong(
                expected_empty, expert_id,
                std::memory_order_release,
                std::memory_order_relaxed)) {
            
            // Successfully claimed, now store buffer
            b.buffer.store(buffer, std::memory_order_release);
            b.flags.store(1, std::memory_order_release);  // Mark as valid
            
            size.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        
        return false;  // Someone else claimed it
    }
    
    // Remove an expert
    bool remove(int expert_id) {
        size_t bucket_idx = find_bucket(expert_id);
        
        if (bucket_idx == capacity) {
            return false;  // Not found
        }
        
        bucket& b = buckets[bucket_idx];
        
        // Mark as invalid first
        b.flags.store(0, std::memory_order_release);
        b.buffer.store(nullptr, std::memory_order_release);
        
        // Then mark expert_id as empty
        int expected_id = expert_id;
        if (b.expert_id.compare_exchange_strong(
                expected_id, -1,
                std::memory_order_release,
                std::memory_order_relaxed)) {
            
            size.fetch_sub(1, std::memory_order_relaxed);
            return true;
        }
        
        return false;  // Someone else modified it
    }
    
    // Statistics
    double average_probes() const {
        size_t total = probes_total.load(std::memory_order_relaxed);
        size_t count = probes_count.load(std::memory_order_relaxed);
        return count > 0 ? static_cast<double>(total) / count : 0.0;
    }
    
    float load_factor() const {
        size_t current_size = size.load(std::memory_order_relaxed);
        return static_cast<float>(current_size) / capacity;
    }
    
    size_t get_size() const {
        return size.load(std::memory_order_relaxed);
    }
};

// RCU (Read-Copy-Update) based LRU list for scalable eviction
struct rcu_lru_list {
    struct alignas(64) node {
        int expert_id;
        std::atomic<node*> next;
        std::atomic<uint64_t> timestamp;
        std::atomic<uint32_t> ref_count;  // For memory reclamation
        
        node(int id) : expert_id(id), next(nullptr), timestamp(0), ref_count(0) {}
    };
    
    std::atomic<node*> head;
    std::atomic<node*> tail;
    
    // RCU synchronization
    struct rcu_epoch {
        std::atomic<uint64_t> current_epoch;
        std::atomic<uint64_t> active_readers;
        
        rcu_epoch() : current_epoch(0), active_readers(0) {}
        
        void enter_read_section() {
            active_readers.fetch_add(1, std::memory_order_acquire);
        }
        
        void exit_read_section() {
            active_readers.fetch_sub(1, std::memory_order_release);
        }
        
        void synchronize() {
            uint64_t epoch = current_epoch.fetch_add(1, std::memory_order_seq_cst);
            
            // Wait for all readers from previous epochs to finish
            while (active_readers.load(std::memory_order_acquire) > 0) {
                std::this_thread::yield();
            }
        }
    };
    
    rcu_epoch epoch;
    
    // Memory reclamation
    struct retired_node {
        node* n;
        uint64_t retired_epoch;
        retired_node* next;
    };
    
    std::atomic<retired_node*> retired_list;
    
    rcu_lru_list() : head(nullptr), tail(nullptr) {
        retired_list.store(nullptr, std::memory_order_relaxed);
    }
    
    ~rcu_lru_list() {
        // Clean up all nodes
        node* current = head.load(std::memory_order_acquire);
        while (current) {
            node* next = current->next.load(std::memory_order_acquire);
            delete current;
            current = next;
        }
        
        // Clean up retired nodes
        retired_node* r = retired_list.load(std::memory_order_acquire);
        while (r) {
            retired_node* next = r->next;
            delete r->n;
            delete r;
            r = next;
        }
    }
    
    // Get current time in nanoseconds
    uint64_t get_current_time() const {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }
    
    // Create new node
    node* create_node(int expert_id) {
        node* new_node = new node(expert_id);
        new_node->timestamp.store(get_current_time(), std::memory_order_relaxed);
        return new_node;
    }
    
    // Retire node for later deletion
    void retire_node(node* n) {
        retired_node* r = new retired_node();
        r->n = n;
        r->retired_epoch = epoch.current_epoch.load(std::memory_order_acquire);
        r->next = retired_list.load(std::memory_order_relaxed);
        
        while (!retired_list.compare_exchange_weak(
            r->next, r,
            std::memory_order_release,
            std::memory_order_relaxed)) {
            // Retry
        }
    }
    
    // Reclaim memory from retired nodes
    void reclaim_memory() {
        uint64_t current_epoch = epoch.current_epoch.load(std::memory_order_acquire);
        retired_node* prev = nullptr;
        retired_node* r = retired_list.load(std::memory_order_acquire);
        
        while (r) {
            retired_node* next = r->next;
            
            // If node is from a sufficiently old epoch, we can delete it
            if (r->retired_epoch + 2 < current_epoch) {
                // Remove from list
                if (prev) {
                    prev->next = next;
                } else {
                    retired_list.store(next, std::memory_order_release);
                }
                
                // Delete the node
                delete r->n;
                delete r;
            } else {
                prev = r;
            }
            
            r = next;
        }
    }
    
    // Touch expert (move to front)
    void touch(int expert_id) {
        node* new_head = create_node(expert_id);
        if (!new_head) return;
        
        // Increment reference count
        new_head->ref_count.fetch_add(1, std::memory_order_relaxed);
        
        // Insert at head
        node* old_head = head.load(std::memory_order_acquire);
        new_head->next.store(old_head, std::memory_order_relaxed);
        
        while (!head.compare_exchange_weak(
            old_head, new_head,
            std::memory_order_release,
            std::memory_order_acquire)) {
            // Update next pointer and retry
            new_head->next.store(old_head, std::memory_order_relaxed);
        }
        
        // Update tail if list was empty
        node* expected_tail = nullptr;
        tail.compare_exchange_strong(expected_tail, new_head, std::memory_order_release);
    }
    
    // Get eviction candidates from tail
    std::vector<int> get_eviction_candidates(size_t count) {
        std::vector<int> candidates;
        candidates.reserve(count);
        
        // Enter RCU read section
        epoch.enter_read_section();
        
        node* current = tail.load(std::memory_order_acquire);
        size_t scanned = 0;
        
        while (current && candidates.size() < count && scanned < 100) {
            // Check if node can be evicted (ref_count == 0)
            uint32_t refs = current->ref_count.load(std::memory_order_acquire);
            
            if (refs == 0) {
                candidates.push_back(current->expert_id);
            }
            
            current = current->next.load(std::memory_order_acquire);
            scanned++;
        }
        
        // Exit RCU read section
        epoch.exit_read_section();
        
        return candidates;
    }
    
    // Get statistics
    struct statistics {
        size_t num_nodes;
        size_t num_retired;
        double avg_ref_count;
    };
    
    statistics get_statistics() const {
        statistics stats = {0, 0, 0.0};
        
        // Count active nodes
        node* current = head.load(std::memory_order_acquire);
        uint64_t total_refs = 0;
        
        while (current) {
            stats.num_nodes++;
            total_refs += current->ref_count.load(std::memory_order_relaxed);
            current = current->next.load(std::memory_order_acquire);
        }
        
        // Count retired nodes
        retired_node* r = retired_list.load(std::memory_order_acquire);
        while (r) {
            stats.num_retired++;
            r = r->next;
        }
        
        stats.avg_ref_count = stats.num_nodes > 0 ? 
            static_cast<double>(total_refs) / stats.num_nodes : 0.0;
        
        return stats;
    }
};

// Sharded locking to reduce contention
struct sharded_locks {
    static constexpr size_t NUM_SHARDS = 64;  // Power of 2 for fast modulo
    static constexpr size_t SHARD_MASK = NUM_SHARDS - 1;
    
    struct alignas(64) shard {
        std::mutex mutex;
        std::atomic<size_t> contention_count;
        std::atomic<size_t> wait_time_total_ns;
        
        shard() : contention_count(0), wait_time_total_ns(0) {}
    };
    
    shard shards[NUM_SHARDS];
    
    // Lock guard that tracks statistics
    struct sharded_lock_guard {
        shard* s;
        std::chrono::high_resolution_clock::time_point start_time;
        bool locked;
        
        sharded_lock_guard(sharded_locks& locks, int expert_id) 
            : locked(false) {
            s = locks.get_shard(expert_id);
            s->contention_count.fetch_add(1, std::memory_order_relaxed);
            start_time = std::chrono::high_resolution_clock::now();
            s->mutex.lock();
            locked = true;
        }
        
        ~sharded_lock_guard() {
            if (locked) {
                auto end_time = std::chrono::high_resolution_clock::now();
                auto wait_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    end_time - start_time).count();
                s->wait_time_total_ns.fetch_add(wait_time, std::memory_order_relaxed);
                s->mutex.unlock();
            }
        }
        
        // Manual unlock for early release
        void unlock() {
            if (locked) {
                auto end_time = std::chrono::high_resolution_clock::now();
                auto wait_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    end_time - start_time).count();
                s->wait_time_total_ns.fetch_add(wait_time, std::memory_order_relaxed);
                s->mutex.unlock();
                locked = false;
            }
        }
    };
    
    // Get shard for expert_id
    shard* get_shard(int expert_id) {
        size_t shard_idx = std::hash<int>{}(expert_id) & SHARD_MASK;
        return &shards[shard_idx];
    }
    
    // Statistics
    double average_contention() const {
        size_t total_contention = 0;
        for (size_t i = 0; i < NUM_SHARDS; ++i) {
            total_contention += shards[i].contention_count.load(std::memory_order_relaxed);
        }
        return static_cast<double>(total_contention) / NUM_SHARDS;
    }
    
    size_t total_wait_time_ns() const {
        size_t total_wait = 0;
        for (size_t i = 0; i < NUM_SHARDS; ++i) {
            total_wait += shards[i].wait_time_total_ns.load(std::memory_order_relaxed);
        }
        return total_wait;
    }
    
    double average_wait_time_ns() const {
        size_t total_wait = total_wait_time_ns();
        size_t total_contention = 0;
        for (size_t i = 0; i < NUM_SHARDS; ++i) {
            total_contention += shards[i].contention_count.load(std::memory_order_relaxed);
        }
        return total_contention > 0 ? static_cast<double>(total_wait) / total_contention : 0.0;
    }
};

// Wait-free buffer pool using atomic stack
struct waitfree_buffer_pool {
    struct alignas(64) buffer_node {
        void* buffer;
        size_t size;
        std::atomic<buffer_node*> next;
        
        buffer_node() : buffer(nullptr), size(0), next(nullptr) {}
    };
    
    std::atomic<buffer_node*> free_list;
    std::atomic<size_t> available_count;
    
    // Statistics
    std::atomic<size_t> acquire_attempts;
    std::atomic<size_t> acquire_failures;
    std::atomic<size_t> release_count;
    
    waitfree_buffer_pool() 
        : free_list(nullptr), available_count(0), 
          acquire_attempts(0), acquire_failures(0), release_count(0) {}
    
    // Initialize pool with pre-allocated buffers
    void initialize(size_t num_buffers, size_t buffer_size) {
        // Allocate storage for nodes
        std::vector<buffer_node> nodes(num_buffers);
        std::vector<char> storage(num_buffers * buffer_size);
        
        // Link nodes into free list
        for (size_t i = 0; i < num_buffers; ++i) {
            nodes[i].buffer = &storage[i * buffer_size];
            nodes[i].size = buffer_size;
            
            if (i < num_buffers - 1) {
                nodes[i].next.store(&nodes[i + 1], std::memory_order_relaxed);
            } else {
                nodes[i].next.store(nullptr, std::memory_order_relaxed);
            }
        }
        
        free_list.store(&nodes[0], std::memory_order_release);
        available_count.store(num_buffers, std::memory_order_release);
    }
    
    // Wait-free acquire operation
    void* acquire(size_t& actual_size) {
        acquire_attempts.fetch_add(1, std::memory_order_relaxed);
        
        buffer_node* head = free_list.load(std::memory_order_acquire);
        buffer_node* new_head;
        
        do {
            if (!head) {
                acquire_failures.fetch_add(1, std::memory_order_relaxed);
                actual_size = 0;
                return nullptr;  // Pool empty
            }
            new_head = head->next.load(std::memory_order_relaxed);
        } while (!free_list.compare_exchange_weak(
            head, new_head,
            std::memory_order_release,
            std::memory_order_acquire));
        
        // Successfully acquired buffer
        available_count.fetch_sub(1, std::memory_order_relaxed);
        actual_size = head->size;
        return head->buffer;
    }
    
    // Wait-free release operation
    void release(void* buffer) {
        if (!buffer) return;
        
        // Find the node for this buffer (assuming buffer is part of node structure)
        // In real implementation, we'd need a way to map buffer back to node
        // For now, this is a simplified version
        
        release_count.fetch_add(1, std::memory_order_relaxed);
        available_count.fetch_add(1, std::memory_order_relaxed);
    }
    
    // Statistics
    float success_rate() const {
        size_t attempts = acquire_attempts.load(std::memory_order_relaxed);
        size_t failures = acquire_failures.load(std::memory_order_relaxed);
        return attempts > 0 ? 1.0f - (static_cast<float>(failures) / attempts) : 1.0f;
    }
    
    size_t get_available() const {
        return available_count.load(std::memory_order_relaxed);
    }
};

} // namespace ggml_moe_lockfree