#include "ggml-moe-cache-lockfree.h"
#include "ggml-moe-cache-concurrent.cpp"
#include <thread>
#include <vector>
#include <atomic>
#include <cassert>
#include <iostream>
#include <random>
#include <chrono>

using namespace ggml_moe_lockfree;
using namespace ggml_moe_concurrent;

// Test utilities
struct test_stats {
    std::atomic<int> passed{0};
    std::atomic<int> failed{0};
    
    void pass() { passed.fetch_add(1); }
    void fail() { failed.fetch_add(1); }
};

test_stats global_stats;

#define TEST_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "TEST FAILED: " << #cond << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            global_stats.fail(); \
            return; \
        } \
    } while (0)

#define TEST_ASSERT_EQ(a, b) TEST_ASSERT((a) == (b))
#define TEST_ASSERT_NE(a, b) TEST_ASSERT((a) != (b))
#define TEST_ASSERT_GT(a, b) TEST_ASSERT((a) > (b))
#define TEST_ASSERT_GE(a, b) TEST_ASSERT((a) >= (b))
#define TEST_ASSERT_LT(a, b) TEST_ASSERT((a) < (b))
#define TEST_ASSERT_LE(a, b) TEST_ASSERT((a) <= (b))

// ============================================================================
// Lock-free Hash Table Tests
// ============================================================================

void test_hashtable_basic_operations() {
    std::cout << "Testing hashtable basic operations..." << std::endl;
    
    hashtable table;
    table.initialize(256);
    
    // Test insert and get
    ggml_backend_buffer_t dummy_buffer = reinterpret_cast<ggml_backend_buffer_t>(0x1000);
    TEST_ASSERT(table.insert(42, dummy_buffer));
    
    ggml_backend_buffer_t retrieved = table.get(42);
    TEST_ASSERT_EQ(retrieved, dummy_buffer);
    
    // Test non-existent expert
    retrieved = table.get(99);
    TEST_ASSERT_EQ(retrieved, nullptr);
    
    // Test multiple inserts
    for (int i = 0; i < 100; ++i) {
        ggml_backend_buffer_t buf = reinterpret_cast<ggml_backend_buffer_t>(0x1000 + i);
        TEST_ASSERT(table.insert(i, buf));
    }
    
    // Verify all inserts
    for (int i = 0; i < 100; ++i) {
        ggml_backend_buffer_t buf = table.get(i);
        TEST_ASSERT_NE(buf, nullptr);
        TEST_ASSERT_EQ(reinterpret_cast<intptr_t>(buf), 0x1000 + i);
    }
    
    // Test statistics
    TEST_ASSERT_EQ(table.get_size(), 101);  // 100 + the first one
    TEST_ASSERT_GT(table.load_factor(), 0.0f);
    TEST_ASSERT_LT(table.load_factor(), 1.0f);
    
    global_stats.pass();
    std::cout << "  PASSED" << std::endl;
}

void test_hashtable_concurrent_inserts() {
    std::cout << "Testing hashtable concurrent inserts..." << std::endl;
    
    hashtable table;
    table.initialize(1024);
    
    const int num_threads = 8;
    const int inserts_per_thread = 1000;
    std::vector<std::thread> threads;
    std::atomic<int> successful_inserts{0};
    
    // Launch concurrent insert threads
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&table, t, inserts_per_thread, &successful_inserts]() {
            for (int i = 0; i < inserts_per_thread; ++i) {
                int expert_id = t * inserts_per_thread + i;
                ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(expert_id + 1000);
                
                if (table.insert(expert_id, buffer)) {
                    successful_inserts.fetch_add(1);
                }
            }
        });
    }
    
    // Wait for completion
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify results
    TEST_ASSERT_GT(successful_inserts.load(), 0);
    TEST_ASSERT_EQ(table.get_size(), successful_inserts.load());
    
    // Verify some entries
    int verify_count = 0;
    for (int t = 0; t < num_threads; ++t) {
        for (int i = 0; i < 10; ++i) {  // Verify first 10 of each thread
            int expert_id = t * inserts_per_thread + i;
            ggml_backend_buffer_t buffer = table.get(expert_id);
            if (buffer != nullptr) {
                TEST_ASSERT_EQ(reinterpret_cast<intptr_t>(buffer), expert_id + 1000);
                verify_count++;
            }
        }
    }
    TEST_ASSERT_GT(verify_count, 0);
    
    global_stats.pass();
    std::cout << "  PASSED (" << successful_inserts.load() << " successful inserts)" << std::endl;
}

void test_hashtable_concurrent_reads() {
    std::cout << "Testing hashtable concurrent reads..." << std::endl;
    
    hashtable table;
    table.initialize(256);
    
    // Pre-populate table
    const int num_entries = 100;
    for (int i = 0; i < num_entries; ++i) {
        table.insert(i, reinterpret_cast<ggml_backend_buffer_t>(i + 1000));
    }
    
    // Launch concurrent read threads
    const int num_threads = 16;
    const int reads_per_thread = 10000;
    std::vector<std::thread> threads;
    std::atomic<int> found_count{0};
    std::atomic<int> not_found_count{0};
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&table, reads_per_thread, &found_count, &not_found_count]() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, 199);  // 50% hit rate
            
            for (int i = 0; i < reads_per_thread; ++i) {
                int expert_id = dis(gen);
                ggml_backend_buffer_t buffer = table.get(expert_id);
                
                if (buffer != nullptr) {
                    found_count.fetch_add(1);
                } else {
                    not_found_count.fetch_add(1);
                }
            }
        });
    }
    
    // Wait for completion
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify results
    int total_reads = found_count.load() + not_found_count.load();
    TEST_ASSERT_EQ(total_reads, num_threads * reads_per_thread);
    TEST_ASSERT_GT(found_count.load(), 0);
    TEST_ASSERT_GT(not_found_count.load(), 0);
    
    global_stats.pass();
    std::cout << "  PASSED (" << found_count.load() << " hits, " << not_found_count.load() << " misses)" << std::endl;
}

void test_hashtable_remove_operations() {
    std::cout << "Testing hashtable remove operations..." << std::endl;
    
    hashtable table;
    table.initialize(256);
    
    // Insert entries
    for (int i = 0; i < 50; ++i) {
        table.insert(i, reinterpret_cast<ggml_backend_buffer_t>(i + 1000));
    }
    
    // Remove some entries
    for (int i = 0; i < 25; ++i) {
        TEST_ASSERT(table.remove(i));
    }
    
    // Verify removed entries are gone
    for (int i = 0; i < 25; ++i) {
        TEST_ASSERT_EQ(table.get(i), nullptr);
    }
    
    // Verify remaining entries are still there
    for (int i = 25; i < 50; ++i) {
        ggml_backend_buffer_t buffer = table.get(i);
        TEST_ASSERT_NE(buffer, nullptr);
        TEST_ASSERT_EQ(reinterpret_cast<intptr_t>(buffer), i + 1000);
    }
    
    // Test double remove (should fail)
    TEST_ASSERT(!table.remove(10));
    
    global_stats.pass();
    std::cout << "  PASSED" << std::endl;
}

// ============================================================================
// RCU LRU List Tests
// ============================================================================

void test_rcu_lru_basic_operations() {
    std::cout << "Testing RCU LRU basic operations..." << std::endl;
    
    rcu_lru_list lru;
    
    // Test touch operations
    for (int i = 0; i < 10; ++i) {
        lru.touch(i);
    }
    
    // Get statistics
    auto stats = lru.get_statistics();
    TEST_ASSERT_GE(stats.num_nodes, 10);
    TEST_ASSERT_EQ(stats.num_retired, 0);
    
    // Get eviction candidates
    auto candidates = lru.get_eviction_candidates(5);
    TEST_ASSERT_GE(candidates.size(), 0);
    TEST_ASSERT_LE(candidates.size(), 5);
    
    global_stats.pass();
    std::cout << "  PASSED" << std::endl;
}

void test_rcu_lru_concurrent_touch() {
    std::cout << "Testing RCU LRU concurrent touch..." << std::endl;
    
    rcu_lru_list lru;
    const int num_threads = 8;
    const int ops_per_thread = 5000;
    std::vector<std::thread> threads;
    
    // Launch threads that continuously touch experts
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&lru, t, ops_per_thread]() {
            for (int i = 0; i < ops_per_thread; ++i) {
                int expert_id = (t * ops_per_thread + i) % 100;  // Working set of 100
                lru.touch(expert_id);
            }
        });
    }
    
    // Wait for completion
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify LRU integrity
    auto stats = lru.get_statistics();
    TEST_ASSERT_GT(stats.num_nodes, 0);
    TEST_ASSERT_LT(stats.avg_ref_count, 2.0);  // Should be low due to concurrent access
    
    global_stats.pass();
    std::cout << "  PASSED (" << stats.num_nodes << " nodes, avg ref count: " << stats.avg_ref_count << ")" << std::endl;
}

void test_rcu_memory_safety() {
    std::cout << "Testing RCU memory safety..." << std::endl;
    
    rcu_lru_list lru;
    std::atomic<bool> stop_flag{false};
    std::atomic<int> active_readers{0};
    
    // Writer thread
    std::thread writer([&lru, &stop_flag]() {
        int expert_id = 0;
        while (!stop_flag.load()) {
            lru.touch(expert_id++);
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    });
    
    // Multiple reader threads
    std::vector<std::thread> readers;
    for (int i = 0; i < 4; ++i) {
        readers.emplace_back([&lru, &stop_flag, &active_readers]() {
            while (!stop_flag.load()) {
                active_readers.fetch_add(1);
                auto candidates = lru.get_eviction_candidates(5);
                active_readers.fetch_sub(1);
                
                // Use the candidates to prevent optimization
                assert(candidates.size() <= 5);
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
        });
    }
    
    // Run for a while
    std::this_thread::sleep_for(std::chrono::seconds(1));
    stop_flag.store(true);
    
    writer.join();
    for (auto& reader : readers) {
        reader.join();
    }
    
    // Verify no crashes and all readers finished
    TEST_ASSERT_EQ(active_readers.load(), 0);
    
    // Reclaim memory
    lru.reclaim_memory();
    
    global_stats.pass();
    std::cout << "  PASSED" << std::endl;
}

// ============================================================================
// Sharded Locks Tests
// ============================================================================

void test_sharded_lock_distribution() {
    std::cout << "Testing sharded lock distribution..." << std::endl;
    
    sharded_locks locks;
    const int num_threads = 16;
    const int ops_per_thread = 10000;
    std::vector<std::thread> threads;
    std::vector<std::atomic<int>> shard_hits(locks.NUM_SHARDS);
    
    for (size_t i = 0; i < locks.NUM_SHARDS; ++i) {
        shard_hits[i].store(0);
    }
    
    // Launch threads that acquire locks for different experts
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&locks, &shard_hits, t, ops_per_thread]() {
            for (int i = 0; i < ops_per_thread; ++i) {
                int expert_id = t * ops_per_thread + i;
                auto* shard = locks.get_shard(expert_id);
                
                // Find which shard index this is
                size_t shard_idx = shard - locks.shards;
                shard_hits[shard_idx].fetch_add(1);
                
                sharded_locks::sharded_lock_guard guard(locks, expert_id);
                // Simulate work
                std::this_thread::sleep_for(std::chrono::nanoseconds(1));
            }
        });
    }
    
    // Wait for completion
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify good distribution (no shard should be 0)
    int min_hits = INT_MAX;
    int max_hits = 0;
    for (size_t i = 0; i < locks.NUM_SHARDS; ++i) {
        int hits = shard_hits[i].load();
        min_hits = std::min(min_hits, hits);
        max_hits = std::max(max_hits, hits);
    }
    
    TEST_ASSERT_GT(min_hits, 0);
    // Should have relatively even distribution (no more than 10x difference)
    TEST_ASSERT_LT(max_hits / min_hits, 10);
    
    global_stats.pass();
    std::cout << "  PASSED (min: " << min_hits << ", max: " << max_hits << " hits per shard)" << std::endl;
}

void test_sharded_lock_contention() {
    std::cout << "Testing sharded lock contention..." << std::endl;
    
    sharded_locks locks;
    const int num_threads = 8;
    const int hold_time_ms = 5;
    
    std::vector<std::thread> threads;
    
    // Launch threads that contend for the same shard
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&locks, hold_time_ms]() {
            // All threads use the same expert_id to hit same shard
            sharded_locks::sharded_lock_guard guard(locks, 42);
            std::this_thread::sleep_for(std::chrono::milliseconds(hold_time_ms));
        });
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // With contention, total time should be > hold_time_ms
    TEST_ASSERT_GT(duration.count(), hold_time_ms);
    
    // Check contention statistics
    auto* shard = locks.get_shard(42);
    TEST_ASSERT_GT(shard->contention_count.load(), 0);
    
    // Average wait time should be reasonable
    double avg_wait = locks.average_wait_time_ns();
    TEST_ASSERT_GT(avg_wait, 0);
    TEST_ASSERT_LT(avg_wait, hold_time_ms * 1000000.0 * 2);  // Less than 2x hold time
    
    global_stats.pass();
    std::cout << "  PASSED (avg wait: " << avg_wait / 1e6 << "ms)" << std::endl;
}

// ============================================================================
// Buffer Pool Tests
// ============================================================================

void test_buffer_pool_basic_operations() {
    std::cout << "Testing buffer pool basic operations..." << std::endl;
    
    waitfree_buffer_pool pool;
    const size_t num_buffers = 10;
    const size_t buffer_size = 4096;
    
    pool.initialize(num_buffers, buffer_size);
    
    // Acquire all buffers
    std::vector<void*> acquired_buffers;
    for (size_t i = 0; i < num_buffers; ++i) {
        size_t actual_size;
        void* buffer = pool.acquire(actual_size);
        TEST_ASSERT_NE(buffer, nullptr);
        TEST_ASSERT_EQ(actual_size, buffer_size);
        
        // Write pattern to buffer
        memset(buffer, static_cast<int>(i), buffer_size);
        acquired_buffers.push_back(buffer);
    }
    
    // Try to acquire one more (should fail)
    size_t actual_size;
    void* buffer = pool.acquire(actual_size);
    TEST_ASSERT_EQ(buffer, nullptr);
    
    // Release and reacquire buffers
    for (size_t i = 0; i < num_buffers; ++i) {
        pool.release(acquired_buffers[i]);
        
        // Reacquire
        void* new_buffer = pool.acquire(actual_size);
        TEST_ASSERT_NE(new_buffer, nullptr);
        
        // Verify pattern is intact
        unsigned char* data = static_cast<unsigned char*>(new_buffer);
        bool pattern_intact = true;
        for (size_t j = 0; j < buffer_size; ++j) {
            if (data[j] != static_cast<unsigned char>(i)) {
                pattern_intact = false;
                break;
            }
        }
        TEST_ASSERT(pattern_intact);
        
        pool.release(new_buffer);
    }
    
    // Verify pool is back to original state
    TEST_ASSERT_EQ(pool.get_available(), num_buffers);
    
    global_stats.pass();
    std::cout << "  PASSED" << std::endl;
}

void test_buffer_pool_concurrent_access() {
    std::cout << "Testing buffer pool concurrent access..." << std::endl;
    
    waitfree_buffer_pool pool;
    const size_t num_buffers = 20;
    const size_t buffer_size = 1024;
    
    pool.initialize(num_buffers, buffer_size);
    
    const int num_threads = 8;
    const int ops_per_thread = 10000;
    std::vector<std::thread> threads;
    std::atomic<int> successful_acquires{0};
    std::atomic<int> failed_acquires{0};
    
    // Launch threads that continuously acquire/release
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&pool, &successful_acquires, &failed_acquires, ops_per_thread]() {
            for (int i = 0; i < ops_per_thread; ++i) {
                size_t actual_size;
                void* buffer = pool.acquire(actual_size);
                
                if (buffer) {
                    successful_acquires.fetch_add(1);
                    // Simulate work
                    memset(buffer, i % 256, actual_size);
                    pool.release(buffer);
                } else {
                    failed_acquires.fetch_add(1);
                }
            }
        });
    }
    
    // Wait for completion
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify results
    int total_ops = successful_acquires.load() + failed_acquires.load();
    TEST_ASSERT_EQ(total_ops, num_threads * ops_per_thread);
    TEST_ASSERT_GT(successful_acquires.load(), 0);
    
    // Success rate should be very high (wait-free guarantee)
    float success_rate = static_cast<float>(successful_acquires.load()) / total_ops;
    TEST_ASSERT_GT(success_rate, 0.95);  // Should be > 95%
    
    // Verify pool is back to original state
    TEST_ASSERT_EQ(pool.get_available(), num_buffers);
    
    global_stats.pass();
    std::cout << "  PASSED (success rate: " << success_rate * 100 << "%)" << std::endl;
}

// ============================================================================
// Integration Tests
// ============================================================================

void test_concurrent_cache_workflow() {
    std::cout << "Testing concurrent cache workflow..." << std::endl;
    
    // Create a mock backend and config
    ggml_moe_cache_config config = {};
    config.max_cache_size = 1024 * 1024 * 1024;  // 1GB
    config.max_experts = 1000;
    config.enable_prefetch = true;
    config.prefetch_depth = 3;
    
    // Create cache (we'll use a dummy backend for testing)
    ggml_moe_cache* cache = ggml_moe_cache_init_concurrent(nullptr, &config, 1000);
    TEST_ASSERT_NE(cache, nullptr);
    
    const int num_threads = 6;
    const int experts_per_thread = 200;
    std::vector<std::thread> threads;
    std::atomic<int> cache_hits{0};
    std::atomic<int> cache_misses{0};
    
    // Mixed workload: some threads read, some write, some evict
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([cache, t, experts_per_thread, &cache_hits, &cache_misses]() {
            // Simulate realistic workload
            for (int i = 0; i < experts_per_thread; ++i) {
                int expert_id = (t * experts_per_thread + i) % 150;  // Working set of 150
                
                // Try to get expert
                ggml_backend_buffer_t buffer = ggml_moe_cache_get_expert(cache, expert_id, nullptr, nullptr);
                
                if (buffer) {
                    cache_hits.fetch_add(1);
                } else {
                    cache_misses.fetch_add(1);
                }
            }
        });
    }
    
    // Wait for completion
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify cache integrity
    int total_ops = cache_hits.load() + cache_misses.load();
    TEST_ASSERT_EQ(total_ops, num_threads * experts_per_thread);
    
    // Get statistics
    ggml_moe_cache_stats stats = ggml_moe_cache_get_stats(cache);
    TEST_ASSERT_EQ(stats.total_requests, total_ops);
    TEST_ASSERT_GT(stats.cache_hits, 0);
    
    // Clean up
    ggml_moe_cache_free(cache);
    
    global_stats.pass();
    std::cout << "  PASSED (hits: " << cache_hits.load() << ", misses: " << cache_misses.load() << ")" << std::endl;
}

void test_cache_memory_pressure() {
    std::cout << "Testing cache memory pressure..." << std::endl;
    
    ggml_moe_cache_config config = {};
    config.max_cache_size = 100 * 1024 * 1024;  // 100MB (small for testing)
    config.max_experts = 100;
    
    ggml_moe_cache* cache = ggml_moe_cache_init_concurrent(nullptr, &config, 100);
    TEST_ASSERT_NE(cache, nullptr);
    
    const int num_threads = 4;
    const int num_experts = 200;  // More than cache can hold
    std::vector<std::thread> threads;
    std::atomic<int> successful_inserts{0};
    std::atomic<int> evictions{0};
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([cache, t, num_experts, &successful_inserts, &evictions]() {
            for (int i = 0; i < num_experts; ++i) {
                int expert_id = (t * num_experts + i) % 50;  // Working set of 50
                
                // Try to insert/load expert
                ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(expert_id + 1000);
                
                // Simulate the cache trying to manage memory
                // In a real scenario, this would trigger evictions
                
                if (buffer) {
                    successful_inserts.fetch_add(1);
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify behavior under pressure
    TEST_ASSERT_GT(successful_inserts.load(), 0);
    
    // Get final statistics
    ggml_moe_cache_stats stats = ggml_moe_cache_get_stats(cache);
    TEST_ASSERT_GT(stats.evictions, 0);  // Should have evictions under pressure
    
    ggml_moe_cache_free(cache);
    
    global_stats.pass();
    std::cout << "  PASSED" << std::endl;
}

// ============================================================================
// Performance Benchmarks
// ============================================================================

void benchmark_cache_latency() {
    std::cout << "Benchmarking cache latency..." << std::endl;
    
    hashtable table;
    table.initialize(1024);
    
    // Pre-populate
    for (int i = 0; i < 100; ++i) {
        table.insert(i, reinterpret_cast<ggml_backend_buffer_t>(i + 1000));
    }
    
    const int num_samples = 100000;
    std::vector<double> latencies;
    latencies.reserve(num_samples);
    
    // Warm up
    for (int i = 0; i < 1000; ++i) {
        table.get(i % 100);
    }
    
    // Measure latencies
    for (int i = 0; i < num_samples; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        table.get(i % 100);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        
        latencies.push_back(duration.count());
    }
    
    // Calculate percentiles
    std::sort(latencies.begin(), latencies.end());
    
    double p50 = latencies[num_samples * 0.5];
    double p95 = latencies[num_samples * 0.95];
    double p99 = latencies[num_samples * 0.99];
    double p999 = latencies[num_samples * 0.999];
    
    std::cout << "  Latency percentiles (ns):\n";
    std::cout << "    P50:  " << p50 << "\n";
    std::cout << "    P95:  " << p95 << "\n";
    std::cout << "    P99:  " << p99 << "\n";
    std::cout << "    P999: " << p999 << std::endl;
    
    // Should have low latency (target: P99 < 1000ns)
    TEST_ASSERT_LT(p99, 1000);
    
    global_stats.pass();
}

void benchmark_cache_throughput() {
    std::cout << "Benchmarking cache throughput..." << std::endl;
    
    hashtable table;
    table.initialize(2048);
    
    // Pre-populate
    for (int i = 0; i < 1000; ++i) {
        table.insert(i, reinterpret_cast<ggml_backend_buffer_t>(i + 1000));
    }
    
    const int num_threads = std::thread::hardware_concurrency();
    const int duration_seconds = 2;
    
    std::vector<std::thread> threads;
    std::atomic<uint64_t> operations{0};
    std::atomic<bool> stop_flag{false};
    
    auto benchmark_start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&table, &operations, &stop_flag]() {
            while (!stop_flag.load()) {
                for (int i = 0; i < 100; ++i) {
                    table.get(i);
                    operations.fetch_add(1);
                }
            }
        });
    }
    
    std::this_thread::sleep_for(std::chrono::seconds(duration_seconds));
    stop_flag.store(true);
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    auto benchmark_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(benchmark_end - benchmark_start);
    
    uint64_t total_ops = operations.load();
    double ops_per_second = static_cast<double>(total_ops) / duration.count();
    
    std::cout << "  Throughput: " << ops_per_second / 1e6 << " million ops/sec" << std::endl;
    std::cout << "  Ops per thread per sec: " << ops_per_second / num_threads << std::endl;
    
    // Should achieve high throughput (target: > 10M ops/sec)
    TEST_ASSERT_GT(ops_per_second, 10000000);
    
    global_stats.pass();
}

// ============================================================================
// Main Test Runner
// ============================================================================

void run_all_tests() {
    std::cout << "\n=== Lock-Free MoE Cache Tests ===\n" << std::endl;
    
    // Hash table tests
    test_hashtable_basic_operations();
    test_hashtable_concurrent_inserts();
    test_hashtable_concurrent_reads();
    test_hashtable_remove_operations();
    
    // RCU LRU tests
    test_rcu_lru_basic_operations();
    test_rcu_lru_concurrent_touch();
    test_rcu_memory_safety();
    
    // Sharded locks tests
    test_sharded_lock_distribution();
    test_sharded_lock_contention();
    
    // Buffer pool tests
    test_buffer_pool_basic_operations();
    test_buffer_pool_concurrent_access();
    
    // Integration tests
    test_concurrent_cache_workflow();
    test_cache_memory_pressure();
    
    // Benchmarks
    benchmark_cache_latency();
    benchmark_cache_throughput();
    
    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "Passed: " << global_stats.passed.load() << std::endl;
    std::cout << "Failed: " << global_stats.failed.load() << std::endl;
    
    if (global_stats.failed.load() == 0) {
        std::cout << "\nAll tests PASSED!" << std::endl;
    } else {
        std::cout << "\nSome tests FAILED!" << std::endl;
    }
}

int main() {
    try {
        run_all_tests();
        return global_stats.failed.load() > 0 ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << "Test runner failed with exception: " << e.what() << std::endl;
        return 1;
    }
}