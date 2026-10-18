#include "power/ring_buffer.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <thread>
#include <vector>

using power::SpscRing;

namespace {

std::vector<int> drain_all(SpscRing<int>& ring) {
    std::vector<int> out;
    ring.drain(std::back_inserter(out));
    return out;
}

}  // namespace

TEST(SpscRing, CapacityRoundsUpToAPowerOfTwo) {
    EXPECT_EQ(SpscRing<int>(1).capacity(), 1u);
    EXPECT_EQ(SpscRing<int>(5).capacity(), 8u);
    EXPECT_EQ(SpscRing<int>(8).capacity(), 8u);
    EXPECT_EQ(SpscRing<int>(1000).capacity(), 1024u);
}

TEST(SpscRing, ZeroCapacityIsRejected) { EXPECT_THROW(SpscRing<int>(0), std::invalid_argument); }

TEST(SpscRing, DrainsInPushOrder) {
    SpscRing<int> ring(8);
    for (int i = 1; i <= 5; ++i) {
        ASSERT_TRUE(ring.try_push(i));
    }

    EXPECT_EQ(drain_all(ring), (std::vector<int>{1, 2, 3, 4, 5}));
    EXPECT_EQ(ring.size_approx(), 0u);
}

TEST(SpscRing, DrainingAnEmptyRingYieldsNothing) {
    SpscRing<int> ring(4);
    std::vector<int> out;

    EXPECT_EQ(ring.drain(std::back_inserter(out)), 0u);
    EXPECT_TRUE(out.empty());
}

TEST(SpscRing, KeepsOrderAcrossTheWrapAround) {
    // Counters run past the capacity many times over; slots are reused.
    SpscRing<int> ring(4);
    int next = 0;
    for (int round = 0; round < 10; ++round) {
        for (int i = 0; i < 3; ++i) {
            ASSERT_TRUE(ring.try_push(next + i));
        }
        EXPECT_EQ(drain_all(ring), (std::vector<int>{next, next + 1, next + 2}));
        next += 3;
    }
}

TEST(SpscRing, FullRingRefusesTheNewestItem) {
    SpscRing<int> ring(4);
    for (int i = 0; i < 4; ++i) {
        ASSERT_TRUE(ring.try_push(i));
    }

    EXPECT_FALSE(ring.try_push(99));
    EXPECT_EQ(drain_all(ring), (std::vector<int>{0, 1, 2, 3}));  // 99 never got in
    EXPECT_TRUE(ring.try_push(4));                               // room again
}

TEST(SpscRing, ConcurrentProducerAndConsumerLoseNothing) {
    // Run under ThreadSanitizer in CI: a missing acquire/release pair shows
    // up there as a data race even when the values happen to come out right.
    constexpr std::uint32_t kItems = 1u << 20;
    SpscRing<std::uint32_t> ring(1024);  // small, so the producer often hits full

    std::thread producer([&] {
        for (std::uint32_t i = 0; i < kItems; ++i) {
            while (!ring.try_push(i)) {
                std::this_thread::yield();
            }
        }
    });

    std::vector<std::uint32_t> received;
    received.reserve(kItems);
    while (received.size() < kItems) {
        if (ring.drain(std::back_inserter(received)) == 0) {
            std::this_thread::yield();
        }
    }
    producer.join();

    for (std::uint32_t i = 0; i < kItems; ++i) {
        ASSERT_EQ(received[i], i) << "first gap or reordering at index " << i;
    }
}
