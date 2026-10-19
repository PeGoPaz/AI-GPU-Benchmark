// Lock-free single-producer, single-consumer ring buffer.
//
// The sampler thread pushes a reading every couple of milliseconds; Python
// drains the lot once per 250 ms poll. A mutex would work, but it would also
// let a slow drain — the consumer holds the GIL and may be preempted — stall
// the producer mid-interval, which is exactly the jitter the native sampler
// exists to avoid. With one writer and one reader, two atomic counters are
// enough and neither side ever waits for the other.
//
// head_ and tail_ count pushes and pops since construction and are never
// wrapped; a slot is counter & mask_. That makes full and empty unambiguous
// (head - tail == capacity vs. head == tail) without sacrificing a slot.

#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace power {

// MSVC warns (C4324) that the alignas below pads the class. The padding is
// the point: it keeps the two counters on separate cache lines.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4324)
#endif

template <typename T>
class SpscRing {
    static_assert(std::is_trivially_copyable_v<T>,
                  "slots are copied by value on both sides; keep T trivially copyable");

  public:
    // Capacity is rounded up to a power of two so the slot index is a mask
    // rather than a division.
    explicit SpscRing(std::size_t min_capacity)
        : mask_(round_up_pow2(min_capacity) - 1), slots_(new T[mask_ + 1]) {}

    SpscRing(const SpscRing&) = delete;
    SpscRing& operator=(const SpscRing&) = delete;

    std::size_t capacity() const noexcept { return mask_ + 1; }

    // Producer only. Returns false, and stores nothing, when the buffer is
    // full: the newest reading is the one dropped, and the caller counts it.
    // Overwriting the oldest instead would need the producer to move tail_,
    // which only the consumer may touch.
    bool try_push(const T& item) noexcept {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        // acquire: the consumer's reads of the slots it freed happen before
        // we overwrite them.
        const std::size_t tail = tail_.load(std::memory_order_acquire);
        if (head - tail == capacity()) {
            return false;
        }
        slots_[head & mask_] = item;
        // release: the slot's contents are visible before the new head is.
        head_.store(head + 1, std::memory_order_release);
        return true;
    }

    // Consumer only. Copies everything currently available to `out` and
    // returns how many items that was.
    template <typename OutputIt>
    std::size_t drain(OutputIt out) {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        // acquire: pairs with the release in try_push, so every slot below
        // head is fully written.
        const std::size_t head = head_.load(std::memory_order_acquire);
        for (std::size_t i = tail; i != head; ++i) {
            *out++ = slots_[i & mask_];
        }
        // release: our reads are done before the producer may reuse the slots.
        tail_.store(head, std::memory_order_release);
        return head - tail;
    }

    // Either side; exact only when the other side is idle.
    std::size_t size_approx() const noexcept {
        return head_.load(std::memory_order_acquire) - tail_.load(std::memory_order_acquire);
    }

  private:
    static std::size_t round_up_pow2(std::size_t n) {
        if (n == 0) {
            throw std::invalid_argument("SpscRing capacity must be at least 1");
        }
        std::size_t p = 1;
        while (p < n) {
            p <<= 1;
        }
        return p;
    }

    // Each counter on its own cache line: the producer writes head_ and the
    // consumer tail_, and sharing a line would bounce it between the two
    // cores on every operation (false sharing). 64 bytes rather than
    // std::hardware_destructive_interference_size, which is missing or
    // flagged as ABI-unstable on some of the compilers CI uses.
    static constexpr std::size_t kCacheLine = 64;

    const std::size_t mask_;
    const std::unique_ptr<T[]> slots_;
    alignas(kCacheLine) std::atomic<std::size_t> head_{0};
    alignas(kCacheLine) std::atomic<std::size_t> tail_{0};
};

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

}  // namespace power
