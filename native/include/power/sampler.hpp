// The sampler: one thread reading a PowerSource at a fixed cadence into a
// lock-free ring buffer, which Python drains once per telemetry poll.
//
// Nothing in here touches Python. The thread never takes the GIL, so a busy
// interpreter — a chart redraw, a pandas aggregation — cannot delay a read.

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "power/nvml_api.hpp"
#include "power/ring_buffer.hpp"

namespace power {

// One reading as it leaves the sampler. Wall-clock nanoseconds, so it lines
// up with the time.time() stamps on the Python side's telemetry rows.
struct Sample {
    std::int64_t unix_ns;
    std::uint32_t milliwatts;
};

// Where readings come from: NVML in production, a script in tests.
class PowerSource {
  public:
    virtual ~PowerSource() = default;
    // Called from the sampler thread only. nullopt is a failed read.
    virtual std::optional<std::uint32_t> read_milliwatts() = 0;
    // What is being measured, for the report: instant or smoothed power.
    virtual std::string description() const = 0;
};

// Reads the GPU through NVML. Prefers the instantaneous power field and
// falls back to nvmlDeviceGetPowerUsage — smoothed on recent drivers, but
// available on every card that reports power at all.
class NvmlPowerSource final : public PowerSource {
  public:
    struct OpenResult;
    static OpenResult open(
        unsigned int device_index,
        const std::vector<std::string>& library_paths = nvml::Library::default_paths());

    std::optional<std::uint32_t> read_milliwatts() override;
    std::string description() const override;
    bool instant() const noexcept { return instant_; }

  private:
    NvmlPowerSource(std::unique_ptr<nvml::Library> library, nvml::Device device, bool instant)
        : library_(std::move(library)), device_(device), instant_(instant) {}

    std::optional<std::uint32_t> read_instant();

    std::unique_ptr<nvml::Library> library_;
    nvml::Device device_;
    bool instant_;
};

struct NvmlPowerSource::OpenResult {
    std::unique_ptr<NvmlPowerSource> source;  // null on failure
    std::string error;
};

// Replays a fixed list of readings, cycling; nullopt entries are failed
// reads. Optionally sleeps inside each read, to stand in for a slow driver.
class ScriptedSource final : public PowerSource {
  public:
    explicit ScriptedSource(std::vector<std::optional<std::uint32_t>> readings,
                            std::chrono::microseconds read_time = std::chrono::microseconds{0});

    std::optional<std::uint32_t> read_milliwatts() override;
    std::string description() const override { return "scripted"; }

  private:
    std::vector<std::optional<std::uint32_t>> readings_;
    std::chrono::microseconds read_time_;
    std::size_t next_ = 0;
};

struct SamplerStats {
    std::uint64_t reads = 0;    // attempts, failed ones included
    std::uint64_t errors = 0;   // failed reads, not stored
    std::uint64_t dropped = 0;  // good readings lost to a full buffer
    // Reads whose value differed from the one before. NVML refreshes its
    // reading less often than it can be asked, so this, not `reads`, is
    // the rate at which genuinely new information arrived — a lower bound,
    // since a refresh that repeats a value is invisible.
    std::uint64_t changes = 0;
    std::int64_t worst_gap_ns = 0;  // longest time between two reads
};

class Sampler {
  public:
    static constexpr std::size_t kDefaultCapacity = std::size_t{1} << 17;

    Sampler(std::unique_ptr<PowerSource> source, std::chrono::microseconds interval,
            std::size_t capacity = kDefaultCapacity);
    ~Sampler();

    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;

    void start();  // throws std::logic_error if already running
    void stop();   // joins; returns within one wake-up, however long the interval
    bool running() const noexcept { return thread_.joinable(); }

    // Consumer side of the ring buffer: everything sampled since the last call.
    template <typename OutputIt>
    std::size_t drain(OutputIt out) {
        return ring_.drain(out);
    }

    SamplerStats stats() const noexcept;
    std::chrono::microseconds interval() const noexcept { return interval_; }
    const PowerSource& source() const noexcept { return *source_; }

  private:
    void loop();

    std::unique_ptr<PowerSource> source_;
    std::chrono::microseconds interval_;
    SpscRing<Sample> ring_;

    std::thread thread_;
    std::mutex wake_mutex_;
    std::condition_variable wake_;
    bool stop_requested_ = false;  // guarded by wake_mutex_

    std::atomic<std::uint64_t> reads_{0};
    std::atomic<std::uint64_t> errors_{0};
    std::atomic<std::uint64_t> dropped_{0};
    std::atomic<std::uint64_t> changes_{0};
    std::atomic<std::int64_t> worst_gap_ns_{0};
};

}  // namespace power
