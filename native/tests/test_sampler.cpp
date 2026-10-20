#include "power/sampler.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <iterator>
#include <thread>
#include <vector>

#include "fake_nvml_control.hpp"

using namespace std::chrono_literals;
using power::NvmlPowerSource;
using power::Sample;
using power::Sampler;
using power::ScriptedSource;

namespace {

std::unique_ptr<ScriptedSource> scripted(std::vector<std::optional<std::uint32_t>> readings,
                                         std::chrono::microseconds read_time = 0us) {
    return std::make_unique<ScriptedSource>(std::move(readings), read_time);
}

std::vector<Sample> run_for(Sampler& sampler, std::chrono::milliseconds duration) {
    sampler.start();
    std::this_thread::sleep_for(duration);
    sampler.stop();
    std::vector<Sample> samples;
    sampler.drain(std::back_inserter(samples));
    return samples;
}

std::int64_t unix_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

}  // namespace

// Timing bounds are deliberately loose: CI runners are shared machines, and
// these tests are about the mechanism, not about this runner's scheduler.

TEST(Sampler, ReadsAtRoughlyTheRequestedRate) {
    Sampler sampler(scripted({100'000}), 2ms);

    auto samples = run_for(sampler, 200ms);

    // 100 at a perfect 2 ms. The floor only proves the thread keeps
    // sampling: a shared macOS CI runner has delivered a third of the
    // wake-ups, and late slots are skipped by design rather than caught up.
    // The ceiling is the real contract — never a burst beyond the cadence.
    EXPECT_GE(samples.size(), 20u);
    EXPECT_LE(samples.size(), 110u);
    EXPECT_EQ(sampler.stats().reads, samples.size());
}

TEST(Sampler, SamplesCarryIncreasingWallClockStamps) {
    Sampler sampler(scripted({100'000}), 1ms);
    const std::int64_t before = unix_now_ns();

    auto samples = run_for(sampler, 30ms);
    const std::int64_t after = unix_now_ns();

    ASSERT_FALSE(samples.empty());
    EXPECT_GE(samples.front().unix_ns, before);
    EXPECT_LE(samples.back().unix_ns, after);
    for (std::size_t i = 1; i < samples.size(); ++i) {
        EXPECT_LT(samples[i - 1].unix_ns, samples[i].unix_ns);
    }
}

TEST(Sampler, StopIsPromptEvenWithALongInterval) {
    Sampler sampler(scripted({100'000}), 10s);
    sampler.start();
    std::this_thread::sleep_for(20ms);

    const auto t0 = std::chrono::steady_clock::now();
    sampler.stop();
    const auto took = std::chrono::steady_clock::now() - t0;

    EXPECT_LT(took, 500ms) << "stop() waited out the interval instead of waking the thread";
    EXPECT_FALSE(sampler.running());
}

TEST(Sampler, FailedReadsAreCountedAndLeftOut) {
    Sampler sampler(scripted({100'000, std::nullopt, 200'000}), 1ms);

    auto samples = run_for(sampler, 50ms);
    auto stats = sampler.stats();

    EXPECT_GT(stats.errors, 0u);
    EXPECT_EQ(samples.size(), stats.reads - stats.errors);
    for (const Sample& s : samples) {
        EXPECT_TRUE(s.milliwatts == 100'000 || s.milliwatts == 200'000);
    }
}

TEST(Sampler, CountsReadingsThatChanged) {
    // 100, 100, 200, 200, ...: every other read is a change.
    Sampler sampler(scripted({100'000, 100'000, 200'000, 200'000}), 1ms);

    run_for(sampler, 50ms);
    auto stats = sampler.stats();

    EXPECT_GT(stats.changes, 0u);
    EXPECT_LT(stats.changes, stats.reads);
}

TEST(Sampler, FullBufferDropsTheNewestAndSaysSo) {
    Sampler sampler(scripted({100'000}), 1ms, /*capacity=*/4);

    auto samples = run_for(sampler, 50ms);

    EXPECT_EQ(samples.size(), 4u);
    EXPECT_EQ(sampler.stats().dropped, sampler.stats().reads - 4);
}

TEST(Sampler, SlowReadsAreNotMadeUpWithABurst) {
    // Every read takes 10 ms against a 1 ms interval: the missed slots must
    // be skipped, not replayed back to back once the read returns.
    Sampler sampler(scripted({100'000}, 10ms), 1ms);

    run_for(sampler, 100ms);
    auto stats = sampler.stats();

    EXPECT_LE(stats.reads, 15u);
    EXPECT_GE(stats.worst_gap_ns, 9'000'000);
}

TEST(Sampler, StartingTwiceIsAnError) {
    Sampler sampler(scripted({100'000}), 1ms);
    sampler.start();

    EXPECT_THROW(sampler.start(), std::logic_error);
    sampler.stop();
    sampler.stop();  // idempotent
}

TEST(Sampler, CanBeRestartedAfterStopping) {
    Sampler sampler(scripted({100'000}), 1ms);

    run_for(sampler, 10ms);
    auto again = run_for(sampler, 10ms);

    EXPECT_FALSE(again.empty());
}

TEST(Sampler, DestructorStopsTheThread) {
    auto sampler = std::make_unique<Sampler>(scripted({100'000}), 1ms);
    sampler->start();
    sampler.reset();  // would std::terminate on a joinable thread
    SUCCEED();
}

TEST(Sampler, RejectsANonPositiveInterval) {
    EXPECT_THROW(Sampler(scripted({1}), 0us), std::invalid_argument);
}

// --- through NVML (the fake library) ---------------------------------------------

TEST(NvmlPowerSource, PrefersTheInstantField) {
    FakeNvml fake;
    fake.set_power(150'000, 1'000);

    auto opened = NvmlPowerSource::open(0, {FAKE_NVML_PATH});
    ASSERT_NE(opened.source, nullptr) << opened.error;
    EXPECT_TRUE(opened.source->instant());
    EXPECT_NE(opened.source->description().find("instant"), std::string::npos);

    Sampler sampler(std::move(opened.source), 1ms);
    auto samples = run_for(sampler, 30ms);

    ASSERT_GE(samples.size(), 2u);
    EXPECT_GT(samples.back().milliwatts, samples.front().milliwatts);  // the fake ramps
    EXPECT_EQ(sampler.stats().changes, sampler.stats().reads - 1);
}

TEST(NvmlPowerSource, FallsBackToPowerUsageWithoutTheInstantField) {
    FakeNvml fake;
    fake.set_instant_supported(false);

    auto opened = NvmlPowerSource::open(0, {FAKE_NVML_PATH});
    ASSERT_NE(opened.source, nullptr) << opened.error;
    EXPECT_FALSE(opened.source->instant());
    EXPECT_NE(opened.source->description().find("smoothed"), std::string::npos);

    EXPECT_EQ(opened.source->read_milliwatts(), std::optional<std::uint32_t>{100'000});
}

TEST(NvmlPowerSource, UnknownGpuIndexIsAnError) {
    FakeNvml fake;

    auto opened = NvmlPowerSource::open(3, {FAKE_NVML_PATH});

    EXPECT_EQ(opened.source, nullptr);
    EXPECT_NE(opened.error.find("GPU 3"), std::string::npos) << opened.error;
}

TEST(NvmlPowerSource, MissingLibraryIsAnError) {
    auto opened = NvmlPowerSource::open(0, {"/nonexistent/libnvidia-ml.so.1"});

    EXPECT_EQ(opened.source, nullptr);
    EXPECT_FALSE(opened.error.empty());
}

TEST(NvmlPowerSource, DriverErrorsDuringARunAreCounted) {
    FakeNvml fake;
    auto opened = NvmlPowerSource::open(0, {FAKE_NVML_PATH});
    ASSERT_NE(opened.source, nullptr) << opened.error;
    fake.set_read_result(15);  // NVML_ERROR_GPU_IS_LOST

    Sampler sampler(std::move(opened.source), 1ms);
    auto samples = run_for(sampler, 20ms);

    EXPECT_TRUE(samples.empty());
    EXPECT_EQ(sampler.stats().errors, sampler.stats().reads);
}
