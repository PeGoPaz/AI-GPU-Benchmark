#include "power/sampler.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace power {

namespace {

// NVML reports the field in whichever integer type the driver chose; the
// sampler wants milliwatts as uint32 (4 MW is plenty of headroom).
std::optional<std::uint32_t> milliwatts_from(const nvml::FieldValue& field) {
    unsigned long long value = 0;
    switch (field.valueType) {
        case nvml::ValueType::UnsignedInt:
            value = field.value.uiVal;
            break;
        case nvml::ValueType::UnsignedLong:
            value = field.value.ulVal;
            break;
        case nvml::ValueType::UnsignedLongLong:
            value = field.value.ullVal;
            break;
        case nvml::ValueType::SignedInt:
            if (field.value.siVal < 0) return std::nullopt;
            value = static_cast<unsigned long long>(field.value.siVal);
            break;
        case nvml::ValueType::SignedLongLong:
            if (field.value.sllVal < 0) return std::nullopt;
            value = static_cast<unsigned long long>(field.value.sllVal);
            break;
        default:
            return std::nullopt;
    }
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        return std::nullopt;
    }
    return static_cast<std::uint32_t>(value);
}

std::int64_t unix_now_ns() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
}

}  // namespace

// --- NvmlPowerSource -------------------------------------------------------------

NvmlPowerSource::OpenResult NvmlPowerSource::open(unsigned int device_index,
                                                  const std::vector<std::string>& library_paths) {
    auto opened = nvml::Library::open(library_paths);
    if (!opened.library) {
        return {nullptr, opened.error};
    }

    nvml::Device device = nullptr;
    if (nvml::Return r = opened.library->device_by_index(device_index, &device);
        r != nvml::kSuccess) {
        return {nullptr,
                "GPU " + std::to_string(device_index) + ": " + opened.library->error_string(r)};
    }

    // Decided once, here: which reading this card supports does not change
    // while it runs, and probing on every read would double the NVML calls.
    std::unique_ptr<NvmlPowerSource> source(
        new NvmlPowerSource(std::move(opened.library), device, /*instant=*/true));
    if (!source->read_instant()) {
        source->instant_ = false;
        unsigned int milliwatts = 0;
        if (nvml::Return r = source->library_->power_usage(device, &milliwatts);
            r != nvml::kSuccess) {
            return {nullptr,
                    "this GPU reports no power reading: " + source->library_->error_string(r)};
        }
    }
    return {std::move(source), ""};
}

std::optional<std::uint32_t> NvmlPowerSource::read_instant() {
    nvml::FieldValue field{};
    field.fieldId = nvml::kFieldPowerInstant;
    if (library_->field_values(device_, &field, 1) != nvml::kSuccess ||
        field.nvmlReturn != nvml::kSuccess) {
        return std::nullopt;
    }
    return milliwatts_from(field);
}

std::optional<std::uint32_t> NvmlPowerSource::read_milliwatts() {
    if (instant_) {
        return read_instant();
    }
    unsigned int milliwatts = 0;
    if (library_->power_usage(device_, &milliwatts) != nvml::kSuccess) {
        return std::nullopt;
    }
    return milliwatts;
}

std::string NvmlPowerSource::description() const {
    return instant_ ? "instant power (NVML_FI_DEV_POWER_INSTANT)"
                    : "power usage (nvmlDeviceGetPowerUsage, smoothed)";
}

// --- ScriptedSource ----------------------------------------------------------------

ScriptedSource::ScriptedSource(std::vector<std::optional<std::uint32_t>> readings,
                               std::chrono::microseconds read_time)
    : readings_(std::move(readings)), read_time_(read_time) {
    if (readings_.empty()) {
        throw std::invalid_argument("ScriptedSource needs at least one reading");
    }
}

std::optional<std::uint32_t> ScriptedSource::read_milliwatts() {
    if (read_time_.count() > 0) {
        std::this_thread::sleep_for(read_time_);
    }
    const auto reading = readings_[next_];
    next_ = (next_ + 1) % readings_.size();
    return reading;
}

// --- Sampler -----------------------------------------------------------------------

Sampler::Sampler(std::unique_ptr<PowerSource> source, std::chrono::microseconds interval,
                 std::size_t capacity)
    : source_(std::move(source)), interval_(interval), ring_(capacity) {
    if (!source_) {
        throw std::invalid_argument("Sampler needs a power source");
    }
    if (interval_.count() <= 0) {
        throw std::invalid_argument("Sampler interval must be positive");
    }
}

Sampler::~Sampler() { stop(); }

void Sampler::start() {
    if (thread_.joinable()) {
        throw std::logic_error("Sampler is already running");
    }
    {
        std::lock_guard<std::mutex> lock(wake_mutex_);
        stop_requested_ = false;
    }
    thread_ = std::thread([this] { loop(); });
}

void Sampler::stop() {
    if (!thread_.joinable()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(wake_mutex_);
        stop_requested_ = true;
    }
    wake_.notify_one();
    thread_.join();
}

SamplerStats Sampler::stats() const noexcept {
    SamplerStats s;
    s.reads = reads_.load(std::memory_order_relaxed);
    s.errors = errors_.load(std::memory_order_relaxed);
    s.dropped = dropped_.load(std::memory_order_relaxed);
    s.changes = changes_.load(std::memory_order_relaxed);
    s.worst_gap_ns = worst_gap_ns_.load(std::memory_order_relaxed);
    return s;
}

void Sampler::loop() {
    using clock = std::chrono::steady_clock;

    auto next = clock::now();
    std::optional<clock::time_point> previous_read;
    std::optional<std::uint32_t> previous_value;

    std::unique_lock<std::mutex> lock(wake_mutex_);
    while (!stop_requested_) {
        // Read without holding the lock: stop() must never wait on the driver.
        lock.unlock();

        const auto read_at = clock::now();
        const auto reading = source_->read_milliwatts();
        const std::int64_t stamp = unix_now_ns();

        reads_.fetch_add(1, std::memory_order_relaxed);
        if (previous_read) {
            const auto gap =
                std::chrono::duration_cast<std::chrono::nanoseconds>(read_at - *previous_read)
                    .count();
            if (gap > worst_gap_ns_.load(std::memory_order_relaxed)) {
                worst_gap_ns_.store(gap, std::memory_order_relaxed);
            }
        }
        previous_read = read_at;

        if (!reading) {
            errors_.fetch_add(1, std::memory_order_relaxed);
        } else {
            if (previous_value && *previous_value != *reading) {
                changes_.fetch_add(1, std::memory_order_relaxed);
            }
            previous_value = reading;
            if (!ring_.try_push(Sample{stamp, *reading})) {
                dropped_.fetch_add(1, std::memory_order_relaxed);
            }
        }

        // Fixed cadence against the steady clock, so the rate does not drift
        // by the time each read takes. If a read overran whole intervals
        // (a slow driver call, the thread descheduled), those slots are
        // skipped rather than made up with a burst of back-to-back reads,
        // which would only cluster samples around the stall.
        next += interval_;
        const auto now = clock::now();
        if (next <= now) {
            const auto behind = (now - next) / interval_ + 1;
            next += behind * interval_;
        }

        lock.lock();
        wake_.wait_until(lock, next, [this] { return stop_requested_; });
    }
}

}  // namespace power
