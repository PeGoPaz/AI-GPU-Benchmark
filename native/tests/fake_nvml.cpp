// A stand-in libnvidia-ml for the tests.
//
// Built as a shared library and opened through power::nvml::Library exactly
// as the real one would be: dlopen, symbol lookup by name, the C ABI. The
// fake_nvml_* functions let a test script its answers. Built a second time
// with FAKE_NVML_INCOMPLETE, it leaves out an entry point, for the
// missing-symbol path.

#include "power/nvml_api.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>

using power::nvml::FieldValue;
using power::nvml::Return;
using power::nvml::ValueType;

namespace {

struct DeviceState {};
DeviceState g_device;

// Atomics: the sampler thread reads while the test thread scripts.
std::atomic<Return> g_init_result{0};
std::atomic<int> g_init_calls{0};
std::atomic<int> g_shutdown_calls{0};
std::atomic<bool> g_instant_supported{true};
std::atomic<Return> g_read_result{0};
std::atomic<unsigned int> g_power_base_mw{100000};
std::atomic<unsigned int> g_power_step_mw{0};
std::atomic<unsigned int> g_reads{0};

unsigned int next_power() {
    return g_power_base_mw.load() + g_power_step_mw.load() * g_reads.fetch_add(1);
}

#if !defined(FAKE_NVML_INCOMPLETE)
std::int64_t now_us() {
    using namespace std::chrono;
    return duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
}
#endif

}  // namespace

extern "C" {

// --- the NVML surface ----------------------------------------------------------

Return nvmlInit_v2() {
    ++g_init_calls;
    return g_init_result.load();
}

Return nvmlShutdown() {
    ++g_shutdown_calls;
    return 0;
}

Return nvmlDeviceGetHandleByIndex_v2(unsigned int index, power::nvml::Device* device) {
    if (index != 0) {
        return power::nvml::kErrorInvalidArgument;
    }
    *device = reinterpret_cast<power::nvml::Device>(&g_device);
    return 0;
}

#if !defined(FAKE_NVML_INCOMPLETE)
Return nvmlDeviceGetFieldValues(power::nvml::Device, int count, FieldValue* values) {
    for (int i = 0; i < count; ++i) {
        FieldValue& v = values[i];
        if (v.fieldId != power::nvml::kFieldPowerInstant || !g_instant_supported.load()) {
            v.nvmlReturn = power::nvml::kErrorNotSupported;
            continue;
        }
        v.nvmlReturn = g_read_result.load();
        v.valueType = ValueType::UnsignedInt;
        v.timestamp = now_us();
        v.value.uiVal = next_power();
    }
    return 0;
}
#endif

Return nvmlDeviceGetPowerUsage(power::nvml::Device, unsigned int* milliwatts) {
    if (Return r = g_read_result.load(); r != 0) {
        return r;
    }
    *milliwatts = next_power();
    return 0;
}

const char* nvmlErrorString(Return result) {
    switch (result) {
        case 0:
            return "Success";
        case 3:
            return "Not Supported";
        case 9:
            return "Driver Not Loaded";
        default:
            return "Unknown Error";
    }
}

// --- test controls -------------------------------------------------------------

void fake_nvml_reset() {
    g_init_result = 0;
    g_init_calls = 0;
    g_shutdown_calls = 0;
    g_instant_supported = true;
    g_read_result = 0;
    g_power_base_mw = 100000;
    g_power_step_mw = 0;
    g_reads = 0;
}

void fake_nvml_set_init_result(Return result) { g_init_result = result; }
void fake_nvml_set_instant_supported(int supported) { g_instant_supported = supported != 0; }
void fake_nvml_set_read_result(Return result) { g_read_result = result; }

// Each read returns base + step × (reads so far): a step of 0 is a flat line.
void fake_nvml_set_power(unsigned int base_mw, unsigned int step_mw) {
    g_power_base_mw = base_mw;
    g_power_step_mw = step_mw;
}

int fake_nvml_init_calls() { return g_init_calls.load(); }
int fake_nvml_shutdown_calls() { return g_shutdown_calls.load(); }
unsigned int fake_nvml_reads() { return g_reads.load(); }

}  // extern "C"
