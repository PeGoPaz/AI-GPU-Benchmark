// NVML, loaded at run time.
//
// The sampler does not link against libnvidia-ml or include nvml.h. Both come
// with the NVIDIA driver and CUDA Toolkit, so linking would make the module
// unbuildable on a Mac or a CI runner and unimportable on any machine without
// an NVIDIA driver. Instead the library is opened at run time (dlopen, or
// LoadLibraryW on Windows; see dynlib.hpp) — exactly what pynvml does
// through ctypes — and the handful of entry points the sampler
// needs are resolved by name. Their declarations are copied below; NVML keeps
// them ABI-stable, which is why the versioned names (_v2) exist at all.
//
// The tests build a fake libnvidia-ml from tests/fake_nvml.cpp and load it
// through this same code path, so it is exercised without a GPU.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "power/dynlib.hpp"

namespace power::nvml {

// --- the slice of nvml.h the sampler uses -------------------------------------

using Return = int;  // nvmlReturn_t
constexpr Return kSuccess = 0;
constexpr Return kErrorInvalidArgument = 2;
constexpr Return kErrorNotSupported = 3;

struct DeviceTag;
using Device = DeviceTag*;  // nvmlDevice_t: an opaque handle

enum class ValueType : unsigned int {  // nvmlValueType_t
    Double = 0,
    UnsignedInt = 1,
    UnsignedLong = 2,
    UnsignedLongLong = 3,
    SignedLongLong = 4,
    SignedInt = 5,
    UnsignedShort = 6,
};

union Value {  // nvmlValue_t
    double dVal;
    unsigned int uiVal;
    unsigned long ulVal;
    unsigned long long ullVal;
    long long sllVal;
    int siVal;
    unsigned short usVal;
};

struct FieldValue {  // nvmlFieldValue_t
    std::uint32_t fieldId;
    std::uint32_t scopeId;
    std::int64_t timestamp;  // CPU time of the reading, µs since 1970
    std::int64_t latencyUsec;
    ValueType valueType;
    Return nvmlReturn;
    Value value;
};

// The layout NVML expects, checked field by field: a mismatch here would not
// crash, it would silently read garbage from the driver's answer.
static_assert(sizeof(FieldValue) == 40);
static_assert(offsetof(FieldValue, scopeId) == 4);
static_assert(offsetof(FieldValue, timestamp) == 8);
static_assert(offsetof(FieldValue, latencyUsec) == 16);
static_assert(offsetof(FieldValue, valueType) == 24);
static_assert(offsetof(FieldValue, nvmlReturn) == 28);
static_assert(offsetof(FieldValue, value) == 32);

// NVML_FI_DEV_POWER_INSTANT: the board's power draw right now, in mW —
// unlike nvmlDeviceGetPowerUsage, which recent drivers report smoothed.
constexpr std::uint32_t kFieldPowerInstant = 186;

// --- the loader ----------------------------------------------------------------

class Library {
  public:
    struct OpenResult;

    // Tries each path in turn and initialises NVML from the first one that
    // loads. Never throws: on failure the result carries the reason.
    static OpenResult open(const std::vector<std::string>& candidates);

    // Where the driver puts the library on this platform: libnvidia-ml.so.1
    // on Linux, nvml.dll in System32 or NVSMI on Windows; empty where NVML
    // does not exist (macOS).
    static std::vector<std::string> default_paths();

    ~Library();
    Library(const Library&) = delete;
    Library& operator=(const Library&) = delete;

    const std::string& path() const noexcept { return path_; }

    Return device_by_index(unsigned int index, Device* device) const;
    Return field_values(Device device, FieldValue* values, int count) const;
    Return power_usage(Device device, unsigned int* milliwatts) const;
    std::string error_string(Return result) const;

  private:
    Library(DynamicLibrary lib, std::string path) : lib_(std::move(lib)), path_(std::move(path)) {}
    std::string resolve_all();

    using InitFn = Return (*)();
    using ShutdownFn = Return (*)();
    using DeviceByIndexFn = Return (*)(unsigned int, Device*);
    using FieldValuesFn = Return (*)(Device, int, FieldValue*);
    using PowerUsageFn = Return (*)(Device, unsigned int*);
    using ErrorStringFn = const char* (*)(Return);

    DynamicLibrary lib_;
    std::string path_;
    bool initialised_ = false;
    InitFn init_ = nullptr;
    ShutdownFn shutdown_ = nullptr;
    DeviceByIndexFn device_by_index_ = nullptr;
    FieldValuesFn field_values_ = nullptr;
    PowerUsageFn power_usage_ = nullptr;
    ErrorStringFn error_string_ = nullptr;
};

struct Library::OpenResult {
    std::unique_ptr<Library> library;  // null on failure
    std::string error;                 // empty on success
};

}  // namespace power::nvml
