#include "power/nvml_api.hpp"

#include <cstdlib>
#include <optional>
#include <utility>

namespace power::nvml {

#if defined(_WIN32)
namespace {

// An environment variable, if set. MSVC flags getenv as unsafe (C4996),
// which /W4 /WX turns into an error, so this goes through _dupenv_s.
std::optional<std::string> env(const char* name) {
    char* value = nullptr;
    std::size_t length = 0;
    if (_dupenv_s(&value, &length, name) != 0 || value == nullptr) {
        return std::nullopt;
    }
    std::string result(value);
    std::free(value);
    return result;
}

}  // namespace
#endif

std::vector<std::string> Library::default_paths() {
#if defined(_WIN32)
    // The same two places pynvml looks, in the same order: System32 for DCH
    // drivers (every current one), the NVSMI folder for older installs.
    return {env("WINDIR").value_or("C:\\Windows") + "\\System32\\nvml.dll",
            env("ProgramFiles").value_or("C:\\Program Files") +
                "\\NVIDIA Corporation\\NVSMI\\nvml.dll"};
#elif defined(__linux__)
    // The versioned soname: the unversioned symlink is only installed with
    // the CUDA development packages, the .1 file with every driver.
    return {"libnvidia-ml.so.1"};
#else
    return {};
#endif
}

Library::OpenResult Library::open(const std::vector<std::string>& candidates) {
    if (candidates.empty()) {
        return {nullptr, "NVML does not exist on this platform"};
    }

    std::string errors;
    for (const std::string& path : candidates) {
        std::string error;
        DynamicLibrary handle = DynamicLibrary::open(path, error);
        if (!handle.is_open()) {
            errors += (errors.empty() ? "" : "; ") + error;
            continue;
        }

        std::unique_ptr<Library> library(new Library(std::move(handle), path));
        if (std::string missing = library->resolve_all(); !missing.empty()) {
            return {nullptr, path + " is missing " + missing};
        }
        if (Return r = library->init_(); r != kSuccess) {
            return {nullptr, "nvmlInit failed: " + library->error_string(r)};
        }
        library->initialised_ = true;
        return {std::move(library), ""};
    }
    return {nullptr, "could not load NVML: " + errors};
}

std::string Library::resolve_all() {
    // Versioned names, as pynvml uses: the unversioned nvmlInit and
    // nvmlDeviceGetHandleByIndex are the pre-2012 ABI, kept only for old
    // binaries.
    if (!lib_.resolve("nvmlInit_v2", init_)) return "nvmlInit_v2";
    if (!lib_.resolve("nvmlShutdown", shutdown_)) return "nvmlShutdown";
    if (!lib_.resolve("nvmlDeviceGetHandleByIndex_v2", device_by_index_))
        return "nvmlDeviceGetHandleByIndex_v2";
    if (!lib_.resolve("nvmlDeviceGetFieldValues", field_values_)) return "nvmlDeviceGetFieldValues";
    if (!lib_.resolve("nvmlDeviceGetPowerUsage", power_usage_)) return "nvmlDeviceGetPowerUsage";
    if (!lib_.resolve("nvmlErrorString", error_string_)) return "nvmlErrorString";
    return "";
}

Library::~Library() {
    // NVML counts initialisations, so this only drops our reference: pynvml,
    // running in the same process, keeps its own. lib_ closes itself after.
    if (initialised_) {
        shutdown_();
    }
}

Return Library::device_by_index(unsigned int index, Device* device) const {
    return device_by_index_(index, device);
}

Return Library::field_values(Device device, FieldValue* values, int count) const {
    return field_values_(device, count, values);
}

Return Library::power_usage(Device device, unsigned int* milliwatts) const {
    return power_usage_(device, milliwatts);
}

std::string Library::error_string(Return result) const {
    const char* text = error_string_ ? error_string_(result) : nullptr;
    return text ? text : "NVML error " + std::to_string(result);
}

}  // namespace power::nvml
