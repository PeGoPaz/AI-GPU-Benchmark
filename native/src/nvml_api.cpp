#include "power/nvml_api.hpp"

#include <cstring>
#include <utility>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

namespace power::nvml {

namespace {

#if !defined(_WIN32)
// dlsym hands back an object pointer; converting it to a function pointer is
// only conditionally supported by the standard, and some compilers warn on
// the cast under -Wpedantic. Copying the bits is well defined everywhere.
template <typename Fn>
bool resolve(void* handle, const char* name, Fn& out) {
    void* symbol = dlsym(handle, name);
    if (symbol == nullptr) {
        return false;
    }
    static_assert(sizeof(Fn) == sizeof(symbol));
    std::memcpy(&out, &symbol, sizeof(out));
    return true;
}
#endif

}  // namespace

std::vector<std::string> Library::default_paths() {
#if defined(__linux__)
    // The versioned soname: the unversioned symlink is only installed with
    // the CUDA development packages, the .1 file with every driver.
    return {"libnvidia-ml.so.1"};
#else
    return {};
#endif
}

Library::OpenResult Library::open(const std::vector<std::string>& candidates) {
#if defined(_WIN32)
    (void)candidates;
    return {nullptr, "the native sampler does not support Windows yet"};
#else
    if (candidates.empty()) {
        return {nullptr, "NVML does not exist on this platform"};
    }

    std::string errors;
    for (const std::string& path : candidates) {
        void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) {
            const char* reason = dlerror();
            errors += (errors.empty() ? "" : "; ") + std::string(reason ? reason : path);
            continue;
        }

        std::unique_ptr<Library> library(new Library(handle, path));
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
#endif
}

std::string Library::resolve_all() {
#if defined(_WIN32)
    return "everything";
#else
    // Versioned names, as pynvml uses: the unversioned nvmlInit and
    // nvmlDeviceGetHandleByIndex are the pre-2012 ABI, kept only for old
    // binaries.
    if (!resolve(handle_, "nvmlInit_v2", init_)) return "nvmlInit_v2";
    if (!resolve(handle_, "nvmlShutdown", shutdown_)) return "nvmlShutdown";
    if (!resolve(handle_, "nvmlDeviceGetHandleByIndex_v2", device_by_index_))
        return "nvmlDeviceGetHandleByIndex_v2";
    if (!resolve(handle_, "nvmlDeviceGetFieldValues", field_values_))
        return "nvmlDeviceGetFieldValues";
    if (!resolve(handle_, "nvmlDeviceGetPowerUsage", power_usage_))
        return "nvmlDeviceGetPowerUsage";
    if (!resolve(handle_, "nvmlErrorString", error_string_)) return "nvmlErrorString";
    return "";
#endif
}

Library::~Library() {
    // NVML counts initialisations, so this only drops our reference: pynvml,
    // running in the same process, keeps its own.
    if (initialised_) {
        shutdown_();
    }
#if !defined(_WIN32)
    dlclose(handle_);
#endif
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
