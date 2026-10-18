// Test-side handle on the fake libnvidia-ml: opens the same shared library the
// code under test loads (dlopen hands back the same instance) and resolves
// its fake_nvml_* controls.

#pragma once

#include "power/nvml_api.hpp"

#include <dlfcn.h>

#include <cstring>
#include <stdexcept>
#include <string>

class FakeNvml {
  public:
    explicit FakeNvml(const char* path = FAKE_NVML_PATH)
        : handle_(dlopen(path, RTLD_NOW | RTLD_LOCAL)) {
        if (handle_ == nullptr) {
            throw std::runtime_error(std::string("cannot open fake NVML: ") + dlerror());
        }
        bind("fake_nvml_reset", reset_);
        bind("fake_nvml_set_init_result", set_init_result_);
        bind("fake_nvml_set_instant_supported", set_instant_supported_);
        bind("fake_nvml_set_read_result", set_read_result_);
        bind("fake_nvml_set_power", set_power_);
        bind("fake_nvml_init_calls", init_calls_);
        bind("fake_nvml_shutdown_calls", shutdown_calls_);
        bind("fake_nvml_reads", reads_);
        reset_();
    }

    ~FakeNvml() { dlclose(handle_); }
    FakeNvml(const FakeNvml&) = delete;
    FakeNvml& operator=(const FakeNvml&) = delete;

    void set_init_result(power::nvml::Return r) { set_init_result_(r); }
    void set_instant_supported(bool supported) { set_instant_supported_(supported ? 1 : 0); }
    void set_read_result(power::nvml::Return r) { set_read_result_(r); }
    void set_power(unsigned int base_mw, unsigned int step_mw = 0) { set_power_(base_mw, step_mw); }
    int init_calls() { return init_calls_(); }
    int shutdown_calls() { return shutdown_calls_(); }
    unsigned int reads() { return reads_(); }

  private:
    template <typename Fn>
    void bind(const char* name, Fn& out) {
        void* symbol = dlsym(handle_, name);
        if (symbol == nullptr) {
            throw std::runtime_error(std::string("fake NVML lacks ") + name);
        }
        std::memcpy(&out, &symbol, sizeof(out));
    }

    void* handle_;
    void (*reset_)() = nullptr;
    void (*set_init_result_)(power::nvml::Return) = nullptr;
    void (*set_instant_supported_)(int) = nullptr;
    void (*set_read_result_)(power::nvml::Return) = nullptr;
    void (*set_power_)(unsigned int, unsigned int) = nullptr;
    int (*init_calls_)() = nullptr;
    int (*shutdown_calls_)() = nullptr;
    unsigned int (*reads_)() = nullptr;
};
