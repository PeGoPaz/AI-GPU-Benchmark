// Test-side handle on the fake libnvidia-ml: opens the same shared library the
// code under test loads (the OS hands back the same instance) and resolves
// its fake_nvml_* controls.

#pragma once

#include <stdexcept>
#include <string>

#include "power/dynlib.hpp"
#include "power/nvml_api.hpp"

class FakeNvml {
  public:
    explicit FakeNvml(const char* path = FAKE_NVML_PATH) {
        std::string error;
        lib_ = power::DynamicLibrary::open(path, error);
        if (!lib_.is_open()) {
            throw std::runtime_error("cannot open fake NVML: " + error);
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
        if (!lib_.resolve(name, out)) {
            throw std::runtime_error(std::string("fake NVML lacks ") + name);
        }
    }

    power::DynamicLibrary lib_;
    void (*reset_)() = nullptr;
    void (*set_init_result_)(power::nvml::Return) = nullptr;
    void (*set_instant_supported_)(int) = nullptr;
    void (*set_read_result_)(power::nvml::Return) = nullptr;
    void (*set_power_)(unsigned int, unsigned int) = nullptr;
    int (*init_calls_)() = nullptr;
    int (*shutdown_calls_)() = nullptr;
    unsigned int (*reads_)() = nullptr;
};
