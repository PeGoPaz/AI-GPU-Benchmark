// A shared library opened at run time, on POSIX and on Windows.
//
// dlopen/dlsym/dlclose and LoadLibraryW/GetProcAddress/FreeLibrary do the
// same job with different spellings, error reporting and string types. This
// wraps both behind one RAII type, so the NVML loader and the tests' fake
// library controls are written once.

#pragma once

#include <cstring>
#include <string>

namespace power {

class DynamicLibrary {
  public:
    DynamicLibrary() = default;
    ~DynamicLibrary();

    DynamicLibrary(DynamicLibrary&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }
    DynamicLibrary& operator=(DynamicLibrary&& other) noexcept;
    DynamicLibrary(const DynamicLibrary&) = delete;
    DynamicLibrary& operator=(const DynamicLibrary&) = delete;

    // Opens the library at `path` (UTF-8). On failure the result is closed
    // and `error` holds the system's reason.
    static DynamicLibrary open(const std::string& path, std::string& error);

    bool is_open() const noexcept { return handle_ != nullptr; }

    // The address of an exported symbol, or nullptr if there is none.
    void* symbol(const char* name) const;

    // Resolves a symbol straight into a function pointer. The conversion from
    // an object pointer is only conditionally supported by the standard, and
    // some compilers warn about the cast under -Wpedantic; copying the bits is
    // well defined everywhere.
    template <typename Fn>
    bool resolve(const char* name, Fn& out) const {
        void* address = symbol(name);
        if (address == nullptr) {
            return false;
        }
        static_assert(sizeof(Fn) == sizeof(address));
        std::memcpy(&out, &address, sizeof(out));
        return true;
    }

  private:
    explicit DynamicLibrary(void* handle) : handle_(handle) {}
    void close() noexcept;

    void* handle_ = nullptr;  // void* from dlopen, an HMODULE on Windows
};

}  // namespace power
