#include "power/dynlib.hpp"

#include <cstring>
#include <string>
#include <utility>

#if defined(_WIN32)
// Before windows.h: without these it defines min() and max() as macros and
// drags in most of the Win32 API.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace power {

namespace {

#if defined(_WIN32)
std::wstring widen(const std::string& utf8) {
    if (utf8.empty()) {
        return {};
    }
    const int size =
        MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), nullptr, 0);
    std::wstring wide(static_cast<std::size_t>(size), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), wide.data(), size);
    return wide;
}

std::string narrow(const wchar_t* wide) {
    const int size = WideCharToMultiByte(CP_UTF8, 0, wide, -1, nullptr, 0, nullptr, nullptr);
    if (size <= 1) {
        return {};
    }
    std::string utf8(static_cast<std::size_t>(size - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide, -1, utf8.data(), size, nullptr, nullptr);
    return utf8;
}

std::string last_error_text(const std::string& path) {
    const DWORD code = GetLastError();
    wchar_t* buffer = nullptr;
    const DWORD length = FormatMessageW(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, code, 0, reinterpret_cast<wchar_t*>(&buffer), 0, nullptr);
    std::string message = length ? narrow(buffer) : "error " + std::to_string(code);
    if (buffer != nullptr) {
        LocalFree(buffer);
    }
    // FormatMessage ends its text with "\r\n".
    while (!message.empty() && (message.back() == '\n' || message.back() == '\r')) {
        message.pop_back();
    }
    return path + ": " + message;
}
#endif

}  // namespace

DynamicLibrary::~DynamicLibrary() { close(); }

DynamicLibrary& DynamicLibrary::operator=(DynamicLibrary&& other) noexcept {
    if (this != &other) {
        close();
        handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
}

DynamicLibrary DynamicLibrary::open(const std::string& path, std::string& error) {
#if defined(_WIN32)
    HMODULE module = LoadLibraryW(widen(path).c_str());
    if (module == nullptr) {
        error = last_error_text(path);
        return DynamicLibrary();
    }
    return DynamicLibrary(reinterpret_cast<void*>(module));
#else
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        const char* reason = dlerror();
        error = reason ? reason : path + ": cannot be opened";
        return DynamicLibrary();
    }
    return DynamicLibrary(handle);
#endif
}

void* DynamicLibrary::symbol(const char* name) const {
    if (handle_ == nullptr) {
        return nullptr;
    }
#if defined(_WIN32)
    FARPROC address = GetProcAddress(reinterpret_cast<HMODULE>(handle_), name);
    void* result = nullptr;
    static_assert(sizeof(address) == sizeof(result));
    std::memcpy(&result, &address, sizeof(result));
    return result;
#else
    return dlsym(handle_, name);
#endif
}

void DynamicLibrary::close() noexcept {
    if (handle_ == nullptr) {
        return;
    }
#if defined(_WIN32)
    FreeLibrary(reinterpret_cast<HMODULE>(handle_));
#else
    dlclose(handle_);
#endif
    handle_ = nullptr;
}

}  // namespace power
