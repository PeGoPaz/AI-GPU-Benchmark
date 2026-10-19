#include "power/dynlib.hpp"

#include <gtest/gtest.h>

#include <string>
#include <utility>

using power::DynamicLibrary;

TEST(DynamicLibrary, MissingFileIsAnErrorWithAReason) {
    std::string error;
    DynamicLibrary lib = DynamicLibrary::open("/nonexistent/libnothing.so", error);

    EXPECT_FALSE(lib.is_open());
    EXPECT_FALSE(error.empty());
    EXPECT_EQ(lib.symbol("anything"), nullptr);
}

TEST(DynamicLibrary, ResolvesAnExportedFunction) {
    std::string error;
    DynamicLibrary lib = DynamicLibrary::open(FAKE_NVML_PATH, error);
    ASSERT_TRUE(lib.is_open()) << error;

    int (*init_calls)() = nullptr;
    ASSERT_TRUE(lib.resolve("fake_nvml_init_calls", init_calls));
    EXPECT_GE(init_calls(), 0);
}

TEST(DynamicLibrary, UnknownSymbolResolvesToNothing) {
    std::string error;
    DynamicLibrary lib = DynamicLibrary::open(FAKE_NVML_PATH, error);
    ASSERT_TRUE(lib.is_open()) << error;

    void (*missing)() = nullptr;
    EXPECT_FALSE(lib.resolve("no_such_function", missing));
    EXPECT_EQ(missing, nullptr);
}

TEST(DynamicLibrary, MovingHandsOverTheHandle) {
    std::string error;
    DynamicLibrary first = DynamicLibrary::open(FAKE_NVML_PATH, error);
    ASSERT_TRUE(first.is_open()) << error;

    DynamicLibrary second = std::move(first);

    EXPECT_FALSE(
        first.is_open());  // NOLINT(bugprone-use-after-move): checking the moved-from state
    EXPECT_TRUE(second.is_open());
    EXPECT_NE(second.symbol("nvmlInit_v2"), nullptr);
}
