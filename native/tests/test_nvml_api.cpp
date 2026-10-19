#include "power/nvml_api.hpp"

#include <gtest/gtest.h>

#include "fake_nvml_control.hpp"

using power::nvml::FieldValue;
using power::nvml::Library;

TEST(NvmlLibrary, MissingLibraryIsAnErrorNotACrash) {
    auto result = Library::open({"/nonexistent/libnvidia-ml.so.1"});

    EXPECT_EQ(result.library, nullptr);
    EXPECT_NE(result.error.find("could not load NVML"), std::string::npos) << result.error;
}

TEST(NvmlLibrary, NoCandidatesMeansNoNvmlOnThisPlatform) {
    auto result = Library::open({});

    EXPECT_EQ(result.library, nullptr);
    EXPECT_NE(result.error.find("does not exist"), std::string::npos);
}

TEST(NvmlLibrary, LaterCandidatesAreTriedWhenEarlierOnesFail) {
    FakeNvml fake;
    auto result = Library::open({"/nonexistent/libnvidia-ml.so.1", FAKE_NVML_PATH});

    ASSERT_NE(result.library, nullptr) << result.error;
    EXPECT_EQ(result.library->path(), FAKE_NVML_PATH);
}

TEST(NvmlLibrary, InitialisesOnOpenAndShutsDownOnClose) {
    FakeNvml fake;
    {
        auto result = Library::open({FAKE_NVML_PATH});
        ASSERT_NE(result.library, nullptr) << result.error;
        EXPECT_EQ(fake.init_calls(), 1);
        EXPECT_EQ(fake.shutdown_calls(), 0);
    }
    EXPECT_EQ(fake.shutdown_calls(), 1);
}

TEST(NvmlLibrary, FailedInitReportsNvmlsOwnMessage) {
    FakeNvml fake;
    fake.set_init_result(9);  // NVML_ERROR_DRIVER_NOT_LOADED

    auto result = Library::open({FAKE_NVML_PATH});

    EXPECT_EQ(result.library, nullptr);
    EXPECT_NE(result.error.find("Driver Not Loaded"), std::string::npos) << result.error;
    EXPECT_EQ(fake.shutdown_calls(), 0) << "shutdown without a successful init";
}

TEST(NvmlLibrary, MissingEntryPointIsNamed) {
    auto result = Library::open({FAKE_NVML_INCOMPLETE_PATH});

    EXPECT_EQ(result.library, nullptr);
    EXPECT_NE(result.error.find("nvmlDeviceGetFieldValues"), std::string::npos) << result.error;
}

TEST(NvmlLibrary, ReadsTheInstantPowerField) {
    FakeNvml fake;
    fake.set_power(187'500);
    auto result = Library::open({FAKE_NVML_PATH});
    ASSERT_NE(result.library, nullptr) << result.error;

    power::nvml::Device device = nullptr;
    ASSERT_EQ(result.library->device_by_index(0, &device), power::nvml::kSuccess);

    FieldValue value{};
    value.fieldId = power::nvml::kFieldPowerInstant;
    ASSERT_EQ(result.library->field_values(device, &value, 1), power::nvml::kSuccess);

    EXPECT_EQ(value.nvmlReturn, power::nvml::kSuccess);
    EXPECT_EQ(value.valueType, power::nvml::ValueType::UnsignedInt);
    EXPECT_EQ(value.value.uiVal, 187'500u);
    EXPECT_GT(value.timestamp, 0);
}

TEST(NvmlLibrary, ErrorStringsComeFromNvml) {
    FakeNvml fake;
    auto result = Library::open({FAKE_NVML_PATH});
    ASSERT_NE(result.library, nullptr) << result.error;

    EXPECT_EQ(result.library->error_string(power::nvml::kErrorNotSupported), "Not Supported");
}

TEST(NvmlLibrary, LooksWhereThisPlatformsDriverPutsIt) {
    const auto paths = Library::default_paths();
#if defined(_WIN32)
    // System32 for DCH drivers first, then the older NVSMI folder.
    ASSERT_EQ(paths.size(), 2u);
    EXPECT_NE(paths[0].find("\\System32\\nvml.dll"), std::string::npos) << paths[0];
    EXPECT_NE(paths[1].find("\\NVIDIA Corporation\\NVSMI\\nvml.dll"), std::string::npos)
        << paths[1];
#elif defined(__linux__)
    EXPECT_EQ(paths, std::vector<std::string>{"libnvidia-ml.so.1"});
#else
    EXPECT_TRUE(paths.empty());
#endif
}
