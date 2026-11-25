cmake_minimum_required(VERSION 3.22)
project(qolda_engine CXX)

if(NOT DEFINED QOLDA_LITERT_SDK)
    message(FATAL_ERROR "QOLDA_LITERT_SDK is required (pass -PqoldaLiteRtSdk=...)")
endif()

set(LITERT_INCLUDE "${QOLDA_LITERT_SDK}/sdk/litert_cc_sdk")
set(LITERT_LIB "${QOLDA_LITERT_SDK}/jni/arm64-v8a/libLiteRt.so")

if(NOT EXISTS "${LITERT_INCLUDE}/litert/c/litert_compiled_model.h")
    message(FATAL_ERROR "LiteRT C headers not found under ${LITERT_INCLUDE}")
endif()
if(NOT EXISTS "${LITERT_LIB}")
    message(FATAL_ERROR "libLiteRt.so not found at ${LITERT_LIB}")
endif()

add_library(litert SHARED IMPORTED)
set_target_properties(litert PROPERTIES IMPORTED_LOCATION "${LITERT_LIB}")

add_library(qolda_engine SHARED engine.cc)
target_include_directories(qolda_engine PRIVATE "${LITERT_INCLUDE}")
target_compile_options(qolda_engine PRIVATE -std=c++17 -Wall -Wextra -O2 -fvisibility=hidden)
target_link_libraries(qolda_engine litert android log)

# JNI_OnLoad and the native methods must stay exported.
set_target_properties(qolda_engine PROPERTIES CXX_VISIBILITY_PRESET hidden)
