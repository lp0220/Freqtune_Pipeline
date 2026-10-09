cmake_minimum_required(VERSION 3.20)
project(Coarse_Freqtune LANGUAGES CXX)

add_executable(Coarse_Freqtune coarse_freqtune.cpp)
target_compile_features(Coarse_Freqtune PRIVATE cxx_std_17)
target_include_directories(Coarse_Freqtune SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

get_filename_component(PIPELINE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set_target_properties(Coarse_Freqtune PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PIPELINE_ROOT}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PIPELINE_ROOT}")

if(MSVC)
    set_property(TARGET Coarse_Freqtune PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    target_compile_options(Coarse_Freqtune PRIVATE /W4 /utf-8)
else()
    target_compile_options(Coarse_Freqtune PRIVATE -Wall -Wextra)
endif()
