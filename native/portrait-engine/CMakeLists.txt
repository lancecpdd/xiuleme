cmake_minimum_required(VERSION 3.20)
project(portrait_engine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(PE_WITH_OPENCV "Link OpenCV core/imgproc (required from P1 on)" ON)
option(PE_BUILD_TESTS "Build ABI and geometry self tests" ON)
option(PE_BUILD_TOOLS "Build pe_debug CLI" ON)
set(PE_STAGE_DIR "" CACHE PATH "Copy the built library here after each build")
set(PE_ORT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/include" CACHE PATH
    "ONNX Runtime C API headers (headers only; the DLL is loaded at runtime)")

add_library(portrait_engine SHARED
  src/engine.cpp
  src/feature_debug.cpp
  src/feature_slim_arms.cpp
  src/feature_body_narrow.cpp
  src/ort_runtime.cpp
  src/sha256.cpp
)
target_include_directories(portrait_engine
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
  PRIVATE "${PE_ORT_INCLUDE_DIR}"
)
target_compile_definitions(portrait_engine PRIVATE PORTRAIT_ENGINE_EXPORTS)
set_target_properties(portrait_engine PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)
if(MSVC)
  target_compile_options(portrait_engine PRIVATE /W4 /utf-8 /permissive-)
else()
  target_compile_options(portrait_engine PRIVATE -Wall -Wextra -Wpedantic)
endif()
if(APPLE)
  set_target_properties(portrait_engine PROPERTIES INSTALL_RPATH "@loader_path" BUILD_WITH_INSTALL_RPATH TRUE)
endif()

if(PE_WITH_OPENCV)
  find_package(OpenCV REQUIRED COMPONENTS core imgproc)
  target_link_libraries(portrait_engine PRIVATE opencv_core opencv_imgproc)
  target_compile_definitions(portrait_engine PRIVATE PE_HAS_OPENCV=1)
  if(NOT EXISTS "${PE_ORT_INCLUDE_DIR}/onnxruntime_c_api.h")
    message(FATAL_ERROR "ONNX Runtime headers missing at ${PE_ORT_INCLUDE_DIR}")
  endif()
else()
  target_compile_definitions(portrait_engine PRIVATE PE_HAS_OPENCV=0)
endif()

if(PE_STAGE_DIR)
  add_custom_command(TARGET portrait_engine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PE_STAGE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:portrait_engine>" "${PE_STAGE_DIR}/"
    COMMENT "Staging portrait_engine into ${PE_STAGE_DIR}"
  )
endif()

if(PE_BUILD_TESTS)
  enable_testing()
  add_executable(pe_selftest tests/pe_selftest.cpp)
  target_link_libraries(pe_selftest PRIVATE portrait_engine)
  target_compile_definitions(pe_selftest PRIVATE PE_HAS_OPENCV=$<BOOL:${PE_WITH_OPENCV}>)
  if(MSVC)
    target_compile_options(pe_selftest PRIVATE /utf-8)
  endif()
  add_test(NAME pe_selftest COMMAND pe_selftest)

  add_executable(pe_geometry_test tests/pe_geometry_test.cpp)
  target_link_libraries(pe_geometry_test PRIVATE portrait_engine)
  if(MSVC)
    target_compile_options(pe_geometry_test PRIVATE /utf-8)
  endif()
  add_test(NAME pe_geometry_test COMMAND pe_geometry_test)

  if(PE_WITH_OPENCV)
    add_executable(pe_ort_smoke tests/pe_ort_smoke.cpp)
    target_link_libraries(pe_ort_smoke PRIVATE portrait_engine)
    if(MSVC)
      target_compile_options(pe_ort_smoke PRIVATE /utf-8)
    endif()
  endif()
endif()

if(PE_BUILD_TOOLS AND PE_WITH_OPENCV)
  add_executable(pe_debug tools/pe_debug.cpp)
  target_link_libraries(pe_debug PRIVATE portrait_engine opencv_core opencv_imgproc)
  if(MSVC)
    target_compile_options(pe_debug PRIVATE /utf-8)
  endif()
endif()
