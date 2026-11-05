cmake_minimum_required(VERSION 3.16)
project(gpupixel_rr LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Expect GPUPixel already built into vendor/gpupixel (bootstrap + gpupixel cmake).
set(GPUPIXEL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../vendor/gpupixel" CACHE PATH "GPUPixel source root")
set(GPUPIXEL_OUT "${GPUPIXEL_ROOT}/output" CACHE PATH "GPUPixel build output")

if(NOT EXISTS "${GPUPIXEL_ROOT}/src")
  message(FATAL_ERROR "GPUPixel vendor missing at ${GPUPIXEL_ROOT}. Run tools/gpupixel-ref/scripts/bootstrap.sh first.")
endif()

include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${GPUPIXEL_OUT}/include"
  "${GPUPIXEL_ROOT}/src"
  "${GPUPIXEL_ROOT}/src/third_party/glfw/include"
  "${GPUPIXEL_ROOT}/src/third_party/glad/include"
  "${GPUPIXEL_ROOT}/src/third_party/stb"
)

link_directories("${GPUPIXEL_OUT}/library")
if(WIN32)
  link_directories("${GPUPIXEL_OUT}/library/windows")
  link_directories("${GPUPIXEL_ROOT}/src/third_party/glfw/lib-mingw-w64")
elseif(APPLE)
  link_directories("${GPUPIXEL_OUT}/library/macos")
  # Upstream builds gpupixel.framework; VNN frameworks live beside it under vnn/.
  set(GPUPIXEL_FW_DIR "${GPUPIXEL_OUT}/library/macos")
  set(VNN_FW_DIR "${GPUPIXEL_OUT}/library/macos/vnn")
  if(NOT EXISTS "${GPUPIXEL_FW_DIR}/gpupixel.framework")
    # Xcode sometimes stages the framework under the build tree.
    file(GLOB_RECURSE _gp_fw
      "${GPUPIXEL_ROOT}/build/**/gpupixel.framework"
      "${GPUPIXEL_ROOT}/output/**/gpupixel.framework")
    list(LENGTH _gp_fw _gp_fw_len)
    if(_gp_fw_len GREATER 0)
      list(GET _gp_fw 0 _gp_fw0)
      get_filename_component(GPUPIXEL_FW_DIR "${_gp_fw0}" DIRECTORY)
    endif()
  endif()
  if(EXISTS "${GPUPIXEL_FW_DIR}/vnn")
    set(VNN_FW_DIR "${GPUPIXEL_FW_DIR}/vnn")
  elseif(EXISTS "${GPUPIXEL_ROOT}/src/third_party/vnn/libs/macos")
    set(VNN_FW_DIR "${GPUPIXEL_ROOT}/src/third_party/vnn/libs/macos")
  endif()
  message(STATUS "GPUPixel frameworks: ${GPUPIXEL_FW_DIR}")
  message(STATUS "VNN frameworks: ${VNN_FW_DIR}")
  link_directories("${GPUPIXEL_FW_DIR}")
endif()

add_library(gpupixel_rr SHARED src/gpupixel_rr.cc)
target_compile_definitions(gpupixel_rr PRIVATE GPUPIXEL_RR_EXPORTS)

if(WIN32)
  target_link_libraries(gpupixel_rr PRIVATE gpupixel opengl32 glfw3)
  set_target_properties(gpupixel_rr PROPERTIES PREFIX "lib" OUTPUT_NAME "gpupixel_rr")
elseif(APPLE)
  enable_language(OBJCXX)
  set_source_files_properties(src/gpupixel_rr.cc PROPERTIES LANGUAGE OBJCXX)
  find_library(COCOA Cocoa)
  find_library(IOKIT IOKit)
  find_library(COREVIDEO CoreVideo)
  find_library(OPENGL_FRAMEWORK OpenGL)
  find_library(APPKIT AppKit)
  find_library(QUARTZCORE QuartzCore)
  find_library(GPUPIXEL_FRAMEWORK gpupixel PATHS "${GPUPIXEL_FW_DIR}" NO_DEFAULT_PATH)
  if(NOT GPUPIXEL_FRAMEWORK)
    set(GPUPIXEL_FRAMEWORK "${GPUPIXEL_FW_DIR}/gpupixel.framework")
  endif()
  target_link_libraries(gpupixel_rr PRIVATE
    "${GPUPIXEL_FRAMEWORK}"
    ${COCOA} ${IOKIT} ${COREVIDEO} ${OPENGL_FRAMEWORK} ${APPKIT} ${QUARTZCORE}
  )
  target_link_options(gpupixel_rr PRIVATE
    "LINKER:-F${GPUPIXEL_FW_DIR}"
    "LINKER:-F${VNN_FW_DIR}"
    "LINKER:-framework,vnn_kit_osx"
    "LINKER:-framework,vnn_core_osx"
    "LINKER:-framework,vnn_face_osx"
  )
  set_target_properties(gpupixel_rr PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "gpupixel_rr"
    INSTALL_RPATH "@loader_path"
    BUILD_WITH_INSTALL_RPATH TRUE
    XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC NO
  )
  target_compile_options(gpupixel_rr PRIVATE -fno-objc-arc)
elseif(ANDROID)
  message(FATAL_ERROR "Android EGL build: use scripts/build-android.sh with NDK; desktop CMake path is Win/macOS only.")
else()
  target_link_libraries(gpupixel_rr PRIVATE gpupixel GL glfw)
endif()

set(RR_STAGING "${CMAKE_CURRENT_SOURCE_DIR}/../output/rr/staging")
set_target_properties(gpupixel_rr PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${RR_STAGING}"
  LIBRARY_OUTPUT_DIRECTORY "${RR_STAGING}"
  ARCHIVE_OUTPUT_DIRECTORY "${RR_STAGING}"
  RUNTIME_OUTPUT_DIRECTORY_DEBUG "${RR_STAGING}"
  LIBRARY_OUTPUT_DIRECTORY_DEBUG "${RR_STAGING}"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${RR_STAGING}"
  LIBRARY_OUTPUT_DIRECTORY_RELEASE "${RR_STAGING}"
)

if(WIN32)
  set(RR_PLATFORM_OUT "${CMAKE_CURRENT_SOURCE_DIR}/../output/rr/windows")
elseif(APPLE)
  set(RR_PLATFORM_OUT "${CMAKE_CURRENT_SOURCE_DIR}/../output/rr/macos")
else()
  set(RR_PLATFORM_OUT "${CMAKE_CURRENT_SOURCE_DIR}/../output/rr/linux")
endif()

add_custom_command(TARGET gpupixel_rr POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${RR_PLATFORM_OUT}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_FILE:gpupixel_rr>"
    "${RR_PLATFORM_OUT}/"
  COMMENT "Staging libgpupixel_rr into ${RR_PLATFORM_OUT}"
)
