# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

cmake_minimum_required(VERSION 3.17)

project(tritonopenpplbackend LANGUAGES C CXX)

# --------------------------------------------------------------------------- #

# options

option(TRITON_ENABLE_GPU "Enable GPU support in backend" ON)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)
option(HPCC_USE_CUDA "Use for OpenPPL cmake option" ON)

# --------------------------------------------------------------------------- #

# variables

set(TRITON_OPENPPL_LIB_PATHS "" CACHE PATH "Paths to OpenPPL libraries. Multiple paths may be specified by separating them with a semicolon.")
set(TRITON_OPENPPL_INCLUDE_PATHS "" CACHE PATH "Paths to OpenPPL includes. Multiple paths may be specified by separating them with a semicolon.")

set(TRITON_BACKEND_REPO_TAG "r21.12" CACHE STRING "Tag for triton-inference-server/backend repo.")
set(TRITON_CORE_REPO_TAG "r21.12" CACHE STRING "Tag for triton-inference-server/core repo.")
set(TRITON_COMMON_REPO_TAG "r21.12" CACHE STRING "Tag for triton-inference-server/common repo.")
set(OPENPPL_REPO_TAG "r21.12" CACHE STRING "Tag for openppl repo")

set(TRITON_OPENPPL_BACKEND_LIBNAME triton_openppl)
set(TRITON_OPENPPL_BACKEND_INSTALLDIR ${CMAKE_INSTALL_PREFIX}/backends/openppl)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --------------------------------------------------------------------------- #

#
# Dependencies
#
# FetchContent's composibility isn't very good. We must include the
# transitive closure of all repos so that we can override the tag.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

FetchContent_Declare(
  openppl
  GIT_REPOSITORY https://github.com/openppl-public/ppl.nn.git
  GIT_TAG ${OPENPPL_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(openppl)

#
# CUDA
#
if(${TRITON_ENABLE_GPU})
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "Using CUDA ${CUDA_VERSION}")
  set(CUDA_NVCC_FLAGS -std=c++11)
else()
  message( FATAL_ERROR "OpenPPL backend requires TRITON_ENABLE_GPU=1, CMake will exit." )
endif() # TRITON_ENABLE_GPU

#
# Shared library implementing the Triton Backend API
#
configure_file(src/libtriton_openppl.ldscript libtriton_openppl.ldscript COPYONLY)

add_library(
  triton-openppl-backend SHARED
  src/openppl.cc
  src/openppl_utils.cc
  src/openppl_utils.h
)

add_library(
  TritonOpenPPLBackend::triton-openppl-backend ALIAS triton-openppl-backend
)

target_include_directories(
  triton-openppl-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${PPLNN_INCLUDE_DIRECTORIES}
    ${TRITON_OPENPPL_INCLUDE_PATHS}
)

target_compile_features(triton-openppl-backend PRIVATE cxx_std_11)
target_compile_options(
  triton-openppl-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror -Wno-deprecated-declarations>
  $<$<CXX_COMPILER_ID:MSVC>:/Wall /D_WIN32_WINNT=0x0A00 /EHsc>
)

# C/C++ defines that are used directly by this backend.
if (${TRITON_ENABLE_GPU})
target_compile_definitions(
    triton-openppl-backend
    PRIVATE TRITON_ENABLE_GPU=1
)
endif() # TRITON_ENABLE_GPU


FOREACH(p ${TRITON_OPENPPL_LIB_PATHS})
  target_link_directories(
    triton-openppl-backend
    PRIVATE ${p}
  )
ENDFOREACH(p)

target_link_libraries(
  triton-openppl-backend
  PRIVATE
    triton-core-serverapi   # from repo-core
    triton-core-backendapi  # from repo-core
    triton-core-serverstub  # from repo-core
    triton-backend-utils    # from repo-backend
    pplnn_static            # add openppl lib path
)

target_link_libraries(
    triton-openppl-backend
    PRIVATE ${TRITON_OPENPPL_LDFLAGS}
)

target_link_libraries(
    triton-openppl-backend
    PRIVATE
      CUDA::cudart
)


if(WIN32)
  set_target_properties(
    triton-openppl-backend
    PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      OUTPUT_NAME ${TRITON_OPENPPL_BACKEND_LIBNAME}
      SKIP_BUILD_RPATH TRUE
      BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH_USE_LINK_PATH FALSE
      INSTALL_RPATH "$\{ORIGIN\}"
  )
else ()
  set_target_properties(
    triton-openppl-backend 
    PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      OUTPUT_NAME ${TRITON_OPENPPL_BACKEND_LIBNAME}
      SKIP_BUILD_RPATH TRUE
      BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH_USE_LINK_PATH FALSE
      INSTALL_RPATH "$\{ORIGIN\}"
      LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_openppl.ldscript
      LINK_FLAGS "-Wl,--export-dynamic,--version-script libtriton_openppl.ldscript"
  )
endif()

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonOpenPPLBackend)

if (NOT WIN32)
  install(
    TARGETS
      triton-openppl-backend
    EXPORT
      triton-openppl-backend-targets
    LIBRARY DESTINATION ${TRITON_OPENPPL_BACKEND_INSTALLDIR}
    ARCHIVE DESTINATION ${TRITON_OPENPPL_BACKEND_INSTALLDIR}
  )
else()
  install(
    TARGETS
      triton-openppl-backend
    EXPORT
      triton-openppl-backend-targets
    RUNTIME DESTINATION ${TRITON_OPENPPL_BACKEND_INSTALLDIR}
    ARCHIVE DESTINATION ${TRITON_OPENPPL_BACKEND_INSTALLDIR}
  )
endif() # WIN32

install(
  EXPORT
    triton-openppl-backend-targets
  FILE
    TritonOpenPPLBackendTargets.cmake
  NAMESPACE
    TritonOpenPPLBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonOpenPPLBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonOpenPPLBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TritonOpenPPLBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-openppl-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonOpenPPLBackendTargets.cmake
  NAMESPACE TritonOpenPPLBackend::
)

export(PACKAGE TritonOpenPPLBackend)