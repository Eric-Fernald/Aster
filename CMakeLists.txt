cmake_minimum_required(VERSION 3.24)
project(aster VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(AsterOptions)
include(AsterDeps)

file(GLOB_RECURSE ASTER_CORE_SOURCES CONFIGURE_DEPENDS src/*.cpp)
add_library(aster_core ${ASTER_CORE_SOURCES})
target_include_directories(aster_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party>
  $<INSTALL_INTERFACE:include>)
target_compile_definitions(aster_core PUBLIC ${ASTER_DEFINITIONS})
target_link_libraries(aster_core PUBLIC ${ASTER_LINK_LIBS} Threads::Threads)
if(NOT MSVC)
  target_compile_options(aster_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(ASTER_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  file(GLOB_RECURSE ASTER_CUDA_SOURCES CONFIGURE_DEPENDS cuda/*.cu)
  add_library(aster_cuda ${ASTER_CUDA_SOURCES})
  target_include_directories(aster_cuda PUBLIC include third_party)
  target_compile_definitions(aster_cuda PUBLIC ${ASTER_DEFINITIONS})
  set_target_properties(aster_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "${ASTER_CUDA_ARCHS}")
  target_link_libraries(aster_cuda PUBLIC ${ASTER_CUDA_LIBS})
  target_link_libraries(aster_core PUBLIC aster_cuda)
endif()

add_library(aster_c SHARED src/c_api.cpp)
target_link_libraries(aster_c PUBLIC aster_core)
set_target_properties(aster_c PROPERTIES OUTPUT_NAME aster)

add_subdirectory(tools)
add_subdirectory(bench)
if(ASTER_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
if(ASTER_ENABLE_DUCKDB)
  add_subdirectory(extensions/duckdb)
endif()

install(TARGETS aster_core aster_c EXPORT asterTargets)
install(DIRECTORY include/ DESTINATION include)
