cmake_minimum_required(VERSION 3.20)

project(ginkgo_dolfinx C CXX)

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)  # https://cmake.org/cmake/help/latest/policy/CMP0144.html
endif()
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW) # Boost CONFIG mode
endif()

# No C++ language extensions
set(CMAKE_CXX_EXTENSIONS OFF)

if(DEFINED ENV{CUDA_ARCH})
  set(nvidia ON)
elseif(DEFINED ENV{HIP_ARCH})
  set(amd ON)
else()
  message(FATAL_ERROR "No GPU arch selected. Set environment variable CUDA_ARCH or HIP_ARCH.")
endif()

if(amd)
  # autodetect = native MI250X = gfx90a MI300X = gfx942 Radeon7900 = gfx1100
  set(CMAKE_HIP_ARCHITECTURES $ENV{HIP_ARCH})
  set(AMDGPU_TARGETS $ENV{HIP_ARCH})
  enable_language(HIP)
elseif(nvidia)
  # See
  # https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
  # A100 = 80 RTX6000Ada = 89 GH200 = 90
  set(CMAKE_CUDA_ARCHITECTURES $ENV{CUDA_ARCH})
  set(CMAKE_CUDA_STANDARD 20)
  enable_language(CUDA)
endif()

set(CMAKE_CXX_FLAGS "-O3 -ffast-math ${CMAKE_CXX_FLAGS} -DNDEBUG -Wall")
set(CMAKE_C_FLAGS "-O3 -ffast-math ${CMAKE_C_FLAGS} -DNDEBUG -Wall")

if(CMAKE_HIP_COMPILER_VERSION GREATER_EQUAL 18)
  # See
  # https://rocm.docs.amd.com/projects/llvm-project/en/docs-6.2.1/LLVM/clang/html/ReleaseNotes.html#c-specific-potentially-breaking-changes
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -fclang-abi-compat=17")
endif()
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -std=c++20 -Wall -O3 -ffast-math -DNDEBUG -munsafe-fp-atomics")

if(DEFINED ENV{CRAYPE_VERSION})
  message("Detected CrayPE - adding MPI flags")
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3 -ffast-math -I/opt/cray/pe/mpich/8.1.32/ofi/gnu/12.3/include")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++20 -O3 -Wall -DNDEBUG --extended-lambda --expt-relaxed-constexpr")

# ------------------------------------------------------------------------------
# General configuration
find_package(MPI)
find_package(Basix REQUIRED)
find_package(DOLFINX REQUIRED)
find_package(Ginkgo REQUIRED)

if(DEFINED ENV{BOOST_ROOT} OR DEFINED BOOST_ROOT)
  set(Boost_NO_SYSTEM_PATHS on)
endif()
find_package(Boost 1.70 REQUIRED program_options)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Compile UFL code
add_custom_command(
  OUTPUT poisson.c
  COMMAND ffcx --scalar_type float64
          ${CMAKE_CURRENT_SOURCE_DIR}/poisson.py
  VERBATIM
  DEPENDS poisson.py
  COMMENT "Compile poisson.py [f64] using FFCx")

if(amd)
  find_package(rocthrust REQUIRED CONFIG)
endif()

add_executable(${PROJECT_NAME} main.cpp poisson.c)

# if(NOT amd AND NOT nvidia)
# CPU-only source files
# target_sources(${PROJECT_NAME} PUBLIC geometry_cpu.cpp)
# endif()

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_link_libraries(${PROJECT_NAME} dolfinx Basix::basix Ginkgo::ginkgo Boost::program_options)

if(amd)
  target_link_libraries(${PROJECT_NAME} roc::rocthrust)
  add_definitions(-DUSE_HIP=1)
  set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
elseif(nvidia)
  add_definitions(-DUSE_CUDA=1)
  set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
endif()

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
