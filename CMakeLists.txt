cmake_minimum_required(VERSION 3.10)

# Set the project name
project(CuContraction LANGUAGES CXX)

# verbose makefile
set(CMAKE_VERBOSE_MAKEFILE on)

# make O3 the default
set(CMAKE_BUILD_TYPE "Release")

# Set Intel compiler
set(CMAKE_CXX_COMPILER "icpx")

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add position independent code flag
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Option to use float or double
option(USE_FLOAT "Use float instead of double" OFF)
if (USE_FLOAT)
    add_definitions(-DUSE_FLOAT)
endif()

# Option to enable/disable SYCL
option(ENABLE_SYCL "Enable SYCL support" ON)

if (ENABLE_SYCL)
    add_definitions(-DENABLE_SYCL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda")
    
    # Find Intel oneAPI MKL
    find_package(MKL CONFIG REQUIRED)
    
    # Add MKL flags for SYCL
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMKL_ILP64 -DMKL_SYCL")
    
    # Add MKL include directories
    include_directories(${MKL_INCLUDE})
    
    # Ensure linking against MKL SYCL libraries
    list(APPEND CMAKE_PREFIX_PATH $ENV{MKLROOT})
endif()
# Add the include directory
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add the source files
file(GLOB_RECURSE CPP_SOURCES
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

# Note: .cu files should be renamed to .cpp for SYCL
if (ENABLE_SYCL)
    file(GLOB_RECURSE SYCL_SOURCES
        ${PROJECT_SOURCE_DIR}/src/*.cpp
    )
    list(APPEND CPP_SOURCES ${SYCL_SOURCES})
endif()

# Create the shared library
add_library(TTC SHARED ${CPP_SOURCES})

# Link against required libraries
if (ENABLE_SYCL)
    target_link_libraries(TTC PRIVATE MKL::MKL_SYCL MKL::MKL_DPCPP)
endif()

if (OpenMP_CXX_FOUND)
    target_link_libraries(TTC PRIVATE OpenMP::OpenMP_CXX)
endif()

