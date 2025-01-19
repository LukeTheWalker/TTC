cmake_minimum_required(VERSION 3.10)

# Set the project name
project(CuContraction LANGUAGES CXX CUDA)

# verbose makefile
set(CMAKE_VERBOSE_MAKEFILE on)

# make O3 the default
set(CMAKE_BUILD_TYPE "Release")

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

# Option to enable/disable CUDA
option(ENABLE_CUDA "Enable CUDA support" ON)

if (ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)
    add_definitions(-DENABLE_CUDA)

    # Find CUDA Toolkit
    find_package(CUDAToolkit REQUIRED)
    
    # Include cuBLAS
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    list(APPEND CUDA_LIBRARIES CUDA::cublas)
endif()

# Add the include directory
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add the source files
file(GLOB_RECURSE CPP_SOURCES
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

if (ENABLE_CUDA)
    file(GLOB_RECURSE CUDA_SOURCES
        ${PROJECT_SOURCE_DIR}/src/*.cu
    )
    list(APPEND CPP_SOURCES ${CUDA_SOURCES})
endif()

# Create the shared library instead of executable
add_library(TTC SHARED ${CPP_SOURCES})

# Link against required libraries
if (ENABLE_CUDA)
    set_target_properties(TTC PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(TTC PROPERTIES CUDA_ARCHITECTURES "70;80")
    target_link_libraries(TTC PRIVATE CUDA::cublas)
endif()

if (OpenMP_CXX_FOUND)
    target_link_libraries(TTC PRIVATE OpenMP::OpenMP_CXX)
endif()
