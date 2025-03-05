cmake_minimum_required(VERSION 3.10)

# Set the project name
project(TTC LANGUAGES CXX)

# verbose makefile
set(CMAKE_VERBOSE_MAKEFILE on)

# make O3 the default
set(CMAKE_BUILD_TYPE "Release")

# Set Intel compiler
set(CMAKE_C_COMPILER "icx")
set(CMAKE_CXX_COMPILER "icpx")

# Find OpenMP package
find_package(OpenMP REQUIRED)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Option to use float or double
option(USE_FLOAT "Use float instead of double" OFF)
if (USE_FLOAT)
    add_definitions(-DUSE_FLOAT)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda -qopenmp")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx90a  -qopenmp")

include_directories(${PROJECT_SOURCE_DIR}/include)

# Add the source files
file(GLOB_RECURSE CPP_SOURCES
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

# Create the shared library
add_library(TTC SHARED ${CPP_SOURCES})

# Link OpenMP
# target_link_libraries(TTC PUBLIC OpenMP::OpenMP_CXX)

find_package(oneMath REQUIRED)
target_link_libraries(TTC PRIVATE onemath)
