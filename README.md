# TTC (SYCL Implementation)

## Overview
TTC (Tensor Train Contraction) is a high-performance computing (HPC) project that implements tensor contraction operations using **SYCL**, a framework for heterogeneous computing. The goal of this implementation is to leverage SYCL's portability and performance capabilities across different hardware architectures, including CPUs, GPUs, and accelerators.

## Features
- **SYCL-based Implementation**: Uses SYCL to achieve efficient tensor contractions on heterogeneous platforms.
- **HPC Optimization**: Optimized to exploit parallelism and memory hierarchy for high-performance execution.
- **Portable Code**: Runs on diverse architectures, including Intel, AMD, and NVIDIA GPUs (with supported SYCL runtimes).
- **Modular Design**: Well-structured codebase for easy extension and modification.

## Prerequisites
Before building and running TTC, ensure you have the following dependencies installed:

- A SYCL-compliant compiler (such as **Intel oneAPI DPC++** or **hipSYCL**)
- CMake (version 3.16 or later)
- A supported GPU or accelerator (optional, for testing on heterogeneous hardware)
- Python (optional, for visualization and analysis scripts)

## Installation
To build the project, follow these steps:

```sh
# Clone the repository
git clone --branch sycl https://github.com/LukeTheWalker/TTC.git
cd TTC

# Create a build directory
mkdir build && cd build

# Configure and build with CMake
cmake .. -DSYCL_BACKEND=oneAPI   # Change SYCL_BACKEND as needed
make -j$(nproc)
```

## Usage
After building, run the executable as follows:

```sh
./ttc_sycl [options]
```

### Example Execution
```sh
./ttc_sycl --input tensor_data.txt --backend sycl
```

## Performance Optimization
- **Vectorization**: Ensuring efficient memory access patterns for parallel execution.
- **Workgroup Tiling**: Optimizing workloads using local memory.
- **Memory Coalescing**: Reducing global memory accesses for improved performance.

## Contributing
Contributions are welcome! If you'd like to improve this project, please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature-xyz`)
3. Commit your changes (`git commit -m 'Add feature XYZ'`)
4. Push to your branch (`git push origin feature-xyz`)
5. Submit a pull request

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
Special thanks to the SYCL community and contributors to HPC tensor contraction research.

---
For any inquiries or issues, please open an [issue](https://github.com/LukeTheWalker/TTC/issues).

