//  Copyright John Maddock 2016.
//  Copyright Matt Borland 2024.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error
#define BOOST_MATH_PROMOTE_DOUBLE_POLICY false

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/relative_difference.hpp>
#include "cuda_managed_ptr.hpp"
#include "stopwatch.hpp"

// For the CUDA runtime routines (prefixed with "cuda_")
#include <nvrtc.h>
#include <cuda_runtime.h>
#include <cuda.h>

typedef double float_type;

/**
 * CUDA Kernel Device code
 *
 */
const char* cuda_kernel = R"(
extern "C" __global__ 
void cuda_test(const float_type *in1, const float_type * in2, float_type *out, int numElements)
{
    using std::cos;
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        out[i] = boost::math::beta(in1[i], in2[i]);
    }
}
)";

void checkNVRTCError(nvrtcResult result, const char* msg)
{
    if (result != NVRTC_SUCCESS)
    {
        std::cerr << msg << ": " << nvrtcGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() 
{
    nvrtcProgram prog;
    nvrtcResult res;

    // Create NVRTC program
    res = nvrtcCreateProgram(&prog, cuda_kernel, "test_beta_nvrtc_ptx.cu", 0, nullptr, nullptr);
    checkNVRTCError(res, "Failed to create NVRTC program");

    // Compile the program
    res = nvrtcCompileProgram(prog, 0, nullptr);
    if (res != NVRTC_SUCCESS) 
    {
        size_t log_size;
        nvrtcGetProgramLogSize(prog, &log_size);
        char* log = new char[log_size];
        nvrtcGetProgramLog(prog, log);
        std::cerr << "Compilation failed:\n" << log << std::endl;
        delete[] log;
        exit(EXIT_FAILURE);
    }

    // Get PTX from the program
    size_t ptx_size;
    nvrtcGetPTXSize(prog, &ptx_size);
    char* ptx = new char[ptx_size];
    nvrtcGetPTX(prog, ptx);

    // Load PTX into CUDA module
    CUmodule module;
    CUfunction kernel;
    cuModuleLoadDataEx(&module, ptx, 0, 0, 0);
    cuModuleGetFunction(&kernel, module, "beta");

    // Clean up
    nvrtcDestroyProgram(&prog);
    delete[] ptx;

    // Input parameters
    int numElements = 50000;
    float *h_in1, *h_in2, *h_out;
    float *d_in1, *d_in2, *d_out;

    // Allocate memory on the host
    h_in1 = new float[numElements];
    h_in2 = new float[numElements];
    h_out = new float[numElements];

    // Initialize input arrays
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < numElements; ++i) 
    {
        h_in1[i] = static_cast<float>(dist(rng));
        h_in2[i] = static_cast<float>(dist(rng));
    }

    // Allocate memory on the device
    cudaMalloc(&d_in1, numElements * sizeof(float));
    cudaMalloc(&d_in2, numElements * sizeof(float));
    cudaMalloc(&d_out, numElements * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_in1, h_in1, numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in2, h_in2, numElements * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    int blockSize = 256;
    int numBlocks = (numElements + blockSize - 1) / blockSize;
    void* args[] = { &d_in1, &d_in2, &d_out, &numElements };
    cuLaunchKernel(kernel, numBlocks, 1, 1, blockSize, 1, 1, 0, 0, args, 0);

    // Copy result back to host
    cudaMemcpy(h_out, d_out, numElements * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < numElements; ++i) 
    {
        auto res = boost::math::beta(h_in1[i], h_in2[i]);
        if (std::isfinite(res))
        {
            if (res != h_out[i])
            {
                std::cout << "error at line: " << i
                        << "\nParallel: " << h_out[i]
                        << "\n  Serial: " << res << std::endl;
            }
        }
    }

    // Clean up
    cudaFree(d_in1);
    cudaFree(d_in2);
    cudaFree(d_out);
    delete[] h_in1;
    delete[] h_in2;
    delete[] h_out;

    std::cout << "Kernel executed successfully." << std::endl;
    return 0;
}