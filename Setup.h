#pragma once
#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <stdlib.h>
#include <omp.h>

#define CL_HPP_TARGET_OPENCL_VERSION 300

#ifdef __APPLE__
    #include <OpenCL/cl.hpp>
#else
    #include <CL/opencl.hpp>
#endif

#include "opencv2/opencv.hpp"

 class Setup
 {
  public:
    Setup(cl::CommandQueue queue, cl::Buffer inputBuffer, cl::Buffer outputBuffer, size_t bufferSize, cl::NDRange globalSize, cl::NDRange localSize);
    cl::CommandQueue queue;
    cl::Buffer inputBuffer;
    cl::Buffer outputBuffer;
    size_t bufferSize;
    cl::NDRange globalSize;
    cl::NDRange localSize;

    cl_int run_kernel(cl::Kernel kernel, uchar* inputImage, uchar* outputImage);
 };