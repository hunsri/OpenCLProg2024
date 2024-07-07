#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include "Setup.h"

/*
This is a helper function that reads the kernel function from a file
and returns it as a char array
*/
std::string read_kernel(const std::string& filename) {
  std::string kernel_text;

  std::ifstream kernel_reader;
  kernel_reader.open(filename, std::ios::in);

  std::string line;
  while (std::getline(kernel_reader, line)) {
    kernel_text.append(line);
    kernel_text.append("\n");
  }
  kernel_reader.close();

  return kernel_text;
}

cl::Device get_device() {
  // get all platforms (drivers), e.g. NVIDIA
  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);
  cl_int status;

  if (all_platforms.size() == 0) {
    std::cerr << " No platforms found. Check OpenCL installation!" << std::endl;
    exit(1);
  }

  cl::Platform default_platform = all_platforms[0];
  std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

  // get default device (CPUs, GPUs) of the default platform
  std::vector<cl::Device> all_devices;
  default_platform.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);
  if(all_devices.size() == 0) {
    std::cerr << " No devices found. Check OpenCL installation!" << std::endl;
    exit(1);
  }

  cl::Device default_device = all_devices[0];
  std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << std::endl;

  return default_device;
}

cl::Program get_program(cl::Device default_device, cl::Context context) {
  cl::Program::Sources sources;

  std::string ycbcr_kernel_str = read_kernel("kernels/ycbcr_kernel.cl");
  std::string dilation_kernel_str = read_kernel("kernels/dilation_kernel.cl");
  sources.push_back({ ycbcr_kernel_str.c_str(), ycbcr_kernel_str.length() });
  sources.push_back({ dilation_kernel_str.c_str(), dilation_kernel_str.length() });

  cl::Program program(context, sources);

  if (program.build({ default_device }) != CL_SUCCESS) {
    std::cerr << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << std::endl;
    exit(1);
  }

  return program;
}

cv::Mat get_image(int argc, char** argv) {
  cv::Mat image;
  std::string image_path = "./images/human/1.harold_small.jpg";
  if (argc > 1)
    image_path = argv[1];

  image = cv::imread(image_path);
  if (image.empty()) {
    std::cerr << "Could not read the image: " << image_path << std::endl;
    exit(1);
  }
  return image;
}