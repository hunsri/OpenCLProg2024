#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <stdlib.h>
#include <omp.h>

#ifdef __APPLE__
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl2.hpp>
#endif

#include "opencv2/opencv.hpp"
#include "utils.h"

// we'll work with blocks, so set a constant for the block size:
#define BLOCK_SIZE 16

int main(int argc, char** argv)
{ 
  // get all platforms (drivers), e.g. NVIDIA
  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);
  cl_int status;

  if (all_platforms.size() == 0) {
    std::cout <<" No platforms found. Check OpenCL installation!\n";
    exit(1);
  }

  cl::Platform default_platform = all_platforms[0];
  std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

  // get default device (CPUs, GPUs) of the default platform
  std::vector<cl::Device> all_devices;
  default_platform.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);
  if(all_devices.size() == 0) {
    std::cout<<" No devices found. Check OpenCL installation!\n";
    exit(1);
  }

  cl::Device default_device = all_devices[0];
  std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>()<<"\n";

  // a context is like a "runtime link" to the device and platform;
  // i.e. communication is possible
  cl::Context context({ default_device });

  // create the program that we want to execute on the device
  cl::Program::Sources sources;

  std::string kernel = read_kernel("image_kernel.cl");
  sources.push_back({ kernel.c_str(), kernel.length() });

  cl::Program program(context, sources);
  if (program.build({ default_device }) != CL_SUCCESS) {
    std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << std::endl;
    exit(1);
  }

  cl::Kernel processImageKernel(program, "processImage");

  // get the command queue
  cl::CommandQueue queue(context);

  // read image
  cv::Mat image = cv::imread( "./images/human/1.harold_small.jpg" );
	cv::Mat output = cv::Mat::zeros(image.size(), image.type());
  // cv::Mat convertedImage;
  // cv::cvtColor(image, convertedImage, cv::COLOR_BGR2YCrCb);
  size_t bufferSize = sizeof(uchar) * image.total() * image.channels();

  // create buffers on device (allocate space on GPU)
  cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY, bufferSize);
  cl::Buffer outputBuffer(context, CL_MEM_READ_WRITE, bufferSize);

  queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, bufferSize, image.data);

  processImageKernel.setArg(0, inputBuffer);
  processImageKernel.setArg(1, outputBuffer);
  processImageKernel.setArg(2, image.cols);
  processImageKernel.setArg(3, image.rows);
  processImageKernel.setArg(4, image.channels());

  	// run the kernel code
	int numBlocksX = (int)std::ceil((double)image.cols / (double)BLOCK_SIZE);
	int numBlocksY = (int)std::ceil((double)image.rows / (double)BLOCK_SIZE);

  cl::NDRange localSize(BLOCK_SIZE, BLOCK_SIZE, 1);
	cl::NDRange globalSize(BLOCK_SIZE * numBlocksX, BLOCK_SIZE * numBlocksY, 1);

	status = queue.enqueueNDRangeKernel(processImageKernel,
		cl::NullRange,
		globalSize,
		localSize);

  if (status != CL_SUCCESS) {
		std::cout << "Error in kernel: " << status << "\n";
	}

  // copy the results from GPU memory into RAM (CPU memory)
	queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, bufferSize, output.data);

	queue.finish();


  // display and wait for a key-press, then close the window
	cv::imshow("original", image);
	cv::imshow("result", output);

	int key = cv::waitKey(0);
  cv::destroyAllWindows();
}