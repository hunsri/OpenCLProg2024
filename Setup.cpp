#include "Setup.h"

Setup::Setup(cl::CommandQueue queue, cl::NDRange globalSize, cl::NDRange localSize) {
  this->queue = queue;
  this->globalSize = globalSize;
  this->localSize = localSize;
}

cl_int Setup::run_kernel(cl::Kernel kernel, uchar* inputImage, uchar* outputImage, cl::Buffer inputBuffer, cl::Buffer outputBuffer, size_t inputBufferSize, size_t outputBufferSize) {
  cl_int status;

  queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, inputBufferSize, inputImage);

  status = queue.enqueueNDRangeKernel(kernel,
    cl::NullRange,
    globalSize,
    localSize);

  if (status != CL_SUCCESS)
    return status;

  queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, outputBufferSize, outputImage);
  queue.finish();

  return status;
}
