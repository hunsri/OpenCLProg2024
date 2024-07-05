#include "Setup.h"

Setup::Setup(cl::CommandQueue queue, cl::Buffer inputBuffer, cl::Buffer outputBuffer, size_t bufferSize, cl::NDRange globalSize, cl::NDRange localSize) {
  this->queue = queue;
  this->inputBuffer = inputBuffer;
  this->outputBuffer = outputBuffer;
  this->bufferSize = bufferSize;
  this->globalSize = globalSize;
  this->localSize = localSize;
}

cl_int Setup::run_kernel(cl::Kernel kernel, uchar* inputImage, uchar* outputImage) {
  cl_int status;

  queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, bufferSize, inputImage);

  status = queue.enqueueNDRangeKernel(kernel,
    cl::NullRange,
    globalSize,
    localSize);

  if (status != CL_SUCCESS)
    return status;

  queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, bufferSize, outputImage);
  queue.finish();

  return status;
}
