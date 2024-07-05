#include "utils.h"
#include "Setup.h"

#define BLOCK_SIZE 16

cl::Device get_device() {
  // get all platforms (drivers), e.g. NVIDIA
  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);
  cl_int status;

  if (all_platforms.size() == 0) {
    std::cout <<" No platforms found. Check OpenCL installation!" << std::endl;
    exit(1);
  }

  cl::Platform default_platform = all_platforms[0];
  std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

  // get default device (CPUs, GPUs) of the default platform
  std::vector<cl::Device> all_devices;
  default_platform.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);
  if(all_devices.size() == 0) {
    std::cout<<" No devices found. Check OpenCL installation!" << std::endl;
    exit(1);
  }

  cl::Device default_device = all_devices[0];
  std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << std::endl;

  return default_device;
}

int main(int argc, char** argv)
{
  cl::Device default_device = get_device();
  cl::Context context({ default_device });
  cl::Program::Sources sources;
  cl_int status;

  std::string ycbcr_kernel_str = read_kernel("kernels/ycbcr_kernel.cl");
  std::string dilation_kernel_str = read_kernel("kernels/dilation_kernel.cl");
  sources.push_back({ ycbcr_kernel_str.c_str(), ycbcr_kernel_str.length() });
  sources.push_back({ dilation_kernel_str.c_str(), dilation_kernel_str.length() });

  cl::Program program(context, sources);
  cl::CommandQueue queue(context);

  if (program.build({ default_device }) != CL_SUCCESS) {
    std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << std::endl;
    exit(1);
  }

  // read image
  cv::Mat image = cv::imread("./images/human/1.harold_small.jpg");
  cv::Mat ycbcr_output = cv::Mat::zeros(image.size(), image.type());
  cv::Mat grayscale_output = cv::Mat::zeros(image.size(), image.type());
  cv::Mat dilation_output = cv::Mat::zeros(image.size(), image.type());

  size_t bufferSize = sizeof(uchar) * image.total();
  std::cout << "Image size: " << image.size() << std::endl;

  // create buffers on device (allocate space on GPU)
  cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY, bufferSize);
  cl::Buffer outputBuffer(context, CL_MEM_READ_WRITE, bufferSize);

  int numBlocksX = (int) std::ceil((double) image.cols / (double) BLOCK_SIZE);
  int numBlocksY = (int) std::ceil((double) image.rows / (double) BLOCK_SIZE);

  cl::NDRange localSize(BLOCK_SIZE, BLOCK_SIZE, 1);
  cl::NDRange globalSize(BLOCK_SIZE * numBlocksX, BLOCK_SIZE * numBlocksY, 1);

  std::cout << "Local Size: " << BLOCK_SIZE << " x " << BLOCK_SIZE << " x 1, Global Size: " << BLOCK_SIZE * numBlocksX << " x " << BLOCK_SIZE * numBlocksY << " x 1" << std::endl;

  // *** YCbCr Conversion ***

  Setup setup(queue, inputBuffer, outputBuffer, bufferSize, globalSize, localSize);

  cl::Kernel ycbcr_kernel(program, "rgb_to_ycbcr");
  ycbcr_kernel.setArg(0, inputBuffer);
  ycbcr_kernel.setArg(1, outputBuffer);
  ycbcr_kernel.setArg(2, image.cols);
  ycbcr_kernel.setArg(3, image.rows);
  ycbcr_kernel.setArg(4, image.channels());

  double t0 = omp_get_wtime(); // start time

  status = setup.run_kernel(ycbcr_kernel, image.data, ycbcr_output.data);
  if (status != CL_SUCCESS) {
    std::cout << "Error in YCbCr kernel: " << status << "\n";
    exit(1);
  }

  double t1 = omp_get_wtime(); // end time
  std::cout << "Processing for YCbCr took " << (t1 - t0) << " seconds" << std::endl;


  // *** Grayscale ***

  cl::Kernel grayscale_kernel(program, "grayscale");
  grayscale_kernel.setArg(0, inputBuffer);
  grayscale_kernel.setArg(1, outputBuffer);
  grayscale_kernel.setArg(2, image.cols);
  grayscale_kernel.setArg(3, image.rows);
  grayscale_kernel.setArg(4, image.channels());

  t0 = omp_get_wtime(); // start time

  status = setup.run_kernel(grayscale_kernel, image.data, grayscale_output.data);
  if (status != CL_SUCCESS) {
    std::cout << "Error in grayscale kernel: " << status << "\n";
    exit(1);
  }

  t1 = omp_get_wtime();  // end time
  std::cout << "Processing for grayscale took " << (t1 - t0) << " seconds" << std::endl;


  // *** Dilation ***

  cl::Kernel dilation_kernel(program, "dilation");
  dilation_kernel.setArg(0, inputBuffer);
  dilation_kernel.setArg(1, outputBuffer);
  dilation_kernel.setArg(2, image.cols);
  dilation_kernel.setArg(3, image.rows);
  dilation_kernel.setArg(4, image.channels());

  t0 = omp_get_wtime(); // start time

  status = setup.run_kernel(dilation_kernel, image.data, dilation_output.data);
  if (status != CL_SUCCESS) {
    std::cout << "Error in dilation kernel: " << status << "\n";
    exit(1);
  }

  t1 = omp_get_wtime();  // end time
  std::cout << "Processing for dilation took " << (t1 - t0) << " seconds" << std::endl;


  // display and wait for a key-press, then close the window
  cv::imshow("original", image);
  cv::imshow("ycbcr", ycbcr_output);
  cv::imshow("grayscale", grayscale_output);
  cv::imshow("dilation", dilation_output);

  int key = cv::waitKey(0);
  cv::destroyAllWindows();
}