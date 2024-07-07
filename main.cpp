#include "utils.h"

#define BLOCK_SIZE 16

int main(int argc, char** argv)
{
  cl::Device default_device = get_device();
  cl::Context context({ default_device });
  cl::CommandQueue queue(context);
  cl::Program program = get_program(default_device, context);
  cl_int status;
  double t0, t1;

  // read image
  cv::Mat image = get_image(argc, argv);
  cv::Mat ycbcr_output = cv::Mat::zeros(image.size(), image.type());
  cv::Mat grayscale_output = cv::Mat::zeros(image.size(), CV_8UC1);
  cv::Mat dilation_output = cv::Mat::zeros(image.size(), CV_8UC1);

  size_t bufferSizeColor = sizeof(uchar) * image.total() * image.channels();
  size_t bufferSizeGrayscale = sizeof(uchar) * image.total();
  std::cout << "Image size: " << image.size() << std::endl;

  // create buffers on device (allocate space on GPU)
  cl::Buffer inputBufferColor(context, CL_MEM_READ_ONLY, bufferSizeColor);
  cl::Buffer outputBufferColor(context, CL_MEM_READ_WRITE, bufferSizeColor);
  cl::Buffer inputBufferGrayscale(context, CL_MEM_READ_ONLY, bufferSizeGrayscale);
  cl::Buffer outputBufferGrayscale(context, CL_MEM_READ_WRITE, bufferSizeGrayscale);

  int numBlocksX = (int) std::ceil((double) image.cols / (double) BLOCK_SIZE);
  int numBlocksY = (int) std::ceil((double) image.rows / (double) BLOCK_SIZE);

  cl::NDRange localSize(BLOCK_SIZE, BLOCK_SIZE, 1);
  cl::NDRange globalSize(BLOCK_SIZE * numBlocksX, BLOCK_SIZE * numBlocksY, 1);

  std::cout << "Local Size: " << BLOCK_SIZE << " x " << BLOCK_SIZE << " x 1, Global Size: " << BLOCK_SIZE * numBlocksX << " x " << BLOCK_SIZE * numBlocksY << " x 1" << std::endl;

  Setup setup(queue, globalSize, localSize);

  // *** YCbCr ***

  cl::Kernel ycbcr_kernel(program, "rgb_to_ycbcr");
  ycbcr_kernel.setArg(0, inputBufferColor);
  ycbcr_kernel.setArg(1, outputBufferColor);
  ycbcr_kernel.setArg(2, image.cols);
  ycbcr_kernel.setArg(3, image.rows);
  ycbcr_kernel.setArg(4, image.channels());

  t0 = omp_get_wtime(); // start time

  status = setup.run_kernel(ycbcr_kernel, image.data, ycbcr_output.data, inputBufferColor, outputBufferColor, bufferSizeColor, bufferSizeColor);
  if (status != CL_SUCCESS) {
    std::cerr << "Error in YCbCr kernel: " << status << "\n";
    exit(1);
  }

  t1 = omp_get_wtime(); // end time
  std::cout << "Processing for YCbCr took " << (t1 - t0) << " seconds" << std::endl;


  // *** Grayscale ***

  cl::Kernel grayscale_kernel(program, "grayscale");
  grayscale_kernel.setArg(0, inputBufferColor);
  grayscale_kernel.setArg(1, outputBufferGrayscale);
  grayscale_kernel.setArg(2, image.cols);
  grayscale_kernel.setArg(3, image.rows);
  grayscale_kernel.setArg(4, image.channels());

  t0 = omp_get_wtime(); // start time

  status = setup.run_kernel(grayscale_kernel, image.data, grayscale_output.data, inputBufferColor, outputBufferGrayscale, bufferSizeColor, bufferSizeGrayscale);
  if (status != CL_SUCCESS) {
    std::cerr << "Error in grayscale kernel: " << status << "\n";
    exit(1);
  }

  t1 = omp_get_wtime(); // end time
  std::cout << "Processing for grayscale took " << (t1 - t0) << " seconds" << std::endl;


  // *** Dilation ***

  cl::Kernel dilation_kernel(program, "dilation");
  dilation_kernel.setArg(0, inputBufferGrayscale);
  dilation_kernel.setArg(1, outputBufferGrayscale);
  dilation_kernel.setArg(2, image.cols);
  dilation_kernel.setArg(3, image.rows);

  t0 = omp_get_wtime(); // start time

  status = setup.run_kernel(dilation_kernel, grayscale_output.data, dilation_output.data, inputBufferGrayscale, outputBufferGrayscale, bufferSizeGrayscale, bufferSizeGrayscale);
  if (status != CL_SUCCESS) {
    std::cerr << "Error in dilation kernel: " << status << std::endl;
    exit(1);
  }

  t1 = omp_get_wtime(); // end time
  std::cout << "Processing for dilation took " << (t1 - t0) << " seconds" << std::endl;


  // display and wait for a key-press, then close the window
  cv::imshow("original", image);
  cv::imshow("ycbcr", ycbcr_output);
  cv::imshow("grayscale", grayscale_output);
  cv::imshow("dilation", dilation_output);

  int key = cv::waitKey(0);
  cv::destroyAllWindows();
}