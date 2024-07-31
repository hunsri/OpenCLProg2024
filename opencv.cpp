// ONE IMAGE

#include "utils.h"

int main(int argc, char** argv)
{
  // read image
  cv::Mat image = get_image(argc, argv);
  cv::Mat outputImage = cv::Mat::zeros(image.size(), image.type());
  cv::Mat grayscaleImage = cv::Mat::zeros(image.size(), CV_8UC1);
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));

  double t0 = omp_get_wtime(); // start time
  // cv::cvtColor(image, outputImage, cv::COLOR_BGR2YCrCb); // <- YCbCr
  cv::cvtColor(image, grayscaleImage, cv::COLOR_BGR2GRAY);  // <- grayscale
  cv::dilate(grayscaleImage, outputImage, kernel, cv::Point(-1, -1), 1, 1, 1); // <- dilation
  double t1 = omp_get_wtime(); // end time

  std::cout << "Processing took " << (t1 - t0) << " seconds" << std::endl;
  cv::imshow("original", image);
  cv::imshow("output", outputImage);
  int key = cv::waitKey(0);
  cv::destroyAllWindows();
}