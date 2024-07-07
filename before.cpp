// ONE IMAGE

#include <iostream>
#include <cstdio>
#include <omp.h>

#include "opencv2/opencv.hpp"
#include "utils.h"

int main(int argc, char** argv)
{
  // read image
  cv::Mat image = get_image(argc, argv);
  cv::Mat outputImage = cv::Mat::zeros(image.size(), image.type());

  double t0 = omp_get_wtime(); // start time

  // #pragma omp parallel for collapse(2)
  for (int i = 0; i < image.rows; ++i) {
    for (int j = 0; j < image.cols; ++j) {

      // get pixel at [i, j] as a <B,G,R> vector
      cv::Vec3b pixel = image.at<cv::Vec3b>( i, j );

      // extract the pixels as uchar (unsigned 8-bit) types (0..255)
      float B = pixel[0] / 255.0f;
      float G = pixel[1] / 255.0f;
      float R = pixel[2] / 255.0f;
      // std::cout << "threadId: " << omp_get_thread_num() << std::endl;

      // Vorlesung:
      float y = 16 + (65.481f * R + 128.553f * G + 24.966f * B);
      float cb = 128 + (-37.797f * R - 74.203f * G + 112.0f * B);
      float cr = 128 + (112.0f * R - 93.786f * G - 18.214f * B);

      outputImage.at<cv::Vec3b>(i, j) = {y, cr, cb};
    }
  }
  double t1 = omp_get_wtime(); // end time

  std::cout << "Processing took " << (t1 - t0) << " seconds" << std::endl;
  // cv::cvtColor(image, outputImage, cv::COLOR_BGR2YCrCb);
  cv::imshow("original", image);
  cv::imshow("ycbcr", outputImage);
  int key = cv::waitKey(0);
  cv::destroyAllWindows();
}