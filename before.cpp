// ONE IMAGE

#include <iostream>
#include <cstdio>
#include <omp.h>

#include "opencv2/opencv.hpp"

int main(int argc, char** argv)
{
  // read image
  cv::Mat image = cv::imread( "images/human/3.harold_large.jpg", cv::IMREAD_UNCHANGED );
  cv::Mat convertedImage;
  cv::cvtColor(image, convertedImage, cv::COLOR_BGR2YCrCb);

  // display and wait for a key-press, then close the window
  cv::imshow( "image", image );
  int key = cv::waitKey( 0 );
  cv::destroyAllWindows();

  double t0 = omp_get_wtime(); // start time

  #pragma omp parallel for
  for ( int i = 0; i < image.rows; ++i ) {
    for ( int j = 0; j < image.cols; ++j ) {

      // get pixel at [i, j] as a <B,G,R> vector
      cv::Vec3b pixel = image.at<cv::Vec3b>( i, j );

      // extract the pixels as uchar (unsigned 8-bit) types (0..255)
      float B = pixel[0] / 255.0f;
      float G = pixel[1] / 255.0f;
      float R = pixel[2] / 255.0f;

      // Vorlesung:
      float y = 16 + (65.481f * R + 128.553f * G + 24.966f * B);
      float cb = 128 + (-37.797f * R - 74.203f * G + 112.0f * B);
      float cr = 128 + (112.0f * R - 93.786f * G - 18.214f * B);

      image.at<cv::Vec3b>( i, j ) = {y, cr, cb};
    }
  }
  double t1 = omp_get_wtime();  // end time

  std::cout << "Processing took " << (t1 - t0) << " seconds" << std::endl;

  // display and wait for a key-press, then close the window
  image.convertTo(convertedImage, CV_32F, 1.0 / 255, 0);
  cv::imshow( "image", image );
  key = cv::waitKey( 0 );
  cv::destroyAllWindows();
}