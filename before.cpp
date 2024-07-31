#include <iostream>
#include <cstdio>
#include <omp.h>

#include "opencv2/opencv.hpp"
#include "utils.h"

uchar CalculateDilatationForPixel(cv::Mat grayScaleImage, int x, int y)
{
  int width = grayScaleImage.cols;
  int height = grayScaleImage.rows;
  uchar pxlVal = 0;

  for (int i=-3; i<=3; ++i)
  {
    for (int j=-3; j<=3; ++j)
    {
      if (x+i < 0 || y+j < 0 || x+i >= height || y+j >= width)
        continue;

      //looking at a new pixel in the surrounding pixels
      const uchar newY = grayScaleImage.at<uchar>(x+i, y+j);

      //if it is bigger than the previously looked at, keep the bigger value
      pxlVal = std::max(pxlVal, newY);
    }

  }

  return pxlVal;

}

cv::Mat RenderDilatation(cv::Mat image)
{
  cv::Mat outputImageGrayscale = cv::Mat::zeros(image.size(), CV_8UC1);

  // grayscaling

  for (int i = 0; i < image.rows; ++i) {
    for (int j = 0; j < image.cols; ++j) {

      cv::Vec3b pixel = image.at<cv::Vec3b>( i, j );

      float B = pixel[0];
      float G = pixel[1];
      float R = pixel[2];

      float Y = 0.299 * R + 0.587 * G + 0.114 * B;

      outputImageGrayscale.at<uchar>(i, j) = {Y};
    }
  }

  cv::Mat outputImageDilatation = cv::Mat::zeros(image.size(), CV_8UC1);

  for (int i = 0; i < image.rows; ++i) {
    for (int j = 0; j < image.cols; ++j) {

      uchar pixel = CalculateDilatationForPixel(outputImageGrayscale, i, j);

      outputImageDilatation.at<uchar>(i, j) = {pixel};
    }
  }
  return outputImageDilatation;
}

cv::Mat RenderYCbCr(cv::Mat image)
{
  cv::Mat outputImage = cv::Mat::zeros(image.size(), image.type());

  for (int i = 0; i < image.rows; ++i) {
    for (int j = 0; j < image.cols; ++j) {

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

      outputImage.at<cv::Vec3b>(i, j) = {y, cr, cb};
    }
  }
  return outputImage;
}

int main(int argc, char** argv)
{
  // read image
  cv::Mat image = get_image(argc, argv);

  // execute RGB to YCbCr
  double t0 = omp_get_wtime(); // start time
  cv::Mat outputImage = RenderYCbCr(image);
  double t1 = omp_get_wtime(); // end time

  std::cout << "Processing for YCbCr took " << (t1 - t0) << " seconds" << std::endl;

  // execute Dilatation
  t0 = omp_get_wtime(); // start time
  cv::Mat dilatationImage = RenderDilatation(image);
  t1 = omp_get_wtime(); // end time
  std::cout << "Processing for Dilatation took " << (t1 - t0) << " seconds" << std::endl;

  cv::imshow("original", image);
  cv::imshow("ycbcr", outputImage);
  cv::imshow("dilatation", dilatationImage);
  int key = cv::waitKey(0);
  cv::destroyAllWindows();
}