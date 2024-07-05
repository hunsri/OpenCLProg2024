__kernel
void grayscale(__global uchar* inputImage, __global uchar* outputImage, int width, int height, int depth)
{
  int x = get_global_id(0);
  int y = get_global_id(1);

  const unsigned int loc = (y * width + x) * depth;

  if (x >= width || y >= height)
    return;

  float B = inputImage[loc + 0];
  float G = inputImage[loc + 1];
  float R = inputImage[loc + 2];

  float Y = 0.299 * R + 0.587 * G + 0.114 * B;

  outputImage[y * width + x] = Y;
}


__kernel
void dilation(__global uchar* inputImage, __global uchar* outputImage, int width, int height)
{
  int x = get_global_id(0);
  int y = get_global_id(1);

  const unsigned int loc = y * width + x;

  if (x >= width || y >= height)
    return;

  float pxVal = 0;

  for (int i=-3; i<=3; ++i)
  {
    for (int j=-3; j<=3; ++j)
    {
      if (x+i < 0 || y+j < 0 || x+i >= width || y+j >= height)
        continue;

      const unsigned int newLoc = (y+j) * width + (x+i);
      const float newY = inputImage[newLoc];

      pxVal = max(pxVal, newY);
    }
  }


  outputImage[loc] = pxVal;
}