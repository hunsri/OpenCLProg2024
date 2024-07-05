__kernel
void rgb_to_ycbcr( __global uchar* inputImage, __global uchar* outputImage, int width, int height, int depth )
{
  int x = get_global_id(0);
  int y = get_global_id(1);
  int z = get_global_id(2);

  // printf("x: %d, y: %d, z: %d\n", x, y, z);

  const unsigned int loc = (y * width + x) * depth;

  if (x >= width || y >= height)
    return;

  float B = inputImage[loc + 0] / 255.0f;
  float G = inputImage[loc + 1] / 255.0f;
  float R = inputImage[loc + 2] / 255.0f;

  outputImage[loc + 0] = 16 + (65.481f * R + 128.553f * G + 24.966f * B);
  outputImage[loc + 1] = 128 + (112.0f * R - 93.786f * G - 18.214f * B);
  outputImage[loc + 2] = 128 + (-37.797f * R - 74.203f * G + 112.0f * B);
}