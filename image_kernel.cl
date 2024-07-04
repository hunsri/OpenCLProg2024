__kernel
void processImage( __global uchar* inputImage, __global uchar* outputImage, int width, int height, int depth )
{
  // get the thread number in X and Y dimensions
  int x = get_local_size( 0 ) * get_group_id( 0 ) + get_local_id( 0 );
  int y = get_local_size( 1 ) * get_group_id( 1 ) + get_local_id( 1 );

  // but really, we can actually just do this in OpenCL instead:
  // int x = get_global_id( 0 );
  // int y = get_global_id( 1 );

  // the rest is the same as in CUDA:
  const unsigned int loc = (y * width + x) * depth;

  if ( x >= width || y >= height )
    return;

  float B = inputImage[loc + 0] / 255.0f;
  float G = inputImage[loc + 1] / 255.0f;
  float R = inputImage[loc + 2] / 255.0f;

  // outputImage[loc + 0] = inputImage[loc + 2]; // R --> B
  // outputImage[loc + 1] = inputImage[loc + 1]; // G --> G
  // outputImage[loc + 2] = inputImage[loc + 0]; // B --> R
  outputImage[loc + 0] = 16 + (65.481f * R + 128.553f * G + 24.966f * B);
  outputImage[loc + 1] = 128 + (112.0f * R - 93.786f * G - 18.214f * B);
  outputImage[loc + 2] = 128 + (-37.797f * R - 74.203f * G + 112.0f * B);
}