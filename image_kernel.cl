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

  outputImage[loc + 0] = inputImage[loc + 2]; // R --> B
  outputImage[loc + 1] = inputImage[loc + 1]; // G --> G
  outputImage[loc + 2] = inputImage[loc + 0]; // B --> R
}