__kernel
void grayscale( __global uchar* inputImage, __global uchar* outputImage, int width, int height, int depth )
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

  float B = inputImage[loc + 0];
  float G = inputImage[loc + 1];
  float R = inputImage[loc + 2];

  float Y = 0.299 * R + 0.587 * G + 0.114 * B;


  outputImage[loc + 0] = Y;
  outputImage[loc + 1] = Y;
  outputImage[loc + 2] = Y;
}


__kernel
void dilatation( __global uchar* inputImage, __global uchar* outputImage, int width, int height, int depth )
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

  float pxVal = 0;


  for(int i=-3; i<=3; ++i)
	{
		for(int j=-3; j<=3; ++j)
		{
      if(x+i < 0 || x+i >= width || y+j < 0 || y+j >= height)
        continue;

      unsigned int newLoc = ((y+j) * width + (x+i)) * depth;
      float newY = 0.299 * inputImage[newLoc + 2] + 0.587 * inputImage[newLoc + 1] + 0.114 * inputImage[newLoc];
			
      pxVal = max(pxVal, newY);
		}
	}


  outputImage[loc + 0] = pxVal;
  outputImage[loc + 1] = pxVal;
  outputImage[loc + 2] = pxVal;
}