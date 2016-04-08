//*****************************************************************************************//
//  pyramid_kernel.cu - CUDA Pyramidal Transform kernels
//
//  Authors: Ramnarayan Krishnamurthy, University of Colorado (Shreyas.Ramnarayan@gmail.com)
//	         Matthew Demi Vis, Embry-Riddle Aeronautical University (MatthewVis@gmail.com)
//			 
//	This code was used to obtain results documented in the SPIE Sensor and Technologies paper: 
//	S. Siewert, V. Angoth, R. Krishnamurthy, K. Mani, K. Mock, S. B. Singh, S. Srivistava, 
//	C. Wagner, R. Claus, M. Demi Vis, “Software Defined Multi-Spectral Imaging for Arctic 
//	Sensor Networks”, SPIE Algorithms and Technologies for Multipectral, Hyperspectral, and 
//	Ultraspectral Imagery XXII, Baltimore, Maryland, April 2016. 
//
//	This code was developed for, tested and run on a Jetson TK1 development kit by NVIDIA
//  running TODO. 
//	
//	Please use at your own risk. We are sharing so that other researchers and developers can 
//	recreate our results and make suggestions to improve and extend the benchmarks over time.
//
//*****************************************************************************************//
#ifndef _PYRAMID_KERNEL_H_
#define _PYRAMID_KERNEL_H_

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))
#define CLAMP_8bit(x) max(0, min(255, (x)))

#define TILE_WIDTH  	6
#define TILE_HEIGHT 	6

#define FILTER_RADIUS	2

#define FILTER_DIAMETER (2 * FILTER_RADIUS + 1)

#define BLOCK_WIDTH   (TILE_WIDTH + 2*FILTER_RADIUS)
#define BLOCK_HEIGHT  (TILE_HEIGHT + 2*FILTER_RADIUS)

/***************************************************************************************************
 * CUDA equivalent of pyrdown OpenCV function				  									  **
 * refer: http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=pyrdown#pyrdown **
 ***************************************************************************************************/
__global__ void PyrDown(unsigned char* g_DataIn, unsigned char* g_DataOut, int width, int height)
{
   __shared__ unsigned char sharedMem[BLOCK_HEIGHT * BLOCK_WIDTH];
   float gaussianMatrix[25];

    gaussianMatrix[0] = 1;
    gaussianMatrix[1] = 4;
    gaussianMatrix[2] = 6;
    gaussianMatrix[3] = 4;
    gaussianMatrix[4] = 1;
	
    gaussianMatrix[5] = 4;
    gaussianMatrix[6] = 16;
    gaussianMatrix[7] = 24;
    gaussianMatrix[8] = 16;
    gaussianMatrix[9] = 4;

    gaussianMatrix[10] = 6;
    gaussianMatrix[11] = 24;
    gaussianMatrix[12] = 36;
    gaussianMatrix[13] = 24;
    gaussianMatrix[14] = 6;

    gaussianMatrix[15] = 4;
    gaussianMatrix[16] = 16;
    gaussianMatrix[17] = 24;
    gaussianMatrix[18] = 16;
    gaussianMatrix[19] = 4;

    gaussianMatrix[20] = 1;
    gaussianMatrix[21] = 4;
    gaussianMatrix[22] = 6;
    gaussianMatrix[23] = 4;
    gaussianMatrix[24] = 1;
	
	// Global x,y input coordinates
   int x = blockIdx.x * TILE_WIDTH + threadIdx.x ;
   int y = blockIdx.y * TILE_HEIGHT + threadIdx.y ;

   // Global index of input and corresponding output
   int index = y * (width) + x;
   int out_index = y/2 * (width/2) + x/2;

   // Threads outside the image
   if (x >= width || y >= height)
      return;

   // Current block values
   int sharedIndex = threadIdx.y * blockDim.y + threadIdx.x;

   // Border cases of the global image
   if( x < FILTER_RADIUS || y < FILTER_RADIUS) {
       g_DataOut[out_index] = g_DataIn[index];
	   sharedMem[sharedIndex] = g_DataIn[index];
       return;
    }

   if ((x > width - FILTER_RADIUS - 1)&&(x <width)) {
       g_DataOut[out_index] = g_DataIn[index];
	   sharedMem[sharedIndex] = g_DataIn[index];
       return;
    }

    if ((y > height - FILTER_RADIUS - 1)&&(y < height)) {
       g_DataOut[out_index] = g_DataIn[index];
       sharedMem[sharedIndex] = g_DataIn[index];
       return;
    }

    sharedMem[sharedIndex] = g_DataIn[index];
   __syncthreads();

   // Starting Computation
   // Don't do any computation for even rows and even columns
   if(x%2 != 0 || y%2 != 0)
	   return;

   // Don't do any computation for outside the current block
    if(threadIdx.x < FILTER_RADIUS || threadIdx.y < FILTER_RADIUS || threadIdx.x > (BLOCK_WIDTH - FILTER_RADIUS - 1) 
	|| threadIdx.y > (BLOCK_HEIGHT - FILTER_RADIUS - 1)) {
	    return;
	}
	
   // Computation for active threads inside the current block
    float sumX = 0;
	for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++) {
        for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++) {
            float Pixel = (float)(sharedMem[threadIdx.y * BLOCK_WIDTH + threadIdx.x +  (dy * BLOCK_WIDTH + dx)]);
            sumX += Pixel * gaussianMatrix[(dy + FILTER_RADIUS) * FILTER_DIAMETER + (dx+FILTER_RADIUS)];
          }
	}
    g_DataOut[out_index] = CLAMP_8bit(sumX/256);
}

/***************************************************************************************************
 * CUDA equivalent of pyrup OpenCV function				  										  **
 * refer: http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=pyrdown#pyrdown **
 ***************************************************************************************************/
__global__ void PyrUp(unsigned char* g_DataIn, unsigned char* g_DataOut, int width, int height)
{
   __shared__ unsigned char sharedMem[BLOCK_HEIGHT * BLOCK_WIDTH];
   float laplacianMatrix[25];

    laplacianMatrix[0] = 1;
    laplacianMatrix[1] = 4;
    laplacianMatrix[2] = 6;
    laplacianMatrix[3] = 4;
    laplacianMatrix[4] = 1;
	
    laplacianMatrix[5] = 4;
    laplacianMatrix[6] = 16;
    laplacianMatrix[7] = 24;
    laplacianMatrix[8] = 16;
    laplacianMatrix[9] = 4;

    laplacianMatrix[10] = 6;
    laplacianMatrix[11] = 24;
    laplacianMatrix[12] = 36;
    laplacianMatrix[13] = 24;
    laplacianMatrix[14] = 6;

    laplacianMatrix[15] = 4;
    laplacianMatrix[16] = 16;
    laplacianMatrix[17] = 24;
    laplacianMatrix[18] = 16;
    laplacianMatrix[19] = 4;

    laplacianMatrix[20] = 1;
    laplacianMatrix[21] = 4;
    laplacianMatrix[22] = 6;
    laplacianMatrix[23] = 4;
    laplacianMatrix[24] = 1;
	
	// Global x,y input coordinates
   int x = blockIdx.x * TILE_WIDTH + threadIdx.x ;
   int y = blockIdx.y * TILE_HEIGHT + threadIdx.y ;

   // Global index of input and corresponding output
   int index = y/2 * (width/2) + x/2;
   int out_index = y * (width) + x;

   // Threads outside the image
   if (x >= width || y >= height)
      return;

   // Current block values
   int sharedIndex = threadIdx.y * blockDim.y + threadIdx.x;

   // Border cases of the global image
   if( x < FILTER_RADIUS || y < FILTER_RADIUS) {
       g_DataOut[out_index] = g_DataIn[index];
	   sharedMem[sharedIndex] = g_DataIn[index];
       return;
    }

   if ((x > width - FILTER_RADIUS - 1)&&(x <width)) {
       g_DataOut[out_index] = g_DataIn[index];
	   sharedMem[sharedIndex] = g_DataIn[index];
       return;
    }

    if ((y > height - FILTER_RADIUS - 1)&&(y < height)) {
       g_DataOut[out_index] = g_DataIn[index];
       sharedMem[sharedIndex] = g_DataIn[index];
       return;
    }

   if(x%2 != 0 || y%2 != 0) {
	   sharedMem[sharedIndex] = 0;
   }   else {
	 sharedMem[sharedIndex] = g_DataIn[index];  
   } 
   __syncthreads();

   // Starting Computation
   // Don't do any computation for outside the current block
    if(threadIdx.x < FILTER_RADIUS || threadIdx.y < FILTER_RADIUS || threadIdx.x > (BLOCK_WIDTH - FILTER_RADIUS - 1) 
	|| threadIdx.y > (BLOCK_HEIGHT - FILTER_RADIUS - 1)) {
	    return;
	}
	
   // Computation for active threads inside the current block
    float sumX = 0;
	for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++) {
        for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++) {
            float Pixel = (float)(sharedMem[threadIdx.y * BLOCK_WIDTH + threadIdx.x +  (dy * BLOCK_WIDTH + dx)]);
            sumX += Pixel * laplacianMatrix[(dy + FILTER_RADIUS) * FILTER_DIAMETER + (dx+FILTER_RADIUS)];
          }
	}
    g_DataOut[out_index] = CLAMP_8bit(sumX/64);
}

/***************************************************************************************************
 * C equivalent of pyrdown OpenCV function				  										  **
 * refer: http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=pyrdown#pyrdown **
 ***************************************************************************************************/
void CPU_pyrdown(unsigned char* imageIn, unsigned char* imageOut, int width, int height)
{
  int i, j, rows, cols, startCol, endCol, startRow, endRow;
  const float gaussianMatrix[25] = {1,4,6,4,1,4,16,24,16,4,6,24,36,24,6,4,16,24,16,4,1,4,6,4,1};

  rows = height;
  cols = width;

  unsigned char* gaussimage = (unsigned char *)malloc(width * height);

  // Initialize all output pixels to zero
  for(i=0; i<rows; i++) {
    for(j=0; j<cols; j++) {
    	gaussimage[i*width + j] = 0;
    }
  }

  startCol = FILTER_RADIUS;
  endCol = cols - FILTER_RADIUS;
  startRow = FILTER_RADIUS;
  endRow = rows - FILTER_RADIUS;

  // Loop for each pixel
  for(i=startRow; i<endRow; i++) {
    for(j=startCol; j<endCol; j++) {
       // convolution of the image with the filter
       float sumX = 0;
       int dy;
       for(dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++) {
    	   int dx;
          for(dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++) {
             float Pixel = (float)(imageIn[i*width + j +  (dy * width + dx)]);
             sumX += Pixel * gaussianMatrix[(dy + FILTER_RADIUS) * FILTER_DIAMETER + (dx+FILTER_RADIUS)];
          }
	}
       gaussimage[i*width + j] = CLAMP_8bit(sumX/256);
    }
  }

  //sub-sampling
  for(i=0; i<height/2; i++) {
    for(j=0; j<width/2; j++) {
    	imageOut[i*(width/2) + j] = gaussimage[i*width*2 + j*2];
    }
  }

  free(gaussimage);
}

/***************************************************************************************************
 * C equivalent of pyrup OpenCV function				  										  **
 * refer: http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=pyrdown#pyrup   **
 ***************************************************************************************************/
void CPU_pyrup(unsigned char* imageIn, unsigned char* imageOut, int width, int height)
{
  int i, j, rows, cols, startCol, endCol, startRow, endRow;
  const float laplacianMatrix[25] = {1,4,6,4,1,4,16,24,16,4,6,24,36,24,6,4,16,24,16,4,1,4,6,4,1};

  rows = height;
  cols = width;

  unsigned char* laplaceimage = (unsigned char *)malloc(width*2 * height*2);

  // super-sampling
  // Insert 0s between columns
  for(i=0; i<rows*2; i=i+2) {
    for(j=1; j<cols*2; j=j+2) {
    	laplaceimage[i*width*2 + j] = 0;
    }
  }

  // Insert 0s between rows
  for(i=1; i<rows*2; i=i+2) {
    for(j=0; j<cols*2; j++) {
    	laplaceimage[i*width*2 + j] = 0;
    }
  }

  int x=0,y=0;
  // Insert pixels from input image
  for(i=0; i<rows*2; i++,i++,x++) {
	  y=0;
    for(j=0; j<cols*2; j++,j++,y++) {
    	laplaceimage[i*width*2 + j] = imageIn[x*width + y];
    }
  }

  startCol = FILTER_RADIUS;
  endCol = cols*2 - FILTER_RADIUS;
  startRow = FILTER_RADIUS;
  endRow = rows*2 - FILTER_RADIUS;

  // Loop for each pixel
  for(i=startRow; i<endRow; i++) {
    for(j=startCol; j<endCol; j++) {
        // convolution of the image with the filter
       float sumX = 0;
       int dy;
       for(dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++) {
    	   int dx;
          for(dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++) {
             float Pixel = (float)(laplaceimage[i*width*2 + j +  (dy * width*2 + dx)]);
             sumX += Pixel * laplacianMatrix[(dy + FILTER_RADIUS) * FILTER_DIAMETER + (dx+FILTER_RADIUS)];
          }
	}
       imageOut[i*width*2 + j] = CLAMP_8bit(sumX/64);
    }
  }
  free(laplaceimage);
}

void CPU_pyrdiff(unsigned char* image1, unsigned char* image2, unsigned char* diff_image, int width, int height)
{
	int i=0,j=0;
	for(i=0; i<height; i++)
		for(j=0; j<width; j++)
			diff_image[i*width + j] = abs(image1[i*width + j] - image2[i*width + j]);
}

#endif // _PYRAMID_KERNEL_H_


