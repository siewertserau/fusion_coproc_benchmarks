//*****************************************************************************************//
//  houghKernels.cu - CUDA Hough Transform Benchmark
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
//  running Ubuntu 14.04 
//	
//	Please use at your own risk. We are sharing so that other researchers and developers can 
//	recreate our results and make suggestions to improve and extend the benchmarks over time.
//
//*****************************************************************************************//

#include <stdio.h>
#include <math.h>
 
#define MAXRGB 			255


__global__ void sobel(u_char * frame_in, u_char * frame_out, int width, int height )
{
    
	int x = blockDim.x*blockIdx.x+threadIdx.x;
	int y = blockDim.y*blockIdx.y+threadIdx.y;
	int index = x + y*width;
	long int size = width*height;
	
	
	//Sobel Implementation
	int Gx[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};
    int Gy[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
	int G_x=0,G_y=0,G = 0; 
	int i=0,j=0;
    if (index < size && (x>1 && y>1) && (x < (width-1) && y < (height-1 ))   ) 
	{
	
		for(i=0;i<3;i++)
		{
			for(j=0;j<3;j++)
			{
				G_y += (Gy[i][j])*frame_in[(x+j-1)+(width*(y+i-1))];
				G_x += (Gx[i][j])*frame_in[(x+j-1)+(width*(y+i-1))];
			}
		}
		
				
		G = abs(G_x) + abs(G_y);
			
		if(G>MAXRGB)
			frame_out[index] = MAXRGB;
		else
			frame_out[index] = G;
		
    }
}

__global__ void houghTransform(u_char * frame_in, u_char * frame_out,const int hough_h)
{
	int x = blockDim.x*blockIdx.x+threadIdx.x;
	int y = blockDim.y*blockIdx.y+threadIdx.y;
	int width = gridDim.x*blockDim.x;
	int height = gridDim.y*blockDim.y;
	int index = x + y*width;
	double DEG2RAD = 0.0174533;
	
	//double DEG2RAD = 1;
	double center_x = width/2;
	double center_y = height/2;   

//	 frame_out[index] = 127;
	if( frame_in[index] > 250 )			//checking for the values greater than 250, has to be modified if we have different threshold 
	{
		for(int t=0;t<180;t++)  
		{
		double r = ( ((double)x - center_x) * cos((double)t * DEG2RAD)) + (((double)y - center_y) * sin((double)t * DEG2RAD));		//plotting x and y in ro and theta
		frame_out[ (int)((round(r + hough_h) * 180.0)) + t]++;
		
	
 
		}
	
	}
//   __syncthreads();
}

