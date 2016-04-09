//*****************************************************************************************//
//  Sobel_kernal.cu - CUDA Sobel Edge detection benchmark
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

// Standard includes
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdint.h>
#include <errno.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include <sys/io.h>
#include <iostream>

#include <time.h>
#include <pthread.h>
#include <sched.h>

// Project Includes
#include <cuda_runtime.h>
#include "ppm.h"
#include "options.h"

// Project Specific Defines
#define MAXRGB	 	255
#define BLOCK_SIZE 	8
#define DEFAULT_IMAGE "beach.pgm"

//#define DEBUG

// Global variables for RT threads
pthread_attr_t rt_sched_attr;
struct sched_param rt_param;
pid_t mainpid;
pthread_t rt_thread;
int rt_max_prio;

// Global Variables for Transform
unsigned char *h_img_out_array, *h_img_in_array;
struct timespec run_time = {0, 0};
unsigned int img_width, img_height, img_chan;
bool run_once = false;
int freq = 0;
std::string imageFilename = DEFAULT_IMAGE;

//***************************************************************//
// Initialize CUDA hardware
//***************************************************************//
#if __DEVICE_EMULATION__
	bool InitCUDA(void){
		fprintf(stderr, "There is no device.\n");
		return true;
	}
#else
	bool InitCUDA(void)
	{
		int count = 0;
		int i = 0;
		
		cudaGetDeviceCount(&count);
		if(count == 0) {
			fprintf(stderr, "There is no device.\n");
			return false;
		}
		
		for(i = 0; i < count; i++) 
		{
			cudaDeviceProp prop;
			if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) 
			{
				if(prop.major >= 1)
				break;
			}
		}
		
		if(i == count) 
		{
			fprintf(stderr, "There is no device supporting CUDA.\n");
			return false;
		}
		
		cudaSetDevice(i);
		
		printf("CUDA initialized.\n");
		return true;
	}
#endif

//***************************************************************//
// Sobel transform using CUDA hardware
//***************************************************************//
__global__ void CUDA_transform(unsigned char *img_out, unsigned char *img_in, unsigned int width, unsigned int height){
	int x,y;
	unsigned char LUp,LCnt,LDw,RUp,RCnt,RDw;
	int pixel;
	
	x=blockDim.x*blockIdx.x+threadIdx.x;
	y=blockDim.y*blockIdx.y+threadIdx.y;
	
	if( x<width && y<height )
	{
		LUp = (x-1>=0 && y-1>=0) ? img_in[(x-1)+(y-1)*width] : 0;
		LCnt= (x-1>=0)           ? img_in[(x-1)+y*width]:0;
		LDw = (x-1>=0 && y+1<height) ? img_in[(x-1)+(y+1)*width] : 0;
		RUp = (x+1<width && y-1>=0)  ? img_in[(x+1)+(y-1)*width] : 0;
		RCnt= (x+1<width)            ? img_in[(x+1)+y*width] : 0;
		RDw = (x+1<width && y+1<height) ? img_in[(x+1)+(y+1)*width] : 0;
		pixel = -1*LUp  + 1*RUp +
		-2*LCnt + 2*RCnt +
		-1*LDw  + 1*RDw;
		pixel = (pixel<0) ? 0 : pixel;
		pixel = (pixel>MAXRGB) ? MAXRGB : pixel;
		img_out[x+y*width] = pixel;
	}
}

//***************************************************************//
// Sobel transform using the CPU
//***************************************************************//
void CPU_transform(unsigned char *img_out, unsigned char *img_in, unsigned int width, unsigned int height) {
	unsigned char LUp,LCnt,LDw,RUp,RCnt,RDw;
	int pixel;
	for(int y=0; y<height; y++)
	{
		for(int x=0; x<width; x++)
		{
			#ifdef DEBUG
				printf("Pixel X:%d Y:%d\n",x,y);
			#endif
			assert(x+(y*width)<width*height);
			LUp = (x-1>=0 && y-1>=0)? img_in[(x-1)+(y-1)*width]:0;
			LCnt= (x-1>=0)? img_in[(x-1)+y*width]:0;
			LDw = (x-1>=0 && y+1<height)? img_in[(x-1)+(y+1)*width]:0;
			RUp = (x+1<width && y-1>=0)? img_in[(x+1)+(y-1)*width]:0;
			RCnt= (x+1<width)? img_in[(x+1)+y*width]:0;
			RDw = (x+1<width && y+1<height)? img_in[(x+1)+(y+1)*width]:0;
			pixel = -1*LUp  + 1*RUp + -2*LCnt + 2*RCnt + -1*LDw  + 1*RDw;
			pixel=(pixel<0)?0:pixel;
			pixel=(pixel>MAXRGB)?MAXRGB:pixel;
			img_out[x+y*width]=pixel;
			#ifdef DEBUG
				printf("\r%5.2f",100*(float)(y*width+x)/(float)(width*height-1));            
			#endif
		}
	}
#ifdef DEBUG
	printf("\n");
#endif
}

//***************************************************************//
// Take the difference of two timespec structures
//***************************************************************//
void timespec_diff(struct timespec *start, struct timespec *stop,
struct timespec *result, bool check_neg)
{        
	result->tv_sec = stop->tv_sec - start->tv_sec;
	result->tv_nsec = stop->tv_nsec - start->tv_nsec;
	
	if ( check_neg && result->tv_nsec < 0) {
		result->tv_sec = result->tv_sec - 1;
		result->tv_nsec = result->tv_nsec + 1000000000;
		
	}
}

//***************************************************************//
// Convert timespec to double containing time in ms
//***************************************************************//
double timespec2double( struct timespec time_in)
{
	double rv;
	rv = (((double)time_in.tv_sec)*1000)+(((double)time_in.tv_nsec)/1000000);
	return rv;
}

//***************************************************************//
// Transform thread
//***************************************************************//
void *CUDA_transform_thread(void * threadp)
{
	// CUDA transform local variables
	unsigned char *d_img_out_array=NULL, *d_img_in_array=NULL;
	struct timespec start_time, end_time, elap_time, diff_time;
	int errVal;
	double start_time_d, end_time_d, elap_time_d, diff_time_d;
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(img_width / threads.x, img_height / threads.y);

	// Allocate CPU memory
	h_img_out_array = (unsigned char *)malloc(img_width * img_height);
	
	// Allocate CUDA memory for in and out image
	cudaMalloc((void**) &d_img_in_array, sizeof(unsigned char)*img_width*img_height);
	cudaMalloc((void**) &d_img_out_array, sizeof(unsigned char)*img_width*img_height);
	
	// Infinite loop to allow for power measurement
	do
	{
		// Get start of runtime timing
		if(clock_gettime(CLOCK_REALTIME, &start_time) )
		{
			printf("clock_gettime() - start - error.. exiting.\n");
			break;
		}
		start_time_d = timespec2double(start_time);
		
		// Copy the image into CUDA memory 
		cudaMemcpy(d_img_in_array, h_img_in_array, sizeof(unsigned char)*img_width*img_height, cudaMemcpyHostToDevice);
		
		// Complete the Sobel Transform
		CUDA_transform<<<grid, threads, 0>>>(d_img_out_array, d_img_in_array, img_width, img_height);
		cudaThreadSynchronize();
		
		// Copy the transformed image back from the CUDA memory 
		cudaMemcpy(h_img_out_array, d_img_out_array, sizeof(unsigned char)*img_width*img_height, cudaMemcpyDeviceToHost);
		
		// Get end of transform time timing
		if(clock_gettime(CLOCK_REALTIME, &end_time) )
		{
			printf("clock_gettime() - end - error.. exiting.\n");
			break;
		}
		
		if(run_time.tv_nsec != 0)
		{
			// Calculate the timing for nanosleep
			timespec_diff(&start_time, &end_time, &elap_time, true);
			timespec_diff(&elap_time, &run_time, &diff_time, false);
#ifdef DEBUG
			end_time_d = timespec2double(end_time);
			elap_time_d = end_time_d - start_time_d;
			diff_time_d = timespec2double(diff_time);
			printf("DEBUG: Transform runtime: %fms\n       Sleep time:       %fms\n",  elap_time_d, diff_time_d);
#endif	
			if(diff_time.tv_sec < 0 || diff_time.tv_nsec < 0)
			{
				diff_time_d = timespec2double(diff_time);
				printf("TIME OVERRUN by %fms---------\n", -diff_time_d);  
			} else
			{
				// Sleep for time needed to allow for running at known frequency 
				errVal = nanosleep(&diff_time, &end_time);			
				if(errVal == -1)
				{
					printf("\nFreq delay interrupted. Exiting..\n");
					printf("**%d - %s**\n", errno, strerror(errno));
					break;
				}
			}
		}
		
		// Get and calculate end of runtime time
		if(clock_gettime(CLOCK_REALTIME, &end_time) )
		{
			printf("clock_gettime() - end - error.. exiting.\n");
			break;
		}
		end_time_d = timespec2double(end_time);
		elap_time_d = end_time_d - start_time_d;
		printf("     Freq: %f Hz\n", 1000.0/elap_time_d);
	} while(!run_once);
	
	cudaFree(d_img_in_array);
	cudaFree(d_img_out_array);
	
	return NULL;
}

//***************************************************************//
// Transform thread
//***************************************************************//
void *CPU_transform_thread(void * threadp)
{
	// CPU transform local variables
	struct timespec start_time, end_time, elap_time, diff_time;
	int errVal;
	double start_time_d, end_time_d, elap_time_d, diff_time_d;

	// Allocate memory
	h_img_out_array = (unsigned char *)malloc(img_width * img_height);
	
	// Infinite loop to allow for power measurement
	do
	{
		// Get start of runtime timing
		if(clock_gettime(CLOCK_REALTIME, &start_time) )
		{
			printf("clock_gettime() - start - error.. exiting.\n");
			break;
		}
		start_time_d = timespec2double(start_time);
        
		CPU_transform(h_img_out_array, h_img_in_array, img_width, img_height);
		
		// Get end of transform time timing
		if(clock_gettime(CLOCK_REALTIME, &end_time) )
		{
			printf("clock_gettime() - end - error.. exiting.\n");
			break;
		}
		
		if(run_time.tv_nsec != 0)
		{
			// Calculate the timing for nanosleep
			timespec_diff(&start_time, &end_time, &elap_time, true);
			timespec_diff(&elap_time, &run_time, &diff_time, false);
#ifdef DEBUG
			end_time_d = timespec2double(end_time);
			elap_time_d = end_time_d - start_time_d;
			diff_time_d = timespec2double(diff_time);
			printf("DEBUG: Transform runtime: %fms\n       Sleep time:       %fms\n",  elap_time_d, diff_time_d);
#endif	
			if(diff_time.tv_sec < 0 || diff_time.tv_nsec < 0)
			{
				diff_time_d = timespec2double(diff_time);
				printf("TIME OVERRUN by %fms---------\n", -diff_time_d);  
			} else
			{
				// Sleep for time needed to allow for running at known frequency 
				errVal = nanosleep(&diff_time, &end_time);			
				if(errVal == -1)
				{
					printf("\nFreq delay interrupted. Exiting..\n");
					printf("**%d - %s**\n", errno, strerror(errno));
					break;
				}
			}
		}
		
		// Get and calculate end of runtime time
		if(clock_gettime(CLOCK_REALTIME, &end_time) )
		{
			printf("clock_gettime() - end - error.. exiting.\n");
			break;
		}
		end_time_d = timespec2double(end_time);
		elap_time_d = end_time_d - start_time_d;
		printf("     Freq: %f Hz\n", 1000.0/elap_time_d);
	} while(!run_once);
	return NULL;
}

//***************************************************************//
// Main function
//***************************************************************//
int main(int argc, char* argv[])
{
	// Local variables
	Options options(argc, argv);
	int errVal = 0;
	bool use_cuda = 0;
	struct sched_param main_param;
	int tempInt, rv;
	char tempChar[50];
	
	// Check input
	if(options.has("help") || options.has("h") || options.has("?")) 
	{
		std::cout << "Usage: " << argv[0] << " [-continuous [-fps=FPS]] [-img=imageFilename]  [-cuda]" << std::endl;
		exit(EXIT_SUCCESS);
	}
	
	if(options.has("img")) 
	{
		imageFilename = options.get<std::string>("img");
	}
	std::cout << "Img set to " << imageFilename << std::endl;
	
	if(options.has("continuous")) 
	{
		run_once = false;
		std::cout << "Continuous mode" << std::endl;
		if(options.has("fps")) 
		{
			freq = options.get<unsigned int>("fps");
			std::cout << "FPS Limiter set at " << freq << std::endl;
		} else 
		{
			freq = 0;
			std::cout << "FPS Unlimited" << std::endl;
		}
	} else 
	{
		run_once = true;
		std::cout << "Single shot mode" << std::endl;
	}
	
	if (options.has("cuda")) 
	{
		std::cout << "Program will use CUDA for transform." << std::endl;
		use_cuda = true;
	} else 
	{
		use_cuda = false;
		std::cout << "Program will use CPU for transform." << std::endl;
	}
	
#ifdef DEBUG
	printf("DEBUG: Begin Program. \n");
#endif
	
	// Initialize CUDA
	if(!InitCUDA()) {
		return 0;
	};
	
	// Read Input image
	printf("Reading input image...");
	rv = parse_ppm_header((const char *) imageFilename.c_str(), &img_width, &img_height, &img_chan);
	if(!rv) 
	{
		printf("error reading file.\n"); 
		exit(-1); 
	}
	
	h_img_in_array = (unsigned char *)malloc(sizeof(unsigned int) * img_width * img_height);
	readppm(h_img_in_array, &tempInt, 
	tempChar, &tempInt,
	&img_height, &img_width, &img_chan,
	(char *)imageFilename.c_str());
	
	printf("\nWidth:%d  Height:%d\n",img_width, img_height);
	printf("[done]\n");
	
	// Pre-setup for Real-time threads
	mainpid = getpid();
	rt_max_prio = sched_get_priority_max(SCHED_FIFO);
	sched_getparam(mainpid, &main_param);
	main_param.sched_priority = rt_max_prio;
	errVal = sched_setscheduler(mainpid, SCHED_FIFO, &main_param);
	if(errVal < 0) 
	perror("main_param error");
	
	// Setup real-time thread
	pthread_attr_init(&rt_sched_attr);
	pthread_attr_setinheritsched(&rt_sched_attr, PTHREAD_EXPLICIT_SCHED);
	pthread_attr_setschedpolicy(&rt_sched_attr, SCHED_FIFO);
	rt_param.sched_priority=rt_max_prio-1;
	pthread_attr_setschedparam(&rt_sched_attr, &rt_param);
	
	if (freq) {
		run_time.tv_nsec = (1000000000/freq);
		} else {
		run_time.tv_nsec = 0;
	}
	
	// If non-continuous inform user that infinite loop will be entered to allow for power measurement
	if (run_once)
		printf("Program will now transform image once\n");
	else 
		printf("Program will enter an infinite loop, use Ctrl+C to exit program when done.\n");      

	printf("Press enter to proceed...");
	std::cin.ignore(); // Pause to allow user to read
	
	// Start Transform
	if (use_cuda)
	{
		// Start real-time thread
		pthread_create( &rt_thread,  // pointer to thread descriptor
		NULL,     				// use default attributes
		CUDA_transform_thread,		// thread function entry point
		&rt_param);				// parameters to pass in
		} else {
		// Start real-time thread
		pthread_create( &rt_thread,  // pointer to thread descriptor
		NULL,     				// use default attributes
		CPU_transform_thread,		// thread function entry point
		&rt_param);				// parameters to pass in
	}
	
	// Let transform thread run //
	
	// Wait for thread to exit
	pthread_join(rt_thread, NULL);
	
	// Write back result
	dump_ppm_data("sobel_out.ppm", img_width, img_height, img_chan, h_img_out_array);
	
	// Free up memory
#ifdef DEBUG
	printf("Cleaning up memory..\n");
#endif
	free(h_img_out_array);
	free(h_img_in_array);
	
	return 0;
}
