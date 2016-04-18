//*****************************************************************************************//
//  hough.cu - CUDA Hough Transform Benchmark
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

// Standard Includes 
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/io.h>
#include <iostream>

#include <time.h>
#include <pthread.h>
#include <sched.h>

// Project Includes
#include <cuda_runtime.h>
#include "ppm.h"
#include "options.h"

// Project-Specific Defines
#define PRIO_ADJUST 	5
#define BLOCK_SIZE_1 	8	
#define BLOCK_SIZE_2 	6	
#define DEFAULT_IMAGE	"Cross.pgm"

// Debug mode
//#define DEBUG  

// Kernels (in houghKernels.cu)
__global__ void sobel(u_char * frame_in, u_char * frame_out, int width, int height);
__global__ void houghTransform(u_char * frame_in, u_char * frame_out, const int hough_h);

// Global variables for RT threads
pthread_attr_t rt_sched_attr;
int rt_max_prio;
struct sched_param rt_param;
pid_t mainpid;
pthread_t rt_thread;

// Globals for Transform
unsigned int img_width;
unsigned int img_height;
unsigned int img_chan;
u_char* input_image;
u_char *result;
int hough_height;
int hough_width;
struct timespec run_time = {0, 0};
bool run_once = false;
int freq = 0;
std::string imageFilename = DEFAULT_IMAGE;

/***********************************************************
 * Functions to cleanup after code complete		  **
 ***********************************************************/
void Cleanup(void)
{
#ifdef DEBUG
	printf("DEBUG: Cleanup().\n");
#endif
     cudaThreadExit() ;
    exit(0);
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
	struct timespec start_time, end_time, elap_time, diff_time;
	cudaError_t errVal;
	double start_time_d, end_time_d, elap_time_d, diff_time_d;
	u_char* devInImage;
	u_char* devTresholded;
	u_char *A;
	
	// initialize needed variables
	int size = img_width*img_height;
	const int hough_h = (int) (sqrt(2.0) * img_width / 2.0f);
	hough_height = hough_h * 2;
	hough_width = 180;
	
	// Allocate memory for Hough output
    result = (u_char*) malloc(hough_height*hough_width * sizeof(u_char));

	dim3 dimBlock(BLOCK_SIZE_1, BLOCK_SIZE_2);
	dim3 dimGrid(img_width / dimBlock.x, img_height / dimBlock.y);

	printf("Filtering started...\n");
	// Allocate CUDA memory
	errVal = cudaMalloc((void**)&devInImage, size*sizeof(u_char));
	if( errVal != cudaSuccess)
		{ printf("cudaMalloc error. %s\n",cudaGetErrorString(errVal)); exit(-1); }
	errVal = cudaMalloc((void**)&devTresholded, size*sizeof(u_char));
	if( errVal != cudaSuccess)
		{ printf("cudaMalloc error. %s\n",cudaGetErrorString(errVal)); exit(-1); }
        errVal = cudaMalloc((void**)&A,sizeof(u_char)*hough_height*hough_width);
	if( errVal != cudaSuccess)
		{ printf("cudaMalloc error. %s\n",cudaGetErrorString(errVal)); exit(-1); }

	// loop to allow for power measurement
	do
	{
		// Get start of runtime timing
		if(clock_gettime(CLOCK_REALTIME, &start_time) )
		{
		  printf("clock_gettime() - start - error.. exiting.\n");
		  break;
		}
		start_time_d = timespec2double(start_time);

////////////////////////////////// BEGIN TRANSFORM ///////////////////////////////////
		errVal = cudaMemcpy(devInImage, input_image, size*sizeof(u_char), cudaMemcpyHostToDevice);
		if( errVal != cudaSuccess)
			{ printf("cudaMemcpy1 error. %s\n",cudaGetErrorString(errVal)); exit(-1); }

		// Complete the Sobel transform to find edges
		sobel<<<dimGrid,dimBlock>>>(devInImage, devTresholded,img_width,img_height);

		// Complete the Hough transform on the transformed sobel image
        houghTransform<<<dimGrid,dimBlock>>>(devTresholded, A, hough_h);
		
		cudaThreadSynchronize();

		cudaMemcpy(result, A, hough_width*hough_height*sizeof(u_char),  cudaMemcpyDeviceToHost);
		if( errVal != cudaSuccess)
			{ printf("cudaMemcpy5 error. %s\n",cudaGetErrorString(errVal)); exit(-1); }

#ifdef DEBUG
		printf("DEBUG: End transform.\n");
#endif
////////////////////////////////// END TRANSFORM ///////////////////////////////////
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
				int err = nanosleep(&diff_time, &end_time);			
				if(err == -1)
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

	cudaFree(devInImage);
	cudaFree(devTresholded);

	return NULL; // to supress no return warnings.
}

//***************************************************************//
// Main function
//***************************************************************//
int main(int argc, char* argv[])
{
    // Local variables
	Options options(argc, argv);
	int errVal = 0;
	struct sched_param main_param;
	bool use_cuda = true;
	int tempInt, rv;
	char tempChar[25];

	// Check input
	if(options.has("help") || options.has("h") || options.has("?")) 
	{
		std::cout << "Usage: " << argv[0] << " [-continuous [-fps=FPS]] [-img=imageFilename] [-cuda] " << std::endl;
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
	
	// Read Input image
	printf("Reading input image...");
	rv = parse_ppm_header((const char *) imageFilename.c_str(), &img_width, &img_height, &img_chan);
	if(!rv) 
	{
		printf("error reading file.\n"); 
		exit(-1); 
	}

	input_image = (unsigned char *)malloc(sizeof(unsigned int) * img_width * img_height);
	readppm(input_image, &tempInt, 
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

	if (freq) 
		run_time.tv_nsec = (1000000000/freq);
	else 
		run_time.tv_nsec = 0;

	// If continuous inform user that infinite loop will be entered to allow for power measurement
	if (run_once)
		printf("Program will now transform image once\n");
	else
		printf("Program will enter an infinite loop, use Ctrl+C to exit program when done.\n"); 
	
	printf("Press enter to proceed...");
	std::cin.ignore(); // Pause to allow user to read
	
	// Start Transform
	if(use_cuda)
    {					
		pthread_create(&rt_thread,   	// pointer to thread descriptor
			NULL,     		// use default attributes
			CUDA_transform_thread,	// thread function entry point
			&rt_param 		// parameters to pass in
			);
	}
	else // Use CPU version for transform
    {
		printf("No CPU version of Hough transform available. Please use CUDA by specifying '-cuda' in program call.\n");
	}	
	
	// Let transform thread run
	
	// Wait for thread to exit
	pthread_join(rt_thread, NULL);
	
	// Writeback results
	dump_ppm_data("hough.ppm", hough_width, hough_height, img_chan, result);
	
	// Free memory
	free(input_image);
	free(result);

	// Final Cleanup
	Cleanup();

    return 0;
}
