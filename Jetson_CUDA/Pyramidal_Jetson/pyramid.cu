//*****************************************************************************************//
//  pyramid.cu - CUDA Pyramidal Transform Benchmark
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

// Project-Specific Defines
#define TILE_WIDTH     6
#define TILE_HEIGHT    6
#define FILTER_RADIUS  2
#define BLOCK_WIDTH    (TILE_WIDTH + 2*FILTER_RADIUS)
#define BLOCK_HEIGHT   (TILE_HEIGHT + 2*FILTER_RADIUS)
#define DEFAULT_IMAGE	"beach.pgm"

// Debug mode
//#define DEBUG

// Functions
void Cleanup(void);

// Kernels (in pyramid_kernel.cu)
__global__ void PyrDown(unsigned char *g_DataIn, unsigned char *g_DataOut, int width, int height);
__global__ void PyrUp(unsigned char* g_DataIn, unsigned char* g_DataOut, int width, int height);
extern void CPU_pyrdown(unsigned char* imageIn, unsigned char* imageOut, int width, int height);
extern void CPU_pyrup(unsigned char* imageIn, unsigned char* imageOut, int width, int height);
extern void CPU_pyrdiff(unsigned char* image1, unsigned char* image2, unsigned char* diff_image, int width, int height);

/* Device Memory */
unsigned char *d_In, *d_Down, *d_Up, *d_Diff;

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
int img_size;
u_char* input_image;
u_char* pyrdown_image;
u_char* pyrup_image;
u_char* pyrdiff_image;
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
    cudaThreadExit();
    exit(0);
}

//***************************************************************//
// Initialize CUDA 
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
// CUDA hardware Transform thread
//***************************************************************//
void *CUDA_transform_thread(void * threadp)
{
	// CUDA transform local variables
	struct timespec start_time, end_time, elap_time, diff_time;
	int errVal;
	double start_time_d, end_time_d, elap_time_d, diff_time_d;

	// initialize needed variables
	int gridWidth  = (img_width + TILE_WIDTH - 1) / TILE_WIDTH;
	int gridHeight = (img_height + TILE_HEIGHT - 1) / TILE_HEIGHT;
	dim3 dimGrid(gridWidth, gridHeight);

	// Block dimensions
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
#ifdef DEBUG
	printf("DEBUG: Begin transform thread.\n");
#endif

	// Allocate CPU memory
    pyrdown_image = (unsigned char *)malloc(img_width/2 * img_height/2);
	pyrup_image   = (unsigned char *)malloc(img_width * img_height);

	// Allocate CUDA memory
    cudaMalloc( (void **)&d_In, img_width*img_height*sizeof(unsigned char));
    cudaMalloc( (void **)&d_Down, (img_width/2)*(img_height/2)*sizeof(unsigned char));
    cudaMalloc( (void **)&d_Up, img_width*img_height*sizeof(unsigned char));
	
	printf("Filtering started...\n");
	
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

		// Copy the input image into CUDA memory 
		cudaMemcpy(d_In, input_image, img_width*img_height*sizeof(unsigned char), cudaMemcpyHostToDevice);

		// Complete the pyrdown
#ifdef DEBUG
		printf("DEBUG: Running pyrdown on input\n");
#endif
		PyrDown<<< dimGrid, dimBlock >>>(d_In, d_Down, img_width, img_height);

		// Copy the transformed image back from the CUDA memory 
		cudaMemcpy(pyrdown_image, d_Down, (img_width/2)*(img_height/2)*sizeof(unsigned char), cudaMemcpyDeviceToHost);

		// Re copy in the input_image 
	    cudaMemcpy(d_In, input_image, (img_width/2)*(img_height/2)*sizeof(unsigned char), cudaMemcpyHostToDevice);

		// Complete the pyrup
#ifdef DEBUG
	    printf("DEBUG: Running pyrup on pyrdown result\n");
#endif
		PyrUp<<< dimGrid, dimBlock >>>(d_Down, d_Up, img_width, img_height);

	    // Copy image back to host
	    cudaMemcpy(pyrup_image, d_Up, img_width*img_height*sizeof(unsigned char), cudaMemcpyDeviceToHost);

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
	}while(!run_once);

	cudaFree(d_In);
	cudaFree(d_Up);
	cudaFree(d_Down);
	cudaFree(d_Diff);

	return NULL;
}

//***************************************************************//
// CPU hardware Transform thread
//***************************************************************//
void *CPU_transform_thread(void * threadp)
{
	// CUDA transform local variables
	struct timespec start_time, end_time, elap_time, diff_time;
	int errVal;
	double start_time_d, end_time_d, elap_time_d, diff_time_d;

	// Allocate Memory
    pyrdown_image = (unsigned char *)malloc(img_width/2 * img_height/2);
	pyrup_image   = (unsigned char *)malloc(img_width * img_height);
	
	printf("Filtering started...\n");
	
	
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

		// Complete the pyrdown
#ifdef DEBUG
		printf("DEBUG: Running pyrdown on input\n");
#endif
		CPU_pyrdown(input_image, pyrdown_image, img_width, img_height);

		// Complete the pyrup
#ifdef DEBUG
	    printf("Running pyrup on pyrdown result\n");
#endif
		CPU_pyrup(pyrdown_image, pyrup_image, img_width/2, img_height/2);

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
	}while(!run_once);

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

	// If non-continuous inform user that infinite loop will be entered to allow for power measurement
	if (run_once)
		printf("Program will now transform image once\n");
	else
		printf("Program will enter an infinite loop, use Ctrl+C to exit program when done.\n");      

	printf("Press enter to proceed...");
	std::cin.ignore(); // Pause to allow user to read
  
	// Start Transform
	if(use_cuda)
    {		
		// Start real-time CUDA thread
		pthread_create(&rt_thread,  // pointer to thread descriptor
			NULL,     				// use default attributes
			CUDA_transform_thread,	// thread function entry point
			&rt_param); 			// parameters to pass in
	}
	else // Use CPU version for transform
    {
		// Start real-time CUDA thread
		pthread_create(&rt_thread,  // pointer to thread descriptor
			NULL,     				// use default attributes
			CPU_transform_thread,	// thread function entry point
			&rt_param); 			// parameters to pass in
	}
					 
	// Let transform thread run //
	
	// Wait for thread to exit
	pthread_join(rt_thread, NULL);
	
	// Calculate difference
	pyrdiff_image   = (unsigned char *)malloc(img_width * img_height);
	CPU_pyrdiff(input_image, pyrup_image, pyrdiff_image, img_width, img_height);
	
	// Write results 
	dump_ppm_data("pyrdown.ppm", img_width/2, img_height/2, img_chan, pyrdown_image);
	dump_ppm_data("pyrup.ppm", img_width, img_height, img_chan, pyrup_image);
	dump_ppm_data("pyrdiff.ppm", img_width, img_height, img_chan, pyrdiff_image);
	
	// Free memory
	free(input_image);
    free(pyrdown_image);
    free(pyrup_image);
	free(pyrdiff_image);

	// Final Cleanup
	Cleanup();

    return 0;
}
