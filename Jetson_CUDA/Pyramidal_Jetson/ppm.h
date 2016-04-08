//*****************************************************************************************//
//  ppm.h - PPM Reading and writing functions
//
//  Authors: Dr. Sam Siewert (siewerts@erau.edu)
//			 Ryan Claus (clausr@my.erau.edu)
//			 Matthew Demi Vis, Embry-Riddle Aeronautical University (MatthewVis@gmail.com)
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
#ifndef PPM_H
#define PPM_H
/* RYAN */
#include <string>

bool parse_ppm_header(const char *filename, unsigned int *width, unsigned int *height, unsigned int *channels);
bool parse_ppm_data(const char *filename, unsigned int *width, unsigned int *height, unsigned int *channels, unsigned char *data);
void dump_ppm_data(std::string filename, unsigned int width, unsigned int height, unsigned int channels, unsigned char *data);


/* SIEWERT */
void readppm(unsigned char *buffer, int *bufferlen, 
             char *header, int *headerlen,
             unsigned *rows, unsigned *cols, unsigned *chans,
             char *file);

void writeppm(unsigned char *buffer, int bufferlen,
              char *header, int headerlen,
              char *file);
#endif
