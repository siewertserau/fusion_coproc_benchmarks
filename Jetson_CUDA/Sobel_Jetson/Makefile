CXX=g++
SDK_ROOT=/home/ubuntu/NVIDIA_GPU_Computing_SDK/
CUDA_ROOT=/usr/local/cuda/

all:
	sudo ${CXX} -c -lpthread options.cpp ppm.cpp -I./
	nvcc -o sobel_edge options.o ppm.o Sobel_kernel.cu -O0 -I./ -I${SDK_ROOT}/C/common/inc/ -L${CUDA_ROOT} -lcudart -L${SDK_ROOT}/C/lib/
clean:
	rm -f *.o *~ sobel_edge sobel_out.ppm  
