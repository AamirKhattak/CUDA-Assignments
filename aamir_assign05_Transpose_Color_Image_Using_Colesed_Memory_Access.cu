
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// Standard C/C++ Directives
#include <iostream>
#include <conio.h>

// OpenCV directives

#include <highgui\highgui.hpp> 


using namespace std;
using namespace cv;

cudaError_t transposeRGBImageOnGPU(Mat *input, Mat *transpose);

__global__ void transposeKernel(uchar *iChannel, uchar *tChannel, const int WIDTH, const int HEIGHT)
{  
	/*thred index along x-dimension*/
	int tIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
	/*thread index along y dimension*/
	int tIdy = (blockIdx.y * blockDim.y) + threadIdx.y;
	

	int i_input, i_transpose;
	i_transpose = tIdx + ( HEIGHT * tIdy);
	i_input = tIdy + ( WIDTH * tIdx);
	
	int SIZE = WIDTH * HEIGHT;
	/*check that the index does not exceed the bound of the image in the memory*/
	if( i_transpose < SIZE && i_input < SIZE)
	{	
		tChannel[i_transpose] = iChannel[i_input];//4;  	
	}
			 

}

void windowSetting( char* title, int width, int height);

int main()
{
	//Windows Creation and Display Size settings
	/*namedWindow("InputImage", WINDOW_NORMAL);
	resizeWindow("InputImage", 300, 200);
	namedWindow("TransposedImage", WINDOW_NORMAL);
	resizeWindow("TransposedImage", 300, 200)*/;

	windowSetting( "InputImage", 300, 200);


	// ## 1. Memory Allocation and Initilization on HOST
	Mat  h_inputImage = imread("D:\\input1.png",CV_LOAD_IMAGE_ANYCOLOR);

	if( h_inputImage.data == NULL)	{
		cout<<"[ERROR] Input Image is null"<<endl;
	}

	Mat h_transposeImage;
	h_transposeImage.create( h_inputImage.cols, h_inputImage.rows, CV_8UC3);

	// ## 2. STUB function invocation
	cudaError_t cudaStatus = transposeRGBImageOnGPU( &h_inputImage, &h_transposeImage);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}


	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}



	imshow("TransposedImage", h_transposeImage);
	waitKey(0);
	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t transposeRGBImageOnGPU(Mat *inputImage, Mat *transposeImage)
{

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		
	}

	const int iWIDTH = inputImage->rows;
	const int iHEIGHT = inputImage->cols;
	const int SIZE_IMAGE = iWIDTH * iHEIGHT;
	//[OPTIONAL] convert HOST images to float
	//inputImage->create( inputImage->rows, inputImage->cols, CV_32FC3);


	vector<uchar> h_Input_Channels;
	split( *inputImage, h_Input_Channels);

	uchar *h_Transposed_BLUE, *h_Transposed_RED, *h_Transposed_GREEN;

	
	// Device Variables
	uchar *d_BLUE, *d_GREEN, *d_RED;
	uchar *d_Transposed_BLUE, *d_Transposed_GREEN, *d_Transposed_RED;

	// Calculating dim BLOCK and GRID
	int threadX = 32;
	int threadY = 32;
	int blockX = iWIDTH/threadX;
	int blockY = iHEIGHT/threadY;

	dim3 dimBlock( threadX, threadY, 1);
	dim3 dimGrid( blockY, blockX, 1);




	//## A. KERNEL 1 i.e. BLUE
	//# A.1 Mem-Alloc for KERNEL BLUE
	cudaStatus = cudaMalloc( &d_BLUE, sizeof(uchar) * SIZE_IMAGE);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		
	}
	cudaStatus = cudaMalloc( &d_Transposed_BLUE, sizeof(uchar) * SIZE_IMAGE);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		
	}

	//# A.2 Mem-CPY for KERNEL BLUE
	cudaStatus = cudaMemcpy( d_BLUE, &h_Input_Channels[0], sizeof(uchar) * SIZE_IMAGE, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		
	}
	//# A.3 KERNEL LAUNCH for BLUE
	transposeKernel<<< dimGrid, dimBlock>>>( d_BLUE, d_Transposed_BLUE, iWIDTH, iHEIGHT);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		
	}



	//## B. KERNEL 1 i.e. GREEN
	//# A.1 Mem-Alloc for KERNEL GREEN
	cudaMalloc( &d_GREEN, sizeof(uchar) * SIZE_IMAGE);
	cudaMalloc( &d_Transposed_GREEN, sizeof(uchar) * SIZE_IMAGE);
	//# A.2 Mem-CPY for KERNEL GREEN
	cudaMemcpy( d_GREEN, &h_Input_Channels[1], sizeof(uchar) * SIZE_IMAGE, cudaMemcpyHostToDevice);
	//# A.3 KERNEL LAUNCH for GREEN
	transposeKernel<<< dimGrid, dimBlock>>>( d_GREEN, d_Transposed_GREEN, iWIDTH, iHEIGHT);

	//## C. KERNEL 1 i.e. RED
	//# A.1 Mem-Alloc for KERNEL RED
	cudaMalloc( &d_RED, sizeof(uchar) * SIZE_IMAGE);
	cudaMalloc( &d_Transposed_RED, sizeof(uchar) * SIZE_IMAGE);
	//# A.2 Mem-CPY for KERNEL RED
	cudaMemcpy( d_RED, &h_Input_Channels[2], sizeof(uchar) * SIZE_IMAGE, cudaMemcpyHostToDevice);
	//# A.3 KERNEL LAUNCH for RED
	transposeKernel<<< dimGrid, dimBlock>>>( d_RED, d_Transposed_RED, iWIDTH, iHEIGHT);



	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);

	}

	//memcpy  to device
	cudaStatus = cudaMemcpy( h_Transposed_BLUE, d_Transposed_BLUE, sizeof(uchar) * iWIDTH * iHEIGHT, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");

	}


	cudaStatus = cudaMemcpy( h_Transposed_GREEN, d_Transposed_GREEN, sizeof(uchar) * iWIDTH * iHEIGHT, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");

	}


	cudaStatus = cudaMemcpy( h_Transposed_RED, d_Transposed_RED, sizeof(uchar) * iWIDTH * iHEIGHT, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");

	}


	vector<uchar> hh;
	hh.push_back(*h_Transposed_RED);
	hh.push_back( *h_Transposed_GREEN);
	hh.push_back(*h_Transposed_BLUE);


	/*Mat i[3];
	Mat image(Size(iWIDTH, iHEIGHT), CV_8UC1, h_Transposed_BLUE, Mat::AUTO_STEP);
	Mat image2(Size(iWIDTH, iHEIGHT), CV_8UC1, h_Transposed_BLUE, Mat::AUTO_STEP);
	Mat image3(Size(iWIDTH, iHEIGHT), CV_8UC1, h_Transposed_BLUE, Mat::AUTO_STEP);
	i[0] = image;
	i[1] = image2;
	i[2] =image3;*/

	merge( hh, *transposeImage);
	//merge( i,3, *transposeImage);


//Error:
	cudaFree(d_BLUE);
	cudaFree(d_GREEN);
	cudaFree(d_RED);
	cudaFree(d_Transposed_BLUE);
	cudaFree(d_Transposed_GREEN);
	cudaFree(d_Transposed_RED);

	return cudaStatus;
}
