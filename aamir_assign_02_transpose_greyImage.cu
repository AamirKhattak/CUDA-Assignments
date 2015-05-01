/*************************************************
Programmer : Muhammad Aamir Javid


Problem Description:
this code loads a greyscale image and transpose it
the kernel launch configuration is 2D in both grid and block level
*************************************************/

//System Headers
#include <iostream>
#include <stdio.h>

//OpenCV Headers
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//Cuda Headers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace cv;
using namespace std;

 __global__ void kernel(float *inputImage,float *transposeImage,int WIDTH,int HEIGHT)
{  
	/*thred index along x-dimension*/
	int tIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
	/*thread index along y dimension*/
	int tIdy = (blockIdx.y * blockDim.y) + threadIdx.y;
	

	int i_inputImage, i_transposeImage;
	i_transposeImage = tIdx + ( HEIGHT * tIdy);
	i_inputImage = tIdy + ( WIDTH * tIdx);
	
	int SIZE = WIDTH * HEIGHT;
	/*check that the index does not exceed the bound of the image in the memory*/
	if( i_transposeImage < SIZE && i_inputImage < SIZE)
	{	
		transposeImage[i_transposeImage] = inputImage[i_inputImage];//4;  
		//img_t[tIdy + cols*tIdx] = img[tIdx + cols*tIdy];
	
	}
			 

}


int main(void)
{
	int WIDTH = 1024;
	int HEIGHT = 768;


	//## 1. Memory Allocation on HOST and DEVICE
	Mat h_inputImage, h_transposeImage;

	//## 2. MEmory Initialization on HOST
	h_inputImage = imread("Penguins.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	h_transposeImage.create( h_inputImage.cols, h_inputImage.rows, CV_8UC1);

	int SIZE_IMAGE_BYTES = ( h_inputImage.rows * h_inputImage.cols * h_inputImage.channels() * sizeof(float) );
	
	//## 1.b Memory Allocation on DEVICE
	float* d_inputImage = 0;
	float* d_transposeImage = 0;
	
	cudaMalloc((void**)&d_inputImage, SIZE_IMAGE_BYTES);
	cudaMalloc((void**)&d_transposeImage, SIZE_IMAGE_BYTES);
		
	//## 3. Memcpy to Device
	h_inputImage.convertTo(h_inputImage, CV_32FC1);
	h_transposeImage.convertTo(h_transposeImage, CV_32FC1);

	cudaMemcpy( d_inputImage, h_inputImage.data, SIZE_IMAGE_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy( d_transposeImage, h_transposeImage.data, SIZE_IMAGE_BYTES, cudaMemcpyHostToDevice);

	
	//## 4. Kernel Launch
    //Grid Dimensions
	int gridX = WIDTH / 32;  //32
	int gridY = HEIGHT / 32;   //24
	dim3 gridSize(gridY,gridX,1);
	//Block Dimensions: 32 x 32 x 1
	dim3 blockSize(32,32,1);
	
	kernel<<<gridSize,blockSize>>>(d_inputImage, d_transposeImage, WIDTH, HEIGHT);
	cudaDeviceSynchronize();
  	  
	

	//## 5. Memcpy to HOST
	cudaMemcpy( h_transposeImage.data, d_transposeImage, SIZE_IMAGE_BYTES, cudaMemcpyDeviceToHost);
	
	// Converting the float images back to uchar
	h_inputImage.convertTo(h_inputImage,CV_8UC1);
	h_transposeImage.convertTo(h_transposeImage,CV_8UC1);
	
	namedWindow("Original Image", WINDOW_NORMAL);
	namedWindow("Transposed Image", WINDOW_NORMAL);
	
	resizeWindow("Original Image", 300, 300);
	resizeWindow("Transposed Image", 300, 300);

	//Display Results
	imshow("Original Image", h_inputImage);
	imshow("Transposed Image", h_transposeImage);
	//End of program
	waitKey();


	//## 6. Free Memory
	//free( h_inputImage.data);
	//free( h_transposeImage.data);

	cudaFree(d_inputImage);
	cudaFree(d_transposeImage);

	return 0;
}
