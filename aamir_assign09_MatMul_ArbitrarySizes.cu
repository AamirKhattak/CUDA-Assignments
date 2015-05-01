/*
### PROGRAM DESCIPTION ###
## Assigment No. 06
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include< iostream>
#include <ctime>

#include "MyMatrixOperations.h"

using namespace std;

/* 
#CUDA PROGRAM STRUCTURE

1. Memory Allocation on CPU and GPU
2. Initialization of Memory in CPU
3. Memcpy to GPU
4. Kernel Invocation
5. Memcpy to CPU
*/
//----------------[START] CUDA KERNEL CODE ---------------------------
const int TILE_WIDTH = 32;

__global__ void MulKernel(int *A, int *B, int *C,int WIDTH, int HEIGHT, int bWidth, int bHeight, int cWidth, int cHeight)
{
	__shared__ float A_shared[TILE_WIDTH][TILE_WIDTH];
	__shared__ float B_shared[TILE_WIDTH][TILE_WIDTH];

	const int Mat_SIZE = WIDTH * HEIGHT;

	int col = ( blockDim.x * blockIdx.x) + threadIdx.x;
	int row = ( blockDim.y * blockIdx.y) + threadIdx.y;

	int bx = blockIdx.x;  int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	//int index = col + ( WIDTH * row);

	if( row < cHeight && col < cWidth){
		
		int value = 0;
		// will run calculate product for a pixel
		for(int i = 0; i<cWidth/TILE_WIDTH; i++){
			A_shared[ty][tx] = A[row*cWidth + i*TILE_WIDTH + tx];
			B_shared[ty][tx] = B[(i*TILE_WIDTH + ty)*cWidth + col];
			value += A[ (row * cHeight) + i ] * B [ (i * cWidth) + col ];
			__syncthreads();

			for (int j = 0; j < TILE_WIDTH; j++) {
				value += A_shared[ty][j] * B_shared[j][tx];
			}
			__syncthreads();

			C[row*WIDTH+col] = value;
		}
	}
}

//---------------- [END] CUDA KERNEL CODE ----------------------------


int main()
{
	const int A_WIDTH = 256;
	const int A_HEIGHT = 512;

	const int B_WIDTH = 512;
	const int B_HEIGHT = 256;

	const int C_WIDTH = B_WIDTH;
	const int C_HEIGHT = B_HEIGHT;

	int *h_a, *h_b, *h_c;
	int *d_a, *d_b, *d_c;

	//## 1. Memory Allocation on HOST & DEVICE
	//1.a Memory allocation on HOST
	int SIZE_IN_BYTES_A = A_WIDTH * A_HEIGHT * sizeof(int);
	int SIZE_IN_BYTES_B = B_WIDTH * B_HEIGHT * sizeof(int);
	int SIZE_IN_BYTES_C = C_WIDTH * C_HEIGHT * sizeof(int);

	h_a = (int *) malloc( SIZE_IN_BYTES_A); // since square matrix so  A = [ WIDTH * WIDTH]
	h_b = (int *) malloc( SIZE_IN_BYTES_B);
	h_c = (int *) malloc( SIZE_IN_BYTES_C);

	//1.b Memory Allocation on DEVICE
	cudaMalloc( (void **) &d_a, SIZE_IN_BYTES_A);
	cudaMalloc( (void **) &d_b, SIZE_IN_BYTES_B);
	cudaMalloc( (void **) &d_c, SIZE_IN_BYTES_C);

	//## 2. Memory Initialization HOST
	//Initializing Host Arrays
	initializeArray( h_a, A_WIDTH, A_HEIGHT, 50);
	initializeArray( h_b, B_WIDTH, B_HEIGHT, 30);

	//## 3. Memcpy HOST to DEVICE
	cudaMemcpy( d_a, h_a, SIZE_IN_BYTES_A, cudaMemcpyHostToDevice);
	cudaMemcpy( d_b, h_b, SIZE_IN_BYTES_B, cudaMemcpyHostToDevice);

	//## 4. Kernel Invocation
	int mat_size= C_WIDTH * C_HEIGHT;
	int threadX = 32;
	int threadY = 32;
	int blockX = ceil( C_WIDTH/threadX) +1;
	int blockY = ceil( C_HEIGHT/threadY) +1;

	dim3 dimBlock( threadX, threadY, 1);
	dim3 dimGrid( blockX, blockY, 1);

	MulKernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, A_WIDTH, A_HEIGHT, B_WIDTH, B_HEIGHT, C_WIDTH, C_WIDTH);

	////## 5. Memcpy DEVICE to HOST
	cudaMemcpy( h_c, d_c, SIZE_IN_BYTES_C, cudaMemcpyDeviceToHost);


	// for comparing results
	//int *cpu_results; // to store CPU results
	//cpu_results = (int *) malloc( SIZE_IN_BYTES_C);
	//mulKernelCPU( h_a, h_b, cpu_results, A_WIDTH, A_HEIGHT);

	// Displaying Result
	cout<<"Comparing and Displaying Result"<<endl;
	//compareResult( h_c, cpu_results, WIDTH, HEIGHT);

	cout<<endl<<"Showing some data : "<<endl;
	displayArray( "a",h_a, 5, 5);
	displayArray("b", h_b, 5, 5);
	displayArray("c",h_c, 5, 5);
	//displayArray("cpu_result",cpu_results,2,2);

	cudaFree(&d_a);
	cudaFree(&d_b);
	cudaFree(&d_c);

	free(h_a);
	free(h_b);
	free(h_c);


	system("pause");
	return 0;
}
