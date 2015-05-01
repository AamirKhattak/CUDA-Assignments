
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// Standard C/C++ Directives
#include <iostream>
#include <conio.h>

using namespace std;

__global__ void ADD_KERNEL(int *a, int *b, int *result, int WIDTH, int HEIGHT)
{
	int idx = threadIdx.x + ( blockIdx.x * blockDim.x);
	int idy = threadIdx.y + ( blockIdx.y * blockDim.y);

	int index = idx + (idy * WIDTH);
	if( index < WIDTH * HEIGHT)
		result[index] = a[index] + b[index];

}
__global__ void SUB_KERNEL(int *a, int *b, int *result, int WIDTH, int HEIGHT)
{

	int idx = threadIdx.x + ( blockIdx.x * blockDim.x);
	int idy = threadIdx.y + ( blockIdx.y * blockDim.y);

	int index = idx + (idy * WIDTH);

	if( index < WIDTH * HEIGHT)
		result[index] = a[index] - b[index];
	//printf("result [%d] : %d \n", index, result[index]);

}
__global__ void MPY_KERNEL(int *a, int *b, int *result, int WIDTH, int HEIGHT)
{

	int idx = threadIdx.x + ( blockIdx.x * blockDim.x);
	int idy = threadIdx.y + ( blockIdx.y * blockDim.y);

	int index = idx + (idy * WIDTH);
	if( index < WIDTH * HEIGHT)
		result[index] = a[index] * b[index];

}
__global__ void DVD_KERNEL(int *a, int *b, int *result, int WIDTH, int HEIGHT)
{

	int idx = threadIdx.x + ( blockIdx.x * blockDim.x);
	int idy = threadIdx.y + ( blockIdx.y * blockDim.y);

	int index = idx + (idy * WIDTH);
	if( index < WIDTH * HEIGHT){
		if( b[index] != 0)
			result[index] = a[index] / b[index];
		else
			result[index] = 0;
	}
}

__global__ void ParentKernel(int *a, int *b, int *add, int *sub, int *mpy, int *dvd, int WIDTH, int HEIGHT)
{
	int i = threadIdx.x;

	int tX = 32;
	int tY = 32;
	int bX = (WIDTH/tX)+1;
	int bY = (HEIGHT/tY)+1;

	dim3 dimBlock( tX, tY, 1);
	dim3 dimGrid( bX, bY, 1);

	if (i==0){
		ADD_KERNEL<<< dimGrid, dimBlock>>>(a, b, add, WIDTH, HEIGHT);
		cudaDeviceSynchronize();
		__syncthreads();
	}else
		if (i==1)
		{
			SUB_KERNEL<<< dimGrid, dimBlock>>>(a, b, sub, WIDTH, HEIGHT);
			cudaDeviceSynchronize();
			__syncthreads();
		}else
			if (i==2)
			{
				MPY_KERNEL<<< dimGrid, dimBlock>>>(a, b, mpy, WIDTH, HEIGHT);
				cudaDeviceSynchronize();
				__syncthreads();
			}
			else
				if( i == 3)
				{
					DVD_KERNEL<<< dimGrid, dimBlock>>>(a, b, dvd, WIDTH, HEIGHT);
					cudaDeviceSynchronize();

				}

				__syncthreads();

}


void iniVector(int *vec, int SIZE, int seed);
void displayVector(char *vecName, int *vec);

int main(void)
{

	int *h_a, *h_b, *h_add, *h_sub, *h_dvd, *h_mpy;

	int WIDTH = 128;
	int HEIGHT = 128;

	int SIZE_IN_BYTES = WIDTH * HEIGHT * sizeof(int);


	// Memory Allocation on HOST
	h_a = (int *) malloc( SIZE_IN_BYTES);
	h_b = (int *) malloc( SIZE_IN_BYTES);
	h_add = (int *) malloc( SIZE_IN_BYTES);
	h_sub = (int *) malloc( SIZE_IN_BYTES);
	h_dvd = (int *) malloc( SIZE_IN_BYTES);
	h_mpy = (int *) malloc( SIZE_IN_BYTES);


	iniVector( h_a, WIDTH*HEIGHT, 2);
	iniVector( h_b, WIDTH*HEIGHT, 1);

	// Memory Allocation on DEVICE
	int *d_a, *d_b, *d_add, *d_sub, *d_dvd, *d_mpy;

	cudaMalloc( (void**) &d_a, SIZE_IN_BYTES);
	cudaMalloc( (void**) &d_b, SIZE_IN_BYTES);
	cudaMalloc( (void**) &d_add, SIZE_IN_BYTES);
	cudaMalloc( (void**) &d_sub, SIZE_IN_BYTES);
	cudaMalloc( (void**) &d_dvd, SIZE_IN_BYTES);
	cudaMalloc( (void**) &d_mpy, SIZE_IN_BYTES);

	// Memcpy HOST to DEVICE
	cudaMemcpy( d_a, h_a, SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy( d_b, h_b, SIZE_IN_BYTES, cudaMemcpyHostToDevice);

	//KERNEL LAUNCH
	ParentKernel<<<1,4>>>(d_a, d_b, d_add, d_sub, d_mpy, d_dvd, WIDTH, HEIGHT);
	cudaDeviceSynchronize();

	// Memcpy DEVICe to HOST
	cudaMemcpy( h_add, d_add, SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy( h_sub, d_sub, SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy( h_mpy, d_mpy, SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
	cudaMemcpy( h_dvd, d_dvd, SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

	displayVector("a", h_a);
	displayVector("b", h_b);
	displayVector("add", h_add);
	displayVector("sub", h_sub);
	displayVector("mpy", h_mpy);
	displayVector("dvd", h_dvd);

	cudaFree( d_a);
	cudaFree( d_b);
	cudaFree( d_add);
	cudaFree( d_sub);
	cudaFree( d_mpy);
	cudaFree( d_dvd);
	
	free( h_a);
	free( h_b);
	free( h_add);
	free( h_sub);
	free( h_mpy);
	free( h_dvd);

	getch();
	return 0;
}

void displayVector(char *vecName, int *vec){
	cout<<vecName;	
	for( int i=0; i<10; i++){
		cout<<" : "<<vec[i]<<" ";
	}
	cout<<endl;
}



void iniVector(int *vec, int SIZE, int seed){

	for( int i=0; i<SIZE; i++)
		vec[i] = i*seed;
}
