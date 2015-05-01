/*
### PROGRAM DESCIPTION ###
## Assigment No. 1

#Description:
	- create two matrices A and B of 100,000 X 100, 000
	- and store thier sum in matrix C

#Issue:
	- allocating 100,000 X 100, 000 matrix of int
	  requires about 1.25 GB of memory in ram [detail is in attachment]
	- So I am using matrix of width 10,000 X 10,000 
	  it requires about 350 MBS of memory, which is feasible

### DEVELOPER DETAILS ###
#Name:
	- M. Aamir Javid
#Email:
#Date:
	Sept 16, 2014
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include< iostream>
#include <ctime>
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
__global__ void addKernel(int *a, int *b, int *c,int WIDTH)
{
	int idx = ( blockDim.x * blockIdx.x) + threadIdx.x;
	if( idx < WIDTH*WIDTH)
	c[idx] = a[idx] + b[idx];
}


//---------------- [END] CUDA KERNEL CODE ----------------------------



//will show the memory used by one array i.e a/b/c
// viewMemoryUse( WIDTH, HEIGHT) : specific to this program
void viewMemUse(int, int);
// IniArray( ARRAY, WIDTH, HEIGHT, RandomValueSeed)
void initializeArray(int*, int, int, int);
// DisplayArray( arrayNAme i.e H_A, array, width, height)
void displayArray(char*, int *,int,int);
//
void addKernelCPU( int*, int*, int*, int);
//
void compareResult( int *arrayA, int *arrayB, int width);
int main()
{
	const int WIDTH = 10000;
	
	int *h_a, *h_b, *h_c;
	int *d_a, *d_b, *d_c;

	//## 1. Memory Allocation on HOST & DEVICE
	//1.a Memory allocation on HOST
	int SIZE_IN_BYTES = WIDTH * WIDTH * sizeof(int);
	h_a = (int *) malloc( SIZE_IN_BYTES); // since square matrix so  A = [ WIDTH * WIDTH]
	h_b = (int *) malloc( SIZE_IN_BYTES);
	h_c = (int *) malloc( SIZE_IN_BYTES);
	
	//1.b Memory Allocation on DEVICE
	cudaMalloc( (void **) &d_a, SIZE_IN_BYTES);
	cudaMalloc( (void **) &d_b, SIZE_IN_BYTES);
	cudaMalloc( (void **) &d_c, SIZE_IN_BYTES);

	//## 2. Memory Initialization HOST
	//Initialing Host Arrays
	initializeArray( h_a, WIDTH, WIDTH, 50);
	initializeArray( h_b, WIDTH, WIDTH, 50);

	//## 3. Memcpy HOST to DEVICE
	cudaMemcpy( d_a, h_a, SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy( d_b, h_b, SIZE_IN_BYTES, cudaMemcpyHostToDevice);

	//## 4. Kernel Invocation
	int mat_size= WIDTH * WIDTH;
	int threadsPerBlock = 1024;
	int blockPerGrid = ceil(mat_size / threadsPerBlock)+1;//97657
	

	addKernel<<<blockPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, WIDTH);

	////## 5. Memcpy DEVICE to HOST
	cudaMemcpy( h_c, d_c, SIZE_IN_BYTES, cudaMemcpyDeviceToHost);


	// for comparing results
	int *cpu_results; // to store CPU results
	cpu_results = (int *) malloc( SIZE_IN_BYTES);
	addKernelCPU( h_a, h_b, cpu_results, WIDTH);
	// Displaying Result
	cout<<"Comparing and Displaying Result"<<endl;
	compareResult( h_c, cpu_results, WIDTH);
	
	
	displayArray( "a",h_a,2,2);
	displayArray("b", h_b, 2,2);
	displayArray("c",h_c,2,2);
	displayArray("cpu_result",cpu_results,2,2);

	cudaFree(&d_a);
	cudaFree(&d_b);
	cudaFree(&d_c);

	free(h_a);
	free(h_b);
	free(h_c);


	system("pause");
    return 0;
}

void addKernelCPU( int *arrayA, int *arrayB, int *arrayC, int width){
	int arraySize = width * width;
	for(int i=0; i<arraySize; i++){
		arrayC[i] = arrayA[i] + arrayB[i];	
	}
}

void compareResult( int *arrayA, int *arrayB, int width){
	int arraySize = width * width;
	for(int i=0; i<arraySize; i++){
		if( arrayA[i] != arrayB[i]){
			cout<<"arrayA["<<i<<"] != arrayB["<<i<<"]"<<endl;
			break;
		}
	}
	cout<<"Result on CPU and GPU is same"<<endl;
}

void initializeArray(int *array, int width, int height, int randomValueSEED){
	int MAT_SIZE = width * height;
	// Initializing Array with random values
	srand ( time(NULL) );	
	for( int i=0; i<MAT_SIZE; i++){
		int value = rand() % randomValueSEED + 1;
		array[i] = value;
	}
}

void displayArray(char* arrayName,int *array,int width, int height){
	cout<<"Displaying Values of Array: "<<arrayName<<endl;
	for(int i=0; i<width*height; i++){		
		cout<<"Array["<<i<<"] : "<<array[i]<<endl;
	}
}

void viewMemUse(int pWidth, int pHeight){
	int size = pWidth * pHeight * sizeof(int);
	cout<<"Size: of WIDTH * HEIGHT * sizeof(int)"<<endl;
	cout<<"Size = "<<pWidth<<" * "<<pHeight<<" * sizeof(int)"<<endl;
	cout<<"Size: BYTES "<<size<<endl;
	cout<<"Size: KBYTES "<<size/1024<<endl;
	cout<<"Size: MBYTES "<<(size/1024)/1024<<endl;
	float gSize = ((size/1024.0)/1024.0)/1024.0;
	cout<<"Size: GBYTES "<<gSize<<endl;

}
