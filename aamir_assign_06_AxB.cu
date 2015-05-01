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
__global__ void MulKernel(int *A, int *B, int *C,int WIDTH, int HEIGHT)
{
	const int Mat_SIZE = WIDTH * HEIGHT;

	int col = ( blockDim.x * blockIdx.x) + threadIdx.x;
	int row = ( blockDim.y * blockIdx.y) + threadIdx.y;
	
	//int index = col + ( WIDTH * row);
	
	if( row < HEIGHT && col < WIDTH){
		int value = 0;
		// will run calculate product for a pixel
		for(int i = 0; i<WIDTH; i++){
				value += A[ (row * HEIGHT) + i ] * B [ (i * WIDTH) + col ];
		}		
		
		C[row*WIDTH+col] = value;
	}
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
void mulKernelCPU( int*, int*, int*, int, int);
//
void compareResult( int *arrayA, int *arrayB, int width, int height);

int main()
{
	const int WIDTH = 2048;
	const int HEIGHT = 2048;
	
	int *h_a, *h_b, *h_c;
	int *d_a, *d_b, *d_c;

	//## 1. Memory Allocation on HOST & DEVICE
	//1.a Memory allocation on HOST
	int SIZE_IN_BYTES = WIDTH * HEIGHT * sizeof(int);
	h_a = (int *) malloc( SIZE_IN_BYTES); // since square matrix so  A = [ WIDTH * WIDTH]
	h_b = (int *) malloc( SIZE_IN_BYTES);
	h_c = (int *) malloc( SIZE_IN_BYTES);
	
	//1.b Memory Allocation on DEVICE
	cudaMalloc( (void **) &d_a, SIZE_IN_BYTES);
	cudaMalloc( (void **) &d_b, SIZE_IN_BYTES);
	cudaMalloc( (void **) &d_c, SIZE_IN_BYTES);

	//## 2. Memory Initialization HOST
	//Initializing Host Arrays
	initializeArray( h_a, WIDTH, HEIGHT, 50);
	initializeArray( h_b, WIDTH, HEIGHT, 30);

	//## 3. Memcpy HOST to DEVICE
	cudaMemcpy( d_a, h_a, SIZE_IN_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy( d_b, h_b, SIZE_IN_BYTES, cudaMemcpyHostToDevice);

	//## 4. Kernel Invocation
	int mat_size= WIDTH * HEIGHT;
	int threadX = 32;
	int threadY = 32;
	int blockX = ceil( WIDTH/threadX) +1;
	int blockY = ceil( HEIGHT/threadY) +1;
	
	dim3 dimBlock( threadX, threadY, 1);
	dim3 dimGrid( blockX, blockY, 1);
	
	MulKernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, WIDTH, HEIGHT);

	////## 5. Memcpy DEVICE to HOST
	cudaMemcpy( h_c, d_c, SIZE_IN_BYTES, cudaMemcpyDeviceToHost);


	// for comparing results
	int *cpu_results; // to store CPU results
	cpu_results = (int *) malloc( SIZE_IN_BYTES);
	//mulKernelCPU( h_a, h_b, cpu_results, WIDTH, HEIGHT);
	
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

void mulKernelCPU( int *arrayA, int *arrayB, int *arrayC, int width, int height){
	int arraySize = width * height;
	for(int i=0; i<arraySize; i++){
		int value;
		for( int j=0; j<arraySize; j++){
			value = arrayA[j * height + j ] * arrayB[i * width + i];	
		}
		int index = 0;
		arrayC[ index ] = value;
	}
}

/*
value = A[ (row * HEIGHT) + i ] * B [ (i * WIDTH) + col ];
		}		
		C[index] = value;
*/

void compareResult( int *arrayA, int *arrayB, int width, int height){
	
	int arraySize = width * height;
	
	for(int i=0; i<arraySize; i++){
		if( arrayA[i] != arrayB[i]){
			cout<<"arrayA["<<i<<"] != arrayB["<<i<<"]"<<endl;
			cout<<"[NOT SAME] Result on CPU and GPU is not same"<<endl;
			break;
		}
		if (i ==arraySize-1){
			cout<<"Result on CPU and GPU is same"<<endl;
		}
	}
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
		if( i % width == 0)
			cout<<endl;
		//cout<<"Array["<<i<<"] : "<<array[i]<<"  ";
		cout<<"["<<i<<"] : "<<array[i]<<"  ";
	}
	cout<<endl;
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
