
// Standard C/C++ Directives
#include <iostream>
#include <conio.h>

// OpenCV directives
#include <opencv2\opencv.hpp>
#include <highgui\highgui.hpp> 


// cuda directives
#include "cuda.h"
#include "device_functions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;
using namespace cv;


double AVG_FILTER_COEFFICIENT = 0.04; // 1/25 = 0.04



//****************** METHODS DECLERATIONS [DEVICE] ***************************
// KERNEL
__global__ void Kernel_Avg_Filter( float *inputImage, float *filterImage, int WIDTH,
								  int HEIGHT, const double AVG_FILTER_COEFFICIENT);
//Stub function
void avg_Filter_DEVICE( Mat *h_inputImage, Mat *h_filteredImage, float AVG_FILTER_COEFFICIENT);

//****************** METHODS DECLERATIONS [HOST] ***************************
void playingFunWithPixels(Mat *image);
void imageInfo(Mat *image);
void avg_Filter(Mat*, Mat*, float);

int getWidth(Mat *image);

void compareHostDeviceImages(Mat *hostImage, Mat *deviceImage);

int main(){
	char* IMAGE_PATH = "C:\\Users\\Aamir\\Downloads\\grey.jpg";
	namedWindow("InputImage", WINDOW_NORMAL);
	resizeWindow("InputImage", 300, 300);
	//	namedWindow("FilteredImageHOST", WINDOW_NORMAL);
	//	resizeWindow("FilteredImageHOST", 300, 300);
	namedWindow("FilteredImageDEVICE", WINDOW_NORMAL);
	resizeWindow("FilteredImageDEVICE", 300, 300);


	//##1. Reading Images and Allocatin Memory on HOST
	Mat  h_inputImage = imread( IMAGE_PATH,CV_LOAD_IMAGE_GRAYSCALE);
	//Mat  h_inputImage = imread("D:\\grey.jpg",CV_LOAD_IMAGE_ANYCOLOR);
	if( h_inputImage.data == NULL){
		cout<<"[ERROR] Input Image is null"<<endl;	
		return -1;
	}
	
	Mat h_filteredImage,h_DeviceFilteredImage;
	h_filteredImage.create( h_inputImage.rows, h_inputImage.cols, h_inputImage.type());
	h_DeviceFilteredImage.create( h_inputImage.rows, h_inputImage.cols, h_inputImage.type());


	// AVG Filter on HOST
	//avg_Filter( &h_inputImage, &h_filteredImage, AVG_FILTER_COEFFICIENT);
	
	// AVG Filter on DEVICE
	avg_Filter_DEVICE( &h_inputImage, &h_DeviceFilteredImage, AVG_FILTER_COEFFICIENT);

	imshow("InputImage", h_inputImage);
	//imshow("FilteredImageHOST", h_filteredImage);
	imshow("FilteredImageDEVICE", h_DeviceFilteredImage);


	//system("pause");
	waitKey(0);
	return 0;
}



//****************** METHODS DEFINITIONS ****************************

__global__ void Kernel_Avg_Filter( float *inputImage, float *filterImage, int WIDTH,
								  int HEIGHT, const double AVG_FILTER_COEFFICIENT){


	int col = threadIdx.x + ( blockIdx.x * blockDim.x); // idx
	int row = threadIdx.y + ( blockIdx.y * blockDim.y);	// idy

	int index = col + ( WIDTH * row);

	// 3*3 i.e. -1,0,1
	//int value = index-15000;

	if( col < WIDTH && row < HEIGHT){
		int value = 0;
		for( int i=-1; i<2; i++){
			// 3*3 i.e. -1,0,1
			for( int j=-1; j<2; j++){

				int f_index = index + (WIDTH * j) +i;

				if( (f_index > -1) || (f_index < HEIGHT) ||  (f_index < WIDTH))
				{
					value += inputImage[ index + (WIDTH *j)+i ] * AVG_FILTER_COEFFICIENT;
				}
				else{ 
					value +=0;
				}

			}//end-for-j
		}//end-for-i
		filterImage[ index ] = value;
		//filterImage[ index ] = 0;
	}// end-if-col<width && row<height
}


void avg_Filter_DEVICE( Mat *h_inputImage, Mat *h_filteredImage, float AVG_FILTER_COEFFICIENT){

	if( h_inputImage->channels() >1){
		cout<<"[ERROR] avg_Filter_DEVICE only works on 1 channel Images"<<endl;
		return;	
	}


	//**caution** CONVERTION of Mat from uchar TO FLOAT
	h_inputImage->convertTo( *h_inputImage, CV_32FC1);
	h_filteredImage->convertTo( *h_filteredImage, CV_32FC1);

	int width = h_inputImage->cols;
	int height = h_inputImage->rows;
	const int BYTES_INPUT_IMAGE  = width * height * sizeof(float);

	//## 1. DEVICE Memory allocation
	float *d_inputImage, *d_filteredImage;

	cudaMalloc( (void**) &d_inputImage, BYTES_INPUT_IMAGE);
	cudaMalloc( (void**)&d_filteredImage, BYTES_INPUT_IMAGE);

	//## 2. DEVICE Memcpy HOST2DEVICE
	cudaMemcpy( d_inputImage, h_inputImage->data, BYTES_INPUT_IMAGE, cudaMemcpyHostToDevice);
	cudaMemcpy( d_filteredImage, h_filteredImage->data, BYTES_INPUT_IMAGE, cudaMemcpyHostToDevice);

	cout<<"cudaGetLastError():"<<cudaGetLastError()<<endl;
	//## 3. KERNEL Launch
	int threadX = 32;
	int threadY = 32;

	int blockX = ceil( width / threadX)+1;
	int blockY = ceil( height / threadY)+1;

	dim3 dimBlock(threadX, threadY, 1);
	dim3 dimGrid( blockX, blockY, 1);

	Kernel_Avg_Filter<<<dimGrid,dimBlock>>>(d_inputImage, d_filteredImage, width, height, AVG_FILTER_COEFFICIENT);
	cout<<"cudaGetLastError():"<<cudaGetLastError()<<endl;
	cudaDeviceSynchronize();


	cout<<"cudaGetLastError():"<<cudaGetLastError()<<endl;
	//## 4. Device Memcpy DEVICE2HOST

	cudaMemcpy( h_filteredImage->data, d_filteredImage, BYTES_INPUT_IMAGE, cudaMemcpyDeviceToHost);

	//**caution** CONVERTING back to uchar 
	h_inputImage->convertTo( *h_inputImage, CV_8UC1);
	h_filteredImage->convertTo( *h_filteredImage, CV_8UC1);

	//## 5. Memory Free
	cudaFree( d_inputImage);
	cudaFree( d_filteredImage);

}

int getWidth(Mat *image){
	if(	image->channels() == 3)
		return image->cols * 3;
	return image->cols;
}


void imageInfo(Mat *image){

	cout<<"Image Type : "<<image->type()<<endl;
	cout<<"Image Channels : "<<image->channels()<<endl;
	cout<<"Image Rows : "<<image->rows<<endl;
	cout<<"Image Cols : "<<image->cols<<endl;
}


void compareHostDeviceImages(Mat *hostImage, Mat *deviceImage)
{
	int height = hostImage->rows;
	int width = getWidth(hostImage);


	for(int i=0; i<height; i++){

		for( int j=0; j<width; j++){

				int h_val = hostImage->at<uchar>(i,j);
				int d_val = deviceImage->at<uchar>(i,j);
				if( d_val != h_val){
					cout<<"Images are not same"<<endl;
				}
		}

	}


}

void playingFunWithPixels(Mat *image)
{
	int height = image->rows;
	int width = getWidth(image);


	for(int i=0; i<height; i++){

		for( int j=0; j<width; j++){

			if( i%10 == 0)
				image->at<uchar>(i,j) = 255;
		}

	}


}

void avg_Filter(Mat *image, Mat *filterImage, float AVG_FILTER_COEFFICIENT){

	int height = image->rows;
	int width = getWidth(image);


	if( image->cols != width){
		cout<<"[ERROR] this avg_Filter() function only works on 1 channel image"<<endl;
		return;
	}


	const int f_width = 3;
	const int f_height = 3;

	int value = 0;

	//will run on rows of entire image
	for(int row=0; row<height; row++){

		//will run on cols of entire image
		for( int col=0; col<width; col++){
			value = 0;
			//loops to run filter
			for( int i =-1; i<f_height-1; i++){
				for(int j=-1; j<f_width-1; j++){

					int f_row = row+i;
					int f_col = col+j;

					if( f_row > -1 &&  f_row < height && f_col > -1 && f_col < width)
						value += image->at<uchar>( f_row, f_col) * AVG_FILTER_COEFFICIENT;
					else
						value +=0;
				}//end-loop-j
			}//end-loop-i

			filterImage->at<uchar>(row, col) = value;

		}//end-loop-row

	}//end-loop-col

}//end-func-avg_Filter