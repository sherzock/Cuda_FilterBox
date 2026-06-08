#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "timer.h"

cv::Mat imageInputRGBA;
cv::Mat imageOutputRGBA;

uchar4 *d_inputImageRGBA__;
uchar4 *d_outputImageRGBA__;

size_t numRows() { return imageInputRGBA.rows; }
size_t numCols() { return imageInputRGBA.cols; }

/*******  DEFINED IN func.cu *********/
void create_filter(float **h_filter, int *filterWidth, int id_filter);

void box_filter(uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA,
                        const size_t numRows, const size_t numCols,
                        unsigned char *d_redFiltered,
                        unsigned char *d_greenFiltered,
                        unsigned char *d_blueFiltered,
                        int id_filter);

//****************************************************************************
// Also note that we've supplied a helpful debugging function called checkCudaErrors.
// You should wrap your allocation and copying statements like we've done in the
// code we're supplying you. Here is an example of the unsafe way to allocate
// memory on the GPU:
//
// cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols);
//
// Here is an example of the safe way to do the same thing:
//
// checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols));
//****************************************************************************

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

//return types are void since any internal error will be handled by quitting
//no point in returning error codes...
//returns a pointer to an RGBA version of the input image
//and a pointer to the single channel grey-scale output
//on both the host and device
void preProcess(uchar4 **h_inputImageRGBA, uchar4 **h_outputImageRGBA,
                uchar4 **d_inputImageRGBA, uchar4 **d_outputImageRGBA,
                unsigned char **d_redFiltered,
                unsigned char **d_greenFiltered,
                unsigned char **d_blueFiltered,               
                const std::string &filename) {

  //make sure the context initializes ok
  checkCudaErrors(cudaFree(0));

  cv::Mat image = cv::imread(filename.c_str(), cv::IMREAD_COLOR);
  if (image.empty()) {
    std::cerr << "Couldn't open file: " << filename << std::endl;
    exit(1);
  }

  cv::cvtColor(image, imageInputRGBA, cv::COLOR_BGR2RGBA);

  //allocate memory for the output
  imageOutputRGBA.create(image.rows, image.cols, CV_8UC4);

  //This shouldn't ever happen given the way the images are created
  //at least based upon my limited understanding of OpenCV, but better to check
  if (!imageInputRGBA.isContinuous() || !imageOutputRGBA.isContinuous()) {
    std::cerr << "Images aren't continuous!! Exiting." << std::endl;
    exit(1);
  }

  *h_inputImageRGBA  = (uchar4 *)imageInputRGBA.ptr<unsigned char>(0);
  *h_outputImageRGBA = (uchar4 *)imageOutputRGBA.ptr<unsigned char>(0);

  const size_t numPixels = numRows() * numCols();
  //allocate memory on the device for both input and output
  checkCudaErrors(cudaMalloc(d_inputImageRGBA, sizeof(uchar4) * numPixels));
  checkCudaErrors(cudaMalloc(d_outputImageRGBA, sizeof(uchar4) * numPixels));
  checkCudaErrors(cudaMemset(*d_outputImageRGBA, 0, numPixels * sizeof(uchar4))); //make sure no memory is left laying around

  //copy input array to the GPU
  checkCudaErrors(cudaMemcpy(*d_inputImageRGBA, *h_inputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

  d_inputImageRGBA__  = *d_inputImageRGBA;
  d_outputImageRGBA__ = *d_outputImageRGBA;
  
  checkCudaErrors(cudaMalloc(d_redFiltered,    sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMalloc(d_greenFiltered,  sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMalloc(d_blueFiltered,   sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMemset(*d_redFiltered,   0, sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMemset(*d_greenFiltered, 0, sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMemset(*d_blueFiltered,  0, sizeof(unsigned char) * numPixels));
}

void postProcess(const std::string& output_file, uchar4* data_ptr) {
  cv::Mat output(numRows(), numCols(), CV_8UC4, (void*)data_ptr);

  cv::Mat imageOutputBGR;
  cv::cvtColor(output, imageOutputBGR, cv::COLOR_BGR2RGBA);
  //output the image
  cv::imwrite(output_file.c_str(), imageOutputBGR);
}

void cleanUp(void)
{
  cudaFree(d_inputImageRGBA__);
  cudaFree(d_outputImageRGBA__);
}


// An unused bit of code showing how to accomplish this assignment using OpenCV.  It is much faster 
//    than the naive implementation in reference_calc.cpp.
void generateReferenceImage(std::string input_file, std::string reference_file, int kernel_size)
{
	cv::Mat input = cv::imread(input_file);
	// Create an identical image for the output as a placeholder
	cv::Mat reference = cv::imread(input_file);
	cv::GaussianBlur(input, reference, cv::Size2i(kernel_size, kernel_size),0);
	cv::imwrite(reference_file, reference);
}

/*******  Begin main *********/

int main(int argc, char **argv) {
  uchar4 *h_inputImageRGBA,  *d_inputImageRGBA;
  uchar4 *h_outputImageRGBA, *d_outputImageRGBA;
  unsigned char *d_redFiltered, *d_greenFiltered, *d_blueFiltered;

  float *h_filter;
  int    filterWidth;
  int id_filter=0;    

  std::string input_file;
  std::string output_file;
  std::string reference_file;
  bool useEpsCheck = false;
  switch (argc)
  {
  case 2:
    input_file = std::string(argv[1]);
    output_file = "output.png";
    break;
  case 3:
    input_file  = std::string(argv[1]);
    id_filter = atoi(argv[2]);
    output_file = "output.png";
    break;
  case 4:
    input_file  = std::string(argv[1]);
    id_filter = atoi(argv[2]);
    output_file = std::string(argv[3]);
    break;    
  default:
      std::cerr << "Usage: ./box_filter input_file [filter_id] [output_filename]" << std::endl;
      exit(1);
  }
  //load the image and give us our input and output pointers
  preProcess(&h_inputImageRGBA, &h_outputImageRGBA, &d_inputImageRGBA, &d_outputImageRGBA,
             &d_redFiltered, &d_greenFiltered, &d_blueFiltered, input_file);

  GpuTimer timer;
  timer.Start();
  //call the students' code
  box_filter(d_inputImageRGBA, d_outputImageRGBA, numRows(), numCols(),
                     d_redFiltered, d_greenFiltered, d_blueFiltered,id_filter);
  timer.Stop();
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  int err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());

  if (err < 0) {
    //Couldn't print! Probably the student closed stdout - bad news
    std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
    exit(1);
  }

  //write results 

  size_t numPixels = numRows()*numCols();
  //copy the output back to the host
  checkCudaErrors(cudaMemcpy(h_outputImageRGBA, d_outputImageRGBA__, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost));

  postProcess(output_file, h_outputImageRGBA);

  checkCudaErrors(cudaFree(d_redFiltered));
  checkCudaErrors(cudaFree(d_greenFiltered));
  checkCudaErrors(cudaFree(d_blueFiltered));

  cleanUp();

  return 0;
}
