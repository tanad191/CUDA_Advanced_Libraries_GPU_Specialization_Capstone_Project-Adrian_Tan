/**
 * @file main.cpp
 * @brief Main function for CUDA-based image detection.
 *
 * Date: 03.12.25
 * Authors: Ibrahim Binmahfood, Kunjan Vyas, Robert Wilcox
 *
 * This program demonstrates a CUDA-accelerated multi-scale face (or object) detection
 * algorithm using a Haar-like cascade classifier. The steps include:
 *   - Parsing command-line arguments for input/output image paths.
 *   - Checking for CUDA device availability.
 *   - Loading an input image and computing its integral images.
 *   - Initializing and linking a cascade classifier.
 *   - Computing additional offsets for the detection window.
 *   - Performing multi-scale detection using a CUDA detection routine.
 *   - Grouping and drawing detection results (result rectangles).
 *   - Saving the output image and cleaning up allocated resources.
 *
 * Usage: ./program -i [path/to/input/image] -o [path/to/output/image]
 */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <cuda_runtime.h>
#include <npp.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include "image_cuda.h"
#include "haar_cuda.h"
#include "cuda_detect.cuh" // runDetection is declared here.

#include <helper_string.h>
#include <helper_cuda.h>

#define MINNEIGHBORS 1

extern int iter_counter;

inline int myRound(float value); // Prototype for the inline function.

void ScaleImage_Invoker(myCascade *_cascade, float _factor, int sum_row, int sum_col, std::vector<MyRect> &_vec);

extern void nearestNeighbor(ImageData *src, ImageData *dst);

// Helper function to scan classifier data and compute the extra offsets (in x and y)
// that account for all rectangle extents beyond the base detection window.
void computeExtraOffsets(const myCascade *cascade, int *extra_x, int *extra_y);

// Helper function to print a subset of values from an integral image.
void debugPrintIntegralImageGPU(ImageDimensions *img, int numSamples);

void generateTestPgm(const char *filename) {
  const int width = 4;
  const int height = 4;
  const int maxGrey = 255;

  // Sample image data: 4x4 increasing values
  unsigned char data[width * height] = {
      0, 64, 128, 192,
      32, 96, 160, 224,
      16, 80, 144, 208,
      48, 112, 176, 240};

  std::ofstream outFile(filename, std::ios::binary);
  if (!outFile) {
    std::cerr << "Failed to open output file: " << filename << std::endl;
    return;
  }

  // Write PGM header (P5 format)
  outFile << "P5\n"
          << width << " " << height << "\n"
          << maxGrey << "\n";

  // Write binary pixel data
  outFile.write(reinterpret_cast<char *>(data), sizeof(data));
  outFile.close();

  std::cout << "PGM file generated: " << filename << std::endl;
}

int main(int argc, char **argv) {
  // [DEBUG] Generate test .pgm file
  generateTestPgm("test_image.pgm");

  ImageData img;
  if (readPgm((char*)"test_image.pgm", &img) == 0) {
      std::cout << "Image loaded: " << img.width << "x" << img.height << std::endl;
      // You can inspect img.data if needed
      freeImage(&img); // Don't forget to clean up
  }

  // 1. Parse arguments and initialize file input and output paths and performance tracking variables.
  // Variable declarations for option parsing and runtime measurement.
  int opt;
  int rc;

  // Pointers to hold input and output file paths.
  char *sFilename = NULL;
  char *sResultFilename = NULL;

  // Variables to store start and end times for performance measurement.
  struct timespec t_start;
  struct timespec t_end;

  // Parse command-line options using getopt.
  while ((opt = getopt(argc, argv, "i:o:")) != -1) {
    switch (opt) {
      // Option 'i' for input image file path.
      case 'i':
        // Verify that the specified file exists.
        if (access(optarg, F_OK) != 0) {
          std::cerr <<  "ERROR: Input file " << optarg << " does not exist." << std::endl;
          std::cerr << "Usage: " << argv[0] << " -i [path/to/image] -o [path/to/output/image]" << std::endl;
          std::cerr << "Exiting..." << std::endl;
          exit(1);
        }
        // Store valid input file path.
        sFilename = optarg;
        std::cout << "imageSpatialFilterNPP running facial detection on input file: <" << sFilename << ">" << std::endl;
        break;

      // Option 'o' for output image file path.
      case 'o':
        sResultFilename = optarg;
        break;

      // Default case if an unknown option is provided.
      default:
        std::cerr <<  "ERROR: Incorrect parameter format. Proper usage is: "<< argv[0] <<" -i [path/to/image] -o [path/to/output/image]" << std::endl;
        std::cerr <<  "Exiting..." << std::endl;
        exit(1);
      }
  }

  // Indicate entry into the main function.
  std::cout << "Entering main function..." << std::endl;

  // Check for available CUDA devices.
  int deviceCount = 0;
  cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
  if (cudaStatus != cudaSuccess || deviceCount == 0) {
    std::cout << "No CUDA devices found or CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
    return 1;
  }
  std::cout << "Found " << deviceCount << "CUDA device(s)." << std::endl;

  // 2. Load the input image and allocate memory for the corresponding output.
  // First, declare a host image object for an 8-bit grayscale image.
  npp::ImageCPU_8u_C1 hostSrc;
  ImageData imageObj;
  // ImageData *image;
  npp::loadImage(sFilename, hostSrc);
  // Then, declare a device image and upload the host source to the device source.
  npp::ImageNPP_8u_C1 deviceSrc(hostSrc);

  NppiSize srcSize = {(int)deviceSrc.width(), (int)deviceSrc.height()};
  NppiPoint srcOffset = {0, 0};

  // Create a struct with ROI size.
  NppiSize sizeROI = {(int)deviceSrc.width(), (int)deviceSrc.height()};
  // Allocate a device image of appropriately reduced size.
  npp::ImageNPP_8u_C1 deviceDst(sizeROI.width, sizeROI.height);

  int nBufferSize = 0;
  Npp8u *pScratchBufferNPP = 0;

  // Get the necessary scratch buffer size and allocate that much device memory.
  NPP_CHECK_NPP(
      nppiFilterCannyBorderGetBufferSize(sizeROI, &nBufferSize));

  cudaMalloc((void **)&pScratchBufferNPP, nBufferSize);

  // Load the grayscale image from the disk.
  if (readPgm(sFilename, &imageObj) == 0) {
    std::cout << "Image <" << sFilename << "> loaded successfully. Dimensions:" << imageObj.width << "x" << imageObj.height << std::endl;
  } else {
    std::cerr << "ERROR: Unable to open input image <" << sFilename << ">." << std::endl;
    return 1;
  }
  //std::cout << "[DEBUG] Image loaded successfully: <" << sFilename << ">" << std::endl;

  // Compute integral images for fast feature computation.
  ImageDimensions sumObj, sqsumObj;
  ImageDimensions *sum = &sumObj;
  ImageDimensions *sqsum = &sqsumObj;
  createSumImage(deviceSrc.width(), deviceSrc.height(), sum);
  createSumImage(deviceSrc.width(), deviceSrc.height(), sqsum);
  std::cout << "[DEBUG] Sum images loaded successfully." << std::endl;
  std::cout << "Summed image dimensions: " << sum->width << " x " << sum->height << std::endl;
  std::cout << "Square summed image dimensions: " << sqsum->width << " x " << sqsum->height << std::endl;

  integralImages(&imageObj, sum, sqsum);
  std::cout << "[DEBUG] Integral images loaded successfully." << std::endl;
  std::cout << "Summed image dimensions: " << sum->width << " x " << sum->height << std::endl;
  std::cout << "Square summed image dimensions: " << sqsum->width << " x " << sqsum->height << std::endl;

  // Allocate a facial detection image of appropriate size (on the GPU).
  npp::ImageNPP_8u_C1 deviceFacialDetectionImg(deviceSrc.width(), deviceSrc.height());

  // 3. Initialize the cascade classifier parameters.
  myCascade cascadeObj;
  myCascade *cascade = &cascadeObj;
  cascade->n_stages = 25;
  cascade->total_nodes = 2913;
  cascade->orig_window_size.width = 24;
  cascade->orig_window_size.height = 24;
  MySize minSize = {20, 20};
  MySize maxSize = {0, 0};

  // Load the cascade classifier data.
  std::cout << "Loading cascade classifier..." << std::endl;
  readTextClassifier(cascade);
  std::cout << "Cascade classifier loaded..." << std::endl;

  // Validate that the classifier rectangles have been loaded.
  if (cascade->scaled_rectangles_array == NULL) {
    std::cerr << "ERROR: cascade->scaled_rectangles_array is NULL after readTextClassifier." << std::endl;
  } else {
    std::cout << "cascade->scaled_rectangles_array is NOT NULL after readTextClassifier: " << cascade->scaled_rectangles_array << std::endl;
  }

  // 4. Link the computed integral images to the cascade classifier.
  std::cout << "Linking integral images to cascade..." << std::endl;
  setImageForCascadeClassifier(cascade, sum, sqsum);
  std::cout << "Integral images linked to cascade." << std::endl;

  // Compute extra offsets that adjust the detection window dimensions.
  int extra_x = 0, extra_y = 0;
  computeExtraOffsets(cascade, &extra_x, &extra_y);
  std::cout << "Computed extra offsets: extra_x = " << extra_x << ", extra_y = " << extra_y << std::endl;

  // Adjust the detection window size to include extra offsets.
  int adjusted_width = cascade->orig_window_size.width + extra_x;
  int adjusted_height = cascade->orig_window_size.height + extra_y;
  std::cout << "Adjusted detection window size: " << adjusted_width << " x " << adjusted_height << std::endl;

  // Allocate buffers for scaled image and its integral images.
  ImageData scaledImg;
  createImage(imageObj.width, imageObj.height, &scaledImg);
  ImageDimensions scaledSum, scaledSqSum;
  createSumImage(imageObj.width, imageObj.height, &scaledSum);
  createSumImage(imageObj.width, imageObj.height, &scaledSqSum);
  float factor = 1.0f;

  // 5. Run CUDA detection at each scale in the image pyramid.
  // First, prepare a vector to store all result detections from GPU.
  std::vector<MyRect> gpuResults;
  int iter_counter = 1;

  // Then, start the timer for performance measurement.
  rc = clock_gettime(CLOCK_REALTIME, &t_start);
  assert(rc == 0);

  // Loop over different scales of the image.
  while (true) {
    // Calculate new dimensions based on the scaling factor.
    int newWidth = (int)(imageObj.width / factor);
    int newHeight = (int)(imageObj.height / factor);
    int winWidth = myRound(cascade->orig_window_size.width * factor);
    int winHeight = myRound(cascade->orig_window_size.height * factor);
    MySize sz = {newWidth, newHeight};
    MySize winSize = {winWidth, winHeight};
    // Compute the available difference in dimensions for placing the detection window.
    MySize diff = {sz.width - cascade->orig_window_size.width, sz.height - cascade->orig_window_size.height};

    // If the difference is negative, the window no longer fits; exit the loop.
    if (diff.width < 0 || diff.height < 0)
      break;
    // Skip scales that produce a detection window smaller than the minimum allowed size.
    if (winSize.width < minSize.width || winSize.height < minSize.height) {
      factor *= 1.2f;
      continue;
    }

    // Reallocate buffers for the current scale.
    freeImage(&scaledImg);
    freeSumImage(&scaledSum);
    freeSumImage(&scaledSqSum);
    createImage(newWidth, newHeight, &scaledImg);
    createSumImage(newWidth, newHeight, &scaledSum);
    createSumImage(newWidth, newHeight, &scaledSqSum);

    // Scale the input image using nearest neighbor interpolation.
    nearestNeighbor(&imageObj, &scaledImg);
    // Compute the integral images for the scaled image.
    integralImages(&scaledImg, &scaledSum, &scaledSqSum);
    // Link the scaled integral images to the cascade classifier.
    setImageForCascadeClassifier(cascade, &scaledSum, &scaledSqSum);

    // Check if the detection window fits within the scaled integral image dimensions.
    if (factor * (cascade->orig_window_size.width + extra_x) < scaledSum.width &&
        factor * (cascade->orig_window_size.height + extra_y) < scaledSum.height) {
      // Run the CUDA detection for the current scale.
      std::vector<MyRect> gpuresults = runDetection(&scaledSum, &scaledSqSum, cascade, 10000000, factor, adjusted_width, adjusted_height, iter_counter);
      // Merge the current scale's results into the overall result list.
      gpuResults.insert(gpuResults.end(), gpuresults.begin(), gpuresults.end());
    }
    // Increment the scale factor for the next iteration.
    factor *= 1.2f;
  }

  // Group nearby result rectangles to remove duplicates.
  groupResults(gpuResults, MINNEIGHBORS, 0.4f);

  // Stop the timer and calculate the runtime.
  rc = clock_gettime(CLOCK_REALTIME, &t_end);
  assert(rc == 0);

  unsigned long long int runtime = 1000000000 * (t_end.tv_sec - t_start.tv_sec) + t_end.tv_nsec - t_start.tv_nsec;

  // Output the number of results and runtime details.
  std::cout << "CUDA detection found " << gpuResults.size() << " results." << std::endl;
  std::cout << "Time spent on detection: " << runtime / 1000000000 << "." << runtime % 1000000000 << " seconds (" << runtime << " nanoseconds)" << std::endl;

  // Print the coordinates and dimensions for each detected result.
  for (size_t i = 0; i < gpuResults.size(); i++) {
    std::cout << "CUDA face detection result #" << i+1 << ": x=" << gpuResults[i].x << ", y=" << gpuResults[i].y << ", width=" << gpuResults[i].width << ", height=" << gpuResults[i].height << std::endl;
  }

  // 8. Draw detection result boxes on the original image.
  // Create a CPU-side image buffer from the original input.
  npp::ImageCPU_8u_C1 hostDrawImg(deviceSrc.size());
  deviceSrc.copyTo(hostDrawImg.data(), hostDrawImg.pitch());

  // Prepare ImageData wrapper to use with drawResultBox().
  ImageData outputImage;
  outputImage.data = hostDrawImg.data();
  outputImage.width = hostDrawImg.width();
  outputImage.height = hostDrawImg.height();

  // Draw rectangles on the host image
  for (const MyRect &rect : gpuResults) {
    drawResultBox(&outputImage, rect);
  }

  // 9. Save the output image
  std::cout << "Saving output to: " << sResultFilename << std::endl;
  saveImage(sResultFilename, hostDrawImg);
  std::cout << "Image saved as: " << sResultFilename << std::endl;

  nppiFree(deviceSrc.data());
  nppiFree(deviceFacialDetectionImg.data());

  // 10. Clean up and release resources.
  releaseTextClassifier(cascade);
  freeImage(&imageObj);
  freeSumImage(sum);
  freeSumImage(sqsum);

  // Free the temporary buffers allocated for scaled images.
  freeImage(&scaledImg);
  freeSumImage(&scaledSum);
  freeSumImage(&scaledSqSum);

  return 0;
}

// Helper function to scan classifier data and compute the extra offsets (in x and y)
// that account for all rectangle extents beyond the base detection window.
void computeExtraOffsets(const myCascade *cascade, int *extra_x, int *extra_y) {
  *extra_x = 0;
  *extra_y = 0;

  int totalRectElems = cascade->total_nodes * 12;

  for (int i = 0; i < totalRectElems; i += 4)
  {
    int rx = cascade->rectangles_array[i];
    int ry = cascade->rectangles_array[i + 1];
    int rw = cascade->rectangles_array[i + 2];
    int rh = cascade->rectangles_array[i + 3];

    if (rx == 0 && ry == 0 && rw == 0 && rh == 0)
      continue;

    int current_right = rx + rw;
    int current_bottom = ry + rh;

    if (current_right > *extra_x)
      *extra_x = current_right;

    if (current_bottom > *extra_y)
      *extra_y = current_bottom;
  }
}

// Helper function to print a subset of values from an integral image.
void debugPrintIntegralImageGPU(ImageDimensions *img, int numSamples) {
  int width = img->width;
  int height = img->height;
  int total = width * height;

#ifdef FINAL_DEBUG
  std::cout << "GPU Integral image summary: width = " << 0 << ", height = " << 0 << ", total values = " << 0 << "\n", width, height, total);
#endif

#ifdef FINAL_DEBUG
  // Print the four corner values
  std::cout << "Top-left (index 0): " << img->data[0] << std::endl;
  std::cout << "Top-right (index " << width - 1 << "): " << img->data[width - 1] << std::endl;
  std::cout << "Bottom-left (index " << (height - 1) * width << "): " << img->data[(height - 1) * width] << std::endl;
  std::cout << "Bottom-right (index " << total - 1 << "): " << img->data[total - 1] << std::endl;
#endif

  int step = total / numSamples;
  if (step < 1)
    step = 1;

#ifdef FINAL_DEBUG
  std::cout << "Printing " << numSamples << " sample values (every " << step << "-th value)..." << std::endl;
  for (int i = 0; i < total; i += step)
  {
    std::cout << "Index " << i << ": " << img->data[i] << std::endl;
  }
#endif
}