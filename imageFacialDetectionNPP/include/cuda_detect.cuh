// ==============================================================================================
// Filename: cuda_detect.h
//
// Description:
//     Header file declaring the interface for CUDA-based Viola-Jones detection functions.
//     It includes necessary CUDA and project-specific headers, defines conditional compilation
//     for C++, and provides the runDetection function declaration.
//
// Date: 03.12.25
// Authors: Ibrahim Binmahfood, Kunjan Vyas, Robert Wilcox
// -----------------------------------------------------------------------------

#ifndef CUDA_DETECT_H
#define CUDA_DETECT_H

#include "haar_cuda.h"
#include "image_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <string.h>
#include <assert.h>

#ifdef __cplusplus
#include <vector>

// runDetection launches the CUDA detection kernel using the host‐side
// integral images and cascade classifier. It transfers data to the GPU,
// launches the kernel, retrieves results, cleans up device memory,
// and returns a std::vector<MyRect> containing candidate detections.
// Parameters:
//   h_sum       - pointer to the host MyIntImage for the integral image
//   h_sqsum     - pointer to the host MyIntImage for the squared integral image
//   cascade     - pointer to the host cascade classifier (after setImageForCascadeClassifier)
//   maxCandidates - maximum number of candidate detections allocated on the device
//   scaleFactor - current scale factor (e.g., 1.0f)
std::vector<MyRect> runDetection(MyIntImage* h_sum, MyIntImage* h_sqsum,
        myCascade* cascade,
        int maxCandidates, 
        float scaleFactor,
        int extra_x,
        int extra_y,
        int iter_counter);
#endif // __cplusplus

#endif // CUDA_DETECT_H
