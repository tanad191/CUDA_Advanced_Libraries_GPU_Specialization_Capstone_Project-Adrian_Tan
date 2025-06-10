
// ==============================================================================================
// Filename: cuda_detect.cu
//
// Description:
//     Implements a CUDA-accelerated sliding window detector for the Viola-Jones algorithm.
//     This implementation evaluates the real weak classifiers and leverages Unified Memory
//     for integral images, cascade structures, and detection results.
//
// Date: 03.12.25
// Authors: Ibrahim Binmahfood, Kunjan Vyas, Robert Wilcox
// -----------------------------------------------------------------------------

#include "cuda_detect.cuh"  // Include header file for CUDA detection
#include <stdio.h>          // Standard I/O
#include <math.h>           // Math functions
#include <cuda_runtime.h>   // CUDA runtime
#include <device_launch_parameters.h>
#include <vector>           // For std::vector
#include <string.h>         // For memcpy
#include <assert.h>         // For device-side assertions


//#define FINAL_DEBUG

#define DEBUG_CANDIDATE_X 2015
#define DEBUG_CANDIDATE_Y 863

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("[CUDA ERROR] %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return std::vector<MyRect>(); \
    } \
} while(0)

// ---------------------------------------------------------------------
// Constant memory for classifier parameters.
__constant__ int* d_stages_array;
__constant__ float* d_stages_thresh_array;
__constant__ int* d_rectangles_array;
__constant__ int* d_weights_array;
__constant__ int* d_alpha1_array;
__constant__ int* d_alpha2_array;
__constant__ int* d_tree_thresh_array;

// ---------------------------------------------------------------------
// Declaration of atomicCAS.
extern __device__ int atomicCAS(int* address, int compare, int val);


// ---------------------------------------------------------------------
// Global device variable for debug print count
__device__ int d_debug_print_count = 0;

// ---------------------------------------------------------------------
// Device function: Integer square root for the GPU.
// This function replicates the behavior of the CPU's int_sqrt.
__device__ int int_sqrt_device(int value) {
    int i;
    unsigned int a = 0, b = 0, c = 0;
    for (i = 0; i < (32 >> 1); i++) {
        c <<= 2;
        c += (value >> 30); // get the upper 2 bits of value
        value <<= 2;
        a <<= 1;
        b = (a << 1) | 1;
        if (c >= b) {
            c -= b;
            a++;
        }
    }
    return a;
}

// ---------------------------------------------------------------------
// Device function: Rounding function mirroring CPU implementation
__device__ inline int myRound_device(float value) {
    return (int)(value + (value >= 0 ? 0.5f : -0.5f));
}


// ---------------------------------------------------------------------
// Device function: Evaluate a weak classifier for candidate window p.
// Assumes that for each feature, d_rectangles_array stores 12 ints in the order:
// [x_offset1, y_offset1, width1, height1, x_offset2, y_offset2, width2, height2,
//  x_offset3, y_offset3, width3, height3]
__device__ float evalWeakClassifier_device(const myCascade* d_cascade, int variance_norm_factor, MyPoint p,
    int haar_counter, int w_index, int r_index, float scaleFactor)
{

    //printf("[Device] entered evalWeakClassifier_device\n");

    // Print candidate coordinates for every 100th candidate

    int* rect = d_rectangles_array + r_index;

    // --- First Rectangle ---
    int tl1_x = p.x + (int)myRound_device(rect[0]);
    int tl1_y = p.y + (int)myRound_device(rect[1]);
    int br1_x = tl1_x + (int)myRound_device(rect[2]);
    int br1_y = tl1_y + (int)myRound_device(rect[3]);

    // Check bounds
    assert(tl1_x >= 0 && tl1_x < d_cascade->sum.width);
    assert(tl1_y >= 0 && tl1_y < d_cascade->sum.height);
    assert(br1_x >= 0 && br1_x < d_cascade->sum.width);
    assert(br1_y >= 0 && br1_y < d_cascade->sum.height);

    int idx_tl1 = tl1_y * d_cascade->sum.width + tl1_x;
    int idx_tr1 = tl1_y * d_cascade->sum.width + br1_x;
    int idx_bl1 = br1_y * d_cascade->sum.width + tl1_x;
    int idx_br1 = br1_y * d_cascade->sum.width + br1_x;


    int sum1 = d_cascade->p0[idx_br1] - d_cascade->p0[idx_tr1] - d_cascade->p0[idx_bl1] + d_cascade->p0[idx_tl1];
    sum1 = sum1 * d_weights_array[w_index + 0];

#ifdef FINAL_DEBUG
    // Debug Statement for First Rectangle
    if (p.x == DEBUG_CANDIDATE_X && p.y == DEBUG_CANDIDATE_Y && haar_counter == 0) {
        printf("[Device DEBUG] Rect 1: tl=(%d,%d), br=(%d,%d), sum1=%d, weight=%d\n",
            tl1_x, tl1_y, br1_x, br1_y, sum1, d_weights_array[w_index]);
    }
#endif

    // --- Second Rectangle ---
    int tl2_x = p.x + (int)myRound_device(rect[4]);
    int tl2_y = p.y + (int)myRound_device(rect[5]);
    int br2_x = tl2_x + (int)myRound_device(rect[6]);
    int br2_y = tl2_y + (int)myRound_device(rect[7]);

    assert(tl2_x >= 0 && tl2_x < d_cascade->sum.width);
    assert(tl2_y >= 0 && tl2_y < d_cascade->sum.height);
    assert(br2_x >= 0 && br2_x < d_cascade->sum.width);
    assert(br2_y >= 0 && br2_y < d_cascade->sum.height);

    int idx_tl2 = tl2_y * d_cascade->sum.width + tl2_x;
    int idx_tr2 = tl2_y * d_cascade->sum.width + br2_x;
    int idx_bl2 = br2_y * d_cascade->sum.width + tl2_x;
    int idx_br2 = br2_y * d_cascade->sum.width + br2_x;


    int sum2 = d_cascade->p0[idx_br2] - d_cascade->p0[idx_tr2] - d_cascade->p0[idx_bl2] + d_cascade->p0[idx_tl2];
    sum2 = sum2 * d_weights_array[w_index + 1];

#ifdef FINAL_DEBUG
    // Debug Statement for Second Rectangle
    if (p.x == DEBUG_CANDIDATE_X && p.y == DEBUG_CANDIDATE_Y && haar_counter == 0) {
        printf("[Device DEBUG] Rect 2: tl=(%d,%d), br=(%d,%d), sum2=%d, weight=%d\n",
            tl2_x, tl2_y, br2_x, br2_y, sum2, d_weights_array[w_index + 1]);
    }
#endif

    int total_sum = sum1 + sum2;

    int sum3 = 0;

    // --- Third Rectangle (if present) ---
    if (d_weights_array[w_index + 2] != 0)
    {
        int tl3_x = p.x + (int)myRound_device(rect[8]);
        int tl3_y = p.y + (int)myRound_device(rect[9]);
        int br3_x = tl3_x + (int)myRound_device(rect[10]);
        int br3_y = tl3_y + (int)myRound_device(rect[11]);

#ifdef FINAL_DEBUG
        if (p.x == DEBUG_CANDIDATE_X && p.y == DEBUG_CANDIDATE_Y &&
            haar_counter == 0 && w_index == 0 && r_index == 0) {
            printf("[Device DEBUG] Third rectangle: tl=(%d,%d), br=(%d,%d)\n", tl3_x, tl3_y, br3_x, br3_y);
        }
#endif

        assert(tl3_x >= 0 && tl3_x < d_cascade->sum.width);
        assert(tl3_y >= 0 && tl3_y < d_cascade->sum.height);
        assert(br3_x >= 0 && br3_x < d_cascade->sum.width);
        assert(br3_y >= 0 && br3_y < d_cascade->sum.height);

        int idx_tl3 = tl3_y * d_cascade->sum.width + tl3_x;
        int idx_tr3 = tl3_y * d_cascade->sum.width + br3_x;
        int idx_bl3 = br3_y * d_cascade->sum.width + tl3_x;
        int idx_br3 = br3_y * d_cascade->sum.width + br3_x;
        sum3 = d_cascade->p0[idx_br3] - d_cascade->p0[idx_tr3] - d_cascade->p0[idx_bl3] + d_cascade->p0[idx_tl3];
        sum3 *= d_weights_array[w_index + 2];
        total_sum += sum3;

#ifdef FINAL_DEBUG
        // Debug Statement for Third Rectangle (only if it exists)
        if (p.x == DEBUG_CANDIDATE_X && p.y == DEBUG_CANDIDATE_Y && haar_counter == 0) {
            printf("[Device DEBUG] Rect 3: tl=(%d,%d), br=(%d,%d), sum3=%d, weight=%d\n",
                tl3_x, tl3_y, br3_x, br3_y, sum3, d_weights_array[w_index + 2]);
        }
#endif

    }
    int threshold = d_tree_thresh_array[haar_counter] * (variance_norm_factor);

#ifdef FINAL_DEBUG
    // Debug only for the specific candidate and first few features
    if (p.x == DEBUG_CANDIDATE_X && p.y == DEBUG_CANDIDATE_Y && haar_counter < 5) {
        printf("[Device DEBUG] Candidate (%d,%d), Feature %d: sum1=%d, sum2=%d, sum3=%d, total_sum=%d, threshold=%d\n",
            p.x, p.y, haar_counter, sum1, sum2, sum3, total_sum, threshold);
    }
#endif

    return (total_sum >= threshold) ? d_alpha2_array[haar_counter] : d_alpha1_array[haar_counter];
}


__device__ int runCascadeClassifier_device(MyIntImage* d_sum, MyIntImage* d_sqsum,
    const myCascade* d_cascade, MyPoint p, int start_stage, float scaleFactor)
{
    int width = d_cascade->sum.width;

    // Compute candidate offsets
    int p_offset = p.y * width + p.x;
    int pq_offset = p.y * d_cascade->sqsum.width + p.x;

    // Compute the integral image values at the four corners
    int top_left = d_cascade->p0[p_offset];
    int top_right = d_cascade->p1[p_offset];
    int bottom_left = d_cascade->p2[p_offset];
    int bottom_right = d_cascade->p3[p_offset];

    int mean = bottom_right - top_right - bottom_left + top_left;

    int sq_top_left = d_cascade->pq0[pq_offset];
    int sq_top_right = d_cascade->pq1[pq_offset];
    int sq_bottom_left = d_cascade->pq2[pq_offset];
    int sq_bottom_right = d_cascade->pq3[pq_offset];

    int var_norm = (sq_bottom_right - sq_top_right - sq_bottom_left + sq_top_left);
    var_norm = (int)((var_norm * d_cascade->inv_window_area) - mean * mean);

    if (var_norm > 0)
        var_norm = int_sqrt_device(var_norm);
    else
        var_norm = 1;

#ifdef FINAL_DEBUG
    // Integral Debugging 
    if (p.x == DEBUG_CANDIDATE_X && p.y == DEBUG_CANDIDATE_Y) {
		printf("\n--------------------------------------------------\n");
        printf("[Device DEBUG] Candidate (%d,%d):\n", p.x, p.y);
        printf("[Device DEBUG] Integral corners: p0=%d, p1=%d, p2=%d, p3=%d\n",
            d_cascade->p0[p_offset], d_cascade->p1[p_offset], d_cascade->p2[p_offset], d_cascade->p3[p_offset]);
        printf("[Device DEBUG] Squared integral corners: pq0=%d, pq1=%d, pq2=%d, pq3=%d\n",
            d_cascade->pq0[pq_offset], d_cascade->pq1[pq_offset],
            d_cascade->pq2[pq_offset], d_cascade->pq3[pq_offset]);
        printf("[Device DEBUG] mean = %u, var_norm = %u\n", mean, var_norm);
        printf("--------------------------------------------------\n\n");
    }
#endif

    int haar_counter = 0;
    int w_index = 0;
    int r_index = 0;
    float stage_sum = 0.0f;

    for (int i = start_stage; i < d_cascade->n_stages; i++) {
        stage_sum = 0.0f;
        int num_features = d_stages_array[i];

        for (int j = 0; j < num_features; j++) {
            int feature_result = evalWeakClassifier_device(d_cascade, (int)var_norm, p,
                haar_counter, w_index, r_index, scaleFactor);
            stage_sum += feature_result;

            haar_counter++;
            w_index += 3;
            r_index += 12;
        }

#ifdef FINAL_DEBUG
        // Debugging after each stage:
        if (p.x == DEBUG_CANDIDATE_X && p.y == DEBUG_CANDIDATE_Y) {
            printf("[Device DEBUG] Stage %d complete: stage_sum = %f, threshold = %f\n",
                i, stage_sum, 0.4f * d_stages_thresh_array[i]);
        }
#endif

        if (stage_sum < 0.4 * d_stages_thresh_array[i])
            return -i;
    }
    return 1;
}



// ---------------------------------------------------------------------
// CUDA kernel: Each thread processes one candidate window.
__global__ void detectKernel(MyIntImage* d_sum, MyIntImage* d_sqsum,
    myCascade* d_cascade, float scaleFactor,
    int x_max, int y_max,
    MyRect* d_candidates, int* d_candidateCount,
    int maxCandidates)
{
	

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int current_count = atomicAdd(&d_debug_print_count, 1);

#ifdef FINAL_DEBUG
    if (current_count < 10) {
        printf("[GPU Debug: detectKernel] x=%d, y=%d, x_max=%d, y_max=%d)\n",
            x, y, x_max, y_max);
    }
#endif

    // Debug: For the first thread in each block, print x and y
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        //printf("[Device DEBUG] Block (%d,%d): first thread: x=%d, y=%d, x_max=%d, y_max=%d\n",
            //blockIdx.x, blockIdx.y, x, y, x_max, y_max);
    }

    // Check the bounds unconditionally for a few threads
    if (x >= x_max || y >= y_max) {
        
#ifdef FINAL_DEBUG
        // debug print: iterations 4,5,6, limited to first 10 prints
        if (1) {
            // Atomically increment global print count and check limi
            if (current_count < 10) {
                printf("[GPU Debug: (x >= x_max || y >= y_max)] x=%d, y=%d, x_max=%d, y_max=%d)\n",
                    x, y, x_max, y_max);
            }
        }
#endif
        return;
    }
    
#ifdef FINAL_DEBUG
    if (current_count == 2) {
        printf("\nMADE IT PAST if (x >= x_max || y >= y_max)\n");
    }
#endif

    MyPoint p;
    p.x = x;
    p.y = y;

#ifdef FINAL_DEBUG
    if (current_count == 2) {
        printf("\nMADE IT PAST p.x and p.y assignments\n");
    }
#endif

    int result = runCascadeClassifier_device(d_sum, d_sqsum, d_cascade, p, 0, scaleFactor);

#ifdef FINAL_DEBUG
    if (current_count == 2) {
        printf("\nMADE IT PAST p.x and p.y runCascadeClassifier_device. \n RESULT: %d\n\n", result);
    }
#endif

#ifdef FINAL_DEBUG
    if (result > 0) {
        printf("\nResult positiive!\n RESULT: %d\n\n", result);
    }
#endif

    if (result > 0) {
        MyRect r;
        r.x = (int)myRound_device(x * scaleFactor);
        r.y = (int)myRound_device(y * scaleFactor);

        r.width = (int)myRound_device(d_cascade->orig_window_size.width * scaleFactor);
        r.height = (int)myRound_device(d_cascade->orig_window_size.height * scaleFactor);
        int idx = atomicAdd(d_candidateCount, 1);
        if (idx < maxCandidates) {
            d_candidates[idx] = r;
        }

    }
}

// runDetection in cuda_detect.cu
std::vector<MyRect> runDetection(MyIntImage* h_sum, MyIntImage* h_sqsum,
    myCascade* cascade, int maxCandidates,
    float scaleFactor, int extra_x, int extra_y, int iter_counter)
{
    std::vector<MyRect> candidates;

    // --- Step 1: Allocate Unified Memory for sum integral image ---
    int dataSize = h_sum->width * h_sum->height * sizeof(int);
    MyIntImage* d_sum = nullptr;
    CUDA_CHECK(cudaMallocManaged((void**)&d_sum, sizeof(MyIntImage)));
    CUDA_CHECK(cudaMallocManaged((void**)&(d_sum->data), dataSize));
    CUDA_CHECK(cudaMemcpy(d_sum->data, h_sum->data, dataSize, cudaMemcpyHostToDevice));
    d_sum->width = h_sum->width;
    d_sum->height = h_sum->height;

    // --- Step 2: Allocate Unified Memory for squared sum integral image ---
    dataSize = h_sqsum->width * h_sqsum->height * sizeof(int);
    MyIntImage* d_sqsum = nullptr;
    CUDA_CHECK(cudaMallocManaged((void**)&d_sqsum, sizeof(MyIntImage)));
    CUDA_CHECK(cudaMallocManaged((void**)&(d_sqsum->data), dataSize));
    CUDA_CHECK(cudaMemcpy(d_sqsum->data, h_sqsum->data, dataSize, cudaMemcpyHostToDevice));
    d_sqsum->width = h_sqsum->width;
    d_sqsum->height = h_sqsum->height;

    // --- Step 3: Allocate Unified Memory for the cascade structure ---
    myCascade* d_cascade = nullptr;
    CUDA_CHECK(cudaMallocManaged((void**)&d_cascade, sizeof(myCascade)));
    *d_cascade = *cascade;  // copy host cascade to unified memory

    // --- Step 4: Update the cascade with unified memory pointers for integral images ---
    d_cascade->sum = *d_sum;
    d_cascade->sqsum = *d_sqsum;
    d_cascade->sum.data = d_sum->data;   // Use unified memory data buffer of d_sum
    d_cascade->sqsum.data = d_sqsum->data; // Use unified memory data buffer of d_sqsum
    d_cascade->sum.width = d_sum->width;
    d_cascade->sum.height = d_sum->height;
    d_cascade->sqsum.width = d_sqsum->width;
    d_cascade->sqsum.height = d_sqsum->height;

    // Use the original window size for classification
    int winW = d_cascade->orig_window_size.width;
    int winH = d_cascade->orig_window_size.height;
    d_cascade->p0 = d_cascade->sum.data;
    d_cascade->p1 = d_cascade->sum.data + winW - 1;
    d_cascade->p2 = d_cascade->sum.data + d_cascade->sum.width * (winH - 1);
    d_cascade->p3 = d_cascade->sum.data + d_cascade->sum.width * (winH - 1) + (winW - 1);

    d_cascade->pq0 = d_cascade->sqsum.data;
    d_cascade->pq1 = d_cascade->sqsum.data + winW - 1;
    d_cascade->pq2 = d_cascade->sqsum.data + d_cascade->sqsum.width * (winH - 1);
    d_cascade->pq3 = d_cascade->sqsum.data + d_cascade->sqsum.width * (winH - 1) + (winW - 1);

#ifdef FINAL_DEBUG
    printf("Cascade corner pointers:\n");
    printf(" p0 = %p\n", (void*)d_cascade->p0);
    printf(" p1 = %p (offset: %td)\n", (void*)d_cascade->p1, d_cascade->p1 - d_cascade->sum.data);
    printf(" p2 = %p (offset: %td)\n", (void*)d_cascade->p2, d_cascade->p2 - d_cascade->sum.data);
    printf(" p3 = %p (offset: %td)\n", (void*)d_cascade->p3, d_cascade->p3 - d_cascade->sum.data);
    printf(" pq0 = %p\n", (void*)d_cascade->pq0);
    printf(" pq1 = %p (offset: %td)\n", (void*)d_cascade->pq1, d_cascade->pq1 - d_cascade->sqsum.data);
    printf(" pq2 = %p (offset: %td)\n", (void*)d_cascade->pq2, d_cascade->pq2 - d_cascade->sqsum.data);
    printf(" pq3 = %p (offset: %td)\n", (void*)d_cascade->pq3, d_cascade->pq3 - d_cascade->sqsum.data);
#endif

    // --- Step 5: Transfer classifier parameters to device constant memory ---
    // (Allocate device memory for the classifier arrays and copy them from host.)
    int* d_stages_array_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_stages_array_dev, cascade->n_stages * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_stages_array_dev, cascade->stages_array, cascade->n_stages * sizeof(int), cudaMemcpyHostToDevice));

    float* d_stages_thresh_array_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_stages_thresh_array_dev, cascade->n_stages * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_stages_thresh_array_dev, cascade->stages_thresh_array, cascade->n_stages * sizeof(float), cudaMemcpyHostToDevice));

    int* d_rectangles_array_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_rectangles_array_dev, cascade->total_nodes * 12 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_rectangles_array_dev, cascade->rectangles_array, cascade->total_nodes * 12 * sizeof(int), cudaMemcpyHostToDevice));

    int* d_weights_array_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_weights_array_dev, cascade->total_nodes * 3 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_weights_array_dev, cascade->weights_array, cascade->total_nodes * 3 * sizeof(int), cudaMemcpyHostToDevice));

    int* d_alpha1_array_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_alpha1_array_dev, cascade->total_nodes * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_alpha1_array_dev, cascade->alpha1_array, cascade->total_nodes * sizeof(int), cudaMemcpyHostToDevice));

    int* d_alpha2_array_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_alpha2_array_dev, cascade->total_nodes * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_alpha2_array_dev, cascade->alpha2_array, cascade->total_nodes * sizeof(int), cudaMemcpyHostToDevice));

    int* d_tree_thresh_array_dev = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_tree_thresh_array_dev, cascade->total_nodes * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tree_thresh_array_dev, cascade->tree_thresh_array, cascade->total_nodes * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpyToSymbol(d_stages_array, &d_stages_array_dev, sizeof(int*)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_stages_thresh_array, &d_stages_thresh_array_dev, sizeof(float*)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_rectangles_array, &d_rectangles_array_dev, sizeof(int*)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_weights_array, &d_weights_array_dev, sizeof(int*)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_alpha1_array, &d_alpha1_array_dev, sizeof(int*)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_alpha2_array, &d_alpha2_array_dev, sizeof(int*)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_tree_thresh_array, &d_tree_thresh_array_dev, sizeof(int*)));

#ifdef FINAL_DEBUG
    printf("[Host DEBUG] Transferred classifier parameters to device constant memory.\n");
#endif

    // --- Step 6: Allocate Unified Memory for detection results ---
    MyRect* d_candidates = nullptr;
    CUDA_CHECK(cudaMallocManaged((void**)&d_candidates, maxCandidates * sizeof(MyRect)));
    int* d_candidateCount = nullptr;
    CUDA_CHECK(cudaMallocManaged((void**)&d_candidateCount, sizeof(int)));
    *d_candidateCount = 0;

#ifdef FINAL_DEBUG
    printf("[Host DEBUG] d_candidates allocated at %p, d_candidateCount allocated at %p, initial candidate count = %d\n",
        (void*)d_candidates, (void*)d_candidateCount, *d_candidateCount);
#endif

    // --- Step 7: Determine search space dimensions and launch the detection kernel ---
    // Use extra_x and extra_y only to clip the search space so that the sliding window remains in bounds.
    // The classifier window size remains the original size (cascade->orig_window_size) * scaleFactor.
    int baseWidth = cascade->orig_window_size.width;
    int baseHeight = cascade->orig_window_size.height;

    // Compute maximum valid starting positions for the sliding window.
    int x_max = d_sum->width - baseWidth;
    int y_max = d_sum->height - baseHeight;
    if (x_max < 0) x_max = 0;
    if (y_max < 0) y_max = 0;

#ifdef FINAL_DEBUG
    printf("[Host DEBUG] Search space dimensions (with extra margins): x_max=%d, y_max=%d\n", x_max, y_max);
#endif

    dim3 blockDim(16, 16);
    dim3 gridDim((x_max + blockDim.x - 1) / blockDim.x,
        (y_max + blockDim.y - 1) / blockDim.y);

#ifdef FINAL_DEBUG
    printf("[Host] Launching kernel with gridDim=(%d, %d), blockDim=(%d, %d)\n",
        gridDim.x, gridDim.y, blockDim.x, blockDim.y);
#endif

#ifdef FINAL_DEBUG
    printf("\nDetection window: %d x %d, x_max=%d, y_max=%d\n",
        baseWidth, baseHeight, x_max, y_max);
#endif


    // Launch the kernel with the original base window size for classification.
    CUDA_CHECK(cudaDeviceSynchronize());

#ifdef FINAL_DEBUG
    printf("\n-------------------------------------------------------------------------------\n");
    printf("[Kernel Launch Debug] scaleFactor = %.3f, x_max = %d, y_max = %d, maxCandidates = %d\n",
        scaleFactor, x_max, y_max, maxCandidates);
    printf("[Kernel Launch Debug] d_sum = %p, d_sqsum = %p, d_cascade = %p, d_candidates = %p, d_candidateCount = %p\n",
        (void*)d_sum, (void*)d_sqsum, (void*)d_cascade, (void*)d_candidates, (void*)d_candidateCount);
    printf("[Kernel Launch Debug] gridDim = (%d, %d), blockDim = (%d, %d)\n",
        gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    printf("\n-------------------------------------------------------------------------------\n");
#endif

    int zero = 0;
    CUDA_CHECK(cudaMemcpyToSymbol(d_debug_print_count, &zero, sizeof(int)));

    detectKernel << <gridDim, blockDim >> > (d_sum, d_sqsum, d_cascade, scaleFactor, x_max, y_max, d_candidates, d_candidateCount, maxCandidates);
    CUDA_CHECK(cudaGetLastError());
   // printf("[Host DEBUG] Kernel launched.\n");
    CUDA_CHECK(cudaDeviceSynchronize());
   // printf("[Host DEBUG] Kernel execution completed.\n");

    int hostCandidateCount = 0;
    CUDA_CHECK(cudaMemcpy(&hostCandidateCount, d_candidateCount, sizeof(int), cudaMemcpyDeviceToHost));
   // printf("[Host] Detected %d candidate windows.\n", hostCandidateCount);
    for (int i = 0; i < hostCandidateCount; i++) {
        candidates.push_back(d_candidates[i]);
    }

   // printf("[Host DEBUG] Cleaning up Unified Memory and device memory allocated with cudaMalloc.\n");
    cudaFree(d_candidates);
    cudaFree(d_candidateCount);
    cudaFree(d_cascade);
    cudaFree(d_sum->data);
    cudaFree(d_sum);
    cudaFree(d_sqsum->data);
    cudaFree(d_sqsum);
    cudaFree(d_stages_array_dev);
    cudaFree(d_stages_thresh_array_dev);
    cudaFree(d_rectangles_array_dev);
    cudaFree(d_weights_array_dev);
    cudaFree(d_alpha1_array_dev);
    cudaFree(d_alpha2_array_dev);
    cudaFree(d_tree_thresh_array_dev);

   // printf("[Host] runDetection() completed.\n");
    return candidates;
}

