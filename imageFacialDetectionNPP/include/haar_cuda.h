/*
 *  TU Eindhoven
 *  Eindhoven, The Netherlands
 *
 *  Name            :   haar.h
 *
 *  Author          :   Francesco Comaschi (f.comaschi@tue.nl)
 *
 *  Date            :   November 12, 2012
 *
 *  Function        :   Haar features evaluation for face detection
 *
 *  History         :
 *      12-11-12    :   Initial version.
 *
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; If not, see <http://www.gnu.org/licenses/>
 *
 * In other words, you are welcome to use, share and improve this program.
 * You are forbidden to forbid anyone else to use, share and improve
 * what you give them.   Happy coding!
 * 
 * Modified 03.12.25 by: Ibrahim Binmahfood, Kunjan Vyas, Robert Wilcox
 * to include CUDA code for parallel processing
 */

#ifndef __HAAR_CUDA_H__
#define __HAAR_CUDA_H__

#include <stdio.h>
#include <stdlib.h>
#include "image_cuda.h"
#include "rectangles_cuda.h"

#ifdef __cplusplus
#include <vector>
#endif

#define MAXLABELS 50

#ifdef __cplusplus
extern "C" {
#endif

    /* C-Compatible Type Definitions */

    typedef int sumtype;
    typedef int sqsumtype;

    typedef struct {
        int x;
        int y;
    } MyPoint;

    typedef struct {
        int width;
        int height;
    } MySize;

    typedef struct {
        int n_stages;
        int total_nodes;
        float scale;
        MySize orig_window_size;
        float inv_window_area;
        MyIntImage sum;
        MyIntImage sqsum;
        sqsumtype* pq0, * pq1, * pq2, * pq3;
        sumtype* p0, * p1, * p2, * p3;

        // Added members for CUDA implementation:
        int* stages_array;         // Array to store number of weak classifiers per stage
        float* stages_thresh_array;    // Array to store stage thresholds
        int* rectangles_array;       // Array to store rectangle features
        int* weights_array;          // Array to store weights
        int* alpha1_array;         // Array to store alpha1 values
        int* alpha2_array;         // Array to store alpha2 values
        int* tree_thresh_array;      // Array to store tree thresholds
		int** scaled_rectangles_array;  // Array to store scaled rectangles

    } myCascade;

    /* C-Compatible Function Declarations */

    /* Sets images for Haar classifier cascade */
    void setImageForCascadeClassifier(myCascade* cascade, MyIntImage* sum, MyIntImage* sqsum);

    /* Runs the cascade on the specified window */
    int runCascadeClassifier(myCascade* cascade, MyPoint pt, int start_stage);

    /* Reads the classifier file into memory */
    void readTextClassifier(myCascade* cascade);

    /* Releases classifier resources */
    void releaseTextClassifier(myCascade* cascade);

    /* Computes integral images (and squared integral images) from a source image */
    void integralImages(MyImage* src, MyIntImage* sum, MyIntImage* sqsum);

#ifdef __cplusplus
} // End of extern "C"
#endif

#ifdef __cplusplus
/* C++-Only Function Declarations (using std::vector) */

std::vector<MyRect> detectObjects(MyImage* image, MySize minSize, MySize maxSize,
    myCascade* cascade, float scale_factor, int min_neighbors);
#endif

#endif // __HAAR_CUDA_H__
