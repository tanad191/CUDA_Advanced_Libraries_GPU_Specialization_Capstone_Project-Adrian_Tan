#ifndef __RECTANGLES_CUDA_H
#define __RECTANGLES_CUDA_H

#include <stdio.h>
#include <stdlib.h>
#include "haar_cuda.h"
#include "image_cuda.h"

#ifdef __cplusplus
#include <vector>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int x;
    int y;
    int width;
    int height;
} MyRect;

int myMax(int a, int b);

int myMin(int a, int b);

inline  int  myRound( float value );

int myAbs(int n);

int predicate(float eps, MyRect& r1, MyRect& r2);

/* Draws white bounding boxes around detected faces */
void drawRectangle(MyImage* image, MyRect r);
#ifdef __cplusplus
} // End of extern "C"
#endif

#ifdef __cplusplus
/* C++-Only Function Declarations (using std::vector) */

int partition(std::vector<MyRect>& _vec, std::vector<int>& labels, float eps);

void groupRectangles(std::vector<MyRect>& _vec, int groupThreshold, float eps);

#endif

#endif // __RECTANGLES_CUDA_H
