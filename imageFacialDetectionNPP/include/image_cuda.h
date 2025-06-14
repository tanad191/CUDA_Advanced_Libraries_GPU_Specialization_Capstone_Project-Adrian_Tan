/*
 *  TU Eindhoven
 *  Eindhoven, The Netherlands
 *
 *  Name            :   image.h
 *
 *  Author          :   Francesco Comaschi (f.comaschi@tue.nl)
 *
 *  Date            :   November 12, 2012
 *
 *  Function        :   Functions to manage .pgm images and integral images
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
 * along with this program;  If not, see <http://www.gnu.org/licenses/>
 *
 * In other words, you are welcome to use, share and improve this program.
 * You are forbidden to forbid anyone else to use, share and improve
 * what you give them.   Happy coding!
 * 
 * Modified 03.12.25 by: Ibrahim Binmahfood, Kunjan Vyas, Robert Wilcox
 * to include CUDA code for parallel processing
 */

#ifndef __IMAGE_CUDA_H__
#define __IMAGE_CUDA_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct 
{
	int width;
	int height;
	int maxgrey;
	unsigned char* data;
	int flag;
	int allocatedSize;	// track allocated buffer size
}
ImageData;

typedef struct 
{
	int width;
	int height;
	int* data;
	int flag;
}
ImageDimensions;

int readPgm(char *fileName, ImageData* image);
int writePgm(char *fileName, ImageData* image);
int copyPgmData(ImageData *src, ImageData *dst);
void createImage(int width, int height, ImageData *image);
void createSumImage(int width, int height, ImageDimensions *image);
int freeImage(ImageData* image);
int freeSumImage(ImageDimensions* image);
void setImage(int width, int height, ImageData *image);
void setSumImage(int width, int height, ImageDimensions *image);

#ifdef __cplusplus
}
#endif

#endif
