/*
 *  TU Eindhoven
 *  Eindhoven, The Netherlands
 *
 *  Name            :   image.c
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

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <iostream>
#include <fstream>
#include "image_cuda.h"

char* strrev(char* str)
{
	char *p1, *p2;
	if (!str || !*str)
		return str;
	for (p1 = str, p2 = str + strlen(str) - 1; p2 > p1; ++p1, --p2)
	{
		*p1 ^= *p2;
		*p2 ^= *p1;
		*p1 ^= *p2;
	}
	return str;
}

int myatoi (char* string)
{
	int sign = 1;
	// how many characters in the string
	int length = strlen(string);
	int i = 0;
	int number = 0;

	// handle sign
	if (string[0] == '-')
	{
		sign = -1;
		i++;
	}

//	for (i; i < length; i++)
	while(i < length)
	{
		// handle the decimal place if there is one
		if (string[i] == '.')
			break;
		number = number * 10 + (string[i]- 48);
		i++;
	}

	number *= sign;

	return number;
}

void itochar(int x, char* szBuffer, int radix)
{
	int i = 0, n, xx;
	n = x;
	while (n > 0)
	{
		xx = n%radix;
		n = n/radix;
		szBuffer[i++] = '0' + xx;
	}
	szBuffer[i] = '\0';
	strrev(szBuffer);
}

int readPgm(char *fileName, ImageData *image) {
	FILE *in_file = fopen(fileName, "rb");
	if (in_file == NULL)
	{
		std::cerr << "ERROR: Unable to open file <" << fileName << ">" << std::endl;
		return -1;
	}

	std::cout << "Reading image file: <" << fileName << ">" << std::endl;

	// Check magic number: should start with "P5"
	char ch = fgetc(in_file);
	if (ch != 'P')
	{
		std::cerr << "ERROR: <" << fileName << "> is not a valid PGM file (missing 'P')" << std::endl;
		fclose(in_file);
		return -1;
	}

	ch = fgetc(in_file);
	if (ch != '5')
	{
		std::cerr << "ERROR: <" << fileName << "> is not P5 format. Only P5 (binary) PGM is supported." << std::endl;
		fclose(in_file);
		return -1;
	}

	// Skip whitespace after magic number
	do
	{
		ch = fgetc(in_file);
	} while (isspace(ch));

	ungetc(ch, in_file); // Put back the first non-space character

	char line[256];

	// Helper lambda to read a non-comment, non-empty line
	auto readNonCommentLine = [&]() -> bool {
		do
		{
			if (!fgets(line, sizeof(line), in_file))
				return false;
		} while (line[0] == '#' || line[0] == '\n' || line[0] == '\r');
		return true;
	};

	// Read width and height (can be split across lines)
	int width = -1, height = -1;
	while (width == -1 || height == -1)
	{
		if (!readNonCommentLine())
		{
			std::cerr << "ERROR: Failed to read image dimensions." << std::endl;
			fclose(in_file);
			return -1;
		}
		char *pch = strtok(line, " \t\n\r");
		while (pch != NULL)
		{
			if (width == -1)
			{
				width = atoi(pch);
			}
			else if (height == -1)
			{
				height = atoi(pch);
			}
			pch = strtok(NULL, " \t\n\r");
		}
	}

	// std::cout << "[DEBUG] width = " << width << std::endl;
	// std::cout << "[DEBUG] height = " << height << std::endl;

	if (width <= 0 || height <= 0 || width > 100000 || height > 100000)
	{
		std::cerr << "ERROR: Invalid image dimensions." << std::endl;
		fclose(in_file);
		return -1;
	}

	image->width = width;
	image->height = height;

	// Read max grey value
	int maxgrey = -1;
	while (maxgrey == -1)
    {
        if (!readNonCommentLine())
		{
			std::cerr << "ERROR: Failed to read max grey value." << std::endl;
			fclose(in_file);
			return -1;
		}
		char *pch = strtok(line, " \t\n\r");
        // std::cout << "[DEBUG] pch = " << pch << std::endl;
        if (pch)
		{
			maxgrey = atoi(pch);
		}
    }

    if (maxgrey <= 0 || maxgrey > 255)
	{
		std::cerr << "ERROR: Invalid max grey value (must be 1–255)." << std::endl;
		fclose(in_file);
		return -1;
	}
	// std::cout << "[DEBUG] maxgrey = " << maxgrey << std::endl;

	image->maxgrey = maxgrey;

	// Allocate memory
	size_t totalPixels = (size_t)image->width * image->height;
	image->data = (unsigned char *)malloc(totalPixels);
	if (!image->data)
	{
		std::cerr << "ERROR: Memory allocation failed for image data." << std::endl;
		fclose(in_file);
		return -1;
	}

	// Read binary pixel data
	size_t bytesRead = fread(image->data, 1, totalPixels, in_file);
	// std::cout << "[DEBUG] bytesRead = " << bytesRead << std::endl;
	if (bytesRead != totalPixels)
	{
		std::cerr << "ERROR: Unexpected end of file. Expected " << totalPixels << " bytes, got " << bytesRead << std::endl;
		free(image->data);
		fclose(in_file);
		return -1;
	}

	image->flag = 1;
	fclose(in_file);
	return 0;
}

int writePgm(char *fileName, ImageData *image)
{
	char parameters_str[5];
	int i;
	const char *format = "P5";
	if (image->flag == 0)
	{
		return -1;
	}

	//changed from 'w' to 'wb' to fix sub and cntrl-z errors since windows writes 'w' as text and 'wb' as binary
    //this is similar to the issue resolved in the readPgm() function
	FILE *fp = fopen(fileName, "wb");
	if (!fp)
	{
		std::cerr << "ERROR: Unable to open file <" << fileName << ">" << std::endl;
		return -1;
	}
	fputs(format, fp);
	fputc('\n', fp);

	itochar(image->width, parameters_str, 10);
	fputs(parameters_str, fp);
	parameters_str[0] = 0;
	fputc(' ', fp);

	itochar(image->height, parameters_str, 10);
	fputs(parameters_str, fp);
	parameters_str[0] = 0;
	fputc('\n', fp);

	itochar(image->maxgrey, parameters_str, 10);
	fputs(parameters_str, fp);
	fputc('\n', fp);

	for (i = 0; i < (image->width * image->height); i++)
	{
		fputc(image->data[i], fp);
	}
	fclose(fp);
	return 0;
}

int copyPgmData(ImageData *src, ImageData *dst)
{
	// std::cout << "[DEBUG] Copying data..." << std::endl;
	int i = 0;
	if (src->flag == 0)
	{
		std::cerr << "No data available in the specified source image" << std::endl;
		return -1;
	}
	dst->width = src->width;
	dst->height = src->height;
	dst->maxgrey = src->maxgrey;
	dst->data = (unsigned char*)malloc(sizeof(unsigned char)*(dst->height*dst->width));
	dst->flag = 1;
	for (i = 0; i < (dst->width * dst->height); i++)
	{
		dst->data[i] = src->data[i];
	}
	return 0;
}


void createImage(int width, int height, ImageData *image)
{
	image->width = width;
	image->height = height;
	image->flag = 1;
	image->data = (unsigned char *)malloc(sizeof(unsigned char)*(height*width));
}

void createSumImage(int width, int height, ImageDimensions *image)
{
	image->width = width;
	image->height = height;
	image->flag = 1;
	image->data = (int *)malloc(sizeof(int)*(height*width));
}

int freeImage(ImageData* image)
{
	if (image->flag == 0)
	{
		std::cout << "no image to delete" << std::endl;
		return -1;
	}
	else
	{
		free(image->data);
		return 0;
	}
}

int freeSumImage(ImageDimensions* image) {
	if (image->flag == 0)
	{
		std::cout << "no image to delete" << std::endl;
		return -1;
	}
	else
	{
//		std::cout << "image deleted\n");
		free(image->data);
		return 0;
	}
}

void setImage(int width, int height, ImageData *image)
{
	image->width = width;
	image->height = height;
}

void setSumImage(int width, int height, ImageDimensions *image)
{
	image->width = width;
	image->height = height;
}