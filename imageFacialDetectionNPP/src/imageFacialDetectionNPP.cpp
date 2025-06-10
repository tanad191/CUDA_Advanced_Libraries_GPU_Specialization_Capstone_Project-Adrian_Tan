/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_cuda.h>
#include <helper_string.h>
#include <sys/stat.h>
#include <sys/types.h>

bool printfNPPinfo(int argc, char *argv[])
{
  const NppLibraryVersion *libVer = nppGetLibVersion();

  printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
         libVer->build);

  int driverVersion, runtimeVersion;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);

  printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
         (driverVersion % 100) / 10);
  printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
         (runtimeVersion % 100) / 10);

  // Min spec is SM 1.0 devices
  bool bVal = checkCudaCapabilities(1, 0);
  return bVal;
}

bool featureDetectionToFile(int argc, char *argv[], std::string sFilename)
{
  try
  {
    std::string sResultFilename;

    if (checkCmdLineFlag(argc, (const char **)argv, "output"))
    {
      std::cout << "DEBUG: checkCmdLineFlag is TRUE" << std::endl;
      char *outputFilePath;
      getCmdLineArgumentString(argc, (const char **)argv, "output",
                               &outputFilePath);
      sResultFilename = outputFilePath;
    } else { //fallback if output isn't specified
      std::cout << "DEBUG: checkCmdLineFlag is TRUE" << std::endl;
      std::string::size_type dot = sFilename.rfind('.');
      sResultFilename = (dot != std::string::npos)
                            ? sFilename.substr(0, dot)
                            : sFilename;
      sResultFilename += "_spatialFilter.pgm";
    }

    // declare a host image object for an 8-bit grayscale image
    npp::ImageCPU_8u_C1 oHostSrc;
    // load gray-scale image from disk
    npp::loadImage(sFilename, oHostSrc);
    // declare a device image and copy construct from the host image,
    // i.e. upload host to device
    npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

    // create struct with the ROI size
    NppiSize oSrcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
    NppiPoint oSrcOffset = {0, 0};
    NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};

    // allocate device image of the same size
    npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);

    // run the rotation
    NPP_CHECK_NPP(nppiFilterSharpenBorder_8u_C1R(
        oDeviceSrc.data(),
        oDeviceSrc.pitch(),
        oSrcSize,
        oSrcOffset,
        oDeviceDst.data(),
        oDeviceDst.pitch(),
        oSizeROI,
        NPP_BORDER_REPLICATE));



    // Create a drawing surface to render a single supersampled tile
    // GpuPBuffer pbuffer(resx *ssx, resy *ssy);
    // GpuCanvas canvas(pbuffer);
    // // GpuCanvas == OpenGL context
    // // Create a drawing mode to hold the GPU state vector
    // GpuDrawmode drawmode;
    // float fov = 90;
    // float hither = 0.1;
    // float yon = 10000;
    // Matrix4 cam2ras = compute_cam2ras(fov, hither, yon, resx, resy);
    // // Allocate a 4-channel, fp32, RGBA final output image buffer
    // size_t nbytes = 4 * resx * resy * sizeof(float);
    // float *image = (float *)malloc(nbytes);
    // memset(image, 0, nbytes);
    // // Create the filter we'll need
    // Filter2D *filter = Filter2D::MakeFilter("gaussian", filterx, filtery);
    // // Allocate a 4-channel, fp32, RGBA tile readback buffer
    // int tilew = downsample.padded_tile_width(tx, *filter);
    // int tileh = downsample.padded_tile_height(ty, *filter);
    // float *tile = (float *)malloc(4 * sizeof(float) * tilew * tileh);

    // declare a host image for the result
    npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
    // and copy the device result data into it
    oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

    // Before saving the image
    std::string::size_type slash = sResultFilename.rfind('/');
    if (slash != std::string::npos) {
      std::string outputDir = sResultFilename.substr(0, slash);
      struct stat st;
      if (stat(outputDir.c_str(), &st) != 0) {
        // Directory doesn't exist, try creating it
        if (mkdir(outputDir.c_str(), 0755) != 0) {
          std::cerr << "Failed to create output directory: " << outputDir << std::endl;
          exit(EXIT_FAILURE);
        }
      }
    }

    std::cout << "DEBUG: Will save image to: " << sResultFilename << std::endl;
    saveImage(sResultFilename, oHostDst);
    std::cout << "Saved image: " << sResultFilename << std::endl;

    nppiFree(oDeviceSrc.data());
    nppiFree(oDeviceDst.data());
  }
  catch (...)
  {
    std::cerr << "Filtering error! An unknown type of exception occurred. \n";
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
    return false;
  }

  return true;
}

int main(int argc, char *argv[])
{
  printf("%s Starting...\n\n", argv[0]);

  try
  {
    std::string sFilename;
    char *filePath;

    findCudaDevice(argc, (const char **)argv);

    if (printfNPPinfo(argc, argv) == false)
    {
      exit(EXIT_SUCCESS);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "input"))
    {
      getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
    }
    else
    {
      filePath = sdkFindFilePath("Lena.pgm", argv[0]);
    }

    if (filePath)
    {
      sFilename = filePath;
    }
    else
    {
      sFilename = "Lena.pgm";
    }

    // if we specify the filename at the command line, then we only test
    // sFilename[0].
    int file_errors = 0;
    std::ifstream infile(sFilename.data(), std::ifstream::in);

    if (infile.good())
    {
      std::cout << "imageSpatialFilterNPP opened: <" << sFilename.data()
                << "> successfully!" << std::endl;
      file_errors = 0;
      infile.close();
    }
    else
    {
      std::cout << "imageSpatialFilterNPP unable to open: <" << sFilename.data() << ">"
                << std::endl;
      file_errors++;
      infile.close();
    }

    if (file_errors > 0)
    {
      exit(EXIT_FAILURE);
    }

    bool isFiltered = featureDetectionToFile(argc, argv, sFilename);

    if (isFiltered)
    {
      exit(EXIT_SUCCESS);
    }
    else
    {
      exit(EXIT_FAILURE);
    }
  }
  catch (npp::Exception &rException)
  {
    std::cerr << "Program error! The following exception occurred: \n";
    std::cerr << rException << std::endl;
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
  }
  catch (...)
  {
    std::cerr << "Program error! An unknown type of exception occurred. \n";
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
    return -1;
  }

  return 0;
}