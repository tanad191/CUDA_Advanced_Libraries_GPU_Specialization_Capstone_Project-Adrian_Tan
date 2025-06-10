# CUDA Advanced Libraries GPU Specialization Capstone Project - Adrian Tan
Capstone project repository for John Hopkins University's CUDA Advanced Libraries course on Coursera

## Overview


## Installing the Project

To build and run the code, you will need to have the following installed:

1.  **GNU C++ Compiler** for building the application.

2.  **`make` Utility** for running the build process from the project terminal. Verify that this is installed and in your PATH by opening a command prompt or PowerShell and typing `make --version`. Version information should be displayed should `make` make be installed.

3.  **IrfanView (or another PGM viewer)** for viewing the PGM images that are both used as input and produced as output. You can also use it to convert JPEG/PNG images to the PGM format and vice versa. Download and install this from [https://www.irfanview.com/](https://www.irfanview.com/).

4. **Python Virtual Environment with certain modules** The modules required are in requirements.txt

## Building and Running the Project

Once the above prerequisites are installed, navigate to the project directory:

Then use the provided `Makefile` to build the project with the following command:

```bash
$ make clean build
```

This will remove all binary ISO files and executables as applicable and then (re)create an executable file named `imageFacialDetectionNPP`.

Once the project has been built and the executable generated, run the following command to test the input:

