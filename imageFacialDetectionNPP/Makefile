# C++ Compiler and Flags
CC      := g++
CFLAGS  := -std=c++14 -Wall -lrt

# CUDA Compiler and Flags
# Google Colab offers T4 GPUs which are the sm_75 compute capability. Since we had a
# a GPU on hand with sm_86 compute capability, we prioritized that if it exists
NCC     := nvcc
NLIB    := /usr/local/cuda/lib64
NFLAGS  := -std=c++14 -lrt -gencode arch=compute_86,code=sm_86 -gencode arch=compute_75,code=sm_75 -Xcompiler "-D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS -D_SILENCE_CXX17_ADAPTOR_TYPEDEPRECATION_WARNINGS"
NLIB += -lnppisu_static -lnppif_static -lnppc_static -lculibos -lfreeimage

# Directories
G_SRC_DIR := ./src
G_INC_DIR := ./include

# Common includes and paths for CUDA
INCLUDES_COMMON := -I../Common
INCLUDES_COMMON += -I../Common/include
INCLUDES_COMMON += -I../Common/UtilNPP

# Gather GPU source files and convert to object files list
G_SRCS_C   := $(wildcard $(G_SRC_DIR)/*.c)
G_OBJS_C   := $(subst .c,.o, $(G_SRCS_C))

G_SRCS_CPP := $(wildcard $(G_SRC_DIR)/*.cpp)
G_OBJS_CPP := $(subst .cpp,.o, $(G_SRCS_CPP))

G_SRCS_CU := $(wildcard $(G_SRC_DIR)/*.cu)
G_OBJS_CU := $(subst .cu,.o, $(G_SRCS_CU))

# Gather CPU source files and convert to object files list
C_SRCS_C   := $(wildcard $(C_SRC_DIR)/*.c)
C_OBJS_C   := $(subst .c,.o, $(C_SRCS_C))

C_SRCS_CPP := $(wildcard $(C_SRC_DIR)/*.cpp)
C_OBJS_CPP := $(subst .cpp,.o, $(C_SRCS_CPP))

# Create GPU objs list total
G_OBJS    := $(G_OBJS_C) $(G_OBJS_CPP) $(G_OBJS_CU)

# Create CPU objs list total
C_OBJS    := $(C_OBJS_C) $(C_OBJS_CPP)

# Don't allow files in directory to be names clean or default or build
.PHONY: clean default build test perf

default: build test perf

build: build_gpu

build_gpu: imageFacialDetectionNPP.cpp $(G_OBJS)
	@echo "Building GPU build..."
	$(NCC) $(NFLAGS) -I $(NLIB) -I $(G_INC_DIR) $(INCLUDES_COMMON) $^ -o imageFacialDetectionNPP
	mkdir -p ./logs
	mkdir -p ./output

$(G_SRC_DIR)/%.o: $(G_SRC_DIR)/%.cu $(G_INC_DIR)/%.cuh
	$(NCC) $(NFLAGS) -I $(G_INC_DIR) -I $(NLIB) -c $< -o $@

$(G_SRC_DIR)/%.o: $(G_SRC_DIR)/%.cpp $(G_INC_DIR)/%.h
	$(CC) $(CFLAGS) -I $(G_INC_DIR) -c $< -o $@

$(G_SRC_DIR)/%.o: $(G_SRC_DIR)/%.c $(G_INC_DIR)/%.h
	$(CC) $(CFLAGS) -I $(G_INC_DIR) -c $< -o $@

test_start:
	@echo "Testing GPU runs on images in ./data/..."
	@echo "Logs are stored in ./logs/"
	mkdir -p ./output/
	./imageFacialDetectionNPP -i test_image.pgm -o ./test_image_detected.pgm > ./logs/test_image.log

test_sample:
	@echo "Testing GPU runs on images in ./data/..."
	@echo "Logs are stored in ./logs/"
	mkdir -p ./output/
	./imageFacialDetectionNPP -i ./data/MyHeroAcademia.pgm -o ./output/MyHeroAcademia_detected.pgm > ./logs/MyHeroAcademia.log

test_gpu:
	@echo "Testing GPU runs on images in ./data/..."
	@echo "Logs are stored in ./logs/"
	mkdir -p ./output/
	./imageFacialDetectionNPP -i ./data/MyHeroAcademia.pgm -o ./output/MyHeroAcademia_detected.pgm > ./logs/MyHeroAcademia.log
	./imageFacialDetectionNPP -i ./data/SpyXFamily.pgm -o ./output/SpyXFamily_detected.pgm > ./logs/SpyXFamily.log
	./imageFacialDetectionNPP -i ./data/DemonSlayer.pgm -o ./output/DemonSlayer_detected.pgm > ./logs/DemonSlayer.log
	./imageFacialDetectionNPP -i ./data/DungeonMeshi.pgm -o ./output/DungeonMeshi_detected.pgm > ./logs/DungeonMeshi.log
	./imageFacialDetectionNPP -i ./data/Naruto.pgm -o ./output/Naruto_detected.pgm > ./logs/Naruto.log
	./imageFacialDetectionNPP -i ./data/Bleach.pgm -o ./output/Bleach_detected.pgm > ./logs/Bleach.log
	./imageFacialDetectionNPP -i ./data/FullmetalAlchemist.pgm -o ./output/FullmetalAlchemist_detected.pgm > ./logs/FullmetalAlchemist.log
	./imageFacialDetectionNPP -i ./data/Pokemon.pgm -o ./output/Pokemon_detected.pgm > ./logs/Pokemon.log

perf:
	@echo "Parsing logs..."
	@echo "Plots generated in logs/plots/"
	python3 run_parser.py ./logs ./logs

clean:
	$(RM) -rf ./logs ./output $(G_SRC_DIR)/*.o $(G_INC_DIR)/*.h.gch