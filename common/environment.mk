# Compiler and flags
# Two options:clang and gcc-11
CC := clang

NVCC := nvcc
# Default optimization level is -O1
OPT_LEVEL := -O1
CFLAGS := -Wall -fopenmp $(OPT_LEVEL)
LDFLAGS :=
