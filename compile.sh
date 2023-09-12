#! /bin/bash

nvcc -lcublas matmul.cu -o matmul -g
