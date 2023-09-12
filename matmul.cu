#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>

#include <thrust/device_vector.h>
#include <cmath>

#define CUDA_KERNEL(kernel, grid, block, args...) \
    do {\
      (kernel<M, N, K><<<grid, block>>>(args)); \
      cudaError_t err = cudaGetLastError(); \
      if (err != cudaSuccess) { \
        printf("Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
	exit(EXIT_FAILURE); \
      } \
    } while (0)

#define CUDA_CALL(func) \
    do { \
      cudaError_t err = (func); \
      if(err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
      } \
    } while(0)

#define EPSILON 1e-6

bool areMatricesEqual(float* d_A, float* d_B, int rows, int cols) {
    thrust::device_ptr<float> dev_ptr_A = thrust::device_pointer_cast(d_A);
    thrust::device_ptr<float> dev_ptr_B = thrust::device_pointer_cast(d_B);

    // 使用Thrust的equal函数来判断两个矩阵是否相等
    if (!thrust::equal(dev_ptr_A, dev_ptr_A + rows * cols, dev_ptr_B)) {
        return false;
    }

    // 使用CUBLAS的nrm2函数来计算两个矩阵的差的范数
    cublasHandle_t handle;
    cublasCreate(&handle);
    float* d_diff;
    CUDA_CALL(cudaMalloc(&d_diff, rows * cols * sizeof(float)));
    thrust::transform(dev_ptr_A, dev_ptr_A + rows * cols, dev_ptr_B, thrust::device_pointer_cast(d_diff), thrust::minus<float>());
    float norm;
    cublasSnrm2(handle, rows * cols, d_diff, 1, &norm);
    CUDA_CALL(cudaFree(d_diff));
    cublasDestroy(handle);

    // 判断范数是否小于预设的阈值
    return std::abs(norm) < EPSILON;
}

// 每个thread block内线程为BLOCKSIZE * BLOCKSIZE
#define BLOCK_SIZE_M 32
#define BLOCK_SIZE_N 32
#define BLOCK_SIZE_K 32

template <int M, int N, int K>
__global__ void MatMulKernel_sharedMemory(float* A, float* B, float* C) {
  float value = 0.0;
  int row_block_start = blockIdx.y * BLOCK_SIZE_M;
  int col_block_start = blockIdx.x * BLOCK_SIZE_N;
  for (int i = 0; i < K / BLOCK_SIZE_K; ++i) {
    __shared__ float asub[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ float bsub[BLOCK_SIZE_K][BLOCK_SIZE_N];
    for (int j = 0; j < BLOCK_SIZE_K / BLOCK_SIZE_N; ++j) {
      asub[threadIdx.y][j * BLOCK_SIZE_N + threadIdx.x] = A[(row_block_start + threadIdx.y) * K + i * BLOCK_SIZE_K + j * BLOCK_SIZE_N + threadIdx.x];
    }
    for (int j = 0; j < BLOCK_SIZE_K / BLOCK_SIZE_M; ++j) {
      bsub[j * BLOCK_SIZE_M + threadIdx.y][threadIdx.x] = B[(i * BLOCK_SIZE_K + j * BLOCK_SIZE_M + threadIdx.y) * N + col_block_start + threadIdx.x];
    }
    __syncthreads();
    for (int j = 0; j < BLOCK_SIZE_K; ++j) {
      value += asub[threadIdx.y][j] * bsub[j][threadIdx.x];
    }
    __syncthreads();
  }
  C[(row_block_start + threadIdx.y) * N + (col_block_start + threadIdx.x)] = value;
}

template <int M, int N, int K>
__global__ void MatMulKernel(float* A, float* B, float* C) {
  float value = 0.0;
  int row = blockIdx.y * BLOCK_SIZE_M + threadIdx.y;
  int col = blockIdx.x * BLOCK_SIZE_N + threadIdx.x;
  for (int i = 0; i < K; ++i) {
    value += A[row * K + i] * B[i * N + col];
  }
  C[row * N + col] = value;
}

enum Strategy {
  CUBLAS,
  VANILLA,
  SHAREDMEMORY,
};

template <int M, int K, int N>
float testMatMul(Strategy st, bool checkResult) {
  float* h_A, *h_B, *h_C;
  float* d_A, *d_B, *d_C;

  // 分配主机内存.
  h_A = (float*)malloc(M * K * sizeof(float));
  h_B = (float*)malloc(K * N * sizeof(float));
  h_C = (float*)malloc(M * N * sizeof(float));

  // 分配设备内存
  CUDA_CALL(cudaMalloc((void**)&d_A, M * K * sizeof(float)));
  CUDA_CALL(cudaMalloc((void**)&d_B, K * N * sizeof(float)));
  CUDA_CALL(cudaMalloc((void**)&d_C, M * N * sizeof(float)));

  // 初始化主机内存
  for (int i = 0; i < M * K; ++i) {
    h_A[i] = rand() / (float)RAND_MAX;
  }
  for (int i = 0; i < K * N; ++i) {
    h_B[i] = rand() / (float)RAND_MAX;
  }

  // 将数据从主机内存拷贝到设备内存
  CUDA_CALL(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

  // 创建CUDA事件
  cudaEvent_t start, stop;
  CUDA_CALL(cudaEventCreate(&start));
  CUDA_CALL(cudaEventCreate(&stop));

  switch (st) {
  case CUBLAS: {
    // 创建和设置cublas上下文.
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;
    CUDA_CALL(cudaEventRecord(start));
    cublasSgemm(
	handle,
       	CUBLAS_OP_N,
       	CUBLAS_OP_N,
       	N,
       	N,
       	K,
       	&alpha,
       	d_B,
       	N,
       	d_A,
       	K,
       	&beta,
       	d_C,
       	N
    );
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));
    cublasDestroy(handle);
    break;

  } case VANILLA: {
    dim3 dimBlock(BLOCK_SIZE_N, BLOCK_SIZE_M);
    dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
    CUDA_CALL(cudaEventRecord(start));
    CUDA_KERNEL(MatMulKernel, dimGrid, dimBlock, d_A, d_B, d_C);
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));
    break;
  } case SHAREDMEMORY: {
    dim3 dimBlock(BLOCK_SIZE_N, BLOCK_SIZE_M);
    dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
    CUDA_CALL(cudaEventRecord(start));
    CUDA_KERNEL(MatMulKernel_sharedMemory, dimGrid, dimBlock, d_A, d_B, d_C);
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));
    break;
  } default:;
  }

  float milliseconds = 0;
  CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));

  if (checkResult) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;
    float* d_D = nullptr;
    CUDA_CALL(cudaMalloc((void**)&d_D, M * N * sizeof(float)));
    cublasSgemm(
	handle,
       	CUBLAS_OP_N,
       	CUBLAS_OP_N,
       	N,
       	N,
       	K,
       	&alpha,
       	d_B,
       	N,
       	d_A,
       	K,
       	&beta,
       	d_D,
       	N
    );
    CUDA_CALL(cudaDeviceSynchronize());
    cublasDestroy(handle);
    // 将结果从设备内存复制回主机内存
    // cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    // float* h_D = (float*)malloc(M * N * sizeof(float));
    // cudaMemcpy(h_D, d_D, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    // free(h_D);
    if (!areMatricesEqual(d_C, d_D, M, N)) {
      printf("WARNING: the result is not correct!\n");
    }
    CUDA_CALL(cudaFree(d_D));
  }


  // 清理
  free(h_A);
  free(h_B);
  free(h_C);
  CUDA_CALL(cudaFree(d_A));
  CUDA_CALL(cudaFree(d_B));
  CUDA_CALL(cudaFree(d_C));
  CUDA_CALL(cudaEventDestroy(start));
  CUDA_CALL(cudaEventDestroy(stop));
  return milliseconds;
} 

const char* strategyToString(Strategy st) {
  switch (st) {
    case CUBLAS:
      return "CUBLAS";
    case VANILLA:
      return "VANILLA";
    case SHAREDMEMORY:
      return "SHAREDMEMORY";
    default:
      return "Unknown";
  }
}

int main() {
  cudaDeviceProp prop;
  int count;
  cudaGetDeviceCount(&count);
  for (int i = 0; i < count; i++) {
    cudaGetDeviceProperties(&prop, i);
    std::cout << "Device " << i << ":\n";
    std::cout << "Maximum threads per block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Maximum dimension size of a thread block: (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")\n";
    std::cout << "Maximum dimension size of a grid size: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")\n";
  }

  constexpr int times = 100;
  constexpr Strategy st = SHAREDMEMORY;
  constexpr int M = 3840, K = 2880, N = 3840;
  constexpr bool checkResult = true;
  float accMillis = 0.0;
  for (int i = 0; i <  times; ++i) {
    accMillis += testMatMul<M, K, N>(st, checkResult);
    if (((i + 1) % 10) == 0) {
      printf("Testing process: %d / %d\n", (i + 1), times);
    }
  }
  printf("M=%d, K=%d, N=%d, strategy=%s, bs_m=%d, bs_n=%d, bs_k=%d, MatMul: Totally elapsed time in GPU was %.2f ms, %.2f ms per operation\n", M, K, N, strategyToString(st), BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, accMillis, accMillis / times);
}
