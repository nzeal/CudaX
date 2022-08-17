#include <stdio.h>

void  initMatrix(float *A, int n, int m, float c) {
  int i,j;
  for (i=0; i<n; i++)
    for (j=0; j<m; j++)
      A[i*m+j] = c;
}

__global__ void gpuMatAdd(float *A, float *B, float *C, int N, int M) 
{
  //----- insert CUDA code -----
  // define index
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  //----------------------------

  //----- insert CUDA code -----
  // check that the thread is not out of the vector boundary
  if (i >= M || j >= N) return;
  //----------------------------

  int index = j*M + i; 
  //----- insert CUDA code -----
  // write the operation for the sum of vectors 
  C[index] = A[index] + B[index];
  //----------------------------
}

int main(int argc, char *argv[]) {

   // size of matrix
  const int N = 1000, M = 1000;

  // allocate memory on host
  float * A = (float *) malloc(N*M * sizeof(float));
  float * B = (float *) malloc(N*M * sizeof(float));
  float * C = (float *) malloc(N*M * sizeof(float));

  initMatrix((float *) A, N, M, 1.0);
  initMatrix((float *) B, N, M, 2.0);
  initMatrix((float *) C, N, M, 0.0);


  //----- insert CUDA code -----
  // allocate memory on device
  float *A_dev, *B_dev, *C_dev;
  cudaMalloc((void **) &A_dev, M*N*sizeof(float));
  cudaMalloc((void **) &B_dev, M*N*sizeof(float));
  cudaMalloc((void **) &C_dev, M*N*sizeof(float));
  //----------------------------

  //----- insert CUDA code -----
  // copy data from host to device
  cudaMemcpy(A_dev, A, M*N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_dev, B, M*N*sizeof(float), cudaMemcpyHostToDevice);
  //----------------------------

  dim3 block(32,32);
  dim3 grid(M/block.x + 1, N/block.y + 1);

  //----- insert CUDA code -----
  // define the execution configuration
  gpuMatAdd<<<grid, block>>>(A_dev, B_dev, C_dev, N, M);
  //----------------------------

  //----- insert CUDA code -----
  // copy data from device to host
  cudaMemcpy(C, C_dev, M*N*sizeof(float), cudaMemcpyDeviceToHost);
  //----------------------------

  printf("%f %f %f\n", C[0], C[1], C[3]);

  //----- insert CUDA code -----
  // free resources on device
  cudaFree(A_dev);
  cudaFree(B_dev);
  cudaFree(C_dev);
  //----------------------------

  // free resources on host
  free(A);
  free(B);
  free(C);

  return 0;
}

