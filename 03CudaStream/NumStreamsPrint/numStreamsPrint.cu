#include <stdio.h>
#include <unistd.h>

__global__ void printNumber(int num) {
  printf("Number from thread %d num = %d\n", threadIdx.x, num);
}

int main() {
  for (int i = 0; i < 5; ++i)
  {
    cudaStream_t stream0;
    cudaStreamCreate(&stream0);
    printNumber<<<1, 2, 0, stream0>>>(i);
    cudaStreamDestroy(stream0);
  }
  cudaDeviceSynchronize();
}

