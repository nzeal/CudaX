#include <stdio.h>

void onCPU()
{
  printf("This function prints: Hello on CPU.\n");
}

__global__ void onGPU()
{
  printf("This function prints: Hello on GPU.\n");
}

int main()
{
  onCPU();
  onGPU<<<1, 1>>>();

  cudaDeviceSynchronize();
}
