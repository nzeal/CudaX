#include <stdio.h>

__global__
void doLoop()
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	printf("%d\n",i);
}

int main()
{
    doLoop<<<2,10>>>();
    cudaDeviceSynchronize();
}

