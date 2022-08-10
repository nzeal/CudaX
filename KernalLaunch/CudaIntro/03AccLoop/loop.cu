#include <stdio.h>

__global__
void doLoop() {
    printf("This is iteration number %d\n", threadIdx.x);
}

int main()
{
    doLoop<<<1,10>>>();
    cudaDeviceSynchronize();
}

