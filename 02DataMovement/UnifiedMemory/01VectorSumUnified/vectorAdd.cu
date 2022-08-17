// ======================================
//  Vector addition on GPU using CUDA
//  h_ stands things on the host
//  d_a stands things on the device
// ======================================

#include <stdio.h>
#include <stdlib.h>

void  initVector(double *u, int n, double c) {
  for (int idx=0; idx<n; idx++)
      u[idx] = c;
}

__global__
void vectorAddGPU(double *a, double *b, double *c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;   // handles the data at this index
  if (idx<n)
    c[idx] = a[idx] + b[idx];
}

void checkResult(double *res, int n)
{
	for(int i = 0; i < n; i++){
		if(res[i] !=3){
			printf("ERROR!\n");
			break;
      }
   }
   printf("\n %f, %f,  %f \n",  res[1] , res[2], res[100]);
   printf("\n Completed Successfully!\n");
}

int main(int argc, char *argv[]) {
// Number of elements 
    double n = 1024; 

// Host Pointers 
    double *h_x, *h_y, *h_z;
    
    size_t bytes = n * sizeof(double);
  
  //Allocate memory on host
    cudaMallocManaged( &h_x, bytes );
    cudaMallocManaged( &h_y, bytes );
    cudaMallocManaged( &h_z, bytes );

  // Get the device id for prefetching calls 
    int id = cudaGetDevice(&id);

   // Set some hints about eh data and do some prefetching 
    cudaMemAdvise(h_x, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(h_y, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemPrefetchAsync(h_z, bytes, id);
    
    initVector((double *) h_x, n, 1.0);
    initVector((double *) h_y, n, 2.0);

// Pre-fetch 'h_x' and 'h_y' arrays to the specified device (GPU)
    cudaMemAdvise(h_x, bytes, cudaMemAdviseSetReadMostly, id);
    cudaMemAdvise(h_y, bytes, cudaMemAdviseSetReadMostly, id);
    cudaMemPrefetchAsync(h_x, bytes, id);
    cudaMemPrefetchAsync(h_y, bytes, id);

   //init block and grid size
   int num_threads = 1<<10;
   int block_size = ( n + num_threads - 1) / num_threads;
   int grid_size  = (int)ceil((float) n /block_size);
   printf("Grid size is %d\n", grid_size);


  // z = u + v
  vectorAddGPU<<<grid_size, block_size>>>(h_x, h_y, h_z, n);

    cudaDeviceSynchronize();

  // Prefetch to the host (CPU)
  cudaMemPrefetchAsync(h_x, bytes, cudaCpuDeviceId);
  cudaMemPrefetchAsync(h_y, bytes, cudaCpuDeviceId);
  cudaMemPrefetchAsync(h_z, bytes, cudaCpuDeviceId);

  // display the results
  checkResult(h_z,n);

  // free memory 

  cudaFree(h_x); cudaFree(h_y); cudaFree(h_z);

  return 0;
}
