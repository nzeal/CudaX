// ======================================
//  Vector addition on GPU using CUDA
//  h_ stands things on the host
//  d_a stands things on the device
// ======================================

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__
void vectorAddGPU(double *a, double *b, double *c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;   // handles the data at this index
  if (idx<n) {
    c[idx] = a[idx] + b[idx];
  }
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
//   printf("\n Completed Successfully!\n");
}

void initVector(double *u, int n, double c);
void vectorAddCPU(double *a, double *b, double *c, int n);

int main (void) {
// Number of elements 
    double n = 2097152; 
     printf("Number of elements: %f\n", n);
    size_t bytes = n * sizeof(double);

// Host and device pointers 
    double *h_x, *h_y, *h_z;
    double *d_x, *d_y,  *d_z;
    
// CudaEvent records 
  cudaEvent_t start, end;
  float eventEtime;

//Allocate memory on host
    h_x = (double*)malloc(bytes);
    h_y = (double*)malloc(bytes);
    h_z = (double*)malloc(bytes);

// Initialization of host buffers    
    initVector((double *) h_x, n, 1.0);
    initVector((double *) h_y, n, 2.0); 

//Allocate memory on device 
   cudaMalloc(&d_x,bytes);
   cudaMalloc(&d_y,bytes);
   cudaMalloc(&d_z,bytes);

// creation of cuda events: start, end
   cudaEventCreate(&start);
   cudaEventCreate(&end);

   printf ("\nGPU computation ... ");
   cudaEventRecord(start,0);

// Insert CUDA code ---   
   cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
   cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);

//init block 
  int nThreads = 256;
  int nBlocks;

// calculate block number
  nBlocks = (n-1) / nThreads + 1;
  printf("GPU execution with %d blocks each one of %d threads\n", nBlocks, nThreads);


// z = u + v
  vectorAddGPU<<<nBlocks,nThreads>>>(d_x, d_y, d_z, n);

// copy back of results from device  
  cudaMemcpy(h_z, d_z, bytes, cudaMemcpyDeviceToHost);
//------------------------------------>

  cudaEventRecord(end,0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&eventEtime, start, end);

  printf ("ok\n");
  printf("Elapsed time on GPU: %.2f ms\n", eventEtime);

//-- Host computation --- 
  printf("\nCPU computation ... ");
  double *cpuResult;
  float eventTimeCPU;
  cudaMallocHost((void**)&cpuResult, bytes);
  cudaEventRecord(start,0);

  vectorAddCPU(h_x, h_y, cpuResult, n);

  cudaEventRecord(end,0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&eventTimeCPU, start, end);
  printf ("ok\n");
  printf("Elapsed time on CPU: %.2f ms\n", eventTimeCPU);
  printf("\nSpeed UP CPU/GPU %.1fx\n", eventTimeCPU/eventEtime);

  printf("\nCheck results on GPU:\n");
  checkResult( h_z, n);
  printf("\nCheck results on CPU:\n");
  checkResult( cpuResult, n);


  // free resources on device
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);

  // free memory 
  free(h_x); free(h_y); free(h_z);

  return 0;
}

void initVector(double *u, int n, double c) {
  for (int idx=0; idx<n; idx++)
      u[idx] = c;
}

void vectorAddCPU(double *a, double *b, double *c, int n) {
	for(int idx=0; idx < n; ++idx)
		c[idx] = a[idx] + b[idx];
}

