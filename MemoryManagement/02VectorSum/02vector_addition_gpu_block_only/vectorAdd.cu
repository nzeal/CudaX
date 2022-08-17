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
void vectorAddGPU(double *a, double *b, double *c) {
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

void checkResult(double *res, int n)
{
	for(int idx = 0; idx < n; idx++){
		if(res[idx] !=3){
			printf("ERROR!\n");
			break;
      }
   }
   printf("\n %f, %f,  %f \n",  res[1] , res[2], res[1023]);
   printf("\n Completed Successfully!\n");
}

int main(int argc, char *argv[]) {
// Number of elements 
    double n = 1024; 

// Host Pointers 
    double *h_x, *h_y, *h_z;

// Device Pointers 
    double *d_x, *d_y,  *d_z;
    
    size_t bytes = n * sizeof(double);
  
  //Allocate memory on host
    h_x = (double*)malloc(bytes);
    h_y = (double*)malloc(bytes);
    h_z = (double*)malloc(bytes);
    
    initVector((double *) h_x, n, 1.0);
    initVector((double *) h_y, n, 2.0);
//    initVector((double *) h_z, n, 0.0);

  //Allocate memory on device 
   cudaMalloc(&d_x,bytes);
   cudaMalloc(&d_y,bytes);
   cudaMalloc(&d_z,bytes);
   
   cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
   cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);

  // z = u + v
  vectorAddGPU<<<n, 1>>>(d_x, d_y, d_z);
  cudaMemcpy(h_z, d_z, bytes, cudaMemcpyDeviceToHost);

  // display the results
  checkResult(h_z,n);

  // free memory 
  free(h_x); free(h_y); free(h_z);
  cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);

  return 0;
}
