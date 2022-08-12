#include <stdio.h>
#include <stdlib.h>

void  initVector(double *u, int n, double c) {
  for (int idx=0; idx<n; idx++)
      u[idx] = c;
}

void vectorAddCPU(double *a, double *b, double *c, int n) {
  for(int idx=0;idx<n;idx++)
    c[idx] = a[idx] + b[idx];
}


int main(int argc, char *argv[]) {
  double *x, *y, *z;
  const int N = 1000;
  int size = N * sizeof(double);

  x = (double *) malloc(size);
  y = (double *) malloc(size);
  z = (double *) malloc(size);

  initVector((double *) x, N, 1.0);
  initVector((double *) y, N, 2.0);
  initVector((double *) z, N, 0.0);

  // z = u + v

  vectorAddCPU(x, y, z, N);
  
  for(int idx=0;idx<N;idx++)
    printf("\n %f + %f  = %f",  x[idx] , y[idx], z[idx]);

  free(x); free(y); free(z);
  return 0;
}
