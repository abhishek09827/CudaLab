#include <stdio.h>
#include <cuda_runtime.h>

#define N 3

__global__ void symmetry_check_kernel(float* mat, int* is_symmetric, int n) {
    int i = threadIdx.x;
    int j = threadIdx.y;

    if (i < n && j < n && i < j) {
        if (mat[i * n + j] != mat[j * n + i]) {
            *is_symmetric = 0;
        }
    }
}

int main() {
    float h_matrix[N * N] = {4, 1, 2, 1, 5, 3, 2, 3, 6};
    int h_is_symmetric = 1;
    float *d_matrix;
    int *d_is_symmetric;

    cudaMalloc(&d_matrix, N * N * sizeof(float));
    cudaMalloc(&d_is_symmetric, sizeof(int));

    cudaMemcpy(d_matrix, h_matrix, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_is_symmetric, &h_is_symmetric, sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(N, N);
    symmetry_check_kernel<<<1, threadsPerBlock>>>(d_matrix, d_is_symmetric, N);
    cudaMemcpy(&h_is_symmetric, d_is_symmetric, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_is_symmetric) {
        printf("Matrix is symmetric.\n");
    } else {
        printf("Matrix is not symmetric.\n");
    }

    cudaFree(d_matrix);
    cudaFree(d_is_symmetric);
    return 0;
}
