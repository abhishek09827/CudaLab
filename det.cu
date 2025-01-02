#include <stdio.h>
#include <cuda_runtime.h>

#define N 3

__global__ void determinant_kernel(float* mat, float* det, int n) {
    int idx = threadIdx.x;
    if (idx >= n) return;

    // Simple row elimination for LU decomposition
    for (int k = 0; k < n; ++k) {
        if (idx > k && mat[k * n + k] != 0) {
            float factor = mat[idx * n + k] / mat[k * n + k];
            for (int j = k; j < n; ++j) {
                mat[idx * n + j] -= factor * mat[k * n + j];
            }
        }
        __syncthreads();
    }

    if (idx == 0) {
        *det = 1.0f;
        for (int i = 0; i < n; ++i) {
            *det *= mat[i * n + i];
        }
    }
}

int main() {
    float h_matrix[N * N] = {4, 1, 2, 1, 5, 3, 2, 3, 6};
    float h_det = 0;
    float *d_matrix, *d_det;

    cudaMalloc(&d_matrix, N * N * sizeof(float));
    cudaMalloc(&d_det, sizeof(float));

    cudaMemcpy(d_matrix, h_matrix, N * N * sizeof(float), cudaMemcpyHostToDevice);

    determinant_kernel<<<1, N>>>(d_matrix, d_det, N);
    cudaMemcpy(&h_det, d_det, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Determinant: %f\n", h_det);

    cudaFree(d_matrix);
    cudaFree(d_det);
    return 0;
}
