#include <stdio.h>
#include <cuda_runtime.h>

#define N 3

__global__ void inverse_matrix_kernel(float* mat, float* inv, int n) {
    int idx = threadIdx.x;
    if (idx >= n) return;

    for (int k = 0; k < n; ++k) {
        float pivot = mat[k * n + k];
        if (idx == k) {
            for (int j = 0; j < n; ++j) {
                mat[k * n + j] /= pivot;
                inv[k * n + j] /= pivot;
            }
        }
        __syncthreads();

        if (idx != k) {
            float factor = mat[idx * n + k];
            for (int j = 0; j < n; ++j) {
                mat[idx * n + j] -= factor * mat[k * n + j];
                inv[idx * n + j] -= factor * inv[k * n + j];
            }
        }
        __syncthreads();
    }
}

int main() {
    float h_matrix[N * N] = {4, 1, 2, 1, 5, 3, 2, 3, 6};
    float h_inverse[N * N] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    float *d_matrix, *d_inverse;

    cudaMalloc(&d_matrix, N * N * sizeof(float));
    cudaMalloc(&d_inverse, N * N * sizeof(float));

    cudaMemcpy(d_matrix, h_matrix, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inverse, h_inverse, N * N * sizeof(float), cudaMemcpyHostToDevice);

    inverse_matrix_kernel<<<1, N>>>(d_matrix, d_inverse, N);
    cudaMemcpy(h_inverse, d_inverse, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Inverse Matrix:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%f ", h_inverse[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(d_matrix);
    cudaFree(d_inverse);
    return 0;
}
