#include <stdio.h>

#define N 10

__global__ void matrix_add(int *a, int *b, int *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        int idx = row * n + col;
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int a[N][N], b[N][N], c[N][N], *d_a, *d_b, *d_c;
    int size = N * N * sizeof(int);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            a[i][j] = i + j;
            b[i][j] = i - j;
        }
    printf("Matrix A:\n");
    for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        printf("%d ", a[i][j]);
    }
    printf("\n");
}

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);
    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    printf("Result matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", c[i][j]);
        }
        printf("\n");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
