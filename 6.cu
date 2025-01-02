#include <stdio.h>

#define N 10

__global__ void matrix_multiply(int *a, int *b, int *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

__global__ void matrix_transpose(int *input, int *output, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        output[col * n + row] = input[row * n + col];
    }
}

__global__ void matrix_compare(int *a, int *b, int *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n && a[idx] != b[idx]) {
        *result = 0;
    }
}

int main() {
    int a[N][N], b[N][N], c[N][N], ta[N][N], tb[N][N], tc[N][N];
    int *d_a, *d_b, *d_c, *d_ta, *d_tb, *d_tc;
    int size = N * N * sizeof(int);
    int result = 1, *d_result;

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            a[i][j] = i + 1;
            b[i][j] = j + 1;
        }

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    cudaMalloc(&d_ta, size);
    cudaMalloc(&d_tb, size);
    cudaMalloc(&d_tc, size);
    cudaMalloc(&d_result, sizeof(int));

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &result, sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);

    // Transpose A and B
    matrix_transpose<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_ta, N);
    matrix_transpose<<<blocksPerGrid, threadsPerBlock>>>(d_b, d_tb, N);

    // C = A * B
    matrix_multiply<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // TC = TA * TB
    matrix_multiply<<<blocksPerGrid, threadsPerBlock>>>(d_ta, d_tb, d_tc, N);

    // Compare C and TC
    matrix_compare<<<(N * N + 255) / 256, 256>>>(d_c, d_tc, d_result, N);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(tc, d_tc, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Matrix C:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", c[i][j]);
        }
        printf("\n");
    }

    printf("Matrix TC:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", tc[i][j]);
        }
        printf("\n");
    }

    printf("C and TC are %sequal.\n", result ? "" : "not ");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_ta);
    cudaFree(d_tb);
    cudaFree(d_tc);
    cudaFree(d_result);

    return 0;
}
