#include <stdio.h>

#define N 1000

__global__ void vector_dot_product(int *a, int *b, int *result) {
    __shared__ int temp[1024];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_id = threadIdx.x;
    temp[thread_id] = (idx < N) ? a[idx] * b[idx] : 0;
    __syncthreads();

    if (thread_id == 0) {
        int sum = 0;
        for (int i = 0; i < blockDim.x; i++) sum += temp[i];
        atomicAdd(result, sum);
    }
}

int main() {
    int a[N], b[N], result = 0;
    int *d_a, *d_b, *d_result;

    for (int i = 0; i < N; i++) {
        a[i] = i + 1;
        b[i] = N - i;
    }

    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));
    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &result, sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vector_dot_product<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result);
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Dot product: %d\n", result);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_result);
    return 0;
}
