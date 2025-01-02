#include <stdio.h>
#include <omp.h>

#define N 500

int main() {
    int A[N][N], B[N][N], C[N][N] = {0};

    // Initialize matrices A and B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = i + j;
            B[i][j] = i - j;
        }
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    printf("C[0][0] = %d\n", C[0][0]);

    return 0;
}
