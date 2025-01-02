#include <stdio.h>
#include <omp.h>

#define SIZE 1000

int main() {
    int a[SIZE], b[SIZE], sum[SIZE], product[SIZE];

    for (int i = 0; i < SIZE; i++) {
        a[i] = i;
        b[i] = SIZE - i;
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        for (int i = 0; i < SIZE; i++)
          sum[i] = a[i] + b[i];

        #pragma omp section
        for (int i = 0; i < SIZE; i++)
          product[i] = a[i] * b[i];
    }

    printf("sum[0]=%d, product[0]=%d\n", sum[0], product[0]);
    return 0;
}
