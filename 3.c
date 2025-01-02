#include <stdio.h>
#include <omp.h>

#define SIZE 1000

int main() {
    int a[SIZE], b[SIZE], c[SIZE];

    for (int i = 0; i < SIZE; i++) {
        a[i] = i;
        b[i] = SIZE - i;
    }

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < SIZE; i++) {
        c[i] = a[i] + b[i];
    }

    printf("First 10 elements of array C:\n");
    for (int i = 0; i < 10; i++) {
        printf("c[%d] = %d\n", i, c[i]);
    }

    return 0;
}
