#include <stdio.h>
#include <omp.h>

#define SIZE 1000

int main() {
    int vector[SIZE];
    int sum = 0;
    for (int i = 0; i < SIZE; i++) {
        vector[i] = i + 1;
    }
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < SIZE; i++) {
        sum += vector[i];
    }

    printf("Sum of vector elements: %d\n", sum);
    return 0;
}
