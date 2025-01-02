#include <stdio.h>
#include <omp.h>

int main() {
    int x = 10; // Shared variable

    printf("Initial value of x: %d\n", x);

    #pragma omp parallel firstprivate(x)
    {
        int thread_id = omp_get_thread_num();
        x += thread_id;
        printf("Thread %d: x = %d\n", thread_id, x);
    }

    printf("Value of x after parallel region: %d\n", x);

    return 0;
}
