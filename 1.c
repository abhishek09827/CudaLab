#include <stdio.h>
#include <omp.h>

int main() {
    long num_steps = 1000000;
    double step = 1.0 / num_steps, pi = 0.0;

    #pragma omp parallel
    {
        double sum = 0.0;
        #pragma omp for
        for (long i = 0; i < num_steps; i++)
            sum += 4.0 / (1.0 + ((i + 0.5) * step) * ((i + 0.5) * step));

        #pragma omp critical
        pi += sum * step;
    }

    printf("PI: %.15f\n", pi);
    return 0;
}
