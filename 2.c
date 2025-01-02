#include <stdio.h>
#include <omp.h>

int main() {
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int total_threads = omp_get_num_threads();
        int max_threads = omp_get_max_threads();
        int num_procs = omp_get_num_procs();
        int in_parallel = omp_in_parallel();

        #pragma omp critical
        {
            printf("Thread ID: %d\n", thread_id);
            printf("Total Threads: %d\n", total_threads);
            printf("Max Threads: %d\n", max_threads);
            printf("Number of Processors: %d\n", num_procs);
            printf("In Parallel: %d\n", in_parallel);
            printf("---------------------------\n");
        }
    }

    return 0;
}
