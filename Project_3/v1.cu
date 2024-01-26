#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>


// Μeasure time using CUDA events
void measureTimeCUDAEvent(cudaEvent_t start, cudaEvent_t stop) {
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("Time Elapsed: %f ms\n", elapsedTime);
}

// Initialize Ising model with random initial state
void initialize(int* lattice, int n) {
    srand(time(NULL));
    // Random +1 or -1
    for (int i = 0; i < n * n; i++) {
        lattice[i] = rand() % 2 * 2 - 1;
    }
}

// Print the current state of the Ising model
void printLattice(int* lattice, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%2d ", lattice[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Perform Ising model simulation
__global__ void simulate(int* current, int* next, int n) {

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i >= n) || (j >= n))
        return;
    
    int sum = current[((i - 1 + n) % n) * n + j] +
                        current[((i + 1) % n) * n + j] +
                        current[i * n + j] +
                        current[i * n + (j - 1 + n) % n] +
                        current[i * n + (j + 1) % n];

    // FOR CORRECTION
    // printf("Block: %d,%d - Thread: %d,%d - i: %d, j: %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, i, j);

    next[i * n + j] = (sum > 0) - (sum < 0);
}

// Saving array for correction later
void saveArrayToFile(int* array, int size, const char* filename) {
    FILE* file = fopen(filename, "wb");

    if (file != NULL) {
        fwrite(array, sizeof(int), size, file);

        fclose(file);

    } else {
        fprintf(stderr, "Error opening file: %s\n", filename);
    }
}


int main() {
    int n, k;

    int BLOCK_WIDTH;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    

    // Get the size and number of iters
    printf("Enter the width of the Ising model (n): ");
    scanf("%d", &n);
    printf("Enter the number of iterations (k): ");
    scanf("%d", &k);
    printf("Define Block Width (BLOCK_WIDTH): ");
    scanf("%d", &BLOCK_WIDTH);


    size_t size = n * n * sizeof(int);

    // Memery allocation
    int* lattice1 = (int*)malloc(size);
    int* lattice2 = (int*)malloc(size);


    // Initialize Ising Model 
    initialize(lattice1, n);
    printf("\nInitial state:\n");
    //printLattice(lattice1, n);

    // GPU arrays
    int *d_lattice1, *d_lattice2;
    cudaMalloc(&d_lattice1, size);
    cudaMalloc(&d_lattice2, size);

    cudaMemcpy(d_lattice1, lattice1, size, cudaMemcpyHostToDevice);


    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    dim3 dimGrid(ceil(n / (float)BLOCK_WIDTH), ceil(n / (float)BLOCK_WIDTH), 1);


    // Record the start event
    cudaEventRecord(start, 0);

    for (int i = 0; i < k; i++) {
        //printf("Iteration %d:\n", i);
        //printLattice(lattice1, n);

        simulate<<<dimGrid, dimBlock>>>(d_lattice1, d_lattice2, n);

        // Swap lattices
        int* temp = d_lattice1;
        d_lattice1 = d_lattice2;
        d_lattice2 = temp;

        cudaDeviceSynchronize();
    }
    
    // Record the stop event
    cudaEventRecord(stop, 0);

    // Measure and print the execution time
    measureTimeCUDAEvent(start, stop);

    cudaMemcpy(lattice1, d_lattice1, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(lattice2, d_lattice2, n * n * sizeof(int), cudaMemcpyDeviceToHost);

    // Final state
    printf("\nFinal state:\n");
    //printLattice(lattice1, n);

    // Save array to file
    saveArrayToFile(lattice1, n * n, "v1.bin");

    // Cleanup
    free(lattice1);
    free(lattice2);

    cudaFree(d_lattice1);
    cudaFree(d_lattice2);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}