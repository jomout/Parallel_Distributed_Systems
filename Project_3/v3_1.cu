#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

// Function to measure time using CUDA events
void measureTimeCUDAEvent(cudaEvent_t start, cudaEvent_t stop) {
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("Time Elapsed: %f ms\n", elapsedTime);
}


// Initialize Ising model with random initial state
void initialize(int* lattice, int n) {
    srand(1);
    //time(NULL)
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

    extern __shared__ int mem[];

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if ((i >= n) || (j >= n))
        return; 

    int MEM_WIDTH = blockDim.x + 2;

    int memIdx = (threadIdx.y + 1) * MEM_WIDTH  + (threadIdx.x + 1);

    mem[memIdx] = current[i * n + j];

    // Load additional boundary data to shared memory
    if (threadIdx.y == 0) {
        mem[memIdx - MEM_WIDTH] = current[((i - 1 + n) % n) * n + j];
    }

    if (threadIdx.y == blockDim.y - 1 || i == n - 1) {
        mem[memIdx + MEM_WIDTH] = current[((i + 1) % n) * n + j];
    }

    if (threadIdx.x == 0) {
        mem[memIdx - 1] = current[i * n + (j - 1 + n) % n];
    }

    if (threadIdx.x == blockDim.x - 1 || j == n - 1) {
        mem[memIdx + 1] = current[i * n + (j + 1) % n];
    }
    __syncthreads();

    int sum = mem[memIdx - MEM_WIDTH] +
              mem[memIdx + MEM_WIDTH] +
              mem[memIdx - 1] +
              mem[memIdx + 1] +
              mem[memIdx];

    next[i * n + j] = (sum > 0) - (sum < 0);

    // FOR CORRECTION
    // printf("Block: %d,%d - Thread: %d,%d - i: %d, j: %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, i, j);
}

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

    // Memory allocation
    int* lattice1 = (int*)malloc(size);
    int* lattice2 = (int*)malloc(size);


    // Initialize Ising Model 
    initialize(lattice1, n);
    printf("\nInitial state:\n");
    //printLattice(lattice1, n);


    int *d_lattice1, *d_lattice2;
    cudaMalloc(&d_lattice1, size);
    cudaMalloc(&d_lattice2, size);

    cudaMemcpy(d_lattice1, lattice1, size, cudaMemcpyHostToDevice);

    int memSize = (BLOCK_WIDTH + 2) * (BLOCK_WIDTH + 2) * sizeof(int);

    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    dim3 dimGrid(ceil(n / (float)BLOCK_WIDTH), ceil(n / (float)BLOCK_WIDTH), 1);

    // Record the start event
    cudaEventRecord(start, 0);

    for (int i = 0; i < k; i++) {
        //printf("Iteration %d:\n", i);
        //printLattice(lattice1, n);

        simulate<<<dimGrid, dimBlock, memSize>>>(d_lattice1, d_lattice2, n);

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

    cudaMemcpy(lattice1, d_lattice1, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(lattice2, d_lattice2, size, cudaMemcpyDeviceToHost);

    // Final state
    printf("\nFinal state:\n");
    //printLattice(lattice1, n);

    // Save array to file
    saveArrayToFile(lattice1, n * n, "v3_1.bin");

    // Cleanup
    free(lattice1);
    free(lattice2);

    cudaFree(d_lattice1);
    cudaFree(d_lattice2);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}