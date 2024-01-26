#include <stdio.h>
#include <stdlib.h>
#include <time.h>


int mysign(int x){
    return (x > 0) - (x < 0);
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
void simulate(int* current, int* next, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int sum = current[((i - 1 + n) % n) * n + j] +
                      current[((i + 1) % n) * n + j] +
                      current[i * n + j] +
                      current[i * n + (j - 1 + n) % n] +
                      current[i * n + (j + 1) % n];

            next[i * n + j] = (sum > 0) - (sum < 0);
        }
    }
}

void saveArrayToFile(int* array, int size, const char* filename) {
    FILE* file = fopen(filename, "wb");

    if (file != NULL) {
        // Write array data to the file
        fwrite(array, sizeof(int), size, file);

        // Close the file
        fclose(file);

        printf("Array saved to file: %s\n", filename);
    } else {
        fprintf(stderr, "Error opening file: %s\n", filename);
    }
}

int main() {
    int n, k;

    // Get the size and number of iters
    printf("Enter the size of the Ising model (n): ");
    scanf("%d", &n);
    printf("Enter the number of iterations (k): ");
    scanf("%d", &k);

    // Mem allocation
    int* lattice1 = (int*)malloc(n * n * sizeof(int));
    int* lattice2 = (int*)malloc(n * n * sizeof(int));

    // Initialize Ising Model 
    initialize(lattice1, n);
    printf("\nIsing Model:\nInitial State:\n");
    //printLattice(lattice1, n);

    for (int i = 0; i < k; i++) {
        printf("Iteration %d:\n", i);
        //printLattice(lattice1, n);

        simulate(lattice1, lattice2, n);

        // Swap lattices
        int* temp = lattice1;
        lattice1 = lattice2;
        lattice2 = temp;
    }

    // Final state
    printf("\nIsing Model:\nFinal State:\n");
    //printLattice(lattice1, n);

    // Save array to file
    saveArrayToFile(lattice1, n * n, "array_data.bin");

    // Free mem
    free(lattice1);
    free(lattice2);

    return 0;
}