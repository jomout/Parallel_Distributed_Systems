#include <stdio.h>
#include <stdlib.h>

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

void loadArrayFromFile(int** array, int* size, const char* filename) {
    FILE* file = fopen(filename, "rb");

    if (file != NULL) {
        // Determine the size of the file
        fseek(file, 0, SEEK_END);
        *size = ftell(file) / sizeof(int);
        fseek(file, 0, SEEK_SET);

        // Allocate memory for the array
        *array = (int*)malloc(*size * sizeof(int));

        // Read array data from the file
        fread(*array, sizeof(int), *size, file);

        // Close the file
        fclose(file);

        printf("Array loaded from file: %s\n", filename);
    } else {
        fprintf(stderr, "Error opening file: %s\n", filename);
    }
}

int main() {

    // Load array from file
    int* seq = NULL;
    int seqSize = 0;
    loadArrayFromFile(&seq, &seqSize, "array_data.bin");

    int* v = NULL;
    int vSize = 0;
    loadArrayFromFile(&v, &vSize, "v3_2.bin");

    // Compare arrays
    int arraysEqual = 1;
    for (int i = 0; i < (seqSize < vSize ? seqSize : vSize); ++i) {
        if (seq[i] != v[i]) {
            arraysEqual = 0;
            break;
        }
    }

    if (arraysEqual) {
        printf("Arrays are equal.\n");
    } else {
        printf("Arrays are not equal.\n");
    }

    // Free allocated memory
    free(seq);
    free(v);

    return 0;
}