#include <iostream>
#include <stdlib.h>
#include <omp.h>
#include <chrono>

#include "csr.cpp"


struct COO getGraphMinor(struct CSR &A, int &num_cl, int c[]);

int *getClusters(int size, int cl);


int main(){

    // Number of clusters
    int num_cl = 4000;

    // Initializes Threads
    omp_set_num_threads(4);

    struct COO A;

    char file[] = "/enter/path/to/matrix";

    getMatrix(&A, file);

    // Converts matrix A to CSR format
    struct CSR B = COOtoCSR(A);
        

    int *c = getClusters(A.M, num_cl);
        
    // Starts Timer
    auto s = std::chrono::high_resolution_clock::now();

    struct COO o = getGraphMinor(B, num_cl, c);

    // Stops Timer
    auto e = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(e - s);

    std::cout << "Duration: " << duration.count() << " microseconds\n";
    free(c);

    // Prints output matrix to an output.txt file
    printMatrix(o);
        

    return 0;

}


// Initialization of clusters which will have to be controllable to compare with other implementations
// So no random initiazation
int *getClusters(int size, int cl){
    int *c = (int*) malloc(sizeof(int) * size);

    for(int i = 0; i < size; i++){

        c[i] = i % cl;

    }
    return c;
}



struct COO getGraphMinor(struct CSR &A, int &num_cl, int c[]){

    // Initializing output
    struct COO output = {NULL, NULL, NULL, num_cl, num_cl, 0};

    // Allocating a large chunk of memory to make fast computations
    output.I = (int*) malloc(sizeof(int) * num_cl * num_cl);
    output.J = (int*) malloc(sizeof(int) * num_cl * num_cl);
    output.val = (long double*) malloc(sizeof(long double) * num_cl * num_cl);

    // Initiallizing the array which contains the groups of the rows according to their cluster
    int* a = (int*) malloc(sizeof(int) * num_cl * A.M);

    // Initiallizing the array which contains the sizes of each group
    int* s = (int*) malloc(sizeof(int) * num_cl);
    memset(s, 0, sizeof(int) * num_cl);
    
    // Grouping the rows
    for (int i = 0; i < A.M; i++){
        a[c[i] + (num_cl * s[c[i]])] = i;
        s[c[i]]++;
    }
    
    #pragma omp parallel for
    for (int cluster = 0; cluster < num_cl; cluster++) {

        // Allocating a large chunk of memory to store the sums of the elements that belong to the same cluster
        long double *p = (long double*) malloc(sizeof(long double) * num_cl);
        memset(p, 0, sizeof(long double) * num_cl);

        // The indexes of the array p, in which positions the p contains the sums
        int* l = (int*) malloc(sizeof(int)*num_cl);
        int nz = 0;

        // iterating over the rows of each cluster
        for (int r = 0; r < s[cluster]; r++){
            int row = a[cluster + (num_cl * r)];

            for (int k = A.I_ptr[row]; k < A.I_ptr[row+1]; k++) {
                if (p[c[A.J[k]]] == 0){
                    l[nz] = k;
                    nz++;
                }
                        
                p[c[A.J[k]]] += A.val[k];
            }
        }

        // Concatenation of the results
        #pragma omp critical
        {
            for (int k = 0; k < nz; k++){
                output.I[output.nz + k] = cluster;
                output.J[output.nz + k] = c[A.J[l[k]]];
                output.val[output.nz + k] = p[c[A.J[l[k]]]];
            }
            output.nz += nz;
        }
        
        free(p);
        free(l);
    }
    
    // Shrinking the output arrays
    output.I = (int*) realloc(output.I, sizeof(int) * output.nz);
    output.J = (int*) realloc(output.J, sizeof(int) * output.nz);
    output.val = (long double*) realloc(output.val, sizeof(long double) * output.nz);


    return output;
}