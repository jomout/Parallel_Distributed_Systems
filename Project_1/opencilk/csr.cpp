#include <iostream>
#include "coo.cpp"

struct CSR{

    int *I_ptr;
    int *J;
    long double *val;

    int M;
    int N;
    int nz;
};

// Converts a COO matrix to CSR
struct CSR COOtoCSR(struct COO A){

    struct CSR output;
    output.nz = A.nz;
    output.M = A.M;
    output.N = A.N;

    output.I_ptr = (int *) malloc((output.M + 1) * sizeof(int));
    output.J = (int *) malloc(output.nz * sizeof(int));
    output.val = (long double *) malloc(output.nz * sizeof(long double));

    for (int i = 0; i < (output.M + 1); i++){
        output.I_ptr[i] = 0;
    }

    for (int i = 0; i < A.nz; i++){
        output.val[i] = A.val[i];
        output.J[i] = A.J[i];
        output.I_ptr[A.I[i] + 1]++;
    }

    for (int i = 0; i < output.M; i++){
        output.I_ptr[i + 1] += output.I_ptr[i];
    }

    return output;
}