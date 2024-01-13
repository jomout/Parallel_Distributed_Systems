#include <iostream>
#include <stdlib.h>

#include <cilk/cilk.h>

#include <chrono>

#include "csr.cpp"

struct CL{
    int cl;
    int idx;
};

struct COO getGraphMinor(struct CSR &A, int &num_cl, int c[]);


int *getClusters(int size, int cl);


pthread_mutex_t mutex;

int main(){


    pthread_mutex_init(&mutex, NULL);

    int num_cl = 4000;


    struct COO A;

    char file[] = "/enter/path/to/matrix";

    getMatrix(&A, file);


    struct CSR B = COOtoCSR(A);
        

    int *c = getClusters(A.M, num_cl);
        

    // Starts Timer
    auto s = std::chrono::high_resolution_clock::now();


    struct COO o = getGraphMinor(B, num_cl, c);

    // Ends Timer
    auto e = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(e - s);

    std::cout << "Duration: " << duration.count() << " microseconds\n";
    free(c);

    pthread_mutex_destroy(&mutex);

    printMatrix(o);
    
    return 0;
}



int *getClusters(int size, int cl){
    int *c = (int*) malloc(sizeof(int) * size);

    for(int i = 0; i < size; i++){
        c[i] = i % cl;
    }
    return c;

}



struct COO getGraphMinor(struct CSR &A, int &num_cl, int c[]){


    struct COO output = {NULL, NULL, NULL, num_cl, num_cl, 0};
    output.I = (int*) malloc(sizeof(int) * num_cl * num_cl);
    output.J = (int*) malloc(sizeof(int) * num_cl * num_cl);
    output.val = (long double*) malloc(sizeof(long double) * num_cl * num_cl);


    int* a = (int*) malloc(sizeof(int) * num_cl * A.M);
    int* s = (int*) malloc(sizeof(int) * num_cl);
    memset(s, 0, sizeof(int) * num_cl);
    

    cilk_for (int i = 0; i < A.M; i++){
        a[c[i] + (num_cl * s[c[i]])] = i;
        s[c[i]]++;
    }
    

    cilk_for (int cluster = 0; cluster < num_cl; cluster++) {

        long double *p = (long double*) malloc(sizeof(long double) * num_cl);
        memset(p, 0, sizeof(long double) * num_cl);

        int* l = (int*) malloc(sizeof(int)*num_cl);
        int nz = 0;

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
        
        pthread_mutex_lock(&mutex);
            for (int k = 0; k < nz; k++){
                output.I[output.nz + k] = cluster;
                output.J[output.nz + k] = c[A.J[l[k]]];
                output.val[output.nz + k] = p[c[A.J[l[k]]]];
            }
            output.nz += nz;
        pthread_mutex_unlock(&mutex);
        
        free(p);
        free(l);
    }
    

    output.I = (int*) realloc(output.I, sizeof(int) * output.nz);
    output.J = (int*) realloc(output.J, sizeof(int) * output.nz);
    output.val = (long double*) realloc(output.val, sizeof(long double) * output.nz);

    
    return output;

}