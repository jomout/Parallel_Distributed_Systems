#include <iostream>
#include <stdlib.h>
#include <pthread.h>
#include <chrono>


#include "csr.cpp"

struct Thread_Params{
    int start;
    int end;
    int num_cl;
    int workLoad;

    pthread_mutex_t *mutex;

    struct COO* o;
    struct CSR* a;
    int* v;
    int* s;

    int *c;

};

struct Thread{
    pthread_t th;

    struct Thread_Params params;

};


struct COO getGraphMinor(struct CSR &A, int &num_cl, int c[]);


int *getClusters(int size, int cl);



int main(){

    // Number of clusters
    int num_cl = 4000;

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



void* clusters_function(void* arg) {

    struct Thread_Params params = *(struct Thread_Params*)arg;

    // Allocating a large chunk of memory to store the sums of the elements that belong to the same cluster
    long double *p = (long double*) malloc(sizeof(long double) * params.num_cl);

    // The indexes of the array p, in which positions the p contains the sums
    int* l = (int*) malloc(sizeof(int) * params.num_cl);

    for (int cluster = params.start; cluster < params.end; cluster++) {

        memset(p, 0, sizeof(long double) * params.num_cl);
        int nz = 0;

        for (int r = 0; r < params.s[cluster]; r++){
            int row = params.v[cluster + (params.num_cl * r)];
            
                for (int k = params.a->I_ptr[row]; k < params.a->I_ptr[row+1]; k++) {
                    if (p[params.c[params.a->J[k]]] == 0){
                        l[nz] = k;
                        nz++;
                    }
                    
                    p[params.c[params.a->J[k]]] += params.a->val[k];
                }
        }

        // Concatenation Process
        pthread_mutex_lock(params.mutex);
        for (int k = 0; k < nz; k++){
            params.o->I[params.o->nz + k] = cluster;
            params.o->J[params.o->nz + k] = params.c[params.a->J[l[k]]];
            params.o->val[params.o->nz + k] = p[params.c[params.a->J[l[k]]]];
        }
        params.o->nz += nz;

        pthread_mutex_unlock(params.mutex);
 
    }

    free(p);
    free(l);

    return NULL;
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
    
    // Allocating a large chunk of memory for fast computations
    output.I = (int*) malloc(sizeof(int) * num_cl * num_cl);
    output.J = (int*) malloc(sizeof(int) * num_cl * num_cl);
    output.val = (long double*) malloc(sizeof(long double) * num_cl * num_cl);

    int threads_num = 4;
    
    // Sets up the threads
    struct Thread thread[threads_num];

    int workLoad = num_cl / threads_num;
    int extra_work = num_cl % threads_num;
    int start = 0;
    
    // Initiallizes Mutex
    pthread_mutex_t mutex;
    pthread_mutex_init(&mutex, NULL);

    // Sets up the thread operations
    for (int i = 0; i < threads_num; i++) {

        // Sets the shared data among the threads
        thread[i].params.workLoad = workLoad;
        thread[i].params.o = &output;
        thread[i].params.v = a;
        thread[i].params.s = s;
        thread[i].params.a = &A;
        thread[i].params.c = c;
        thread[i].params.num_cl = num_cl;
        thread[i].params.mutex = &mutex;

        if (extra_work > 0){
            thread[i].params.workLoad++;
            extra_work--;
        }

        thread[i].params.start = start;
        thread[i].params.end = start + thread[i].params.workLoad;

        start += thread[i].params.workLoad;

        if (pthread_create(&(thread[i].th), NULL, clusters_function, &(thread[i].params)) != 0){
           perror("Failed to create thread");
        }
    }

    // Joins the threads
    for (int i = 0; i < threads_num; i++) {
        if (pthread_join(thread[i].th, NULL) != 0) {
           perror("Failed to join thread");
        }
    }

    //Destroys mutex
    pthread_mutex_destroy(&mutex);
    
    // Shrinks the output arrays
    output.I = (int*) realloc(output.I, sizeof(int) * output.nz);
    output.J = (int*) realloc(output.J, sizeof(int) * output.nz);
    output.val = (long double*) realloc(output.val, sizeof(long double) * output.nz);

    return output;
}