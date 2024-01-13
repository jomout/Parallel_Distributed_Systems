#include <stdio.h>
#include "mmio.c"

struct COO{
    int *I;
    int *J;
    long double *val;

    int M;
    int N;
    int nz;
};

void getMatrix(struct COO *A, char *file){

    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nz;   
    int i;

    
    if ((f = fopen(file, "r")) == NULL) 
        exit(1);


    if (mm_read_banner(f, &matcode) != 0) {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }


    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && mm_is_sparse(matcode) ) {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    /* find out size of sparse matrix .... */

    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
        exit(1);

    A->M = M;
    A->N = N;
    A->nz = nz;

    /* reseve memory for matrices */

    A->I = (int *) malloc(nz * sizeof(int));
    A->J = (int *) malloc(nz * sizeof(int));
    A->val = (long double *) malloc(nz * sizeof(long double));


    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i=0; i<nz; i++) {
        fscanf(f, "%d %d %Lf\n", &A->J[i], &A->I[i], &A->val[i]);
        A->I[i]--;  /* adjust from 1-based to 0-based */
        A->J[i]--;
    }

    if (f != stdin)
        fclose(f);
}


void printMatrix(struct COO A){

    FILE *outputFile;

    // Open a text file for writing
    outputFile = fopen("output.txt", "w");
    
    printf("------------------------------------\n");
    printf("Printing Matrix\n");
    printf("------------------------------------\n");

    for (int i = 0; i < A.nz; i++){
        fprintf(outputFile, "%d %d %Lf\n", A.I[i], A.J[i], A.val[i]);
    }
    
    fclose(outputFile);
}