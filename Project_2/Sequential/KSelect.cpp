#include <iostream>

typedef struct Pointers{
    int p;
    int q;
} Pointers;

Pointers partitioninplace(int A[], int v, int left, int right);

void swap(int *a, int *b);

int kselect(int A[], int k, int left, int right);

int main(){

    int arr[] = { 10, 4, 5, 8, 6, 11, 26, 9, 23, 56, 2, 11 }; 
    int n = sizeof(arr) / sizeof(arr[0]); 
    int k = 0; 

    int value = kselect(arr, k, 0, n - 1);

    fprintf(stdout, "The %d-th smallest value is: %d\n", k, value);

    int arr2[] = { 10, 4, 5, 8, 6, 11, 26, 9, 23, 56, 2, 11 }; 
    int m = sizeof(arr2) / sizeof(arr[0]); 

    auto res = partitioninplace(arr2, 0, 0, m-1);
    



    return 0;
}

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}



Pointers partitioninplace(int A[], int v, int left, int right){
    
    Pointers res;

    int i = left;
    int j = right;
    
    int s = i;
    int e = j;
    
    while(1){
        while (i <=j && A[j] >= v){
            if (A[j] == v){
                swap(&A[j], &A[e]);
                e--;
            }
            j--;
        }
        while (i <= j && A[i] <= v){
            if (A[i] == v){
                swap(&A[i], &A[s]);
                s++;
            }
            i++;
        }
        if (i > j)
            break;
        
        swap(&A[i], &A[j]);
        i++;
        j--;
    }
    while(e < right){
        e++;
        swap(&A[i], &A[e]);
        i++;
    }

    while(s > left){
        s--;
        swap(&A[j], &A[s]);
        j--;
    }

    res.p = j + 1;
    res.q = i;

    return res;
 
}

int kselect(int A[], int k, int left, int right){
    int n = right + 1;
    if (k < 0 || k >= n){
        printf("Error: k out of bounds");
        exit(1);
    }
    
    Pointers res = partitioninplace(A, A[k], left, right);

    if (k < res.p){
        return kselect(A, k, left, res.p - 1);
    }  
    else if (k >= res.q){
        return kselect(A, k , res.q, right);
    }
    
    return A[res.p];
}