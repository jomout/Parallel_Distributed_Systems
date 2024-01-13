import numpy as np
import random


def partitionInPlace(A, v):
    n = len(A)
    
    i = 0
    j = n - 1
    
    s = i
    e = j
    
    while(True):
        while (i <=j and A[j] >= v):
            if A[j] == v:
                A[j], A[e] = A[e], A[j]
                e -= 1
            j -= 1
        while (i <= j and A[i] <= v):
            if A[i] == v:
                A[i], A[s] = A[s], A[i]
                s += 1
            i += 1
         
        if (i > j):
            break
        
        A[i], A[j] = A[j], A[i]
        i += 1
        j -= 1
    
    while(e < n - 1):
        e += 1
        A[e], A[i] = A[i], A[e]
        i += 1
        
    while(s > 0):
        s -= 1
        A[s], A[j] = A[j], A[s]
        j -= 1

    return j + 1, i
    


def kselect(A, k):
    if k < 0 or k >= len(A):
        print("Error: k out of bounds")
        exit(1)
    
    p, q = partitionInPlace(A, A[k])
    
    
    if p <= k and k < q:
        return A[p]
    elif k < p:
        return kselect(A[:p], k)
    elif k >= q:
        return kselect(A[q:], k - q)
        


if __name__ == "__main__":
    
    arr = [random.randint(-1000, 1000) for _ in range(100)]
    
    print(f"Array: {arr}")
    print()
    
    k = 49
    result = kselect(arr, k)
    print(f"The {k}th smallest element is:", result)
    print()
    
    a = sorted(arr)
    print(f"sorted arr[{k}]: {a[k]}")