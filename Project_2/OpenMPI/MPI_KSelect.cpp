#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <vector>
#include <algorithm>
#include <memory>
#include <random>
#include <ctime>
#include <chrono>
#include <limits>
#include <iostream>
#include <cstdint>

#include "mpi.h"

#define COMPUTE 10
#define STOP -10
#define LEFT -1
#define RIGHT 1


typedef struct Pointers{
    int64_t p;
    int64_t q;
} Pointers;



Pointers partitioninplace(std::vector<uint32_t> &A, uint32_t v, int64_t left, int64_t right);

std::vector<uint32_t> getfile(const char* file_path, int rank, int size);



int main(int argc, char *argv[]) {

    // Filename of data array
    const char *filename = "/path/to/dataArray";

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the total number of tasks
    int num_tasks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);

    // Get the task ID
    int task_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &task_id);

    // Master process
    int master = 0;

    // Receive buffer
    std::vector<uint32_t> buffer = getfile(filename, task_id, num_tasks);
    int64_t left = 0;
    int64_t right = buffer.size() - 1;

    uint32_t pivot;
    uint32_t answer;

    // Flags
    int flag = COMPUTE;
    int operation;
    int done = 0;

    // Pointers of struct
    Pointers global;
    Pointers local;

    // K-select
    int64_t k = 0;

    // Boundaries of values
    uint32_t low = std::numeric_limits<uint32_t>::min();
    uint32_t high = std::numeric_limits<uint32_t>::max();

    // Random Generator
    std::random_device rd;
    std::mt19937 gen(rd());

    // Synchronizing processes before search
    MPI_Barrier(MPI_COMM_WORLD);

    // Starting Timer
    auto start = std::chrono::high_resolution_clock::now();

    while(flag != STOP){
        if (task_id == master){
            std::uniform_int_distribution<uint32_t> distribution(low, high);

            pivot = distribution(gen);
        }

        // Broadcasting pivot
        MPI_Bcast(&pivot, 1, MPI_UINT32_T, master, MPI_COMM_WORLD);

        if (!done){
	        local = partitioninplace(buffer, pivot, left, right);
        }
            
        // Getting the global pointers
        MPI_Reduce(&local.p, &global.p, 1, MPI_INT64_T, MPI_SUM, master, MPI_COMM_WORLD);
        MPI_Reduce(&local.q, &global.q, 1, MPI_INT64_T, MPI_SUM, master, MPI_COMM_WORLD);
        
        // Check if answer found, else choose direction
        if (task_id == master){
            if (global.p <= k && k < global.q){
                answer = pivot;
                flag = STOP;
            }
            else if (k < global.p){
                operation = LEFT;
                high = pivot;
            }
            else if (k >= global.q){
                operation = RIGHT;
                low = pivot;
            }
        }

        // Broadcasting the FLAG message and the OPERATION message
        MPI_Bcast(&flag, 1, MPI_INT, master, MPI_COMM_WORLD);
        MPI_Bcast(&operation, 1, MPI_INT, master, MPI_COMM_WORLD);

        // Determine which sub-array to search
        if (!done && operation == LEFT){
            right = local.p - 1;
        }
        else if (!done && operation == RIGHT){
            left = local.q;
        }

        // Checking if process has nothing left to do
        if (!done && left > right){
            done = 1;
            local.p = left;
            local.q = left;
        }
    }

    // Stopping Timer
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    
    // Master prints the result
    if (task_id == master){
        std::cout << "The " << k << "-th smallest element is: " << answer << "\n";
        std::cout << "Time taken by function: "
         << duration.count() << " microseconds" << std::endl;
    }

    // Finish our MPI work
    MPI_Finalize();
    return 0;
}



Pointers partitioninplace(std::vector<uint32_t> &A, uint32_t v, int64_t left, int64_t right){
    
    Pointers res;
        
    // Pointers of the subarray
    int64_t i = left;
    int64_t j = right;
    
    // Pointers of the elements equal to pivot at start and end of the subarray
    int64_t s = i;
    int64_t e = j;
    
    while(1){
        while (i <=j && A[j] >= v){
            if (A[j] == v){
                std::swap(A[j], A[e]);
                e--;
            }
            j--;
        }
        while (i <= j && A[i] <= v){
            if (A[i] == v){
                std::swap(A[i], A[s]);
                s++;
            }
            i++;
        }
        if (i > j)
            break;
        
        std::swap(A[i], A[j]);
        i++;
        j--;
    }

    // Moving each element equal to pivot in the middle 
    while(e < right){
        e++;
        std::swap(A[i], A[e]);
        i++;
    }
    while(s > left){
        s--;
        std::swap(A[j], A[s]);
        j--;
    }

    res.p = j + 1;
    res.q = i;

    return res;
}



std::vector<uint32_t> getfile(const char* file_path, int rank, int size) {
    MPI_File mpi_file;
    MPI_Offset file_size;

    // Open the file collectively
    MPI_File_open(MPI_COMM_WORLD, file_path, MPI_MODE_RDONLY, MPI_INFO_NULL, &mpi_file);

    // Get the size of the file
    MPI_File_get_size(mpi_file, &file_size);

    // Determine the number of uint32_t elements
    size_t num_elements = file_size / sizeof(uint32_t);

    // Determine the size of each partition
    size_t partition_size = num_elements / size;

    // Calculate the start and end offsets for every process
    MPI_Offset start_element = rank * partition_size;
    MPI_Offset end_element = (rank == size - 1) ? num_elements : (rank + 1) * partition_size;

    // Calculate the size of the partition for every process
    size_t local_partition_size = end_element - start_element;

    // Create a buffer for the local partition
    std::vector<uint32_t> local_partition(local_partition_size);


    // Write the partition into the buffer
    MPI_File_set_view(mpi_file, start_element * sizeof(uint32_t), MPI_UINT32_T, MPI_UINT32_T, "native", MPI_INFO_NULL);
    MPI_File_read_all(mpi_file, local_partition.data(), local_partition_size, MPI_UINT32_T, MPI_STATUS_IGNORE);

    // Close file
    MPI_File_close(&mpi_file);


    return local_partition;
}