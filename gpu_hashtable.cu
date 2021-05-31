#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>
#include "test_map.hpp"
#include "gpu_hashtable.hpp"

using namespace std;

#define DEFAULT_KEY 0
#define DEFAULT_VALUE 0
#define NUM_THREADS 1024
#define MAX_LOAD 0.9f
#define MIN_LOAD 0.7f

int numBlocks;

/*
Allocate CUDA memory only through glbGpuAllocator
cudaMalloc -> glbGpuAllocator->_cudaMalloc
cudaMallocManaged -> glbGpuAllocator->_cudaMallocManaged
cudaFree -> glbGpuAllocator->_cudaFree
*/

__device__ int hashFunc(int value, int limit) {
    return ((value * 31) % 113) % limit;
}            

__global__ void kernel_insertBatch(HashTable *htb, int *keys, int* values, int numKeys) {
	// Compute the global element index this thread should process
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Keys and values must be grater that 0
	if (idx >= numKeys || keys[idx] <= 0 || values[idx] <= 0) {
		return;
	}

	// Get key, value and initial hash code
	unsigned int key = keys[idx];
	unsigned int value = values[idx];
	int hashCode = hashFunc(key, htb->size);

	int count = 0;		// Variable to make sure it doesn't loop

	while (count < htb->size) {
		// Exchange the old key with the new one
		unsigned int old_key = atomicExch(&(htb->table[hashCode].key), key);

		// This position in the HashTable was not used, therefore add also the value
		if (old_key == DEFAULT_KEY) {
			atomicExch(&(htb->table[hashCode].value), value);
			break;
		}

		// The key already exists, update the value
		else if (old_key == key) {
			atomicExch(&(htb->table[hashCode].value), value);
			break;
		}

		// This position is already used, put the old key back
		atomicExch(&(htb->table[hashCode].key), old_key);

		// Test the next element to the right
		hashCode = (hashCode + 1) % htb->size;
		count++;
	}

	htb->occupied += numKeys;
}

__global__ void kernel_reshape(HashTable *htb, HashNode *new_table, int new_size) {
	// Compute the global element index this thread should process
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Get key, value and initial hash code
	unsigned int key = htb->table[idx].key;
	unsigned int value = htb->table[idx].value;
	int hashCode = hashFunc(key, new_size);

	int count = 0;		// Variable to make sure it doesn't loop

	while (count < new_size) {
		// Exchange the old key with the new one
		unsigned int old_key = atomicExch(&(new_table[hashCode].key), key);

		// This position in the HashTable was not used, therefore add also the value
		if (old_key == DEFAULT_KEY) {
			atomicExch(&(new_table[hashCode].value), value);
			break;
		}

		// This position is already used, put the old key back
		atomicExch(&(new_table[hashCode].key), old_key);

		// Test the next element to the right
		hashCode = (hashCode + 1) % new_size;
		count++;
	}
}

__global__ void kernel_getBatch(HashTable *htb, int* keys, int* values, int numKeys) {
	// Compute the global element index this thread should process
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Keys must be grater that 0
	if (idx >= numKeys || keys[idx] <= 0) {
		return;
	}

	// Get key and initial hash code
	unsigned int key = keys[idx];
	int hashCode = hashFunc(key, htb->size);

	int count = 0;		// Variable to make sure it doesn't loop

	while (count < htb->size) {
		// The keys match, put the correct value in the vector
		if (htb->table[hashCode].key == key) {
			values[idx] = htb->table[hashCode].value;
			break;
		}

		// Test the next element to the right
		hashCode = (hashCode + 1) % htb->size;
		count++;
	}
}




/**
 * Function constructor GpuHashTable
 * Performs init
 * Example on using wrapper allocators _cudaMalloc and _cudaFree
 */
GpuHashTable::GpuHashTable(int size) {
	glbGpuAllocator->_cudaMalloc((void **) &htb.table, size * sizeof(HashNode));
	cudaCheckError();

	htb.size = size;
	htb.occupied = 0;

	// Get the number of blocks per thread
	numBlocks = htb.size / NUM_THREADS;
	if (htb.size % NUM_THREADS) {
		numBlocks++;
	}
}

/**
 * Function desctructor GpuHashTable
 */
GpuHashTable::~GpuHashTable() {
	glbGpuAllocator->_cudaFree(htb.table);
	htb.table = nullptr;

	htb.size = 0;
	htb.occupied = 0;
	numBlocks = 0;
}

/**
 * Function reshape
 * Performs resize of the hashtable based on load factor
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	HashNode *new_table;

	glbGpuAllocator->_cudaMalloc((void **) &new_table, numBucketsReshape * sizeof(HashNode));
	cudaCheckError();
	cudaMemset(new_table, DEFAULT_VALUE, numBucketsReshape * sizeof(HashNode));

	// If the structure in empty
	if (htb.occupied == 0) {
		glbGpuAllocator->_cudaFree(htb.table);
		htb.table = new_table;
		htb.size = numBucketsReshape;
	}

	else {
		// Move the elements in the new table
		kernel_reshape<<<numBlocks, NUM_THREADS>>>(&htb, new_table, numBucketsReshape);

		cudaDeviceSynchronize();

		glbGpuAllocator->_cudaFree(htb.table);
		htb.table = new_table;
		htb.size = numBucketsReshape;
	}	

	// Get the new number of blocks per thread
	numBlocks = htb.size / NUM_THREADS;
	if (numBucketsReshape % NUM_THREADS) {
		numBlocks++;
	}
}

/**
 * Function insertBatch
 * Inserts a batch of key:value, using GPU and wrapper allocators
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *device_keys; int *device_values;

	glbGpuAllocator->_cudaMalloc((void **) &device_keys, numKeys * sizeof(int));
	cudaCheckError();
	glbGpuAllocator->_cudaMalloc((void **) &device_values, numKeys * sizeof(int));
	cudaCheckError();

	// Copy the keys and values from host to device
	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);


	// Calculate the current loading of the HashTable and reshape if nedded
	float capacity = (float) (numKeys + htb.occupied) / htb.size;
	if (capacity > MAX_LOAD) {
		reshape((int) ((numKeys + htb.occupied) / MIN_LOAD));
	}

	// Insert the pairs in the HashTable
	kernel_insertBatch<<<numBlocks, NUM_THREADS>>>(&htb, device_keys, device_values, numKeys);

	cudaDeviceSynchronize();

	// Free the memory 
	glbGpuAllocator->_cudaFree(device_keys);
	glbGpuAllocator->_cudaFree(device_values);

	return true;
}

/**
 * Function getBatch
 * Gets a batch of key:value, using GPU
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
    int *device_keys; int *device_values;

    glbGpuAllocator->_cudaMalloc((void **) &device_keys, numKeys * sizeof(int));
    cudaCheckError();
    glbGpuAllocator->_cudaMalloc((void **) &device_values, numKeys * sizeof(int));
    cudaCheckError();

	// Copy the keys from host to device
    cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	// Get the values coresponding to the keys
    kernel_getBatch<<<numBlocks, NUM_THREADS>>>(&htb, device_keys, device_values, numKeys);

    cudaDeviceSynchronize();

    // Copy the values into host memory 
    int *values = (int *) malloc(numKeys * sizeof(int));
    cudaMemcpy(values, device_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

    // Free the memory 
    glbGpuAllocator->_cudaFree(device_keys);
    glbGpuAllocator->_cudaFree(device_values);

    return values;

}
