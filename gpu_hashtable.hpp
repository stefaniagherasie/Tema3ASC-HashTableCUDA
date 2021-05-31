#ifndef _HASHCPU_
#define _HASHCPU_

#include <vector>

using namespace std;

#define cudaCheckError() { \
	cudaError_t e=cudaGetLastError(); \
	if(e!=cudaSuccess) { \
		cout << "Cuda failure " << __FILE__ << ", " << __LINE__ << ", " << cudaGetErrorString(e); \
		exit(0); \
	 }\
}


#define DIE(assertion, call_description) \
	do {	\
		if (assertion) {	\
		fprintf(stderr, "(%s, %d): ",	\
		__FILE__, __LINE__);	\
		perror(call_description);	\
		exit(errno);	\
		}	\
	} while (0)

//int hashFunc(int value, int limit) {
//        return ((long) abs(value) * 421439781llu) % 271862205833llu % limit;
//}


typedef struct hash_node {
	unsigned int key;
	unsigned int value;
} HashNode;


typedef struct hash_table {
	HashNode *table;
	int occupied;
	int size;
} HashTable;


/**
 * Class GpuHashTable to implement functions
 */
class GpuHashTable
{
	public:
		HashTable htb;

		GpuHashTable(int size);
		void reshape(int sizeReshape);
		
		bool insertBatch(int *keys, int* values, int numKeys);
		int* getBatch(int* key, int numItems);
	
		~GpuHashTable();
};

#endif

