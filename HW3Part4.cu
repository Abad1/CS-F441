#include "stdio.h"
#define COLUMNS 16
#define ROWS 16

const int THREADS_PER_BLOCK = 16;
const int NUM_BLOCKS = 16;

__global__ void sumColumn(int arr[], int sum[]) {
	
	__shared__ int blockSum[THREADS_PER_BLOCK];
	int x = threadIdx.x;
	blockSum[x] += arr[x +(COLUMNS * blockIdx.x)];
	int i = THREADS_PER_BLOCK/2;
	while(i > 0) {
		if (x < i) {
			blockSum[x] += blockSum[x + i];
		}
		__syncthreads();
		i /= 2;
	}
	sum[blockIdx.x] = blockSum[0];
}

int main() {

	int total = 0;
	int arr[ROWS][COLUMNS];
	int columnSum[COLUMNS];
	int *devArr;
	int *devColumnSum;

	cudaMalloc((void**)&devArr, ROWS * COLUMNS * sizeof(int));
	cudaMalloc((void**)&devColumnSum, COLUMNS * sizeof(int));

	for (int y = 0; y < ROWS; y++) { // Fill Array
		for (int x = 0; x < COLUMNS; x++) {
			arr[y][x] = x + y;
		}
	}

	cudaMemcpy(devArr, arr, ROWS * COLUMNS * sizeof(int), cudaMemcpyHostToDevice);

	sumColumn<<<ROWS,COLUMNS>>>(devArr,devColumnSum);

	cudaMemcpy(columnSum, devColumnSum, COLUMNS * sizeof(int), cudaMemcpyDeviceToHost);

	printf("Sums computed by array:");
	for(int i = 0; i < COLUMNS; i++){
		printf("%d, ",columnSum[i]);
	}
	printf("\n");

	//Array Values
	for (int y = 0; y < ROWS; y++) {
		for (int x = 0; x < COLUMNS; x++) {
			printf("[%d][%d]=%d ", y, x, arr[y][x]);
		}
		printf("\n");
	}

	for(int i = 0; i < COLUMNS; i++) {
		total += columnSum[i];
	}
	printf("Sum of array: %d\n",total);
	cudaFree(devArr);
	cudaFree(devColumnSum);
	return 0;
}
