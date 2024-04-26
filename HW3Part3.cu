#include "stdio.h"
#define COLUMNS 4
#define ROWS 3

__global__ void sumColumn(int arr[], int sum[]) {
	int x = threadIdx.x;

	for(int i = 0; i < ROWS; i++){
		sum[x] += arr[x + (COLUMNS * i)];
	}

}

int main() {

	int total = 0;
	int arr[ROWS][COLUMNS];
	int columnSum[COLUMNS] = {1,2,3,4};
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

	sumColumn<<<1,COLUMNS>>>(devArr,devColumnSum);

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
