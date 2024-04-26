#include <stdio.h>
#include <time.h>
#include <limits.h>

#define THREADS 8

__global__ void findMin(int arr[],int minsArr[], int N){

	int numToFind = N/THREADS;
	int low = threadIdx.x * numToFind;
	int high = low + numToFind;
	if(threadIdx.x == (THREADS - 1)){
		high = N;
	}

	int min = INT_MAX;

	for(int i = low; i < high; i++){
		if (arr[i] < min) {
			min = arr[i];
		}
	}

	minsArr[threadIdx.x] = min;
}

__global__ void printArr(int arr[], int size){

	for(int i = 0; i < size; i++){
		printf("%d, ",arr[i]);
	}
	printf("\n");

}

int findMin(int arr[], int size){
	int min = INT_MAX;
	for(int i = 0; i < size; i++){
		if(arr[i] < min){
			min = arr[i];
		}
	}
	return min;
}

int main() {

	int size = 8000000;
	int min;
	int hostMin;

	int *randArr = (int*)malloc(size*sizeof(int));
	int *foundMins = (int*)malloc(THREADS*sizeof(int));

	//Generate random array
	srand((int)time(NULL));
	for(int i = 0; i < size; i++){
		randArr[i] = rand() % 1000000000;
	}

	//Device Variables
	int *devArr;
	int *devMins;
	cudaMalloc((void**)&devArr, size*sizeof(int));
	cudaMalloc((void**)&devMins,THREADS*sizeof(int));

	cudaMemcpy(devArr, randArr, (size*sizeof(int)), cudaMemcpyHostToDevice);

	findMin<<<1,THREADS>>>(devArr,devMins,size);

	cudaMemcpy(foundMins,devMins, (THREADS*sizeof(int)), cudaMemcpyDeviceToHost);

	//print array of mins found in each thread
	printf("GPU found: ");
	for(int i = 0; i < THREADS; i++) {
		printf("%d, ",foundMins[i]);
	}
	printf("\n");

	//Find overall min from device
	min = findMin(foundMins,THREADS);
	printf("Overall minimum: %d\n",min);

	//Find min on host
	hostMin = findMin(randArr,size);
	if (min == hostMin) {
		printf("GPU has found the min successfully\n");
	} else{
		printf("bruh\n");
		printf("actual min was %d\n", hostMin);
	}

	cudaFree(devArr);
	cudaFree(devMins);

	free(randArr);
	free(foundMins);

	return 0;
}

