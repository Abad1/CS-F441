#include <mpi.h>
#include <stdio.h>
#include <limits.h>
#include <stdlib.h>
#include <time.h>

int findMin(int arr[],int low,int high){
	int min = INT_MAX;
	for(int i = low; i < high; i++){
		if(arr[i] < min) {
			min = arr[i];
		}
	}
	return min;
}

int main(int argc, char *argv[]) {
	MPI_Status status;
	MPI_Init(&argc, &argv);
	int *randArray;
	int myRank;
	int p;
	int size = 8000000;
	int tag = 0;

	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	
	randArray = (int*)malloc(sizeof(int) * size);
	
	if(myRank == 0){
		//Generate rand array
		srand(time(NULL));
		for(int i = 0; i < size; i++) {
			randArray[i] = (rand() % 1000000000);
		}
	}

	MPI_Bcast(randArray,size,MPI_INT,0,MPI_COMM_WORLD);

	int numToFind = size / p;
	int low = myRank * numToFind;
	int high = low + numToFind + 1;

	if(high > size)
		high--;
	int foundMin = findMin(randArray,low,high);
	//Wait for all threads to finish
	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Send(&foundMin, 1, MPI_INT,0,tag,MPI_COMM_WORLD);
	if(myRank == 0){
		int multiMin;
		int singleMin;
		int temp[8];
		for(int i = 0; i < p; i++){
			MPI_Recv(&foundMin, 1, MPI_INT,i,tag,MPI_COMM_WORLD, &status);
			temp[i] = foundMin;
		}
		printf("Pretty sure everything else should be done.\n");
		printf("Each min from threads: ");
		for(int i = 0; i < p; i++){
			printf("%d, ",temp[i]);
		}

		multiMin = findMin(temp,0,8);
		singleMin = findMin(randArray,0,size);

		printf("\n");
		printf("Min found by multiple threads: %d\n",multiMin);
		printf("Min found by 1 thread: %d\n", singleMin);
		
		if(multiMin == singleMin){
			printf("Multithreaded min found succesfully ohhh my goodness\n");
		}else{
			printf("Multithreaded min is incorrect\n");
		}
	}
	
	free(randArray);
	MPI_Finalize();
	return 0;
}
