#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <mpi.h>

struct knnresult{
  int * nidx;
  double * ndist;
  int m;
  int k;
};

int main(int argc, char * argv[]){

  int tid, tNum, err;
  MPI_Init(&argc,&argv);
  MPI_Comm_size( MPI_COMM_WORLD, &tNum);
  MPI_Comm_rank( MPI_COMM_WORLD, &tid);
  MPI_Request mpireq1, mpireq2;
  MPI_Status mpistat;

  struct timespec ts_start, ts_end, dg1, dg2;
  struct knnresult knn;
  int N, d, k;
  double *X;

  if (argc == 5){
    N = atoi(argv[2]);
    d = atoi(argv[3]);
    k = atoi(argv[4]);

    X = (double *)malloc(N*d*sizeof(double));
    if(tid == 0){
      char *fname = argv[1];
      FILE *file = fopen(fname,"r");

      char line[1024];

      int i = 0;
      while(fgets(line, 1024, file) && (i < N)){
        char *tmp = strdup(line);

        int j = 0;
        const char* tok;
        for(tok = strtok(line, " "); tok && *tok; j++, tok = strtok(NULL, " ")){
          if(j < d) X[i*d+j] = atof(tok);
        }
        free(tmp);
        i++;
      }
      fclose(file);
    }
  } else {
    if(tid == 0){
      printf("Incorrect amount of arguments please try:\n./a <file> <N> <d> <k>\n\n");
    }
    MPI_Finalize();
    return 0;
  }

  clock_gettime(CLOCK_MONOTONIC, &ts_start);

  int max_usable = N%tNum;

  int n = N/tNum;
  int idx = max_usable*(n+1) + n*(tid-max_usable);
  if(tid < max_usable){
     n++;
     idx = n*tid;
  }

  if(tid > 0) {
    MPI_Recv(X, N*d, MPI_DOUBLE, tid-1, 10+tid, MPI_COMM_WORLD, &mpistat);
  }
  if(tid < tNum-1){
    MPI_Isend(X, N*d, MPI_DOUBLE, tid+1, 11+tid, MPI_COMM_WORLD, &mpireq1);
  }

  knn.ndist = (double *)calloc(k*(N-idx),sizeof(double));
  knn.nidx = (int *)calloc(k*(N-idx),sizeof(int));

  int limit = 2000;
  int segments = N/limit;
  int remaining = N%limit;
  N = limit;

  double *D = (double *)calloc(n*N,sizeof(double));
  double *V = (double *)calloc(N,sizeof(double));
  double *U = (double *)calloc(n,sizeof(double));

  int *mx_idx = (int *)calloc(n,sizeof(int));
  for(int seg = 0; seg <= segments; seg++){
    if(seg == segments){
      N = remaining;
      D = (double *)realloc(D,n*N*sizeof(double));
      V = (double *)realloc(V,N*sizeof(double));
      if(N == 0) break;
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, N, d, -2, X + idx*d, d, X + seg*limit*d, d, 0, D, N);

    for(int i = 0; i < N; i++){
      V[i]=0;
    }

    for(int i = 0; i < n; i++){
      U[i]=0;
    }

    for(int i = 0; i < N*d; i++){
      V[i/d] += X[i]*X[i];
    }
    for(int i = 0; i < n*d; i++){
      U[i%d] += X[i]*X[i];
    }

    for(int i = 0; i < n; i++){
      for(int j = 0; j < N; j++){
        D[i*N + j] += V[j] + U[i];
        if((j < k) && (seg == 0)){

          knn.ndist[i*k + j] = D[i*N + j];
          knn.nidx[i*k + j] = seg*limit + j;

          if(knn.ndist[i*k + j] > knn.ndist[i*k + mx_idx[i]]){
            mx_idx[i] = j;
          }
        } else if(D[i*N + j] < knn.ndist[i*k + mx_idx[i]]){
          knn.ndist[i*k + mx_idx[i]] = D[i*N + j];
          knn.nidx[i*k + mx_idx[i]] = seg*limit + j;

          mx_idx[i] = 0;
          for(int l = 1; l < k; l++){
            if(knn.ndist[i*k + l] > knn.ndist[i*k + mx_idx[i]]){
              mx_idx[i] = l;
            }
          }
        }
      }
    }
  }

  N = segments*limit+remaining;

  if(tid < tNum-1) {
    MPI_Irecv(knn.ndist + n*k, (N - idx - n)*k, MPI_DOUBLE, tid+1, 21+tid, MPI_COMM_WORLD, &mpireq1);
    MPI_Irecv(knn.nidx + n*k, (N - idx - n)*k, MPI_INT, tid+1, 31+tid, MPI_COMM_WORLD, &mpireq2);

    MPI_Wait(&mpireq1, &mpistat);
    MPI_Wait(&mpireq2, &mpistat);
  }

  if(tid > 0){
    MPI_Isend(knn.ndist, (N - idx)*k, MPI_DOUBLE, tid-1, 20+tid, MPI_COMM_WORLD, &mpireq1);
    MPI_Isend(knn.nidx, (N - idx)*k, MPI_INT, tid-1, 30+tid, MPI_COMM_WORLD, &mpireq2);
  }
  clock_gettime(CLOCK_MONOTONIC, &ts_end);

  MPI_Barrier(MPI_COMM_WORLD);
  if(tid == 0){
    /*if(N > 100) N = 100;
    for(int i = 0; i < N; i++){
      printf("Point: %d\t%d-NN:\t",i,k);
      for(int j = 0; j < k; j++){
        printf("%d\t",knn.nidx[i*k+j]);
      }
      printf("\n");
    }*/

    printf("Time: %ld.%ld\n", (ts_end.tv_sec - ts_start.tv_sec), (abs(ts_end.tv_nsec - ts_start.tv_nsec)));
  }
  MPI_Finalize();

  return 0;
}
