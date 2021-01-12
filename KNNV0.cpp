#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <string.h>

struct knnresult{
  int * nidx;
  double * ndist;
  int m;
  int k;
};

int main(int argc, char * argv[]){

  struct timespec ts_start, ts_end;

  struct knnresult knn;
  int N, d, k;
  double *X;

  if (argc == 5){

    N = atoi(argv[2]);
    d = atoi(argv[3]);
    k = atoi(argv[4]);

    X = (double *)malloc(N*d*sizeof(double));

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
  } else {
    printf("Incorrect amount of arguments please try:\n./a <file> <N> <d> <k>\n\n");
  }

  clock_gettime(CLOCK_MONOTONIC, &ts_start);

  knn.ndist = (double *)malloc(k*N*sizeof(double));
  knn.nidx = (int *)malloc(k*N*sizeof(int));

  int limit = 2000;
  int segments = N/limit;
  int remaining = N%limit;
  int Qsize, Csize;
  Qsize = limit;
  Csize = limit;

  double *D = (double *)malloc(Qsize*Csize*sizeof(double));
  double *V = (double *)malloc(Csize*sizeof(double));
  double *U = (double *)malloc(Qsize*sizeof(double));

  int *mx_idx = (int *)calloc(Qsize,sizeof(int));
  for(int Qseg = 0; Qseg <= segments; Qseg++){
    Csize = limit;
    if(Qseg == segments){
      if(remaining == 0) break;
      Qsize = remaining;
    }

    for(int Cseg = 0; Cseg <= segments; Cseg++){
      if(Cseg == segments){
        if(remaining == 0) break;
        Csize = remaining;
      }
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Qsize, Csize, d, -2, X + Qseg*limit*d, d, X + Cseg*limit*d, d, 0, D, Csize);

      for(int i = 0; i < Csize; i++){
        V[i] = 0;
      }
      for(int i = 0; i < Qsize; i++){
        U[i] = 0;
      }

      for(int i = 0; i < Csize*d; i++){
        V[i/d] += X[i]*X[i];
      }
      for(int i = 0; i < Qsize*d; i++){
        U[i%d] += X[i]*X[i];
      }

      for(int i = 0; i < Qsize; i++){
        for(int j = 0; j < Csize; j++){
          D[i*Csize + j] += V[j] + U[i];
          if((j < k) && (Cseg == 0)){

            knn.ndist[(Qseg+i)*k + j] = D[i*Csize + j];
            knn.nidx[(Qseg+i)*k + j] = Cseg*limit + j;

            if(knn.ndist[(Qseg+i)*k + j] > knn.ndist[(Qseg+i)*k + mx_idx[i]]){
              mx_idx[i] = j;
            }
          } else if(D[i*Csize + j] < knn.ndist[(Qseg+i)*k + mx_idx[i]]){
            knn.ndist[(Qseg+i)*k + mx_idx[i]] = D[i*Csize + j];
            knn.nidx[(Qseg+i)*k + mx_idx[i]] = Cseg*limit + j;

            mx_idx[i] = 0;
            for(int l = 1; l < k; l++){
              if(knn.ndist[(Qseg+i)*k + l] > knn.ndist[(Qseg+i)*k + mx_idx[i]]){
                mx_idx[i] = l;
              }
            }
          }
        }
      }
    }
  }

  free(D);
  free(V);
  free(U);

  clock_gettime(CLOCK_MONOTONIC, &ts_end);
  free(X);

  /*if(N > 100) N = 100;
  for(int i = 0; i < N; i++){
    printf("Point: %d\t%d-NN:\t",i,k);
    for(int j = 0; j < k; j++){
      printf("%d\t",knn.nidx[i*k+j]);
    }
    printf("\n");
  }*/

  printf("Time: %ld.%ld\n", (ts_end.tv_sec - ts_start.tv_sec), (abs(ts_end.tv_nsec - ts_start.tv_nsec)));
  return 0;
}
