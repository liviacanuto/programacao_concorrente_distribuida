#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

/* ---------- util CSV 1D: cada linha tem 1 número ---------- */
static int count_rows(const char *path){
    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n", path); exit(1); }
    int rows=0; char line[8192];
    while(fgets(line,sizeof(line),f)){
        int only_ws=1;
        for(char *p=line; *p; p++){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(!only_ws) rows++;
    }
    fclose(f);
    return rows;
}

static double *read_csv_1col(const char *path, int *n_out){
    int R = count_rows(path);
    if(R<=0){ fprintf(stderr,"Arquivo vazio: %s\n", path); exit(1); }
    double *A = (double*)malloc((size_t)R * sizeof(double));
    if(!A){ fprintf(stderr,"Sem memoria para %d linhas\n", R); exit(1); }

    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n", path); free(A); exit(1); }

    char line[8192];
    int r=0;
    while(fgets(line,sizeof(line),f)){
        int only_ws=1;
        for(char *p=line; *p; p++){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(only_ws) continue;

        /* aceita vírgula/ponto-e-vírgula/espaco/tab, pega o primeiro token numérico */
        const char *delim = ",; \t";
        char *tok = strtok(line, delim);
        if(!tok){ fprintf(stderr,"Linha %d sem valor em %s\n", r+1, path); free(A); fclose(f); exit(1); }
        A[r] = atof(tok);
        r++;
        if(r>R) break;
    }
    fclose(f);
    *n_out = R;
    return A;
}

static void write_assign_csv(const char *path, const int *assign, int N){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    for(int i=0;i<N;i++) fprintf(f, "%d\n", assign[i]);
    fclose(f);
}

static void write_centroids_csv(const char *path, const double *C, int K){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    for(int c=0;c<K;c++) fprintf(f, "%.6f\n", C[c]);
    fclose(f);
}

/* ---------- assignment local ---------- */
double assignment_local(const double *X, const double *C,
                        int *assign_local, int local_N, int K,
                        double *sum_local, int *cnt_local)
{
    double SSE_local = 0.0;

    for(int c=0;c<K;c++){
        sum_local[c] = 0.0;
        cnt_local[c] = 0;
    }

    for(int i=0;i<local_N;i++){
        int best = -1;
        double bestd = 1e300;
        for(int c=0;c<K;c++){
            double diff = X[i] - C[c];
            double d = diff*diff;
            if(d < bestd){ bestd = d; best = c; }
        }

        assign_local[i] = best;
        SSE_local += bestd;

        sum_local[best] += X[i];
        cnt_local[best] += 1;
    }

    return SSE_local;
}
/* ---------- kmeans MPI ---------- */

int kmeans_1d_mpi(double *X_local, int local_N, double *C, int K,
                   int max_iter, double eps,
                   int rank, int size,
                   int *assign_local, double reference_point, double *final_SSE)
{
    double *sum_local  = malloc(K * sizeof(double));
    int    *cnt_local  = malloc(K * sizeof(int));
    double *sum_global = malloc(K * sizeof(double));
    int    *cnt_global = malloc(K * sizeof(int));

    double prev_SSE = 1e300;
    int it;

    for(it=0; it<max_iter; it++){
        double SSE_local = assignment_local(X_local, C, assign_local,
                                            local_N, K, sum_local, cnt_local);

        double SSE_global;
        MPI_Allreduce(&SSE_local, &SSE_global, 1,
              MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        MPI_Allreduce(sum_local, sum_global, K,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        MPI_Allreduce(cnt_local, cnt_global, K,
                      MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if(rank == 0) {
            for(int c=0;c<K;c++){
                if(cnt_global[c] > 0)
                    C[c] = sum_global[c] / cnt_global[c];
                else
                    C[c] = reference_point ;
            }
        }
        MPI_Bcast(C, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        double rel;
        if(rank == 0)
            rel = fabs(SSE_global - prev_SSE) / (prev_SSE > 0.0 ? prev_SSE : 1.0);
        MPI_Bcast(&rel, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if(rel < eps){ it++; break;};

        if(rank == 0) {
            prev_SSE = SSE_global;
            *final_SSE = SSE_global;
        }

    }

    free(sum_local);
    free(cnt_local);
    free(sum_global);
    free(cnt_global);
    return it;
}

/* ---------- MAIN MPI ---------- */

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(argc < 3){
        if(rank == 0)
            printf("Uso: mpirun -np P %s dados.csv centroides.csv [max_iter] [eps] [assign.csv] [centroids.csv]\n", argv[0]);
        MPI_Finalize();
        return 0;
    }

    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc>3)? atoi(argv[3]) : 50;
    double eps   = (argc>4)? atof(argv[4]) : 1e-4;
    const char *outAssign   = (argc>5)? argv[5] : NULL;
    const char *outCentroid = (argc>6)? argv[6] : NULL;

    int N=0, K=0;
    double *X_full = NULL;
    double *C = NULL;

    /*----Apenas o master lê os dados*/
    if(rank == 0){
        X_full = read_csv_1col(pathX, &N);
        C      = read_csv_1col(pathC, &K);
        printf("K-means 1D (naive)\n");
        printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, eps);
    }

    
    double reference_point;
    double final_SSE;
    if(rank == 0){
        reference_point = X_full[0];
    }
    /*----Master distribuí dados necessários*/
    MPI_Bcast(&reference_point, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank != 0)
        C = malloc(K * sizeof(double));

    MPI_Bcast(C, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int *counts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));

    int base = N / size;
    int r = N % size;

    /* cada rank pega uma parte, a divisão é feita igual */
    for(int i=0, off=0;i<size;i++){
        counts[i] = base + (i < r);
        displs[i] = off;
        off += counts[i];
    }

    int local_N = counts[rank];
    double *X_local = malloc(local_N * sizeof(double));

    MPI_Scatterv(X_full, counts, displs, MPI_DOUBLE,
                 X_local, local_N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    int *assign_local = malloc(local_N * sizeof(int));
    
    /*------onde chamamos de fato o kmeans----- */
    double t0 = MPI_Wtime();

    int iters = kmeans_1d_mpi(X_local, local_N, C, K, max_iter, eps,
                  rank, size, assign_local, reference_point, &final_SSE);
    
    double t1 = MPI_Wtime();
    double ms = (t1 - t0) * 1000.0;
    /*------fim kmeans----- */
    int *assign_full = NULL;
    if(rank == 0)
        assign_full = malloc(N * sizeof(int));

    MPI_Gatherv(assign_local, local_N, MPI_INT,
                assign_full, counts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    if(rank == 0){
        printf("K-means 1D (MPI)\n");
        printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, eps);
        printf("Iterações: %d | SSE final: %.6f | Tempo: %.1f ms\n",
            iters, final_SSE, ms);
        write_assign_csv(outAssign, assign_full, N);
        write_centroids_csv(outCentroid, C, K);
        free(assign_full);
        free(X_full);
    }

    free(X_local);
    free(assign_local);
    free(C);
    free(counts);
    free(displs);

    MPI_Finalize();
    return 0;
}