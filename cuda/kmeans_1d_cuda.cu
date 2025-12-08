#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda.h>

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

/* ---------- k-means 1D ---------- */
__global__ void assignment_step_1d_cuda(const double *X, const double *C, int *assign, double *sse, int N, int K){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        int best = -1;
        double bestd = 1e300;
        for(int i = 0; i < K; i++) {
            double diff = X[idx] - C[i];
            double d = diff*diff;
            if(d < bestd){ bestd = d; best = i; }
        }

        assign[idx] = best;
        sse[idx] = bestd;
    }
}

__global__ void update_sums_counts(
    const double *X,
    const int *assign,
    double *sum,
    int *cnt,
    int N
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N){
        int c = assign[i];
        atomicAdd(&sum[c], X[i]);
        atomicAdd(&cnt[c], 1);
    }
}

__global__ void update_centroids(double *C, const double *sum, const int *cnt, const double fallback, int K){
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < K){
        if (cnt[c] > 0)
            C[c] = sum[c] / (double)cnt[c];
        else
            C[c] = fallback;
    }
}

static void chama_assigment(const double *X, double *cuda_X,
                            double *C, double *cuda_C,
                            int *assign, int *cuda_Assign,
                            double *v_sse, double *cuda_SSE,
                            int N,
                            int K,
                            int blockSize,
                            int gridSize
                            ) {

    cudaMemcpy(cuda_X, X, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_C, C, K * sizeof(double), cudaMemcpyHostToDevice);

    assignment_step_1d_cuda<<<gridSize, blockSize>>>(cuda_X, cuda_C, cuda_Assign, cuda_SSE, N, K);

    cudaDeviceSynchronize();

    cudaMemcpy(assign, cuda_Assign, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(v_sse, cuda_SSE, N * sizeof(double), cudaMemcpyDeviceToHost);
}



static void chama_update(double *cuda_X,
                        double *C, double *cuda_C,
                        int *cuda_Assign,
                        double *cuda_sum,
                        int *cuda_cnt,
                        int N,
                        int K,
                        int blockSize,
                        int gridSize,
                        int gridK
                        ) {
    cudaMemset(cuda_sum, 0, K * sizeof(double));
    cudaMemset(cuda_cnt, 0, K * sizeof(int));

    double firstX;
    cudaMemcpy(&firstX, cuda_X, sizeof(double), cudaMemcpyDeviceToHost);

    update_sums_counts<<<gridSize, blockSize>>>(cuda_X, cuda_Assign, cuda_sum, cuda_cnt, N);
    update_centroids<<<gridK, blockSize>>>(cuda_C, cuda_sum, cuda_cnt, firstX, K);

    cudaDeviceSynchronize();

    cudaMemcpy(C, cuda_C, K * sizeof(double), cudaMemcpyDeviceToHost);
}


static void kmeans_1d(const double *X, double *C, int *assign,
                      int N, int K, int max_iter, double eps,
                      int *iters_out, double *sse_out)
{
    double prev_sse = 1e300;
    double *v_sse = (double*)malloc((size_t)N * sizeof(double));
    double sse = 0.0;
    int it;

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    int gridK = (K + blockSize - 1) / blockSize;
    double *cuda_X, *cuda_C, *cuda_SSE;
    int *cuda_Assign;

    double *cuda_sum;
    int *cuda_cnt;

    cudaMalloc(&cuda_X, N * sizeof(double));
    cudaMalloc(&cuda_C, K * sizeof(double));
    cudaMalloc(&cuda_SSE, N * sizeof(double));
    cudaMalloc(&cuda_Assign, N * sizeof(int));

    cudaMalloc(&cuda_sum, K * sizeof(double));
    cudaMalloc(&cuda_cnt, K * sizeof(int));

    for(it=0; it<max_iter; it++){

        chama_assigment(X, cuda_X, C, cuda_C, assign, cuda_Assign, v_sse, cuda_SSE, N, K, blockSize, gridSize);

        sse = 0.0;
        for(int j = 0; j < N; j++) {
            sse += v_sse[j];
        }

        double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if(rel < eps){ it++; break; }

        chama_update(cuda_X, C, cuda_C, cuda_Assign, cuda_sum, cuda_cnt, N, K, blockSize, gridSize, gridK);

        prev_sse = sse;
    }

    cudaFree(cuda_sum);
    cudaFree(cuda_cnt);
    cudaFree(cuda_X);
    cudaFree(cuda_C);
    cudaFree(cuda_SSE);
    cudaFree(cuda_Assign);
    free(v_sse);
    *iters_out = it;
    *sse_out = sse;
}

/* ---------- main ---------- */
int main(int argc, char **argv){
    const char *pathX = "/content/dados.csv";
    const char *pathC = "/content/centroides_iniciais.csv";
    int max_iter = (argc>3)? atoi(argv[3]) : 50;
    double eps   = (argc>4)? atof(argv[4]) : 1e-6;
    const char *outAssign   = (argc>5)? argv[5] : "/content/assign_cuda.csv";
    const char *outCentroid = (argc>6)? argv[6] : "/content/centroid_cuda.csv";

    if(max_iter <= 0 || eps <= 0.0){
        fprintf(stderr,"Parâmetros inválidos: max_iter>0 e eps>0\n");
        return 1;
    }

    int N=0, K=0;
    double *X = read_csv_1col(pathX, &N);
    double *C = read_csv_1col(pathC, &K);
    int *assign = (int*)malloc((size_t)N * sizeof(int));
    if(!assign){ fprintf(stderr,"Sem memoria para assign\n"); free(X); free(C); return 1; }

    clock_t t0 = clock();
    int iters = 0; double sse = 0.0;
    kmeans_1d(X, C, assign, N, K, max_iter, eps, &iters, &sse);
    clock_t t1 = clock();
    double ms = 1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    printf("K-means 1D (naive)\n");
    printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, eps);
    printf("Iterações: %d | SSE final: %.6f | Tempo: %.1f ms\n", iters, sse, ms);

    write_assign_csv(outAssign, assign, N);
    write_centroids_csv(outCentroid, C, K);

    free(assign); free(X); free(C);
    return 0;
}