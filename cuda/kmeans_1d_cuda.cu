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
__global__ void assignment_step_1d_cuda(
    const double *X,
    const double *C,
    int *assign,
    double *sse,
    int N,
    int K
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
        int best = 0;
        double bestd = 1e300;

        for(int c = 0; c < K; c++){
            double diff = X[i] - C[c];
            double d = diff * diff;
            if(d < bestd){
                bestd = d;
                best = c;
            }
        }
        assign[i] = best;
        sse[i] = bestd;
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
    if(i < N){
        int c = assign[i];
        atomicAdd(&sum[c], X[i]);
        atomicAdd(&cnt[c], 1);
    }
}

__global__ void update_centroids(
    double *C,
    const double *sum,
    const int *cnt,
    double fallback,
    int K
){
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(c < K){
        C[c] = (cnt[c] > 0) ? sum[c] / cnt[c] : fallback;
    }
}

__global__ void reduce_sse_kernel(
    const double *sse,
    double *out,
    int N
){
    __shared__ double buf[256];

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;

    buf[tid] = (i < N) ? sse[i] : 0.0;
    __syncthreads();

    for(int s = blockDim.x / 2; s > 0; s >>= 1){
        if(tid < s)
            buf[tid] += buf[tid + s];
        __syncthreads();
    }

    if(tid == 0)
        atomicAdd(out, buf[0]);
}

static void kmeans_1d_cuda(
    const double *X,
    double *C,
    int *assign,
    int N,
    int K,
    int max_iter,
    double eps,
    int *iters_out,
    double *sse_out
){
    int blockSize = 256;
    int gridN = (N + blockSize - 1) / blockSize;
    int gridK = (K + blockSize - 1) / blockSize;

    // ===== GPU memory =====
    double *d_X, *d_C, *d_sse, *d_sse_sum, *d_sum;
    int *d_assign, *d_cnt;

    cudaMalloc(&d_X, N * sizeof(double));
    cudaMalloc(&d_C, K * sizeof(double));
    cudaMalloc(&d_sse, N * sizeof(double));
    cudaMalloc(&d_assign, N * sizeof(int));
    cudaMalloc(&d_sum, K * sizeof(double));
    cudaMalloc(&d_cnt, K * sizeof(int));
    cudaMalloc(&d_sse_sum, sizeof(double));

    // copia inicial
    cudaMemcpy(d_X, X, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, K * sizeof(double), cudaMemcpyHostToDevice);

    double prev_sse = 1e300;
    double sse = 0.0;
    int it;

    for(it = 0; it < max_iter; it++){
        // reset acumuladores
        cudaMemset(d_sum, 0, K * sizeof(double));
        cudaMemset(d_cnt, 0, K * sizeof(int));
        cudaMemset(d_sse_sum, 0, sizeof(double));

        // assignment
        assignment_step_1d_cuda<<<gridN, blockSize>>>(
            d_X, d_C, d_assign, d_sse, N, K
        );

        // redução SSE
        reduce_sse_kernel<<<gridN, blockSize>>>(
            d_sse, d_sse_sum, N
        );

        // update
        update_sums_counts<<<gridN, blockSize>>>(
            d_X, d_assign, d_sum, d_cnt, N
        );
        update_centroids<<<gridK, blockSize>>>(
            d_C, d_sum, d_cnt, C[0], K
        );

        // copiar só 1 double
        cudaMemcpy(&sse, d_sse_sum, sizeof(double), cudaMemcpyDeviceToHost);

        double rel = fabs(sse - prev_sse) / prev_sse;
        if(rel < eps){
            it++;
            break;
        }
        prev_sse = sse;
    }

    // resultados finais
    cudaMemcpy(assign, d_assign, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(C, d_C, K * sizeof(double), cudaMemcpyDeviceToHost);

    // free
    cudaFree(d_X);
    cudaFree(d_C);
    cudaFree(d_sse);
    cudaFree(d_assign);
    cudaFree(d_sum);
    cudaFree(d_cnt);
    cudaFree(d_sse_sum);

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
    kmeans_1d_cuda(X, C, assign, N, K, max_iter, eps, &iters, &sse);
    clock_t t1 = clock();
    double ms = 1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    printf("K-means 1D Cuda\n");
    printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, eps);
    printf("Iterações: %d | SSE final: %.6f | Tempo: %.1f ms\n", iters, sse, ms);

    write_assign_csv(outAssign, assign, N);
    write_centroids_csv(outCentroid, C, K);

    free(assign); free(X); free(C);
    return 0;
}