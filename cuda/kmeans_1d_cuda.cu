#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

/* ---------- CSV 1D utility: each line has 1 number ---------- */
static int count_rows(const char *path){
    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"Error opening %s\n", path); exit(1); }
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

        // Accept comma/semicolon/space/tab, get first numeric token
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

/* ---------- Time metrics structure ---------- */
typedef struct {
    double h2d_ms;      // Host-to-Device time
    double d2h_ms;      // Device-to-Host time
    double kernel_ms;   // Total kernel time
    double total_ms;    // Total CUDA time
    double cpu_total_ms;// Total CPU time (including CUDA calls)
    int gridN;          // Number of blocks for data
    int gridK;          // Number of blocks for centroids
    int blockSize;      // Threads per block
} CudaMetrics;

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
    extern __shared__ double buf[];
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
    int blockSize,
    int *iters_out,
    double *sse_out,
    CudaMetrics *metrics
){
    // Create CUDA events for timing
    cudaEvent_t start_total, stop_total;
    cudaEvent_t start_h2d, stop_h2d;
    cudaEvent_t start_kernel, stop_kernel;
    cudaEvent_t start_d2h, stop_d2h;
    
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventCreate(&start_h2d);
    cudaEventCreate(&stop_h2d);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
    cudaEventCreate(&start_d2h);
    cudaEventCreate(&stop_d2h);
    
    // Start total CUDA timer
    cudaEventRecord(start_total);
    
    // Calculate grid size
    int gridN = (N + blockSize - 1) / blockSize;
    int gridK = (K + blockSize - 1) / blockSize;
    
    // Store configuration in metrics
    metrics->gridN = gridN;
    metrics->gridK = gridK;
    metrics->blockSize = blockSize;

    // ===== GPU memory allocation =====
    double *d_X, *d_C, *d_sse, *d_sse_sum, *d_sum;
    int *d_assign, *d_cnt;

    cudaMalloc(&d_X, N * sizeof(double));
    cudaMalloc(&d_C, K * sizeof(double));
    cudaMalloc(&d_sse, N * sizeof(double));
    cudaMalloc(&d_assign, N * sizeof(int));
    cudaMalloc(&d_sum, K * sizeof(double));
    cudaMalloc(&d_cnt, K * sizeof(int));
    cudaMalloc(&d_sse_sum, sizeof(double));

    // ===== HOST-TO-DEVICE transfer (H2D) =====
    cudaEventRecord(start_h2d);
    cudaMemcpy(d_X, X, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, K * sizeof(double), cudaMemcpyHostToDevice);
    cudaEventRecord(stop_h2d);
    cudaEventSynchronize(stop_h2d);
    
    float h2d_ms = 0;
    cudaEventElapsedTime(&h2d_ms, start_h2d, stop_h2d);
    metrics->h2d_ms = (double)h2d_ms;

    double prev_sse = 1e300;
    double sse = 0.0;
    int it;
    
    // ===== Start kernel timer =====
    cudaEventRecord(start_kernel);

    for(it = 0; it < max_iter; it++){
        // Reset accumulators
        cudaMemset(d_sum, 0, K * sizeof(double));
        cudaMemset(d_cnt, 0, K * sizeof(int));
        cudaMemset(d_sse_sum, 0, sizeof(double));

        // 1. Assignment kernel
        assignment_step_1d_cuda<<<gridN, blockSize>>>(
            d_X, d_C, d_assign, d_sse, N, K
        );
        cudaDeviceSynchronize();

        // 2. Reduction kernel (SSE)
        size_t shared_mem_size = blockSize * sizeof(double);
        reduce_sse_kernel<<<gridN, blockSize, shared_mem_size>>>(
            d_sse, d_sse_sum, N
        );
        cudaDeviceSynchronize();

        // 3. Update sums and counts
        update_sums_counts<<<gridN, blockSize>>>(
            d_X, d_assign, d_sum, d_cnt, N
        );
        cudaDeviceSynchronize();
        
        // 4. Update centroids
        update_centroids<<<gridK, blockSize>>>(
            d_C, d_sum, d_cnt, X[0], K
        );
        cudaDeviceSynchronize();

        // Copy SSE to check convergence
        cudaMemcpy(&sse, d_sse_sum, sizeof(double), cudaMemcpyDeviceToHost);

        // Check convergence
        double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if(rel < eps){
            it++;
            break;
        }
        prev_sse = sse;
    }
    
    // ===== Stop kernel timer =====
    cudaEventRecord(stop_kernel);
    cudaEventSynchronize(stop_kernel);
    
    float kernel_ms = 0;
    cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);
    metrics->kernel_ms = (double)kernel_ms;

    // ===== DEVICE-TO-HOST transfer (D2H) =====
    cudaEventRecord(start_d2h);
    cudaMemcpy(assign, d_assign, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(C, d_C, K * sizeof(double), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_d2h);
    cudaEventSynchronize(stop_d2h);
    
    float d2h_ms = 0;
    cudaEventElapsedTime(&d2h_ms, start_d2h, stop_d2h);
    metrics->d2h_ms = (double)d2h_ms;

    // ===== Stop total CUDA timer =====
    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);
    
    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start_total, stop_total);
    metrics->total_ms = (double)total_ms;

    // ===== Free GPU memory =====
    cudaFree(d_X);
    cudaFree(d_C);
    cudaFree(d_sse);
    cudaFree(d_assign);
    cudaFree(d_sum);
    cudaFree(d_cnt);
    cudaFree(d_sse_sum);

    // ===== Destroy events =====
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_h2d);
    cudaEventDestroy(stop_h2d);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);
    cudaEventDestroy(start_d2h);
    cudaEventDestroy(stop_d2h);

    *iters_out = it;
    *sse_out = sse;
}

/* ---------- main ---------- */
int main(int argc, char **argv){
    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc > 3) ? atoi(argv[3]) : 50;
    double eps   = (argc > 4) ? atof(argv[4]) : 1e-4;
    int blockSize = (argc > 5) ? atoi(argv[5]) : 256;
    const char *outAssign   = (argc > 6) ? argv[6] : NULL;
    const char *outCentroid = (argc > 7) ? argv[7] : NULL;

    if(max_iter <= 0 || eps <= 0.0){
        fprintf(stderr,"Parâmetros inválidos: max_iter>0 e eps>0\n");
        return 1;
    }

    // Read data
    int N=0, K=0;
    double *X = read_csv_1col(pathX, &N);
    double *C = read_csv_1col(pathC, &K);
    int *assign = (int*)malloc((size_t)N * sizeof(int));
    if(!assign){ fprintf(stderr,"Sem memoria para assign\n"); free(X); free(C); return 1; }

    // Create structure for metrics
    CudaMetrics metrics;
    memset(&metrics, 0, sizeof(CudaMetrics));
    
    // Measure total CPU time (host wall time)
    clock_t cpu_start = clock();
    
    // Execute k-means CUDA
    int iters = 0; 
    double sse = 0.0;
    kmeans_1d_cuda(X, C, assign, N, K, max_iter, eps, blockSize, &iters, &sse, &metrics);
    
    // Calculate total CPU time (host wall time)
    clock_t cpu_end = clock();
    metrics.cpu_total_ms = 1000.0 * (double)(cpu_end - cpu_start) / (double)CLOCKS_PER_SEC;

    // Calculate throughput in Mpts/s (million points per second)
    // Throughput = (N * iterations) / (total time in seconds) / 1e6
    double total_points_processed = (double)N * iters;
    double total_time_seconds = metrics.total_ms / 1000.0;
    double throughput = (total_points_processed / total_time_seconds) / 1e6;
    
    // Print simplified output
    printf("Block size: %d\n", metrics.blockSize);
    printf("Iterações: %d | SSE final: %.6f | Tempo (host wall): %.1f ms\n", 
           iters, sse, metrics.cpu_total_ms);
    printf("Grid: %d blocks\n", metrics.gridN);
    printf("H2D: %.3f ms\n", metrics.h2d_ms);
    printf("Kernel: %.3f ms [total across iters]\n", metrics.kernel_ms);
    printf("D2H: %.3f ms\n", metrics.d2h_ms);
    printf("Total: %.3f ms\n", metrics.total_ms);
    printf("Throughput: %.3f Mpts/s\n", throughput);
    
    // Save results if specified
    if(outAssign) {
        write_assign_csv(outAssign, assign, N);
    }
    
    if(outCentroid) {
        write_centroids_csv(outCentroid, C, K);
    }

    // Save metrics to CSV file for analysis (optional)
    FILE *metrics_file = fopen("cuda_metrics.csv", "a");
    if(metrics_file) {
        // Header if new file
        fseek(metrics_file, 0, SEEK_END);
        if(ftell(metrics_file) == 0) {
            fprintf(metrics_file, "N,K,blockSize,gridN,gridK,iters,sse,cpu_total_ms,total_ms,h2d_ms,kernel_ms,d2h_ms,throughput_mpts_s\n");
        }
        
        fprintf(metrics_file, "%d,%d,%d,%d,%d,%d,%.6f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                N, K, metrics.blockSize, metrics.gridN, metrics.gridK, 
                iters, sse, metrics.cpu_total_ms, metrics.total_ms,
                metrics.h2d_ms, metrics.kernel_ms, metrics.d2h_ms,
                throughput);
        fclose(metrics_file);
    }

    // Free memory
    free(assign); free(X); free(C);
    return 0;
}
