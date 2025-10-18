/* kmeans_1d_omp.c
   K-means 1D (C99), implementação paralelizada com OpenMP:
   - Lê X (N linhas, 1 coluna) e C_init (K linhas, 1 coluna) de CSVs sem cabeçalho.
   - Itera assignment + update até max_iter ou variação relativa do SSE < eps.
   - Salva (opcional) assign (N linhas) e centróides finais (K linhas).

   Compilar: gcc -O2 -std=c99 -fopenmp kmeans_1d_omp.c -o kmeans_1d_omp -lm
   Uso:      OMP_NUM_THREADS=4 ./kmeans_1d_omp dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]
*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

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

/* ---------- k-means 1D paralelizado ---------- */
/* assignment: para cada X[i], encontra c com menor (X[i]-C[c])^2 */
static double assignment_step_1d_omp(const double *X, const double *C, int *assign, int N, int K){
    double sse = 0.0;
    
    #pragma omp parallel for reduction(+:sse)
    for(int i=0;i<N;i++){
        int best = -1;
        double bestd = 1e300;
        for(int c=0;c<K;c++){
            double diff = X[i] - C[c];
            double d = diff*diff;
            if(d < bestd){ bestd = d; best = c; }
        }
        assign[i] = best;
        sse += bestd;
    }
    return sse;
}

/* update: média dos pontos de cada cluster (1D) usando acumuladores por thread */
static void update_step_1d_omp(const double *X, double *C, const int *assign, int N, int K){
    int num_threads = omp_get_max_threads();
    
    /* Alocar acumuladores por thread */
    double **sum_thread = (double**)malloc((size_t)num_threads * sizeof(double*));
    int **cnt_thread = (int**)malloc((size_t)num_threads * sizeof(int*));
    
    if(!sum_thread || !cnt_thread){ 
        fprintf(stderr,"Sem memoria para acumuladores de thread\n"); 
        exit(1); 
    }
    
    for(int t=0; t<num_threads; t++){
        sum_thread[t] = (double*)calloc((size_t)K, sizeof(double));
        cnt_thread[t] = (int*)calloc((size_t)K, sizeof(int));
        if(!sum_thread[t] || !cnt_thread[t]){
            fprintf(stderr,"Sem memoria para acumuladores da thread %d\n", t);
            exit(1);
        }
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        
        /* Cada thread acumula em seus próprios arrays */
        #pragma omp for
        for(int i=0;i<N;i++){
            int a = assign[i];
            cnt_thread[tid][a] += 1;
            sum_thread[tid][a] += X[i];
        }
    }
    
    /* Redução sequencial dos acumuladores das threads */
    double *sum = (double*)calloc((size_t)K, sizeof(double));
    int *cnt = (int*)calloc((size_t)K, sizeof(int));
    
    if(!sum || !cnt){ 
        fprintf(stderr,"Sem memoria para arrays de redução\n"); 
        exit(1); 
    }
    
    for(int t=0; t<num_threads; t++){
        for(int c=0; c<K; c++){
            cnt[c] += cnt_thread[t][c];
            sum[c] += sum_thread[t][c];
        }
        free(sum_thread[t]);
        free(cnt_thread[t]);
    }
    
    free(sum_thread);
    free(cnt_thread);
    
    /* Calcular novos centróides */
    for(int c=0;c<K;c++){
        if(cnt[c] > 0) C[c] = sum[c] / (double)cnt[c];
        else           C[c] = X[0]; /* simples: cluster vazio recebe o primeiro ponto */
    }
    
    free(sum);
    free(cnt);
}

/* Versão alternativa do update com seção critica */
static void update_step_1d_omp_critical(const double *X, double *C, const int *assign, int N, int K){
    double *sum = (double*)calloc((size_t)K, sizeof(double));
    int *cnt = (int*)calloc((size_t)K, sizeof(int));
    if(!sum || !cnt){ fprintf(stderr,"Sem memoria no update\n"); exit(1); }

    #pragma omp parallel
    {
        double *sum_local = (double*)calloc((size_t)K, sizeof(double));
        int *cnt_local = (int*)calloc((size_t)K, sizeof(int));
        
        #pragma omp for
        for(int i=0;i<N;i++){
            int a = assign[i];
            cnt_local[a] += 1;
            sum_local[a] += X[i];
        }
        
        #pragma omp critical
        {
            for(int c=0; c<K; c++){
                cnt[c] += cnt_local[c];
                sum[c] += sum_local[c];
            }
        }
        
        free(sum_local);
        free(cnt_local);
    }
    
    for(int c=0;c<K;c++){
        if(cnt[c] > 0) C[c] = sum[c] / (double)cnt[c];
        else           C[c] = X[0];
    }
    free(sum);
    free(cnt);
}

static void kmeans_1d_omp(const double *X, double *C, int *assign,
                      int N, int K, int max_iter, double eps,
                      int *iters_out, double *sse_out)
{
    double prev_sse = 1e300;
    double sse = 0.0;
    int it;
    for(it=0; it<max_iter; it++){
        sse = assignment_step_1d_omp(X, C, assign, N, K);
        /* parada por variação relativa do SSE */
        double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if(rel < eps){ 
            it++;
            break;
        }
        update_step_1d_omp(X, C, assign, N, K);
        prev_sse = sse;
    }
    *iters_out = it;
    *sse_out = sse;
}

/* ---------- main ---------- */
int main(int argc, char **argv){
    if(argc < 3){
        printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]\n", argv[0]);
        printf("Obs: arquivos CSV com 1 coluna (1 valor por linha), sem cabeçalho.\n");
        printf("Compilar com: gcc -O2 -std=c99 -fopenmp kmeans_1d_omp.c -o kmeans_1d_omp -lm\n");
        return 1;
    }
    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc>3)? atoi(argv[3]) : 50;
    double eps   = (argc>4)? atof(argv[4]) : 1e-4;
    const char *outAssign   = (argc>5)? argv[5] : NULL;
    const char *outCentroid = (argc>6)? argv[6] : NULL;

    if(max_iter <= 0 || eps <= 0.0){
        fprintf(stderr,"Parâmetros inválidos: max_iter>0 e eps>0\n");
        return 1;
    }

    int N=0, K=0;
    double *X = read_csv_1col(pathX, &N);
    double *C = read_csv_1col(pathC, &K);
    int *assign = (int*)malloc((size_t)N * sizeof(int));
    if(!assign){ fprintf(stderr,"Sem memoria para assign\n"); free(X); free(C); return 1; }

    printf("K-means 1D Paralelizado (OpenMP)\n");
    printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, eps);
    printf("Threads disponíveis: %d\n", omp_get_max_threads());

    clock_t t0 = clock();
    int iters = 0; double sse = 0.0;
    kmeans_1d_omp(X, C, assign, N, K, max_iter, eps, &iters, &sse);
    clock_t t1 = clock();
    double ms = 1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    printf("Iterações: %d | SSE final: %.6f | Tempo: %.1f ms\n", iters, sse, ms);

    write_assign_csv(outAssign, assign, N);
    write_centroids_csv(outCentroid, C, K);

    free(assign); free(X); free(C);
    return 0;
}