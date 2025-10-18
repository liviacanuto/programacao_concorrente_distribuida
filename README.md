### K-means 1D Paralelizado

Este projeto implementa o algoritmo K-means em 1 dimensão paralelizado como parte do trabalho final da disciplina Programação Concorrente e Distribuída. O código lê os dados e centróides iniciais de arquivos CSV, realiza as etapas de assignment e update iterativamente até convergir ou atingir o número máximo de iterações, e salva os resultados (opcionalmente) em arquivos de saída.


### Pré-requisitos

Certifique-se de ter instalado:

- **GCC** com suporte a OpenMP  
- Arquivos de entrada:
  - dados.csv — contém os pontos (1 por linha)
  - centroides_iniciais.csv — contém os centróides iniciais (1 por linha)

### Como executar

- **Clone o repositório**
   ```bash
   git https://github.com/liviacanuto/programacao_concorrente_distribuida.git
   cd programacao_concorrente_distribuida
   ```

- **Garanta que os arquivos de código e de dados estejam na mesma pasta**

#### Etapa 1 - OpenMP

#### 1.1. **Compile o código**
   ```bash
   gcc -O2 -std=c99 -fopenmp kmeans_1d_omp.c -o kmeans_1d_omp -lm
   ```

#### 1.2 **Execute o programa**
   ```bash
   OMP_NUM_THREADS=4 ./kmeans_1d_omp dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]
   ```

   - `OMP_NUM_THREADS` → define o número de threads utilizadas.  
   - `max_iter` → número máximo de iterações (opcional, padrão = 50).  
   - `eps` → tolerância para parada (opcional, padrão = 1e-4).  
   - `assign.csv` → (opcional) arquivo de saída com as atribuições de clusters.  
   - `centroids.csv` → (opcional) arquivo de saída com os centróides finais.  

   Exemplo:
   ```bash
   OMP_NUM_THREADS=8 ./kmeans_1d_omp dados.csv centroides_iniciais.csv 100 1e-5 resultado_assign.csv resultado_centroides.csv
   ```

#### 1.3 Alternar entre versões do update

Por padrão, o código utiliza a versão **com acumuladores por thread** (`update_step_1d_omp`), que oferece melhor desempenho.
Caso queira visualizar a **versão com seção crítica** (`update_step_1d_omp_critical`), siga os passos:

1.3.1 No arquivo `kmeans_1d_omp.c`, vá até a função:
   ```c
   kmeans_1d_omp(...)
   ```
1.3.2 Substitua a chamada:
   ```c
   update_step_1d_omp(X, C, assign, N, K);
   ```
   por:
   ```c
   update_step_1d_omp_critical(X, C, assign, N, K);
   ```

1.3.3 **Recompile o código:**
   ```bash
   gcc -O2 -std=c99 -fopenmp kmeans_1d_omp.c -o kmeans_1d_omp -lm
   ```

⚠️ **Importante:** sempre recompile após alterar a função de update.
