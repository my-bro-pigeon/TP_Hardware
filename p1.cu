#include <stdlib.h>
#include <time.h>
#include <stdio.h>

void MatrixInit(float *M, int n, int p){
    
    for(int i=0; i<n; i++){
        for(int j=0; j<p;j++){
            M[i*p+j] = ((float)rand() / RAND_MAX)*2.0f - 1.0f;
        }
    }
}

void MatrixPrint(float *M, int n, int p){
    for(int i=0; i<n; i++){
        for(int j=0; j<p;j++){
            printf("%.2f\t", M[i*p+j]);
        }
        printf("\n");
    }
}


// Fonction d'addition de deux matrices sur CPU
void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            Mout[i * p + j] = M1[i * p + j] + M2[i * p + j];
        }
    }
}

// Fonction de multiplication de deux matrices NxN sur CPU
void MatrixMult(float *M1, float *M2, float *Mout, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += M1[i * n + k] * M2[k * n + j];
            }
            Mout[i * n + j] = sum;
        }
    }
}

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < p) {
        Mout[i * p + j] = M1[i * p + j] + M2[i * p + j];
    }
}

__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += M1[row * n + k] * M2[k * n + col];
        }
        Mout[row * n + col] = sum;
    }
}
/// TESTER LAFFICHAGE D4UNE MATRICE 


// int main() {
//     int n = 3;
//     int p = 4;
//     float *matrix;
//     float *d_matrix; // Pointeur pour la matrice sur le GPU

//     // Allocation de mémoire pour la matrice sur le GPU
//     cudaMalloc((void **)&d_matrix, n * p * sizeof(float));

//     // Allocation et initialisation de la matrice sur le CPU
//     matrix = (float *)malloc(n * p * sizeof(float));

//     MatrixInit(matrix, n, p);

//     // Copie de la matrice du CPU vers le GPU
//     cudaMemcpy(d_matrix, matrix, n * p * sizeof(float), cudaMemcpyHostToDevice);
//     // Affichage de la matrice sur le CPU
//     printf("Matrice :\n");

//     MatrixPrint(matrix, n, p);
//     // Libération de la mémoire sur le CPU et le GPU
//     free(matrix);
//     cudaFree(d_matrix);
//     return 0;

// }


/// TESTER LE TEMPS NECESSAIRE POUR LA SOMME ET LA MULTIPLICATION DE DEUX MATRICES AVEC ET SANS GPU

int main() {
    int n = 1000; // Taille des matrices (n x n)
    int p = 500;  // Nombre de colonnes pour les matrices

    // Allocation de mémoire pour les matrices sur le CPU
    float *h_matrix1, *h_matrix2, *h_result_cpu_add, *h_result_cpu_mult;
    h_matrix1 = (float *)malloc(n * p * sizeof(float));
    h_matrix2 = (float *)malloc(n * p * sizeof(float));
    h_result_cpu_add = (float *)malloc(n * p * sizeof(float));
    h_result_cpu_mult = (float *)malloc(n * n * sizeof(float));

    // Initialisation des matrices sur le CPU (ajustez selon votre méthode d'initialisation)
    // ...

    // Allocation de mémoire sur le GPU
    float *d_matrix1, *d_matrix2, *d_result_gpu_add, *d_result_gpu_mult;
    cudaMalloc((void **)&d_matrix1, n * p * sizeof(float));
    cudaMalloc((void **)&d_matrix2, n * p * sizeof(float));
    cudaMalloc((void **)&d_result_gpu_add, n * p * sizeof(float));
    cudaMalloc((void **)&d_result_gpu_mult, n * n * sizeof(float));

    // Copie des données du CPU vers le GPU
    cudaMemcpy(d_matrix1, h_matrix1, n * p * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix2, h_matrix2, n * p * sizeof(float), cudaMemcpyHostToDevice);

    // Configuration des dimensions de la grille et des blocs pour l'addition sur GPU
    dim3 blockDim_add(16, 16); // Par exemple, 16x16 threads par bloc
    dim3 gridDim_add((n + blockDim_add.x - 1) / blockDim_add.x, (p + blockDim_add.y - 1) / blockDim_add.y);

    // Mesure du temps pour la somme sur CPU
    clock_t start_cpu_add = clock();
    MatrixAdd(h_matrix1, h_matrix2, h_result_cpu_add, n, p);
    clock_t end_cpu_add = clock();
    double cpu_time_add = ((double)(end_cpu_add - start_cpu_add)) / CLOCKS_PER_SEC;

    // Mesure du temps pour la somme sur GPU
    clock_t start_gpu_add = clock();
    cudaMatrixAdd<<<gridDim_add, blockDim_add>>>(d_matrix1, d_matrix2, d_result_gpu_add, n, p);
    cudaDeviceSynchronize(); // Attente de la fin du kernel
    clock_t end_gpu_add = clock();
    double gpu_time_add = ((double)(end_gpu_add - start_gpu_add)) / CLOCKS_PER_SEC;

    // Configuration des dimensions de la grille et des blocs pour la multiplication sur GPU
    dim3 blockDim_mult(16, 16); // Par exemple, 16x16 threads par bloc
    dim3 gridDim_mult((n + blockDim_mult.x - 1) / blockDim_mult.x, (n + blockDim_mult.y - 1) / blockDim_mult.y);

    // Mesure du temps pour la multiplication sur CPU
    clock_t start_cpu_mult = clock();
    MatrixMult(h_matrix1, h_matrix2, h_result_cpu_mult, n);
    clock_t end_cpu_mult = clock();
    double cpu_time_mult = ((double)(end_cpu_mult - start_cpu_mult)) / CLOCKS_PER_SEC;

    // Mesure du temps pour la multiplication sur GPU
    clock_t start_gpu_mult = clock();
    cudaMatrixMult<<<gridDim_mult, blockDim_mult>>>(d_matrix1, d_matrix2, d_result_gpu_mult, n);
    cudaDeviceSynchronize(); // Attente de la fin du kernel
    clock_t end_gpu_mult = clock();
    double gpu_time_mult = ((double)(end_gpu_mult - start_gpu_mult)) / CLOCKS_PER_SEC;

    // Affichage des temps pour chaque opération
    printf("Temps pour la somme de matrices sur CPU : %.5f secondes\n", cpu_time_add);
    printf("Temps pour la somme de matrices sur GPU : %.5f secondes\n", gpu_time_add);
    printf("Temps pour la multiplication de matrices sur CPU : %.5f secondes\n", cpu_time_mult);
    printf("Temps pour la multiplication de matrices sur GPU : %.5f secondes\n", gpu_time_mult);

    // Libération de la mémoire sur le CPU et le GPU
    free(h_matrix1);
    free(h_matrix2);
    free(h_result_cpu_add);
    free(h_result_cpu_mult);

    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_result_gpu_add);
    cudaFree(d_result_gpu_mult);

    return 0;
}

/// TESTER LA PUISSANCE DU GPU ///

// int main() {
//     int max_size = 10000; // Taille maximale de la matrice
//     int step = 100; // Pas d'incrémentation de la taille de la matrice

//     for (int n = step; n <= max_size; n += step) {
//         float *h_matrix1, *h_matrix2, *h_result;
//         float *d_matrix1, *d_matrix2, *d_result;

//         // Allocation et initialisation des matrices sur CPU
//         h_matrix1 = (float *)malloc(n * n * sizeof(float));
//         h_matrix2 = (float *)malloc(n * n * sizeof(float));
//         h_result = (float *)malloc(n * n * sizeof(float));

//         // Initialisation des matrices sur CPU
//         // ... (code pour l'initialisation des matrices h_matrix1 et h_matrix2)

//         // Allocation de mémoire sur le GPU
//         cudaMalloc((void **)&d_matrix1, n * n * sizeof(float));
//         cudaMalloc((void **)&d_matrix2, n * n * sizeof(float));
//         cudaMalloc((void **)&d_result, n * n * sizeof(float));

//         // Copie des données du CPU vers le GPU
//         cudaMemcpy(d_matrix1, h_matrix1, n * n * sizeof(float), cudaMemcpyHostToDevice);
//         cudaMemcpy(d_matrix2, h_matrix2, n * n * sizeof(float), cudaMemcpyHostToDevice);

//         // Configuration des dimensions de la grille et des blocs pour la multiplication
//         dim3 blockDim(16, 16); // Par exemple, 16x16 threads par bloc
//         dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

//         // Mesure du temps pour la multiplication sur GPU
//         clock_t start = clock();
//         cudaMatrixMult<<<gridDim, blockDim>>>(d_matrix1, d_matrix2, d_result, n);
//         cudaDeviceSynchronize(); // Attente de la fin du kernel
//         clock_t end = clock();

//         double time_spent = ((double)(end - start)) / CLOCKS_PER_SEC;
//         printf("Taille de la matrice : %d x %d, Temps écoulé : %.5f secondes\n", n, n, time_spent);

//         // Libération de la mémoire sur le GPU
//         cudaFree(d_matrix1);
//         cudaFree(d_matrix2);
//         cudaFree(d_result);

//         // Libération de la mémoire sur le CPU
//         free(h_matrix1);
//         free(h_matrix2);
//         free(h_result);
//     }

//     return 0;
// }