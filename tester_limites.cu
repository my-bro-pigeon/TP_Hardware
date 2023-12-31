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


/// TESTER LA PUISSANCE DU GPU ///

int main() {
    int max_size = 10000; // Taille maximale de la matrice
    int step = 100; // Pas d'incrémentation de la taille de la matrice

    for (int n = step; n <= max_size; n += step) {
        float *h_matrix1, *h_matrix2, *h_result;
        float *d_matrix1, *d_matrix2, *d_result;

        // Allocation et initialisation des matrices sur CPU
        h_matrix1 = (float *)malloc(n * n * sizeof(float));
        h_matrix2 = (float *)malloc(n * n * sizeof(float));
        h_result = (float *)malloc(n * n * sizeof(float));

        // Initialisation des matrices sur CPU
        // ... (code pour l'initialisation des matrices h_matrix1 et h_matrix2)

        // Allocation de mémoire sur le GPU
        cudaMalloc((void **)&d_matrix1, n * n * sizeof(float));
        cudaMalloc((void **)&d_matrix2, n * n * sizeof(float));
        cudaMalloc((void **)&d_result, n * n * sizeof(float));

        // Copie des données du CPU vers le GPU
        cudaMemcpy(d_matrix1, h_matrix1, n * n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_matrix2, h_matrix2, n * n * sizeof(float), cudaMemcpyHostToDevice);

        // Configuration des dimensions de la grille et des blocs pour la multiplication
        dim3 blockDim(16, 16); // Par exemple, 16x16 threads par bloc
        dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

        // Mesure du temps pour la multiplication sur GPU
        clock_t start = clock();
        cudaMatrixMult<<<gridDim, blockDim>>>(d_matrix1, d_matrix2, d_result, n);
        cudaDeviceSynchronize(); // Attente de la fin du kernel
        clock_t end = clock();

        double time_spent = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("Taille de la matrice : %d x %d, Temps écoulé : %.5f secondes\n", n, n, time_spent);

        // Libération de la mémoire sur le GPU
        cudaFree(d_matrix1);
        cudaFree(d_matrix2);
        cudaFree(d_result);

        // Libération de la mémoire sur le CPU
        free(h_matrix1);
        free(h_matrix2);
        free(h_result);
    }

    return 0;
}