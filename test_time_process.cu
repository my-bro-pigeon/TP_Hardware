#include <stdlib.h>
#include <time.h>
#include <stdio.h>

//val = 1 : initialise la matrice à 0, 
//val = 0 : initialise la matrice avec des valeurs comprises entre 0-1, 

void MatrixInit3D(float *M, int n, int p, int l,int val) { 
    if(val==0){
        for (int i = 0; i < l; i++) {
            for (int j = 0; j < p; j++) {
                for (int k = 0; k < n; k++) {
                    //M[i * p * n + j * n + k] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
                    M[i * p * n + j * n + k] = 1;
                }
            }
        }
    }
    else{
        for (int i = 0; i < l; i++) {
            for (int j = 0; j < p; j++) {
                for (int k = 0; k < n; k++) {
                    M[i * p * n + j * n + k] = 0;
                    }
                }
            }
    }
}

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

void MatrixPrint3D(float *M, int n, int p, int l) {
    for (int i = 0; i < l; ++i) {
        printf("Matrice %d :\n", i + 1);
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < p; ++k) {
                printf("%.2f\t", M[i * p * n + k * n + j]);
            }
            printf("\n");
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
    int i = blockIdx.x;
    int j = threadIdx.x;

    if (i < n && j < p) {
        Mout[i * p + j] = M1[i * p + j] + M2[i * p + j];
    }
}

__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += M1[row * n + k] * M2[k * n + col];
        }
        Mout[row * n + col] = sum;
    }
}

__device__ float activation_tanh(float M) {
    return tanh(M);
}


__global__ void Convolution2DGPU(float *input, float *kernels, float *output,int inputWidth, int kernelSize) { 

    int n = gridDim.x;
    int p = gridDim.y;
    //int l = gridDim.z;
    int outputIdx = blockIdx.z * n * p + blockIdx.y * n + blockIdx.x;
    int x = blockIdx.x;
    int y = blockIdx.y;
    int z = blockIdx.z;

    float sum=0.0f;
    for (int ky = 0; ky < kernelSize; ++ky) {
        for (int kx = 0; kx < kernelSize; ++kx) {
            int inputX = x + kx;
            int inputY = y + ky;
            int inputIdx = inputY * inputWidth + inputX;
            int kernelIdx = z*kernelSize*kernelSize + ky*kernelSize+kx;
            sum += input[inputIdx] * kernels[kernelIdx];
        }
    }
    output[outputIdx] = activation_tanh(sum);
}
    

__global__ void Moyennage2DGPU(float *input, float *output,int inputWidth) { 

    int n = gridDim.x; //14
    int p = gridDim.y; //14
    //int n = 14; //14
    //int p = 14; 
    int outputIdx = blockIdx.z * n * p + blockIdx.y * n + blockIdx.x; //pour parcourir mat sortie
    int x = blockIdx.x;
    int y = blockIdx.y;
    int z = blockIdx.z;

    float sum=0.0f;
    for (int ky = 0; ky < 2; ++ky) {
        for (int kx = 0; kx < 2; ++kx) {
            int inputX = 2*x + kx;
            int inputY = 2*y + ky;
            int inputIdx = z*inputWidth*inputWidth + inputY * inputWidth + inputX;
            sum += input[inputIdx];
        }
    }
    output[outputIdx] = sum/4;
}


/ TESTER LE TEMPS NECESSAIRE POUR LA SOMME ET LA MULTIPLICATION DE DEUX MATRICES AVEC ET SANS GPU

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