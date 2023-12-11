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



/// tester  la somme et la multiplication de matrices  

int main() {
    
    float *raw_data;
    float *d_raw_data; // Pointeur pour la matrice sur le GPU
    int nr=32;
    int pr=32;
    int lr=1;
    float *C1_data;
    float *d_C1_data;
    int nC1=28;
    int pC1=28;
    int lC1=6;
    float *S1_data;
    float *d_S1_data;
    int nS1=14;
    int pS1=14;
    int lS1=6;
    float *C1_kernel;
    float *d_C1_kernel;
    int nk=5;
    int pk=5;
    int lk=6;
    srand(time(NULL));

    // Allocation de mémoire pour la matrice sur le GPU
    cudaMalloc((void **)&d_raw_data, nr * pr * lr*sizeof(float));
    cudaMalloc((void **)&d_C1_data, nC1 * pC1 * lC1* sizeof(float));
    cudaMalloc((void **)&d_S1_data, nS1 * pS1 * lS1* sizeof(float));
    cudaMalloc((void **)&d_C1_kernel, nk * pk * lk* sizeof(float));

    // Allocation et initialisation de la matrice sur le CPU
    raw_data = (float *)malloc(nr * pr * lr*sizeof(float));
    C1_data = (float *)malloc(nC1 * pC1 * lC1*sizeof(float));
    S1_data = (float *)malloc(nS1 * pS1 * lS1*sizeof(float));
    C1_kernel = (float *)malloc(nk * pk * lk* sizeof(float));
    
    MatrixInit3D(raw_data, nr,pr,lr,0);
    MatrixInit3D(C1_data, nC1,pC1,lC1,1);
    MatrixInit3D(S1_data, nS1,pS1,lS1,1);
    MatrixInit3D(C1_kernel, nk,pk,lk,1);

    C1_kernel[12]=2;
    C1_kernel[25+12]=1;

    
    // Copie de la matrice du CPU vers le GPU
    cudaMemcpy(d_raw_data, raw_data, nr * pr * lr*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_data, C1_data, nC1 * pC1 * lC1* sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_S1_data, S1_data, nS1 * pS1 * lS1*  sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, nk * pk * lk* sizeof(float), cudaMemcpyHostToDevice);

    dim3 gridDim(nC1,nC1,lC1);
    Convolution2DGPU<<<gridDim,1>>>(d_raw_data, d_C1_kernel, d_C1_data,nr,nk);

    dim3 gridDim2(nS1,pS1,lS1);
    Moyennage2DGPU<<<gridDim2,1>>>(d_C1_data, d_S1_data, nC1);

    cudaMemcpy(raw_data, d_raw_data, nr * pr * lr* sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(C1_data, d_C1_data, nC1 * pC1 * lC1*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(S1_data, d_S1_data, nS1 * pS1 * lS1* sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(C1_kernel, d_C1_kernel, nk * pk * lk* sizeof(float), cudaMemcpyDeviceToHost);

    // Affichage de la matrice sur le CPU
    printf("Matrice raw_data :\n");

    MatrixPrint3D(raw_data, nr, pr,lr);

    // printf("Matrice C1_data :\n");

    // MatrixPrint3D(C1_data, nC1, pC1,lC1);

    printf("Matrice S1_data :\n");

    MatrixPrint3D(S1_data, nS1, pS1,lS1);

    printf("Matrice C1_kernel :\n");

    MatrixPrint3D(C1_kernel, nk, pk,lk);

    // Libération de la mémoire sur le CPU et le GPU
    free(raw_data);
    cudaFree(d_raw_data);

    free(C1_data);
    cudaFree(d_C1_data);

    free(S1_data);
    cudaFree(d_S1_data);

    free(C1_kernel);
    cudaFree(d_C1_kernel);

    return 0;

}

// / tester  la somme et la multiplication de matrices  
// int main() {
//     int n = 4;
//     int p = 4;
//     float *matrix;
//     float *d_matrix; // Pointeur pour la matrice sur le GPU

//     float *matrix2;
//     float *d_matrix2;

//     float *matrixOut;
//     float *d_matrixOut;

//     float *matrixOutMult;
//     float *d_matrixOutMult;
//     srand(time(NULL));

//     // Allocation de mémoire pour la matrice sur le GPU
//     cudaMalloc((void **)&d_matrix, n * p * sizeof(float));
//     cudaMalloc((void **)&d_matrix2, n * p * sizeof(float));
//     cudaMalloc((void **)&d_matrixOut, n * p * sizeof(float));
//     cudaMalloc((void **)&d_matrixOutMult, n * p * sizeof(float));

//     // Allocation et initialisation de la matrice sur le CPU
//     matrix = (float *)malloc(n * p * sizeof(float));
//     matrix2 = (float *)malloc(n * p * sizeof(float));
//     matrixOut = (float *)malloc(n * p * sizeof(float));
//     matrixOutMult = (float *)malloc(n * p * sizeof(float));

//     MatrixInit(matrix, n, p);
//     MatrixInit(matrix2, n, p);

//     // Copie de la matrice du CPU vers le GPU
//     cudaMemcpy(d_matrix, matrix, n * p * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_matrix2, matrix2, n * p * sizeof(float), cudaMemcpyHostToDevice);
//     //dim3 gridDim(n,p);

//     cudaMatrixAdd<<<n,p>>>(d_matrix, d_matrix2, d_matrixOut, n, p);
//     cudaMatrixMult<<<n,p>>>(d_matrix, d_matrix2, d_matrixOutMult, n);

//     cudaMemcpy(matrixOut, d_matrixOut, n * p * sizeof(float), cudaMemcpyDeviceToHost);
//     cudaMemcpy(matrixOutMult, d_matrixOutMult, n * p * sizeof(float), cudaMemcpyDeviceToHost);

//     // Affichage de la matrice sur le CPU
//     printf("Matrice 1 :\n");

//     MatrixPrint(matrix, n, p);

//     printf("Matrice 2 :\n");

//     MatrixPrint(matrix2, n, p);

//     printf("Somme :\n");

//     MatrixPrint(matrixOut, n, p);

//     printf("Mult :\n");

//     MatrixPrint(matrixOutMult, n, p);
//     // Libération de la mémoire sur le CPU et le GPU
//     free(matrix);
//     cudaFree(d_matrix);

//     free(matrix2);
//     cudaFree(d_matrix2);

//     free(matrixOut);
//     cudaFree(d_matrixOut);

//     free(matrixOutMult);
//     cudaFree(d_matrixOutMult);

//     return 0;

// }

/// TESTER LE TEMPS NECESSAIRE POUR LA SOMME ET LA MULTIPLICATION DE DEUX MATRICES AVEC ET SANS GPU

// int main() {
//     int n = 1000; // Taille des matrices (n x n)
//     int p = 500;  // Nombre de colonnes pour les matrices

//     // Allocation de mémoire pour les matrices sur le CPU
//     float *h_matrix1, *h_matrix2, *h_result_cpu_add, *h_result_cpu_mult;
//     h_matrix1 = (float *)malloc(n * p * sizeof(float));
//     h_matrix2 = (float *)malloc(n * p * sizeof(float));
//     h_result_cpu_add = (float *)malloc(n * p * sizeof(float));
//     h_result_cpu_mult = (float *)malloc(n * n * sizeof(float));

//     // Initialisation des matrices sur le CPU (ajustez selon votre méthode d'initialisation)
//     // ...

//     // Allocation de mémoire sur le GPU
//     float *d_matrix1, *d_matrix2, *d_result_gpu_add, *d_result_gpu_mult;
//     cudaMalloc((void **)&d_matrix1, n * p * sizeof(float));
//     cudaMalloc((void **)&d_matrix2, n * p * sizeof(float));
//     cudaMalloc((void **)&d_result_gpu_add, n * p * sizeof(float));
//     cudaMalloc((void **)&d_result_gpu_mult, n * n * sizeof(float));

//     // Copie des données du CPU vers le GPU
//     cudaMemcpy(d_matrix1, h_matrix1, n * p * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_matrix2, h_matrix2, n * p * sizeof(float), cudaMemcpyHostToDevice);

//     // Configuration des dimensions de la grille et des blocs pour l'addition sur GPU
//     dim3 blockDim_add(16, 16); // Par exemple, 16x16 threads par bloc
//     dim3 gridDim_add((n + blockDim_add.x - 1) / blockDim_add.x, (p + blockDim_add.y - 1) / blockDim_add.y);

//     // Mesure du temps pour la somme sur CPU
//     clock_t start_cpu_add = clock();
//     MatrixAdd(h_matrix1, h_matrix2, h_result_cpu_add, n, p);
//     clock_t end_cpu_add = clock();
//     double cpu_time_add = ((double)(end_cpu_add - start_cpu_add)) / CLOCKS_PER_SEC;

//     // Mesure du temps pour la somme sur GPU
//     clock_t start_gpu_add = clock();
//     cudaMatrixAdd<<<gridDim_add, blockDim_add>>>(d_matrix1, d_matrix2, d_result_gpu_add, n, p);
//     cudaDeviceSynchronize(); // Attente de la fin du kernel
//     clock_t end_gpu_add = clock();
//     double gpu_time_add = ((double)(end_gpu_add - start_gpu_add)) / CLOCKS_PER_SEC;

//     // Configuration des dimensions de la grille et des blocs pour la multiplication sur GPU
//     dim3 blockDim_mult(16, 16); // Par exemple, 16x16 threads par bloc
//     dim3 gridDim_mult((n + blockDim_mult.x - 1) / blockDim_mult.x, (n + blockDim_mult.y - 1) / blockDim_mult.y);

//     // Mesure du temps pour la multiplication sur CPU
//     clock_t start_cpu_mult = clock();
//     MatrixMult(h_matrix1, h_matrix2, h_result_cpu_mult, n);
//     clock_t end_cpu_mult = clock();
//     double cpu_time_mult = ((double)(end_cpu_mult - start_cpu_mult)) / CLOCKS_PER_SEC;

//     // Mesure du temps pour la multiplication sur GPU
//     clock_t start_gpu_mult = clock();
//     cudaMatrixMult<<<gridDim_mult, blockDim_mult>>>(d_matrix1, d_matrix2, d_result_gpu_mult, n);
//     cudaDeviceSynchronize(); // Attente de la fin du kernel
//     clock_t end_gpu_mult = clock();
//     double gpu_time_mult = ((double)(end_gpu_mult - start_gpu_mult)) / CLOCKS_PER_SEC;

//     // Affichage des temps pour chaque opération
//     printf("Temps pour la somme de matrices sur CPU : %.5f secondes\n", cpu_time_add);
//     printf("Temps pour la somme de matrices sur GPU : %.5f secondes\n", gpu_time_add);
//     printf("Temps pour la multiplication de matrices sur CPU : %.5f secondes\n", cpu_time_mult);
//     printf("Temps pour la multiplication de matrices sur GPU : %.5f secondes\n", gpu_time_mult);

//     // Libération de la mémoire sur le CPU et le GPU
//     free(h_matrix1);
//     free(h_matrix2);
//     free(h_result_cpu_add);
//     free(h_result_cpu_mult);

//     cudaFree(d_matrix1);
//     cudaFree(d_matrix2);
//     cudaFree(d_result_gpu_add);
//     cudaFree(d_result_gpu_mult);

//     return 0;
// }

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

/// TESTER LAFFICHAGE D4UNE MATRICE 


// int main() {
//     int n = 4;
//     int p = 8;
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