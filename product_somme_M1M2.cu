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




/ tester  la somme et la multiplication de matrices  
int main() {
    int n = 4;
    int p = 4;
    float *matrix;
    float *d_matrix; // Pointeur pour la matrice sur le GPU

    float *matrix2;
    float *d_matrix2;

    float *matrixOut;
    float *d_matrixOut;

    float *matrixOutMult;
    float *d_matrixOutMult;
    srand(time(NULL));

    // Allocation de mémoire pour la matrice sur le GPU
    cudaMalloc((void **)&d_matrix, n * p * sizeof(float));
    cudaMalloc((void **)&d_matrix2, n * p * sizeof(float));
    cudaMalloc((void **)&d_matrixOut, n * p * sizeof(float));
    cudaMalloc((void **)&d_matrixOutMult, n * p * sizeof(float));

    // Allocation et initialisation de la matrice sur le CPU
    matrix = (float *)malloc(n * p * sizeof(float));
    matrix2 = (float *)malloc(n * p * sizeof(float));
    matrixOut = (float *)malloc(n * p * sizeof(float));
    matrixOutMult = (float *)malloc(n * p * sizeof(float));

    MatrixInit(matrix, n, p);
    MatrixInit(matrix2, n, p);

    // Copie de la matrice du CPU vers le GPU
    cudaMemcpy(d_matrix, matrix, n * p * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix2, matrix2, n * p * sizeof(float), cudaMemcpyHostToDevice);
    //dim3 gridDim(n,p);

    cudaMatrixAdd<<<n,p>>>(d_matrix, d_matrix2, d_matrixOut, n, p);
    cudaMatrixMult<<<n,p>>>(d_matrix, d_matrix2, d_matrixOutMult, n);

    cudaMemcpy(matrixOut, d_matrixOut, n * p * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(matrixOutMult, d_matrixOutMult, n * p * sizeof(float), cudaMemcpyDeviceToHost);

    // Affichage de la matrice sur le CPU
    printf("Matrice 1 :\n");

    MatrixPrint(matrix, n, p);

    printf("Matrice 2 :\n");

    MatrixPrint(matrix2, n, p);

    printf("Somme :\n");

    MatrixPrint(matrixOut, n, p);

    printf("Mult :\n");

    MatrixPrint(matrixOutMult, n, p);
    // Libération de la mémoire sur le CPU et le GPU
    free(matrix);
    cudaFree(d_matrix);

    free(matrix2);
    cudaFree(d_matrix2);

    free(matrixOut);
    cudaFree(d_matrixOut);

    free(matrixOutMult);
    cudaFree(d_matrixOutMult);

    return 0;

}