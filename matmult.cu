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
                    M[i * p * n + j * n + k] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
                    //M[i * p * n + j * n + k] = 1;
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



void MatrixPrint3D(float *M, int n, int p, int l) {
    for (int i = 0; i < l; ++i) {
        printf("Matrice %d :\n", i + 1);
        for (int j = 0; j < p; ++j) {
            for (int k = 0; k < n; ++k) {
                printf("%.2f\t", M[i * p * n + j * n + k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

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

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout) {
    int n = gridDim.x;
    int p = gridDim.y;
    int l = gridDim.z;
    int size = n*p*l;
    int idx = blockIdx.z * n * p + blockIdx.y * n + blockIdx.x;
    if (idx < size) {
        Mout[idx] = M1[idx]+M2[idx];
    }
}


__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int len) {
    int n = gridDim.x;
    int p = gridDim.y;

    int k = blockIdx.x;
    int j = blockIdx.y;

    if (k < n && j < p) {
        float sum = 0.0f;
        for (int h = 0; h < len; ++h) {
            sum += M1[j * p + h] * M2[h * n + k];
        }
        Mout[j * n + k] = sum;
    }
}

__device__ float activation_tanh(float M) {
    return tanh(M);
}

int main() {
    int n = 500;
    int p = 500;

    int n1 = 500;
    int p1 = 500;

    int nout = 500;
    int pout = 500;
    float *matrix;
    float *d_matrix; // Pointeur pour la matrice sur le GPU

    float *matrix2;
    float *d_matrix2;

    float *cpu_out;

    // float *matrixOut;
    // float *d_matrixOut;

    float *matrixOutMult;
    float *d_matrixOutMult;
    srand(time(NULL));

    // Allocation de mémoire pour la matrice sur le GPU
    cudaMalloc((void **)&d_matrix, n * p * sizeof(float));
    cudaMalloc((void **)&d_matrix2, n1 * p1 * sizeof(float));
    //cudaMalloc((void **)&d_matrixOut, n * p * sizeof(float));
    cudaMalloc((void **)&d_matrixOutMult, nout * pout * sizeof(float));

    // Allocation et initialisation de la matrice sur le CPU
    matrix = (float *)malloc(n * p * sizeof(float));
    matrix2 = (float *)malloc(n1 * p1 * sizeof(float));
    cpu_out = (float *)malloc(n * p * sizeof(float));
    matrixOutMult = (float *)malloc(nout * pout  * sizeof(float));

    MatrixInit3D(matrix, n, p,1,0);
    MatrixInit3D(matrix2, n1, p1,1,0);
    MatrixInit3D(cpu_out, n, p,1,0);
    MatrixInit3D(matrixOutMult, nout, pout,1,0);

    // Mesure du temps pour la multiplication sur CPU
    clock_t start_cpu_mult = clock();
    MatrixMult(matrix, matrix2, cpu_out, n);
    clock_t end_cpu_mult = clock();
    double cpu_time_mult = ((double)(end_cpu_mult - start_cpu_mult)) / CLOCKS_PER_SEC;


    // Copie de la matrice du CPU vers le GPU
    cudaMemcpy(d_matrix, matrix, n * p * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix2, matrix2, n1 * p1 * sizeof(float), cudaMemcpyHostToDevice);
    //dim3 gridDim(n,p);

    //cudaMatrixAdd<<<n,p>>>(d_matrix, d_matrix2, d_matrixOut, n, p);
    clock_t start_gpu_mult = clock();
    dim3 gridDim(nout,pout,1);
    cudaMatrixMult<<<gridDim,1>>>(d_matrix, d_matrix2, d_matrixOutMult, n);
    cudaDeviceSynchronize(); // Attente de la fin du kernel
    clock_t end_gpu_mult = clock();
    double gpu_time_mult = ((double)(end_gpu_mult - start_gpu_mult)) / CLOCKS_PER_SEC;

    
    //cudaMemcpy(matrixOut, d_matrixOut, n * p * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(matrixOutMult, d_matrixOutMult, nout*pout * sizeof(float), cudaMemcpyDeviceToHost);

    // Affichage de la matrice sur le CPU
    // printf("Matrice 1 :\n");

    // MatrixPrint3D(matrix, n, p,1);

    // printf("Matrice 2 :\n");

    // MatrixPrint3D(matrix2, n1, p1,1);

    // // printf("Somme :\n");

    // // MatrixPrint(matrixOut, n, p);
    // printf("Mult CPU:\n");

    // MatrixPrint3D(cpu_out, n, p,1);

    // printf("Mult GPU:\n");
    // MatrixPrint3D(matrixOutMult, nout, pout,1);

    printf("Temps pour la multiplication de matrices sur CPU : %.5f secondes\n", cpu_time_mult);
    printf("Temps pour la multiplication de matrices sur GPU : %.5f secondes\n", gpu_time_mult);

    
    // Libération de la mémoire sur le CPU et le GPU
    free(matrix);
    cudaFree(d_matrix);

    free(matrix2);
    cudaFree(d_matrix2);

    // free(matrixOut);
    // cudaFree(d_matrixOut);

    free(cpu_out);

    free(matrixOutMult);
    cudaFree(d_matrixOutMult);

    return 0;

}

