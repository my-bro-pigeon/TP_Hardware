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