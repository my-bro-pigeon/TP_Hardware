#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#define N 32
#define P 32
#define Q 6
#define K 5
#define WIDTH 28
#define HEIGHT 28
#define NO_IMG 5

void readFile(char* path, double * out){
    FILE *f = fopen(path, "r");

    if (f == NULL)
    {
        printf("Error: could not open file %s", path);
    }
    int i =0;

    while ((fscanf(f,"%lf", &out[i])) != EOF){
        i++;
    }
    fclose(f);
}

void readImage(double * data, int no_img){
    FILE *fptr;
    unsigned int magic, nbImg, nbRows, nbCols;
    unsigned char val;

    //Open File
    if((fptr = fopen("train-images.idx3-ubyte","rb")) == NULL){
        printf("Can't open file");
        exit(1);
    }

    //Read File
    fread(&magic, sizeof(int), 1, fptr);
    fread(&nbImg, sizeof(int), 1, fptr);
    fread(&nbRows, sizeof(int), 1, fptr);
    fread(&nbCols, sizeof(int), 1, fptr);

    for(int k=0; k<no_img; k++){

        for(int i=2; i<WIDTH+2; i++){
            for(int j=2; j<HEIGHT+2; j++){ 
                fread(&val, sizeof(unsigned char), 1, fptr);  
                data[i*P+j]=(double)val/255;
            }
        }
    }  
}


//val = 1 : initialise la matrice à 0,
//val = 0 : initialise la matrice avec des valeurs comprises entre 0-1,

void MatrixInit3D(double *M, int n, int p, int l,int val) {
    if(val==0){
        for (int i = 0; i < l; i++) {
            for (int j = 0; j < p; j++) {
                for (int k = 0; k < n; k++) {
                    //M[i * p * n + j * n + k] = ((double)rand() / RAND_MAX) * 2.0f - 1.0f;
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

void MatrixPrint3D(double *M, int n, int p, int l) {
    for (int i = 0; i < l; ++i) {
        printf("Matrice %d :\n", i + 1);
        for (int j = 0; j < p; ++j) {
            for (int k = 0; k < n; ++k) {
                printf("%lf\t", M[i * p * n + j * n + k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}


__global__ void cudaMatrixAdd(double *M1, double *M2, double *Mout) {
    int n = gridDim.x;
    int p = gridDim.y;
    int l = gridDim.z;
    int size = n*p*l;
    int idx = blockIdx.z * n * p + blockIdx.y * n + blockIdx.x;
    if (idx < size) {
        Mout[idx] = M1[idx]+M2[idx];
    }
}

__global__ void cudaMatrixMult(double *M1, double *M2, double *Mout, int len) {
    int n = gridDim.x;
    int p = gridDim.y;

    int k = blockIdx.x;
    int j = blockIdx.y;

    if (k < n && j < p) {
        double sum = 0.0;
        for (int h = 0; h < len; ++h) {
            sum += M1[j * p + h] * M2[h * n + k];
        }
        Mout[j * n + k] = sum;
    }
}
__device__ double activation_tanh(double M) {
    return tanh(M);
}


__global__ void Convolution2DGPU(double *input, double *kernels, double *output,int inputWidth, int kernelSize) {

    int n = gridDim.x;
    int p = gridDim.y;
    //int l = gridDim.z;
    int outputIdx = blockIdx.z * n * p + blockIdx.y * n + blockIdx.x;
    int x = blockIdx.x;
    int y = blockIdx.y;
    int z = blockIdx.z;

    double sum=0.0f;
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

__global__ void Convolution3DGPU(double *input, double *kernels, double *output,int inputWidth, int kernelSize, int kerneldepth) {

    int n = gridDim.x;
    int p = gridDim.y;
    //int l = gridDim.z;
    int outputIdx = blockIdx.z * n * p + blockIdx.y * n + blockIdx.x;
    int x = blockIdx.x;
    int y = blockIdx.y;
    int z = blockIdx.z;

    double sum=0.0f;
    for(int kz = 0; kz < kerneldepth; ++kz){ 
        for (int ky = 0; ky < kernelSize; ++ky) {
            for (int kx = 0; kx < kernelSize; ++kx) {
                int inputX = x + kx;
                int inputY = y + ky;
                int inputIdx = inputY * inputWidth + inputX + kz*inputWidth*inputWidth;
                int kernelIdx = z*kernelSize*kernelSize*kerneldepth + kz*kernelSize*kernelSize + ky*kernelSize + kx;
                sum += input[inputIdx] * kernels[kernelIdx];
            }
        }
    } 
    output[outputIdx] = activation_tanh(sum);
}

__global__ void Moyennage2DGPU(double *input, double *output,int inputWidth) {

    int n = gridDim.x; //14
    int p = gridDim.y; //14
    //int n = 14; //14
    //int p = 14;
    int outputIdx = blockIdx.z * n * p + blockIdx.y * n + blockIdx.x; //pour parcourir mat sortie
    int x = blockIdx.x;
    int y = blockIdx.y;
    int z = blockIdx.z;

    double sum=0.0f;
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

__global__ void Flatten(double *input, double *output) {
    int n = gridDim.x;
    int p = gridDim.y;
    int l = gridDim.z;
    int size = n*p*l;
    int idx = blockIdx.z * n * p + blockIdx.y * n + blockIdx.x;
    if (idx < size) {
        output[idx] = input[idx];
    }
}

/// modèle global

int main() {


    double *raw_data;
    double *d_raw_data; // Pointeur pour la matrice sur le GPU
    int nr=32;
    int pr=32;
    int lr=1;
    double *C1_data;
    double *d_C1_data;
    int nC1=28;
    int pC1=28;
    int lC1=6;
    double *S1_data;
    double *d_S1_data;
    int nS1=14;
    int pS1=14;
    int lS1=6;
    double *C1_kernel;
    double *d_C1_kernel;
    int nk=5;
    int pk=5;
    int lk=6;
    double *Conv2_data;
    double *d_Conv2_data;
    int nC2=10;
    int pC2=10;
    int lC2=16;
    double *S2_data;
    double *d_S2_data;
    int nS2=5;
    int pS2=5;
    int lS2=16;
    double *Conv2_kernel;
    double *d_Conv2_kernel;
    int nk2=5;
    int pk2=5;
    int lk2=16;
    int depthk2=6;
    double *flatten_data;
    double *d_flatten_data;
    int nf=400;
    int pf=1;
    int lf=1;

    //couche dense 1
    double *dense1_weight;
    double *d_dense1_weight;
    int ndw1=120;
    int pdw1=400;
    int ldw1=1;
    double *dense1_bias;
    double *d_dense1_bias;
    int ndb1=120;
    int pdb1=1;
    int ldb1=1;
    double *dense1_data;
    double *d_dense1_data;
    int ndd1=120;
    int pdd1=1;
    int ldd1=1;

    //couche dense 2
    double *dense2_weight;
    double *d_dense2_weight;
    int ndw2=84;
    int pdw2=120;
    int ldw2=1;
    double *dense2_bias;
    double *d_dense2_bias;
    int ndb2=84;
    int pdb2=1;
    int ldb2=1;
    double *dense2_data;
    double *d_dense2_data;
    int ndd2=84;
    int pdd2=1;
    int ldd2=1;

    //couche dense 3
    double *dense3_weight;
    double *d_dense3_weight;
    int ndw3=10;
    int pdw3=84;
    int ldw3=1;
    double *dense3_bias;
    double *d_dense3_bias;
    int ndb3=10;
    int pdb3=1;
    int ldb3=1;
    double *dense3_data;
    double *d_dense3_data;
    int ndd3=10;
    int pdd3=1;
    int ldd3=1;


    srand(time(NULL));

    // Allocation de mémoire pour la matrice sur le GPU
    cudaMalloc((void **)&d_raw_data, nr * pr * lr*sizeof(double));
    cudaMalloc((void **)&d_C1_data, nC1 * pC1 * lC1* sizeof(double));
    cudaMalloc((void **)&d_S1_data, nS1 * pS1 * lS1* sizeof(double));
    cudaMalloc((void **)&d_C1_kernel, nk * pk * lk* sizeof(double));
    cudaMalloc((void **)&d_Conv2_data, nC2 * pC2 * lC2* sizeof(double));
    cudaMalloc((void **)&d_S2_data, nS2 * pS2 * lS2* sizeof(double));
    cudaMalloc((void **)&d_Conv2_kernel, nk2 * pk2 * lk2* depthk2* sizeof(double));
    cudaMalloc((void **)&d_flatten_data, nf * pf * lf* sizeof(double));

    cudaMalloc((void **)&d_dense1_weight, ndw1 * pdw1 * ldw1* sizeof(double));
    cudaMalloc((void **)&d_dense2_weight, ndw2 * pdw2 * ldw2* sizeof(double));
    cudaMalloc((void **)&d_dense3_weight, ndw3 * pdw3 * ldw3* sizeof(double));

    cudaMalloc((void **)&d_dense1_bias, ndb1 * pdb1 * ldb1* sizeof(double));
    cudaMalloc((void **)&d_dense2_bias, ndb2 * pdb2 * ldb2* sizeof(double));
    cudaMalloc((void **)&d_dense3_bias, ndb3 * pdb3 * ldb3* sizeof(double));

    cudaMalloc((void **)&d_dense1_data, ndd1 * pdd1 * ldd1* sizeof(double));
    cudaMalloc((void **)&d_dense2_data, ndd2 * pdd2 * ldd2* sizeof(double));
    cudaMalloc((void **)&d_dense3_data, ndd3 * pdd3 * ldd3* sizeof(double));

    // Allocation et initialisation de la matrice sur le CPU
    raw_data = (double *)malloc(nr * pr * lr*sizeof(double));
    
    C1_data = (double *)malloc(nC1 * pC1 * lC1*sizeof(double));
    S1_data = (double *)malloc(nS1 * pS1 * lS1*sizeof(double));
    C1_kernel = (double *)malloc(nk * pk * lk* sizeof(double));
    

    Conv2_data = (double *)malloc(nC2 * pC2 * lC2*sizeof(double));
    S2_data = (double *)malloc(nS2 * pS2 * lS2*sizeof(double));
    Conv2_kernel = (double *)malloc(nk2 * pk2 * lk2*depthk2* sizeof(double));
    flatten_data = (double *)malloc(nf * pf * lf* sizeof(double));

    dense1_weight = (double *)malloc(ndw1 * pdw1 * ldw1* sizeof(double));
    dense1_bias = (double *)malloc(ndb1 * pdb1 * ldb1* sizeof(double));
    dense1_data = (double *)malloc(ndd1 * pdd1 * ldd1* sizeof(double));

    dense2_weight = (double *)malloc(ndw2 * pdw2 * ldw2* sizeof(double));
    dense2_bias = (double *)malloc(ndb2 * pdb2 * ldb2* sizeof(double));
    dense2_data = (double *)malloc(ndd2 * pdd2 * ldd2* sizeof(double));

    dense3_weight = (double *)malloc(ndw3 * pdw3 * ldw3* sizeof(double));
    dense3_bias = (double *)malloc(ndb3 * pdb3 * ldb3* sizeof(double));
    dense3_data = (double *)malloc(ndd3 * pdd3 * ldd3* sizeof(double));

    MatrixInit3D(raw_data, nr,pr,lr,1);
    readImage(raw_data, 5);
    MatrixInit3D(C1_data, nC1,pC1,lC1,1);
    MatrixInit3D(S1_data, nS1,pS1,lS1,1);
    MatrixInit3D(C1_kernel, nk,pk,lk,1);
    readFile((char *)"weights_nobias/k1.h", C1_kernel);

    MatrixInit3D(Conv2_data, nC2,pC2,lC2,1);
    MatrixInit3D(S2_data, nS2,pS2,lS2,1);
    MatrixInit3D(Conv2_kernel, nk2,pk2,lk2*depthk2,1);
    readFile((char *)"weights_nobias/k2.h", Conv2_kernel);

    MatrixInit3D(flatten_data, nf,pf,lf,1);

    MatrixInit3D(dense1_weight, ndw1,pdw1,ldw1,1);
    readFile((char *)"weights_nobias/w1.h", dense1_weight);
    MatrixInit3D(dense1_bias, ndb1,pdb1,ldb1,1);
    readFile((char *)"weights_nobias/b1.h", dense1_bias);
    MatrixInit3D(dense1_data, ndd1,pdd1,ldd1,1);

    MatrixInit3D(dense2_weight, ndw2,pdw2,ldw2,1);
    readFile((char *)"weights_nobias/w2.h", dense2_weight);
    MatrixInit3D(dense2_bias, ndb2,pdb2,ldb2,1);
    readFile((char *)"weights_nobias/b2.h", dense2_bias);
    MatrixInit3D(dense2_data, ndd2,pdd2,ldd2,1);

    MatrixInit3D(dense3_weight, ndw3,pdw3,ldw3,1);
    readFile((char *)"weights_nobias/w3.h", dense3_weight);
    MatrixInit3D(dense3_bias, ndb3,pdb3,ldb3,1);
    readFile((char *)"weights_nobias/b3.h", dense3_bias);
    MatrixInit3D(dense3_data, ndd3,pdd3,ldd3,1);

    //Pour tester mes convolutions, création de kernels naifs : 
    // C1_kernel[12]=2;
    // C1_kernel[5*25+12]=1;

    // Conv2_kernel[12]=1;
    // Conv2_kernel[25+12]=1;
    // Conv2_kernel[92*25+12]=1;

    // Copie de la matrice du CPU vers le GPU
    cudaMemcpy(d_raw_data, raw_data, nr * pr * lr*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_data, C1_data, nC1 * pC1 * lC1* sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_S1_data, S1_data, nS1 * pS1 * lS1*  sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, nk * pk * lk* sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_Conv2_data, Conv2_data, nC2 * pC2 * lC2* sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_S2_data, S2_data, nS2 * pS2 * lS2*  sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Conv2_kernel, Conv2_kernel, nk2 * pk2 * lk2*depthk2* sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flatten_data, flatten_data, nf * pf * lf* sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_dense1_weight, dense1_weight, ndw1 * pdw1 * ldw1* sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dense1_bias, dense1_bias, ndb1 * pdb1 * ldb1* sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dense1_data, dense1_data, ndd1 * pdd1 * ldd1* sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_dense2_weight, dense2_weight, ndw2 * pdw2 * ldw2* sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dense2_bias, dense2_bias, ndb2 * pdb2 * ldb2* sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dense2_data, dense2_data, ndd2 * pdd2 * ldd2* sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_dense3_weight, dense3_weight, ndw3 * pdw3 * ldw3* sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dense3_bias, dense3_bias, ndb3 * pdb3 * ldb3* sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dense3_data, dense3_data, ndd3 * pdd3 * ldd3* sizeof(double), cudaMemcpyHostToDevice);

    dim3 gridDim(nC1,nC1,lC1);
    Convolution2DGPU<<<gridDim,1>>>(d_raw_data, d_C1_kernel, d_C1_data,nr,nk);

    dim3 gridDim2(nS1,pS1,lS1);
    Moyennage2DGPU<<<gridDim2,1>>>(d_C1_data, d_S1_data, nC1);

    dim3 gridDim3(nC2,nC2,lC2);
    Convolution3DGPU<<<gridDim3,1>>>(d_S1_data, d_Conv2_kernel, d_Conv2_data,nS1,nk2,depthk2);

    dim3 gridDim4(nS2,pS2,lS2);
    Moyennage2DGPU<<<gridDim4,1>>>(d_Conv2_data, d_S2_data, nC2);

    dim3 gridDim5(nS2,pS2,lS2);
    Flatten<<<gridDim5,1>>>(d_S2_data, d_flatten_data);

    dim3 gridDim6(ndd1,pdd1,ldd1);
    cudaMatrixMult<<<gridDim6,1>>>(d_flatten_data, d_dense1_weight, d_dense1_data, nf);

    dim3 gridDim7(ndd1,pdd1,ldd1);
    cudaMatrixAdd<<<gridDim7,1>>>(d_dense1_data, d_dense1_bias, d_dense1_data);

    dim3 gridDim8(ndd2,pdd2,ldd2);
    cudaMatrixMult<<<gridDim8,1>>>(d_dense1_data, d_dense2_weight, d_dense2_data, ndd1);

    dim3 gridDim9(ndd2,pdd2,ldd2);
    cudaMatrixAdd<<<gridDim9,1>>>(d_dense2_data, d_dense2_bias, d_dense2_data);

    dim3 gridDim10(ndd3,pdd3,ldd3);
    cudaMatrixMult<<<gridDim10,1>>>(d_dense2_data, d_dense3_weight, d_dense3_data, ndd2);

    dim3 gridDim11(ndd3,pdd3,ldd3);
    cudaMatrixAdd<<<gridDim11,1>>>(d_dense3_data, d_dense3_bias, d_dense3_data);

    cudaMemcpy(raw_data, d_raw_data, nr * pr * lr* sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(C1_data, d_C1_data, nC1 * pC1 * lC1*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(S1_data, d_S1_data, nS1 * pS1 * lS1* sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(C1_kernel, d_C1_kernel, nk * pk * lk* sizeof(double), cudaMemcpyDeviceToHost);

    cudaMemcpy(Conv2_data, d_Conv2_data, nC2 * pC2 * lC2*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(S2_data, d_S2_data, nS2 * pS2 * lS2* sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Conv2_kernel, d_Conv2_kernel, nk2 * pk2 * lk2*depthk2* sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(flatten_data, d_flatten_data, nf * pf * lf* sizeof(double), cudaMemcpyDeviceToHost);

    cudaMemcpy(dense1_weight, d_dense1_weight, ndw1 * pdw1 * ldw1* sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(dense1_bias, d_dense1_bias, ndb1 * pdb1 * ldb1* sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(dense1_data, d_dense1_data, ndd1 * pdd1 * ldd1* sizeof(double), cudaMemcpyDeviceToHost);

    cudaMemcpy(dense2_weight, d_dense2_weight, ndw2 * pdw2 * ldw2* sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(dense2_bias, d_dense2_bias, ndb2 * pdb2 * ldb2* sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(dense2_data, d_dense2_data, ndd2 * pdd2 * ldd2* sizeof(double), cudaMemcpyDeviceToHost);

    cudaMemcpy(dense3_weight, d_dense3_weight, ndw3 * pdw3 * ldw3* sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(dense3_bias, d_dense3_bias, ndb3 * pdb3 * ldb3* sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(dense3_data, d_dense3_data, ndd3 * pdd3 * ldd3* sizeof(double), cudaMemcpyDeviceToHost);


    // Affichage de la matrice sur le CPU
    // printf("Matrice raw_data :\n");

    // MatrixPrint3D(raw_data, nr, pr,lr);

    // printf("Matrice C1_data :\n");

    // MatrixPrint3D(C1_data, nC1, pC1,lC1);

    // printf("Matrice S1_data :\n");

    // MatrixPrint3D(S1_data, nS1, pS1,lS1);

    // printf("Matrice C1_kernel :\n");

    // MatrixPrint3D(C1_kernel, nk, pk,lk);

    // printf("Matrice Conv2_data :\n");

    // MatrixPrint3D(Conv2_data, nC2, pC2,lC2);

    // printf("Matrice S2_data :\n");

    // MatrixPrint3D(S2_data, nS2, pS2,lS2);

    // printf("Matrice Conv2_kernel :\n");

    // MatrixPrint3D(Conv2_kernel, nk2, pk2,lk2*depthk2);

    // printf("Matrice flatten_matrix :\n");

    // MatrixPrint3D(flatten_data, nf, pf,lf);

    // printf("Matrice dense1_weight :\n");

    // MatrixPrint3D(dense1_weight, ndw1, pdw1,ldw1);

    // printf("Matrice dense1_bias :\n");

    // MatrixPrint3D(dense1_bias, ndb1, pdb1,ldb1);

    // printf("Matrice dense1_data :\n");

    // MatrixPrint3D(dense1_data, ndd1, pdd1,ldd1);

    // printf("Matrice dense3_weight :\n");

    // MatrixPrint3D(dense3_weight, ndw3, pdw3,ldw3);

    // printf("Matrice dense3_bias :\n");

    // MatrixPrint3D(dense3_bias, ndb3, pdb3,ldb3);

    printf("Matrice dense3_data :\n");

    MatrixPrint3D(dense3_data, ndd3, pdd3,ldd3);

    // Libération de la mémoire sur le CPU et le GPU
    free(raw_data);
    cudaFree(d_raw_data);

    free(C1_data);
    cudaFree(d_C1_data);

    free(S1_data);
    cudaFree(d_S1_data);

    free(C1_kernel);
    cudaFree(d_C1_kernel);

    free(Conv2_data);
    cudaFree(d_Conv2_data);

    free(S2_data);
    cudaFree(d_S2_data);

    free(Conv2_kernel);
    cudaFree(d_Conv2_kernel);

    free(flatten_data);
    cudaFree(d_flatten_data);

    free(dense1_weight);
    cudaFree(d_dense1_weight);
    free(dense2_weight);
    cudaFree(d_dense2_weight);
    free(dense3_weight);
    cudaFree(d_dense3_weight);
    
    free(dense1_bias);
    cudaFree(d_dense1_bias);
    free(dense2_bias);
    cudaFree(d_dense2_bias);
    free(dense3_bias);
    cudaFree(d_dense3_bias);

    free(dense1_data);
    cudaFree(d_dense1_data);
    free(dense2_data);
    cudaFree(d_dense2_data);
    free(dense3_data);
    cudaFree(d_dense3_data);

    return 0;

}

