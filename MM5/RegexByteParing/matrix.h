#ifndef MATRIX_H
#define MATRIX_H

float *AxB(float *A,float *B,int rows,int cols);
float *multAB(float *A, float *B, int ai, int aj, int bi, int bj);
void multMxV(float *M,float *V,int rows,int cols);
float *multMV(float *M,float *V,int rows,int cols);
float *multVxM(float *V,float *M,int rows,int cols);
float *multVM(float *V,float *M,int rows,int cols);
float *AxBsum(float *A,float *B,int rows,int cols);
float *VxMsum(float *V, float *M, int rows, int cols);
float *transpose(float *a,int r, int c);
void addV2V(float *to,float *fm,int length);
float *multVxV(float *to,float *fm,int length);
void addM2M(float *to,float *fm, int rows, int cols);
void addV2M(float *fm, float *to, int rows, int cols);
float *sumCols(float *M, int rows, int cols);
float *getColumn(float *M, int rows, int cols, int j);
float sumKahan(float *M, int len);
float KahanBabushkaNeumaierSum(float *M, int len);
float sumIntegers(float data[], int len);
float pairWiseSum(float arr[], int start, int end);

#endif