#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "matrix.h"

float *AxB(float *A,float *B,int rows,int cols){
    // element-wise multiplication of C = A x B
    float *C = (float *)calloc(rows*cols,sizeof(float));
    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
            *(C+i*cols+j) = *(A+i*cols+j) * *(B+i*cols+j);
    return C;
}
float *multAB(float *A, float *B, int ai, int aj, int bi, int bj) {
    // matrix multiplication
    float sum;
    float *C = (float *)calloc(ai*bj,sizeof(float));
    for(int i=0;i<ai;i++)
        for(int j=0;j<bj;j++){
            sum = 0.0;
            for(int k=0;k<aj;k++){
                sum += *(A+i*bi+k) * *(B+k*bj+j);
            }
            *(C+i*bj+j) = sum;
        }
    return C;
}
void multMxV(float *M,float *V,int rows,int cols){
    // element-wise multiplication of a Matrix by a Vector
    // length of Vector == length of Matrix row
    // result is M
    for(int i=0;i<rows;i++){
       // printf("i = %d\n",i);
        for(int j=0;j<cols;j++)
            *(M+i*cols+j) = *(M+i*cols+j) * V[j];
    }
    return;
}
float *multMV(float *M,float *V,int rows,int cols){
    // element-wise multiplication of a Matrix by a Vector
    // length of Vector === length of Matrix row
    // returns a Matrix
    float *result=(float *)calloc(rows*cols,sizeof(float));
    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
            *(result+i*cols+j) = V[j] * *(M+i*cols+j);
    return result;
}
float *multVxM(float *V,float *M,int rows,int cols){
    // element-wise multiplication of a Vector by a Matrix
    // length of Vector === length of Matrix row
    // result is a Matrix
    float *R = (float *)calloc(rows*cols,sizeof(float)); 
    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
            *(R+i*cols+j) += V[j] * *(M+i*cols+j);
    return R;
}
float *multVM(float *V,float *M,int rows,int cols){
    // element-wise multiplication of a Vector by a Matrix
    // length of Vector === length of Matrix row
    // result is a Vector
    float *result=(float *)calloc(cols,sizeof(float));
    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
            result[j] += V[j] * *(M+i*cols+j);
    return result;
}
float *AxBsum(float *A,float *B,int rows,int cols){
    // element-wise multiplication of C = A x B
    float *C = (float *)calloc(rows*cols,sizeof(float));
    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
            *(C+i*cols+j) = *(A+i*cols+j) * *(B+i*cols+j);
    // sum the columns giving R
    float *R=(float *)calloc(cols,sizeof(float));
    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
            R[j] += *(C+i*cols+j);
    free(C);
    return R;
}
float *VxMsum(float *V, float *M, int rows, int cols){
    // multiply vector X matrix and sum the columns
    float *R=(float *)calloc(cols,sizeof(float));
    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
            R[j] += V[j] * *(M+i*cols+j);
    return R;
}
float *transpose(float *a,int r, int c){
    // transpose a matrix
    float *t = (float *)calloc(r*c,sizeof(float));
    for (int i=0;i<r;i++) {
		for (int j=0;j<c;j++)
			*(t+j*r+i) = *(a+i*c+j);
	}
    return t;
}
void addV2V(float *to,float *fm,int length){
    for(int i=0;i<length;i++)
        to[i] += fm[i];
    return;
}
float *multVxV(float *to,float *fm,int length){
    float *R=(float *)calloc(length,sizeof(float));
    for(int i=0;i<length;i++)
        R[i] = to[i] * fm[i];
    return R;
}
void addM2M(float *to,float *fm, int rows, int cols){
    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
            *(to+i*cols+j) += *(fm+i*cols+j);
    return;
}
void addV2M(float *fm, float *to, int rows, int cols){
    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
            *(to+i*cols+j) += fm[j];
    return;
}
float *sumCols(float *M, int rows, int cols){
    float *tot=(float *)calloc(cols,sizeof(float));
    float *aColumn;
    for(int j=0;j<cols;j++){
        aColumn = getColumn(M,rows,cols,j);
        //tot[j] = sumKahan(aColumn,rows);
        //tot[j] = KahanBabushkaNeumaierSum(aColumn,rows);
        //tot[j] = sumIntegers(aColumn,rows);
        tot[j] = pairWiseSum(aColumn,0,rows-1);
        free(aColumn);
    }
    return tot;
}
float *getColumn(float *M, int rows, int cols, int j){
    float *aColumn=(float *)calloc(rows,sizeof(float));
    for(int i=0;i<rows;i++)
        aColumn[i] = *(M+i*cols+j);
    return aColumn;
}
float sumKahan(float *M, int len){
    // Kahan summation algorithm
    float sum = 0.0;
    float c = 0.0;
    float y;
    float t;
    for(int i=0;i<len;i++){
        y = *(M+i) - c;
        t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}
float KahanBabushkaNeumaierSum(float *M, int len){
    float sum = 0.0;
    float c = 0.0;
    float t;
    for(int i=0;i<len;i++){
        t = sum + M[i];
        if(fabsf(sum) >= fabsf(M[i]))
            c += (sum - t) + M[i];
        else
            c += (M[i] - t) + sum;
        sum = t;
    }
    return sum + c;
}
float sumIntegers(float data[], int len){
	float m[len];
	int e[len];
	int se=100;
	for(int i=0;i<len;i++){
		m[i]=frexpf(data[i],&e[i]);
		if(se > e[i]) se=e[i];
	}
	for(int i=0;i<len;i++){
		if(e[i]!=se){
			m[i]=ldexpf(m[i],e[i]-se);
			e[i]=se;
		}
	}	
	int M[len];
	for(int i=0;i<len;i++){
		m[i] = m[i]*10000000;
		M[i] = m[i];
	}
	int total=0;
	for(int i=0;i<len;i++)
		total+=M[i];
	return ldexpf((float)total/10000000.0,se);
}
float pairWiseSum(float arr[], int start, int end) {
	// Used by np.sum(a,0)
    if (start == end) {
        return arr[start];
    } else if (start + 1 == end) {
        return arr[start] + arr[end];
    } else {
        int mid = (start + end) / 2;
        float leftSum = pairWiseSum(arr, start, mid);
        float rightSum = pairWiseSum(arr, mid + 1, end);
        return leftSum + rightSum;
    }
}
