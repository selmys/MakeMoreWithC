#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "printers.h"
#include "helpers.h"
#include "params.h"

void printInt(const char *title, int *matrix, int rows, int cols){
    printf("========== %s ===========\n",title);
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++)
            printf("%3d ",*(matrix+i*cols+j));
        printf("\n");
    }
    return;
}
void print(const char *title, float *matrix, int rows, int cols){
    printf("========== %s ===========\n",title);
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++)
            printf(" %.6f",*(matrix+i*cols+j));
        printf("\n");
    }
    return;
}
void printE(const char *title, float *matrix, int rows, int cols){
    printf("========== %s ===========\n",title);
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++)
            printf(" %.5e",*(matrix+i*cols+j));
        printf("\n");
    }
    return;
}
void printEncodedBigrams(int *X,int *Y){
    printf("=========== mini batch ============\n");
    for(int i=0;i<BATCHSIZE;i++){
        for(int j=0;j<BLOCKSIZE;j++)
            printf("%3d ",*(X+i*BLOCKSIZE+j));
        printf(" ---> %d\n",*(Y+i));
    }
}
void printEmb(float *emb){
    printf("=========== emb ============\n");
    for(int i=0;i<BATCHSIZE;i++){
        for(int j=0;j<DIMENSIONS*BLOCKSIZE;j++)
            printf("%7.4lf ",*(emb+i*DIMENSIONS*BLOCKSIZE+j));
        printf("\n");
    }
    return;
}
void printHistory(int history[][4]){
    for(int i=0;i<MERGES;i++){
        for(int j=0;j<4;j++)
            printf("%6d ",history[i][j]);
        printf("\n");
    }
    return;
}
void printVocab(char **vocab,int lenVocab){
    for(int i=0;i<lenVocab;i++){
        fprintf(stderr,"%6d '%s'\n",i,vocab[i]);
    }
    return;
}