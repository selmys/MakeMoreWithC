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
    for(int i=0;i<BATCHSIZE;i++){
        for(int j=0;j<BLOCKSIZE;j++)
            printf("%3d",*(X+i*BLOCKSIZE+j));
        printf(" ---> %d\n",*(Y+i));
    }
}
void printBigrams(char *names){
    int numNames=getNumNames(names);
    int k,len;
    char *s=(char *)calloc(30,sizeof(char)); 
    // max length of any name assumed to be 30
    for(int i=1;i<numNames+1;i++){
        char *w=getOneName(i,names);
        printf("%s\n",w);
        *s=0;
        strcat(s,"...");
        strcat(s,w); 
        strcat(s,".");
        len=strlen(w);
        for(int j=0;j<len+1;j++){
            for(k=0;k<BLOCKSIZE;k++)
                printf("%c",s[k]);
            printf(" ---> %c\n",s[k]);
            s=s+1;
        }
    }
}
void printEmb(float *emb, int l, int w, int h){
    printf("=========== emb ============\n");
    for(int i=0;i<l;i++){
        for(int j=0;j<w;j++){
            for(int k=0;k<h;k++){
                printf("%7.4lf ",*(emb+i*w*h+j*h+k));
            }
            printf("\n");
        }
        printf("\n");
    }
    return;
}
