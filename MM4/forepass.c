#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "forepass.h"
#include "helpers.h"
#include "params.h"

void encodeX_Y(char *list,char *alphabet,int *X,int *Y){
    // list is a string of period delimited names
    // encoding bigrams into X and Y
    char *aName=NULL; 
    int lenName;
    int ix;
    int m;
    int pX=0,pY=0;
    int context[BLOCKSIZE];
    int numNames=getNumNames(list);
    for(int i=1;i<=numNames;i++){
        for(int j=0;j<BLOCKSIZE;j++)
            context[j]=0;
        if(aName != NULL)free(aName);
        aName=getOneName(i,list);
        lenName=strlen(aName);
        for(m=0;m<lenName;m++){
            for(int k=0;k<BLOCKSIZE;k++)
                X[pX++]=context[k];
            ix=getIndex(alphabet,aName[m]);
            Y[pY++]=ix;
            for(int l=0;l<BLOCKSIZE-1;l++)
                context[l]=context[l+1];
            context[BLOCKSIZE-1]=ix;
        }
        Y[pY++]=0;
        for(int k=0;k<BLOCKSIZE;k++)
            X[pX++]=context[k];
    }
    return;
}
void embedCX(float *emb,float *c,int *x,int numBigrams){
    // homebrew embed function
    // step through all values in X (i = numBigrams X BLOCKSIZE)
    // take the vector from the lookup table (c)
    // and place it into the embedding table (emb)
    int k=0;
    for(int i=0;i<numBigrams*BLOCKSIZE;i++)
        for(int j=0;j<DIMENSIONS;j++)
            emb[k++] = c[x[i]*DIMENSIONS+j];
    return;
}

void addBias(float *A,float *B,int rows,int cols){
    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
            *(A+i*cols+j) += *(B+j);
    return;
}
float *getCounts(float *logits, int rows, int cols){
    // change log counts (logits) to counts
    // first normalize the logits by subtracting each from the maximum
    // so let's find the maximums first
    float *maxes=(float *)calloc(rows,sizeof(float));
    for(int i=0;i<rows;i++){
        maxes[i] = *(logits+i*cols);
        for(int j=1;j<cols;j++)
            if (*(logits+i*cols+j) > maxes[i])
                maxes[i] = *(logits+i*cols+j);
    }
    // now subtract each logit from the maximum
    float *normal_logits=(float *)calloc(rows*cols,sizeof(float));
    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
            *(normal_logits+i*cols+j) = *(logits+i*cols+j) - maxes[i];
    // finally find the counts by raising exp
    float *counts=(float *)calloc(rows*cols,sizeof(float));
    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
            *(counts+i*cols+j) = expf(*(normal_logits+i*cols+j));
    free(maxes);free(normal_logits);
    return counts;
}
float *getProbs(float *counts, int rows, int cols){
    // change counts to probabilities
    float *probs=(float *)calloc(rows*cols,sizeof(float));
    float sum;
    for(int i=0;i<rows;i++){
        sum=0.0;
        for(int j=0;j<cols;j++)
            sum += *(counts+i*cols+j);
        for(int j=0;j<cols;j++)
            *(probs+i*cols+j) = *(counts+i*cols+j)/sum;
    }
    return probs;
}
void getMiniBatch(char *allNames,int *allX,int *allY,int *X,int *Y){
    int p,q;
    for(int i=0;i<BATCHSIZE;i++){    
        p=rand()%(strlen(allNames)-1); 
        for(q=0;q<BLOCKSIZE;q++)
            *(X+i*BLOCKSIZE+q) = *(allX+p*BLOCKSIZE+q);
        Y[i] = allY[p];
    }
    return;
}
float *softMax(float *logits, int rows, int cols){
    // get the counts
    float *counts=getCounts(logits,rows,cols);
    // get the probabilities
    float *probs=getProbs(counts,rows,cols);
    free(counts);
    return probs;
}
float *getActuals(float *probs, int *Y, int rows, int cols){
    float *probArray=(float *)calloc(rows,sizeof(float));
    for(int i=0;i<rows;i++)
        probArray[i] = *(probs+i*cols+Y[i]);
    return probArray;
}
float crossEntropy(float *logits, int *Y, int rows, int cols){
    // get the probabilities
    float *probs=softMax(logits,rows,cols);
    // get actual results from the nn
    float *A=getActuals(probs,Y,rows,cols);
    // now get the loss
    // add up the negative log likelihood of the result probabilities
    float nll=0.0;
    for(int i=0;i<rows;i++)
        nll += -logf(A[i]);
    // so the average nll is the loss
    float loss = nll/rows;
    free(probs);free(A);
    return loss;
}
void addTanh(float *H,int rows,int cols){
    // tanh is our sigmoid function
    for(int i=0;i<rows;i++)
       for(int j=0;j<cols;j++)
           *(H+i*cols+j) = tanhf(*(H+i*cols+j));
    return;
}
