#ifndef FOREPASS_H
#define FOREPASS_H

void encodeX_Y(char *list,char *alphabet,int *X,int *Y);
void embedCX(float *emb,float *c,int *x,int numBigrams);
void addBias(float *A,float *B,int rows,int cols);
float *getCounts(float *logits, int rows, int cols);
float *getProbs(float *counts, int rows, int cols);
void getMiniBatch(char *allNames,int *allX,int *allY,int *X,int *Y);
float *softMax(float *logits, int rows, int cols);
float *getActuals(float *probs, int *Y, int rows, int cols);
float crossEntropy(float *logits, int *Y, int rows, int cols);
void addTanh(float *H,int rows,int cols);

#endif